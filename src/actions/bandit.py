"""
bandit.py
---------
The adaptive layer: learns, per (cognitive_state, action), whether YOU
actually want that action to keep firing — using Thompson Sampling, a
standard multi-armed-bandit algorithm well-suited to sparse, binary
feedback (accept/reject) rather than the huge data volumes deep RL needs.

WHY THIS SHAPE, SPECIFICALLY:
Rather than picking one "best" action among competitors, each
(state, action) pair is its OWN independent Bernoulli bandit — a coin
with an unknown bias representing "does this user like this action when
in this state." That matches how you actually described the problem:
"which contrasts are better, which notifications should be blocked" —
independent per-action preferences, not a single ranked competition.

HOW IT WORKS:
Each arm's belief is a Beta(alpha, beta) distribution over "probability
you'll accept this action." Starts at Beta(1,1) — a flat, uninformed
prior (equal chance of any acceptance rate). Every piece of feedback
updates it: accept -> alpha += 1, reject -> beta += 1. The distribution
narrows and shifts toward your actual preference as evidence accumulates.

COLD START: for the first `cold_start_trials` firings of a given
(state, action) pair, we always fire it (matching the static config
exactly) rather than gating on an unlearned belief — you need SOME data
before the bandit is allowed to start deviating from the defaults we
already know are research-grounded.

EXPLORATION: Thompson Sampling never fully "shuts off" an arm — even an
arm with mostly rejections still has some (shrinking but nonzero) chance
of firing, because we sample a probability from the *distribution*, not
just its mean. This means the system can still notice if your
preferences change later, without needing a separate epsilon-exploration
mechanism bolted on.

Everything is local SQLite — no data leaves the machine.
"""

import logging
import os
import random
import sqlite3
import threading
import time

logger = logging.getLogger("cognitive_monitor.bandit")


class BanditStore:
    def __init__(self, db_path: str, cold_start_trials: int = 5):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.cold_start_trials = cold_start_trials
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS bandit_arms (
                    state TEXT NOT NULL,
                    action_key TEXT NOT NULL,
                    alpha REAL NOT NULL DEFAULT 1.0,
                    beta REAL NOT NULL DEFAULT 1.0,
                    trials INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (state, action_key)
                )
            """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS action_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    state TEXT NOT NULL,
                    action_key TEXT NOT NULL,
                    feedback TEXT,
                    feedback_timestamp TEXT
                )
            """)
            self._conn.commit()

    # ------------------------------------------------------------------

    def _get_arm(self, state: str, action_key: str):
        with self._lock:
            cur = self._conn.execute(
                "SELECT alpha, beta, trials FROM bandit_arms WHERE state=? AND action_key=?",
                (state, action_key),
            )
            row = cur.fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO bandit_arms (state, action_key, alpha, beta, trials) "
                    "VALUES (?, ?, 1.0, 1.0, 0)",
                    (state, action_key),
                )
                self._conn.commit()
                return (1.0, 1.0, 0)
            return row

    def decide(self, state: str, action_key: str) -> bool:
        """
        Returns True if this action should fire right now. During cold
        start, always True (defers to the static config). After that,
        draws a sample from the learned Beta distribution and fires
        probabilistically based on it (Thompson Sampling).
        """
        alpha, beta, trials = self._get_arm(state, action_key)

        if trials < self.cold_start_trials:
            return True

        try:
            sampled_p = random.betavariate(alpha, beta)
        except ValueError:
            # betavariate requires alpha,beta > 0; guard against any
            # corrupted row rather than crashing the daemon over this.
            return True

        fire = random.random() < sampled_p
        logger.debug(
            "Bandit decision: state=%s action=%s alpha=%.1f beta=%.1f "
            "sampled_p=%.2f -> %s",
            state, action_key, alpha, beta, sampled_p, "FIRE" if fire else "SKIP",
        )
        return fire

    def should_prompt_feedback(self, state: str, action_key: str, max_prompts: int = 15) -> bool:
        """
        Whether to bother asking for feedback on this firing. Tapers off
        once an arm has enough trials to have a reasonably converged
        belief — asking forever, on every single firing, would make the
        feedback mechanism itself the annoyance it's meant to prevent.
        """
        _, _, trials = self._get_arm(state, action_key)
        return trials < max_prompts

    def record_fired(self, state: str, action_key: str) -> int:
        """Call after an action actually fires. Returns an event_id that
        feedback (from a notification button, CLI command, or dashboard)
        can later reference to close the loop."""
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO action_events (timestamp, state, action_key) VALUES (?, ?, ?)",
                (time.strftime("%Y-%m-%d %H:%M:%S"), state, action_key),
            )
            self._conn.commit()
            return cur.lastrowid

    def record_feedback(self, event_id: int, accepted: bool) -> bool:
        """
        Closes the loop on a specific fired action: updates that event's
        row AND updates the arm's belief. Returns False if event_id
        doesn't exist (e.g. typo'd via CLI) rather than raising.
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT state, action_key, feedback FROM action_events WHERE id=?",
                (event_id,),
            )
            row = cur.fetchone()
            if row is None:
                logger.warning("record_feedback: no action_event with id=%d", event_id)
                return False
            state, action_key, existing_feedback = row
            if existing_feedback is not None:
                logger.info("event_id=%d already has feedback (%s); overwriting.",
                            event_id, existing_feedback)

            feedback_str = "up" if accepted else "down"
            self._conn.execute(
                "UPDATE action_events SET feedback=?, feedback_timestamp=? WHERE id=?",
                (feedback_str, time.strftime("%Y-%m-%d %H:%M:%S"), event_id),
            )

            delta_alpha = 1.0 if accepted else 0.0
            delta_beta = 0.0 if accepted else 1.0
            self._conn.execute(
                "UPDATE bandit_arms SET alpha = alpha + ?, beta = beta + ?, trials = trials + 1 "
                "WHERE state=? AND action_key=?",
                (delta_alpha, delta_beta, state, action_key),
            )
            self._conn.commit()
            logger.info("Feedback recorded: event_id=%d state=%s action=%s -> %s",
                        event_id, state, action_key, feedback_str)
            return True

    # ------------------------------------------------------------------

    def get_stats(self):
        """Returns all arms with their current learned acceptance rate —
        used by the CLI stats command and, later, the dashboard."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT state, action_key, alpha, beta, trials FROM bandit_arms "
                "ORDER BY state, action_key"
            )
            rows = cur.fetchall()
        stats = []
        for state, action_key, alpha, beta, trials in rows:
            mean_acceptance = alpha / (alpha + beta)
            stats.append({
                "state": state,
                "action_key": action_key,
                "alpha": alpha,
                "beta": beta,
                "trials": trials,
                "learned_acceptance_rate": mean_acceptance,
            })
        return stats

    def get_pending_events(self, limit: int = 20):
        """Fired actions still awaiting feedback — useful for a CLI/dashboard
        prompt: 'here's what fired recently, tell me what you thought.'"""
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, timestamp, state, action_key FROM action_events "
                "WHERE feedback IS NULL ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            return cur.fetchall()

    def close(self):
        with self._lock:
            self._conn.close()
