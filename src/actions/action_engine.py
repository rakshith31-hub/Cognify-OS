"""
action_engine.py
-----------------
Turns a raw per-window prediction ({state, confidence, all_probabilities})
into actual system actions, applying three guards so the "help" doesn't
itself become a source of interruption:

  1. min_confidence_to_act   - ignore low-confidence predictions entirely.
  2. consecutive_windows_required - require the state to persist across
     N consecutive 10s windows before acting (debounce noisy per-window
     predictions).
  3. action_cooldown_seconds - don't refire the same action more than once
     per cooldown window, and respect `only_once_per_session` for actions
     that should really only happen once until the state changes away and
     back.

When a DND-triggering state ends, it reverts (DND off, brightness restored)
if `revert_on_state_change` is set — see _maybe_revert().
"""

import logging
import time

from src.actions import notifications, ui_controller
from src.actions.bandit import BanditStore

logger = logging.getLogger("cognitive_monitor.action_engine")


class ActionEngine:
    def __init__(self, config: dict, bandit_db_path: str = None, notification_logger=None):
        self.config = config
        self.notification_logger = notification_logger
        general = config.get("general", {})
        self.consecutive_required = general.get("consecutive_windows_required", 3)
        self.cooldown_seconds = general.get("action_cooldown_seconds", 300)
        self.min_confidence = general.get("min_confidence_to_act", 0.55)
        self.revert_on_change = config.get("revert_on_state_change", True)
        # After this many CONSECUTIVE low-confidence windows, treat it as
        # "no longer a confirmed state" and revert whatever was last
        # committed (DND/brightness/etc), instead of leaving stale actions
        # from the last confident reading active indefinitely. This is the
        # common case when the user steps away / isn't visible to the
        # camera: no-face input produces noisy, sub-threshold predictions
        # across several classes rather than one confident state, so it
        # could never otherwise win the confidence+streak gate needed to
        # replace the old committed state.
        self.low_confidence_revert_after = general.get("low_confidence_revert_after", 3)

        self.bandit = None
        if bandit_db_path:
            self.bandit = BanditStore(
                bandit_db_path,
                cold_start_trials=general.get("bandit_cold_start_trials", 5),
            )
        self.bandit_feedback_max_prompts = general.get("bandit_feedback_max_prompts", 15)

        self._streak_state = None
        self._streak_count = 0
        self._low_confidence_streak = 0
        self._last_fired_at = {}  # action key -> timestamp
        self._fired_once_this_session = set()  # action keys with only_once_per_session
        self._dnd_active = False
        self._color_filter_active = False
        self._high_contrast_active = False
        self._taskbar_hidden = False
        self._current_committed_state = None  # the state we last actually acted on

    def process(self, result: dict) -> str:
        """
        Feed one prediction in. Returns a short string describing what
        action (if any) fired, for logging purposes.
        """
        state = result.get("state")
        confidence = result.get("confidence", 0.0)

        if confidence < self.min_confidence:
            self._low_confidence_streak += 1
            logger.debug("Confidence %.2f < min %.2f — no action (%d/%d low-confidence streak).",
                         confidence, self.min_confidence,
                         self._low_confidence_streak, self.low_confidence_revert_after)
            self._reset_streak()
            if (self._low_confidence_streak >= self.low_confidence_revert_after
                    and self._current_committed_state is not None):
                logger.info(
                    "%d consecutive low-confidence windows — reverting stale '%s' actions "
                    "(likely no reliable read: camera/away, or a genuinely ambiguous state).",
                    self._low_confidence_streak, self._current_committed_state,
                )
                self._maybe_revert(previous_state=self._current_committed_state)
                self._current_committed_state = None
                self._low_confidence_streak = 0
            return ""
        self._low_confidence_streak = 0

        # ---- hysteresis: require a persistent streak of the same state ----
        if state == self._streak_state:
            self._streak_count += 1
        else:
            self._streak_state = state
            self._streak_count = 1

        if self._streak_count < self.consecutive_required:
            return ""

        # streak confirmed -> this is now the "committed" state
        if state != self._current_committed_state:
            self._maybe_revert(previous_state=self._current_committed_state)
            self._current_committed_state = state

        fired = self._run_actions_for_state(state)
        return fired

    # ------------------------------------------------------------------

    def _reset_streak(self):
        self._streak_state = None
        self._streak_count = 0

    def _run_actions_for_state(self, state: str) -> str:
        state_cfg = self.config.get("states", {}).get(state)
        if not state_cfg:
            return ""

        fired_descriptions = []
        now = time.time()

        for action in state_cfg.get("actions", []):
            action_type = action.get("type")
            key = f"{state}:{action_type}:{action.get('title', '')}"

            if action.get("only_once_per_session") and key in self._fired_once_this_session:
                continue

            last_fired = self._last_fired_at.get(key, 0)
            if now - last_fired < self.cooldown_seconds and not action.get("only_once_per_session"):
                continue

            if self.bandit is not None and not self.bandit.decide(state, action_type):
                logger.info("Bandit skipped '%s' for state=%s (learned preference from your feedback).",
                            action_type, state)
                continue

            if self._execute(action_type, action, state):
                self._last_fired_at[key] = now
                if action.get("only_once_per_session"):
                    self._fired_once_this_session.add(key)

                event_id = None
                if self.bandit is not None:
                    event_id = self.bandit.record_fired(state, action_type)
                    if self.bandit.should_prompt_feedback(state, action_type, self.bandit_feedback_max_prompts):
                        notifications.request_feedback(
                            event_id,
                            title=f"Was '{action_type}' helpful?",
                            message=f"Fired because you were predicted '{state}'.",
                        )

                fired_descriptions.append(f"{action_type}#{event_id}" if event_id else action_type)

        return ",".join(fired_descriptions)

    def _log_notification(self, kind: str, state: str, title: str, message: str = ""):
        if self.notification_logger is not None:
            try:
                self.notification_logger.log(kind, state, title, message)
            except Exception:
                pass  # logging must never break the action itself

    def _execute(self, action_type: str, action: dict, state: str) -> bool:
        try:
            if action_type == "dnd_on":
                notifications.set_do_not_disturb(True)
                self._dnd_active = True
                return True

            elif action_type == "dnd_off":
                notifications.set_do_not_disturb(False)
                self._dnd_active = False
                return True

            elif action_type == "notify":
                # If our own DND is active, we still show a single state-
                # change notice (that's the point of it) but the daemon
                # never floods notifications while DND is on.
                title = action.get("title", "Cognitive Monitor")
                message = action.get("message", f"State: {state}")
                if action.get("with_tone"):
                    # Same short-tone treatment as break_popup, for states
                    # where a silent toast alone is too easy to miss.
                    notifications.show_break_popup(title, message)
                else:
                    notifications.show_notification(title, message)
                self._log_notification("notify", state, title, message)
                return True

            elif action_type == "brightness_step":
                ui_controller.step_brightness(
                    delta_percent=action.get("delta", -10),
                    floor=action.get("floor", 20),
                )
                return True

            elif action_type == "color_filter":
                ui_controller.set_color_filter(
                    enabled=True,
                    filter_type=action.get("filter_type", 0),
                )
                self._color_filter_active = True
                return True

            elif action_type == "high_contrast":
                ui_controller.set_high_contrast(True)
                self._high_contrast_active = True
                return True

            elif action_type == "taskbar_hide":
                ui_controller.set_taskbar_autohide(True)
                self._taskbar_hidden = True
                return True

            elif action_type == "alarm":
                pattern = action.get("pattern", "triple_beep")
                notifications.play_alarm(pattern)
                self._log_notification("alarm", state, f"alarm:{pattern}")
                return True

            elif action_type == "break_popup":
                title = action.get("title", "Time for a break")
                message = action.get("message", "Consider stepping away for a few minutes.")
                notifications.show_break_popup(title, message)
                self._log_notification("break_popup", state, title, message)
                return True

            elif action_type == "block_domains":
                if not action.get("enabled", False):
                    return False
                domains = action.get("domains", [])
                if not domains:
                    return False
                ui_controller.block_domains(domains, action.get("duration_minutes", 20))
                return True

            elif action_type == "run_command":
                logger.warning(
                    "run_command actions are disabled by default for safety; "
                    "enable explicitly in code if you understand the risk."
                )
                return False

            else:
                logger.warning("Unknown action type in config: %s", action_type)
                return False
        except Exception as e:
            logger.warning("Action '%s' failed (%s): %s — continuing.", action_type, type(e).__name__, e)
            return False

    def _maybe_revert(self, previous_state: str):
        if not self.revert_on_change or previous_state is None:
            return
        # Gate each revert on OUR OWN flag -- "did THIS app turn this
        # setting on" -- not on live OS state. This flag can't drift: it's
        # set true only inside _execute() when we fire the action, and
        # cleared only here when we successfully revert it, so it's purely
        # under this class's own control, unlike the OS-level state itself.
        #
        # CORRECTNESS NOTE (this replaced an earlier, too-broad "always
        # call every revert unconditionally" version): that version fixed
        # the real color-filter drift bug but over-applied the same
        # reasoning to high_contrast/taskbar/DND, which don't have a
        # drift-of-DIRECTION problem the way color filter does (Windows
        # only exposes a *relative* toggle hotkey for color filter, so its
        # own cached on/off flag can genuinely diverge from the real OS
        # state -- see ui_controller.set_color_filter's live registry
        # query, which is the correct fix for THAT specific problem).
        # high_contrast and taskbar use ABSOLUTE OS calls
        # (SystemParametersInfoW / SHAppBarMessage) with no such drift
        # risk, so gating them here is safe -- and necessary: calling them
        # unconditionally on every revert would force off a user's own,
        # pre-existing High Contrast accessibility setting or their own
        # manually-configured taskbar auto-hide preference, even in a
        # session where this app never touched either one.
        if self._dnd_active:
            notifications.set_do_not_disturb(False)
            self._dnd_active = False
        ui_controller.restore_brightness()
        if self._color_filter_active:
            # restore_color_filter() itself is safe to call unconditionally
            # once we've decided WE are the one reverting it -- it queries
            # live OS state internally (see ui_controller.py) rather than
            # trusting a possibly-drifted cached flag for which DIRECTION
            # to toggle.
            ui_controller.restore_color_filter()
            self._color_filter_active = False
        if self._high_contrast_active:
            ui_controller.restore_high_contrast()
            self._high_contrast_active = False
        if self._taskbar_hidden:
            ui_controller.restore_taskbar()
            self._taskbar_hidden = False

    def pause_for_idle(self):
        """
        Called by daemon.py when the person appears to be genuinely away
        (no input, no face, for longer than idle_pause_after_minutes) --
        reverts whatever was last committed (DND, dimming, taskbar hide,
        etc.) the same way a real state change would, WITHOUT going
        through the normal confidence/streak gate in process(). There's
        no reason to keep e.g. a fatigue-triggered dim active while
        you're not even at the desk, and no reason to wait through
        low_confidence_revert_after's several-window debounce for
        something we're already confident about (idle detection isn't a
        noisy classifier call, it's a direct timestamp check).
        Idempotent -- safe to call every idle tick, not just the first.
        """
        if self._current_committed_state is not None:
            self._maybe_revert(previous_state=self._current_committed_state)
            self._current_committed_state = None
        self._reset_streak()
        self._low_confidence_streak = 0

    def shutdown(self):
        """Called on daemon exit: always leave the system as we found it --
        meaning only revert what THIS app itself turned on this session,
        same principle as _maybe_revert() above. Gating on our own flags
        (rather than calling ui_controller.restore_all_expanded_ui()
        unconditionally, as this used to) avoids forcing off a user's own,
        pre-existing High Contrast accessibility setting or manually-
        configured taskbar auto-hide just because this daemon is exiting."""
        if self._dnd_active:
            notifications.set_do_not_disturb(False)
            self._dnd_active = False
        ui_controller.restore_brightness()  # internally a no-op if we never touched brightness
        if self._color_filter_active:
            ui_controller.restore_color_filter()
            self._color_filter_active = False
        if self._high_contrast_active:
            ui_controller.restore_high_contrast()
            self._high_contrast_active = False
        if self._taskbar_hidden:
            ui_controller.restore_taskbar()
            self._taskbar_hidden = False
        ui_controller.unblock_domains()  # scoped to our own marker block, always safe to call
        if self.bandit is not None:
            self.bandit.close()
