"""
daemon.py
---------
The actual background service. Combines:

  EyeTracker (optional)  ─┐
  InputTracker            ├─► FeatureAggregator ─► CognitiveStatePredictor ─► ActionEngine ─► notifications / UI / logs
  (mouse + keyboard)     ─┘

Two run modes:
  --mode full        (default) webcam + mouse/keyboard. Requires
                      opencv-python + mediapipe installed and a camera.
  --mode input-only   mouse/keyboard ONLY, no camera at all. Eye-related
                      features fall back to the FeatureAggregator's neutral
                      defaults (the same fallback path used when no face is
                      detected in full mode), so the exact same trained
                      model is used either way — it's just working with 8
                      of 14 features live instead of 14.

This process is meant to be started once and left running in the
background (see cli.py / scripts/install_*). It writes a PID file so
`cognitive-monitor stop` / `status` can find it, and it restores any
system changes (DND, brightness, hosts file) on exit — including on
Ctrl+C / SIGTERM.
"""

import argparse
import logging
import os
import platform
import signal
import sys
import threading
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import yaml

from src.feature_extraction.input_tracker import InputTracker
from src.feature_extraction.window_tracker import WindowTracker
from src.feature_extraction.feature_aggregator import FeatureAggregator, FEATURE_COLS
from src.model.predict import CognitiveStatePredictor, CNNCognitiveStatePredictor, EnsembleCognitiveStatePredictor
from src.actions.action_engine import ActionEngine
from src.actions.state_logger import StateLogger
from src.actions.app_usage_logger import AppUsageLogger
from src.actions.block_attempt_logger import BlockAttemptLogger
from src.actions.blocked_site_server import BlockedSiteServer
from src.actions.notification_logger import NotificationLogger
from src.actions import ui_controller
from src.actions import notifications
from src.actions.backend_client import BackendClient
from src.actions import session_summary

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "default_config.yaml")
PID_FILE = os.path.join(PROJECT_ROOT, "logs", "daemon.pid")

logger = logging.getLogger("cognitive_monitor.daemon")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def validate_or_exit(config: dict, config_path: str):
    """Run config_validator before touching any hardware/OS state. Hard
    errors block startup entirely (a broken config running for hours
    unattended is worse than refusing to start); warnings are logged and
    the daemon proceeds — see config_validator.py's own docstring for why
    this exists (most of this project's real bugs were silent
    misconfiguration, not exceptions)."""
    from src.config_validator import validate_config
    result = validate_config(config)
    for w in result.warnings:
        logger.warning("[config] %s", w)
    if not result.ok:
        for e in result.errors:
            logger.error("[config] %s", e)
        logger.error(
            "Refusing to start with %d config error(s) in %s. Run "
            "`cognitive-monitor validate-config` for the full report, or "
            "`cognitive-monitor setup` to regenerate a clean config.",
            len(result.errors), config_path,
        )
        sys.exit(1)


def setup_logging(log_dir: str, verbose: bool):
    os.makedirs(log_dir, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [
        logging.FileHandler(os.path.join(log_dir, "daemon.log")),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


class CognitiveMonitorDaemon:
    def __init__(self, config: dict, mode: str = "full", camera_index: int = 0):
        self.config = config
        self.mode = mode
        self.camera_index = camera_index

        general = config.get("general", {})
        self.predict_interval = general.get("predict_interval_seconds", 10)

        self.input_tracker = InputTracker()
        self.window_tracker = WindowTracker()
        self.aggregator = FeatureAggregator(
            window_seconds=self.predict_interval,
            head_yaw_threshold_deg=general.get("head_yaw_threshold_deg", 25.0),
            head_pitch_threshold_deg=general.get("head_pitch_threshold_deg", 18.0),
            head_pose_calibration_frames=general.get("head_pose_calibration_frames", 30),
            head_pose_smoothing_window=general.get("head_pose_smoothing_window", 5),
        )
        self._pose_calibration_logged = False
        self.head_pose_override_enabled = general.get("head_pose_override_enabled", True)
        self.head_pose_off_screen_threshold = general.get("head_pose_off_screen_ratio_threshold", 0.5)

        # ---- idle/away auto-pause ----
        # Genuine AFK time (stepped away, lunch, a walk) still fed into
        # the classifier before this: no input + no face just means every
        # 14 features hit their neutral fallback, producing noisy
        # low-confidence predictions across random classes until
        # low_confidence_revert_after eventually reverts the stale state
        # -- reactive, and wastes a model call every window for a period
        # that was never a real cognitive state to begin with. This
        # checks BEFORE calling the model at all and skips prediction
        # entirely once idle, resuming the instant real activity returns.
        self.idle_pause_enabled = general.get("idle_pause_enabled", True)
        self.idle_pause_after_seconds = general.get("idle_pause_after_minutes", 5) * 60
        self._is_idle = False

        self.model_type = general.get("model_type", "rf")
        if self.model_type == "cnn":
            self.predictor = CNNCognitiveStatePredictor()
        elif self.model_type == "ensemble":
            self.predictor = EnsembleCognitiveStatePredictor(
                cnn_weight=general.get("ensemble_cnn_weight", 0.2)
            )
        elif self.model_type == "rf":
            self.predictor = CognitiveStatePredictor()
        else:
            raise ValueError(
                f"general.model_type='{self.model_type}' is not recognized "
                f"(expected 'rf', 'cnn', or 'ensemble')."
            )
        log_dir_path = os.path.join(PROJECT_ROOT, general.get("log_dir", "logs"))

        # ---- real-label collection (closing the synthetic-data gap) ----
        # The classifier is trained ENTIRELY on synthetic data (see
        # README) and never updates from real usage on its own -- this is
        # the actual data-collection half of that fix. Every real
        # (non-idle) tick's raw feature vector gets written to
        # logs/last_features.json, so `cognitive-monitor label <state>`
        # can attach a real ground-truth label to it at any time within
        # label_max_age_seconds, not just during the periodic reminder
        # below. See src/model/retrain_from_real_data.py for what
        # actually happens with these labels once collected.
        from src.actions.real_label_logger import RealLabelLogger
        self.real_label_logger = RealLabelLogger(log_dir=log_dir_path)
        self.label_prompt_enabled = general.get("label_prompt_enabled", True)
        # Bounded "labeling session" instead of scattered one-off
        # reminders: once every label_session_interval_minutes, fires a
        # burst of quick tap-to-label prompts (see notifications.request_label)
        # every label_session_prompt_interval_seconds, for
        # label_session_duration_minutes, then stops until the next
        # scheduled session. A short concentrated burst is much lower
        # friction than remembering to self-report at random times over
        # days -- you know a session is happening and can either engage
        # with it for a few minutes or ignore it entirely, rather than
        # needing standing vigilance.
        self.label_session_interval_seconds = general.get("label_session_interval_minutes", 120) * 60
        self.label_session_duration_seconds = general.get("label_session_duration_minutes", 5) * 60
        self.label_session_prompt_interval_seconds = general.get(
            "label_session_prompt_interval_seconds", 50)
        self._label_session_active = False
        self._label_session_started_at = 0.0
        self._label_session_last_prompt_ts = 0.0
        self._label_session_end_ts = 0.0  # last session's end, for the next-session timer
        self._last_features_path = os.path.join(log_dir_path, "last_features.json")
        self.auto_retrain_on_startup = general.get("auto_retrain_on_startup", False)
        self.auto_retrain_min_new_samples = general.get("auto_retrain_min_new_samples", 60)
        self._log_dir_path = log_dir_path

        # ---- per-user calibration for all 14 ML features ----
        # Built and unit-tested (src/feature_extraction/personal_calibrator.py)
        # but previously never actually wired into the live prediction path --
        # fixed here. See that module's docstring for why this is a
        # population-scale-preserving transform (population_mean + z*population_std
        # using YOUR OWN mean/std), not naive z-scoring -- the trained model
        # expects population-scale inputs, and naive personal z-scoring would
        # feed it data in a completely different space than it was trained on.
        from src.feature_extraction.personal_calibrator import PersonalCalibrator
        self.personal_calibration_enabled = general.get("personal_calibration_enabled", True)
        self.personal_calibrator = PersonalCalibrator(
            profile_path=os.path.join(log_dir_path, "personal_feature_baseline.json"),
            min_samples=general.get("personal_calibration_min_windows", 30),
        )
        self._personal_calibration_logged = self.personal_calibrator.is_calibrated()

        self.notification_logger = NotificationLogger(log_dir=log_dir_path)
        self.action_engine = ActionEngine(
            config,
            bandit_db_path=os.path.join(log_dir_path, general.get("bandit_db_file", "bandit.db")),
            notification_logger=self.notification_logger,
        )
        self.state_logger = StateLogger(
            log_dir=os.path.join(PROJECT_ROOT, general.get("log_dir", "logs")),
            log_file=general.get("log_file", "state_history.csv"),
        )
        self.app_usage_logger = AppUsageLogger(
            log_dir=os.path.join(PROJECT_ROOT, general.get("log_dir", "logs")),
        )

        # ---- session-wide site blocking (STATE-INDEPENDENT) ----
        # Previously block_domains only ever fired as part of `distracted`'s
        # action list -- gated behind confidence + consecutive-window +
        # cooldown checks like every other action. That means visiting a
        # blocked site while in ANY other state (or while distracted hadn't
        # yet cleared the debounce gate) went straight through unblocked.
        # `site_blocking` here is a separate, top-level config section
        # applied once for the WHOLE SESSION regardless of cognitive state,
        # not tied to the per-state action system at all.
        site_blocking = config.get("site_blocking", {})
        self.site_blocking_enabled = bool(site_blocking.get("enabled", False))
        self.site_blocking_domains = site_blocking.get("domains", [])
        self.site_blocking_dim_on_attempt = bool(site_blocking.get("dim_on_attempt", True))
        self.site_blocking_dim_percent = int(site_blocking.get("dim_percent", 5))
        self.site_blocking_dim_seconds = float(site_blocking.get("dim_duration_seconds", 4.0))

        def _on_block_attempt(domain, protocol, path):
            if self.site_blocking_dim_on_attempt:
                ui_controller.flash_dim_on_block(
                    percent=self.site_blocking_dim_percent,
                    duration_seconds=self.site_blocking_dim_seconds,
                )

        self.block_attempt_logger = BlockAttemptLogger(
            log_dir=os.path.join(PROJECT_ROOT, general.get("log_dir", "logs")),
            on_attempt=_on_block_attempt,
        )
        self.blocked_site_server = BlockedSiteServer(self.block_attempt_logger)

        # ---- optional teacher/supervision backend ----
        # No-op (empty domain list, upload silently skipped) unless the
        # student has run `cognitive-monitor login` + `link` -- see
        # backend_client.py's docstring for the full contract. Teacher-set
        # domains are fetched once here at startup and merged into
        # site_blocking_domains below, so they take effect starting THIS
        # session (not pushed live into one already running) -- and a
        # session summary is uploaded once at _shutdown().
        self.backend_client = BackendClient(log_dir=log_dir_path)

        self.eye_tracker = None
        self._running = False

    def _init_eye_tracker(self):
        if self.mode != "full":
            return
        try:
            from src.feature_extraction.eye_tracker import EyeTracker
            self.eye_tracker = EyeTracker(camera_index=self.camera_index)
            logger.info("Camera initialized (index=%d). Running in FULL mode.", self.camera_index)
        except Exception as e:
            logger.warning(
                "Could not initialize webcam/MediaPipe (%s: %s). "
                "Falling back to INPUT-ONLY mode (mouse + keyboard signals only). "
                "Run with --mode input-only to silence this warning.",
                type(e).__name__, e,
            )
            self.eye_tracker = None
            self.mode = "input-only"

    def _maybe_auto_retrain(self):
        """
        Runs ONCE at startup (never mid-session -- a retrain shouldn't
        change model behavior under you while you're actively working),
        and only if auto_retrain_on_startup is explicitly enabled (off by
        default -- see config comment). Checks whether enough NEW real
        labels have accumulated since the last retrain before running
        retrain_from_real_data.py in a background thread, so a slow
        retrain never delays the daemon actually starting.
        """
        if not self.auto_retrain_on_startup:
            return
        try:
            from src.actions.real_label_logger import RealLabelLogger
            from src.model.retrain_from_real_data import HISTORY_PATH
            import json as _json

            counts = RealLabelLogger(log_dir=self._log_dir_path).count()
            total_now = sum(counts.values())

            last_total = 0
            if os.path.exists(HISTORY_PATH):
                with open(HISTORY_PATH) as f:
                    history = _json.load(f)
                if history.get("retrains"):
                    last_total = history["retrains"][-1].get("n_real_total", 0)

            new_samples = total_now - last_total
            if new_samples < self.auto_retrain_min_new_samples:
                logger.debug("Auto-retrain: only %d new real labels since last retrain "
                             "(need %d) — skipping.", new_samples, self.auto_retrain_min_new_samples)
                return

            logger.info("Auto-retrain: %d new real labels since last retrain — "
                        "retraining in the background now.", new_samples)

            def _run():
                from src.model.retrain_from_real_data import retrain
                result = retrain(self._log_dir_path)
                logger.info("Auto-retrain finished: %s", result["message"])

            threading.Thread(target=_run, daemon=True).start()
        except Exception as e:
            logger.warning("Auto-retrain check failed (%s): %s — continuing normally.",
                           type(e).__name__, e)

    def start(self):
        self._write_pid_file()
        self._init_eye_tracker()
        self.input_tracker.start()
        self.window_tracker.start()
        self.blocked_site_server.start()
        self._session_start = time.time()

        # Retry any session summaries that failed to upload last time
        # (e.g. offline at shutdown) before fetching this session's domain
        # list, so a teacher isn't left waiting on a machine that only
        # reconnects sporadically.
        self.backend_client.retry_pending()

        teacher_domains = self.backend_client.fetch_blocked_domains()
        if teacher_domains:
            merged = list(self.site_blocking_domains)
            for d in teacher_domains:
                if d not in merged:
                    merged.append(d)
            self.site_blocking_domains = merged
            self.site_blocking_enabled = True  # a teacher's list applies even if this student's own config has it off
            logger.info("Merged %d teacher-configured domain(s) into this session's site blocking.",
                        len(teacher_domains))

        if self.site_blocking_enabled and self.site_blocking_domains:
            # Session-length "duration": re-applied on every daemon restart
            # anyway, and explicitly unblocked in _shutdown() below, so this
            # just needs to outlast any realistic single session rather
            # than represent a real expiry.
            ui_controller.block_domains(self.site_blocking_domains, duration_minutes=24 * 60)
            logger.info("Session-wide site blocking ACTIVE for: %s (applies in EVERY "
                        "state, not just 'distracted').", ", ".join(self.site_blocking_domains))

        self._running = True

        self._maybe_auto_retrain()

        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        # BUG FIX: on Windows, os.kill(pid, 15) from cli.py's `stop` command
        # does NOT deliver a real SIGTERM -- cross-process, Windows has no
        # such thing, so Python's os.kill() falls back to TerminateProcess(),
        # which kills this process immediately and skips the try/finally in
        # start() (=> _shutdown() / action_engine.shutdown() never runs, so
        # DND/brightness/color-filter/high-contrast/taskbar/hosts-block never
        # revert). The one cross-process signal Windows actually supports is
        # CTRL_BREAK_EVENT, which Python surfaces as SIGBREAK -- but only to
        # processes in their own console process group (see
        # CREATE_NEW_PROCESS_GROUP in cli.py's cmd_start). Registering it here
        # lets `cognitive-monitor stop` trigger a real graceful shutdown on
        # Windows the same way SIGTERM does on macOS/Linux.
        if platform.system() == "Windows" and hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, self._handle_signal)

        logger.info("Cognitive Monitor daemon started (mode=%s, predict_interval=%ss).",
                    self.mode, self.predict_interval)

        last_predict = time.time()
        eye_poll_interval = 1.0 / 15.0  # ~15fps if camera is active

        try:
            while self._running:
                if self.eye_tracker is not None:
                    frame_feats = self.eye_tracker.read_frame_features()
                    self.aggregator.push_eye_frame(frame_feats)
                    time.sleep(eye_poll_interval)
                else:
                    time.sleep(0.5)

                now = time.time()
                if now - last_predict >= self.predict_interval:
                    self._tick()
                    last_predict = now
        finally:
            self._shutdown()

    def _check_idle_and_maybe_pause(self) -> bool:
        """
        Returns True if this tick should be SKIPPED (person appears to be
        away). Idle = both no input activity AND no face seen for
        idle_pause_after_seconds. In input-only mode there's no camera
        signal at all, so idle is judged on input activity alone.

        Uses time.time() as the fallback "last seen" when a signal has
        never fired yet this process (both trackers report 0.0 in that
        case) -- daemon just started, not "idle since epoch".
        """
        now = time.time()
        last_input = self.input_tracker.last_activity_timestamp() or self._session_start
        if self.mode == "full" and self.eye_tracker is not None:
            last_face = self.aggregator.last_face_seen_ts or self._session_start
            last_activity = max(last_input, last_face)
        else:
            last_activity = last_input

        idle_seconds = now - last_activity
        currently_idle = idle_seconds >= self.idle_pause_after_seconds

        if currently_idle and not self._is_idle:
            self._is_idle = True
            logger.info(
                "No input or face detected for %.0fs (>= %.0fs threshold) — pausing "
                "predictions and reverting any active state until activity resumes.",
                idle_seconds, self.idle_pause_after_seconds,
            )
            self.action_engine.pause_for_idle()
        elif not currently_idle and self._is_idle:
            self._is_idle = False
            logger.info("Activity resumed after being idle — predictions resuming.")

        if currently_idle:
            # Logged every idle TICK (not just the transition into idle)
            # so idle_windows in the dashboard reflects actual idle
            # DURATION (windows * predict_interval), the same way real
            # state time is counted -- logging only once per idle period
            # would undercount a multi-hour absence as a single window.
            self.state_logger.log({"state": "idle", "confidence": 0.0, "all_probabilities": {}}, action_fired="")

        return currently_idle

    def _write_last_features(self, feats: dict, result: dict):
        """Overwritten every real tick -- lets `cognitive-monitor label
        <state>` attach a ground-truth label to whatever the most recent
        computed window actually was, at any time, not just when the
        periodic reminder below happens to fire."""
        import json
        data = {
            "timestamp": time.time(),
            "features": feats,
            "predicted_state": result.get("state"),
            "confidence": result.get("confidence"),
        }
        try:
            tmp = self._last_features_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, self._last_features_path)
        except OSError as e:
            logger.debug("Could not write last_features.json: %s", e)

    def _maybe_run_label_session(self):
        """
        Runs a bounded ~5-minute labeling session periodically instead of
        scattered one-off reminders (see the config comment above for why).
        A tap prompt (or, off-Windows/no-BurntToast, a plain text reminder
        with the CLI command) fires roughly every
        label_session_prompt_interval_seconds while a session is active.
        """
        if not self.label_prompt_enabled:
            return
        now = time.time()

        if self._label_session_active:
            if now - self._label_session_started_at >= self.label_session_duration_seconds:
                self._label_session_active = False
                self._label_session_end_ts = now
                logger.info("Labeling session ended.")
                return
            if now - self._label_session_last_prompt_ts >= self.label_session_prompt_interval_seconds:
                self._label_session_last_prompt_ts = now
                notifications.request_label(
                    "Quick check-in",
                    "What are you actually doing right now?",
                )
                self.notification_logger.log("label_prompt", "", "Quick check-in", "")
            return

        # not currently in a session -- check if it's time to start one
        last_end = self._label_session_end_ts or self._session_start
        if now - last_end >= self.label_session_interval_seconds:
            self._label_session_active = True
            self._label_session_started_at = now
            self._label_session_last_prompt_ts = now
            logger.info("Starting a %.0f-minute labeling session (prompts every %.0fs).",
                        self.label_session_duration_seconds / 60,
                        self.label_session_prompt_interval_seconds)
            notifications.request_label(
                "Quick check-in",
                "A few quick check-ins over the next 5 minutes helps improve accuracy "
                "— totally optional, tap a state or ignore.",
            )
            self.notification_logger.log("label_prompt", "", "Quick check-in", "")

    def _predict(self, snapshot: dict, feats_for_prediction: dict) -> dict:
        """
        Dispatches to whichever model_type is configured. CNN/ensemble
        additionally need a (T, 14) sequence -- computed here rather than
        unconditionally in _tick(), since RF (the default) never needs it
        and computing it is wasted work in that case. feats_for_prediction
        is always the aggregated row (already computed in _tick() for
        personal-calibration/real-label-logging purposes regardless of
        model_type) -- only the SEQUENCE is conditional.
        """
        if self.model_type == "rf":
            return self.predictor.predict(feats_for_prediction)

        seq = self.aggregator.compute_sequence_features(snapshot, t_steps=self.predictor.t_steps)
        if self.personal_calibration_enabled:
            seq = self.personal_calibrator.transform_sequence(seq, FEATURE_COLS)

        if self.model_type == "cnn":
            return self.predictor.predict(seq)
        else:  # ensemble
            return self.predictor.predict(feats_for_prediction, seq)

    def _tick(self):
        if self.idle_pause_enabled and self._check_idle_and_maybe_pause():
            return

        snapshot = self.input_tracker.snapshot(window_seconds=self.predict_interval)
        feats = self.aggregator.compute_features(snapshot)

        if self.personal_calibration_enabled:
            self.personal_calibrator.update(feats)
            if self.personal_calibrator.is_calibrated() and not self._personal_calibration_logged:
                logger.info("Personal calibration complete for all 14 features — "
                            "predictions now adjust for your own baseline (see README).")
                self._personal_calibration_logged = True
            # transform() is a per-feature no-op/identity until THAT feature
            # individually has enough samples, so this is always safe to call
            # unconditionally rather than gating on full calibration.
            feats_for_prediction = self.personal_calibrator.transform(feats)
        else:
            feats_for_prediction = feats

        result = self._predict(snapshot, feats_for_prediction)

        self._write_last_features(feats_for_prediction, result)
        self._maybe_run_label_session()

        # ---- rule-based override: head not facing the screen ----
        # The 14-feature ML classifier has no head-pose concept at all (see
        # README) -- gaze_dispersion only measures eyeball movement WITHIN
        # the socket, so bending your head down to a phone or turning to
        # look away often reads as a steady, low-dispersion "focus"-like
        # signal rather than "distracted". This override runs ahead of the
        # model and replaces its output when the head has been off-screen
        # for most of the window, rather than trying to make the trained
        # classifier infer something it was never given the signal for.
        if self.head_pose_override_enabled:
            pose = self.aggregator.compute_off_screen_ratio()
            if pose.get("calibrated") and not self._pose_calibration_logged:
                base = self.aggregator._pose_baseline
                logger.info("Head-pose calibrated: neutral yaw=%.1f pitch=%.1f deg "
                            "(your own baseline -- movements are now judged relative "
                            "to this, not a fixed absolute angle).", base[0], base[1])
                self._pose_calibration_logged = True
            if pose["n_frames"] > 0 and pose["off_screen_ratio"] >= self.head_pose_off_screen_threshold:
                logger.debug(
                    "Head-pose override: off_screen_ratio=%.2f (no_face=%.2f) >= threshold %.2f -> distracted",
                    pose["off_screen_ratio"], pose["no_face_ratio"], self.head_pose_off_screen_threshold,
                )
                result = {
                    "state": "distracted",
                    "confidence": 0.95,
                    "all_probabilities": {"distracted": 0.95, "focus": 0.0, "fatigue": 0.0, "overload": 0.0},
                    "rule_override": "head_pose",
                }

        fired = self.action_engine.process(result)
        self.state_logger.log(result, action_fired=fired)

        window_snapshot = self.window_tracker.snapshot(window_seconds=self.predict_interval)
        self.app_usage_logger.log(window_snapshot, cognitive_state=result["state"])

        logger.info(
            "state=%s confidence=%.1f%% app=%s %s",
            result["state"], result["confidence"] * 100,
            window_snapshot.get("dominant_app") or "n/a",
            f"-> fired: {fired}" if fired else "",
        )

    def _handle_signal(self, signum, frame):
        logger.info("Received signal %s, shutting down...", signum)
        self._running = False

    def _shutdown(self):
        logger.info("Shutting down: reverting any active system changes...")
        self.action_engine.shutdown()
        self.input_tracker.stop()
        self.window_tracker.stop()
        self.blocked_site_server.stop()
        if self.eye_tracker is not None:
            self.eye_tracker.release()
        self._upload_session_summary()
        self._remove_pid_file()
        logger.info("Clean shutdown complete.")
        self._open_session_summary()

    def _upload_session_summary(self):
        """Best-effort upload to the supervision backend, if linked -- see
        backend_client.py. Never raises: a network hiccup here must not
        prevent the rest of _shutdown() (pid file removal, dashboard open)
        from completing."""
        try:
            summary = session_summary.build_session_summary(
                self._log_dir_path, self._session_start, time.time(),
            )
            self.backend_client.upload_session_summary(summary)
        except Exception as e:
            logger.warning("Could not build/upload session summary (%s): %s — "
                           "session stays local-only.", type(e).__name__, e)

    def _open_session_summary(self):
        general = self.config.get("general", {})
        if not general.get("open_dashboard_on_exit", True):
            return
        try:
            from src.dashboard.app import run_dashboard_for_session
            session_start = getattr(self, "_session_start", time.time())
            run_dashboard_for_session(
                config=self.config,
                project_root=PROJECT_ROOT,
                session_start=session_start,
                session_end=time.time(),
                port=general.get("dashboard_port", 5757),
            )
        except Exception as e:
            logger.warning("Could not open end-of-session dashboard (%s): %s — "
                            "run `cognitive-monitor dashboard` manually to view it.",
                            type(e).__name__, e)

    def _write_pid_file(self):
        os.makedirs(os.path.dirname(PID_FILE), exist_ok=True)
        with open(PID_FILE, "w") as f:
            f.write(str(os.getpid()))

    def _remove_pid_file(self):
        try:
            os.remove(PID_FILE)
        except FileNotFoundError:
            pass


def main():
    parser = argparse.ArgumentParser(description="Cognitive Monitor background daemon")
    parser.add_argument("--mode", choices=["full", "input-only"], default="full",
                         help="'full' = webcam + mouse/keyboard. 'input-only' = mouse/keyboard only, no camera.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(os.path.join(PROJECT_ROOT, config.get("general", {}).get("log_dir", "logs")), args.verbose)
    validate_or_exit(config, args.config)

    daemon = CognitiveMonitorDaemon(config, mode=args.mode, camera_index=args.camera_index)
    daemon.start()


if __name__ == "__main__":
    main()
