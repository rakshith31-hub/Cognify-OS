"""
feature_aggregator.py
----------------------
Turns raw buffered signals from EyeTracker (per-frame) and InputTracker
(per-event) into ONE windowed feature vector matching FEATURE_COLS used
during training (see src/model/train_model.py).

A "window" is e.g. the last 10-15 seconds of behavior. Longer windows give
more stable estimates (better for blink rate / typing speed) at the cost of
responsiveness; shorter windows react faster but are noisier. 10s is a
reasonable default.
"""

import time
import numpy as np


PAUSE_THRESHOLD_SECONDS = 3.0  # a gap between keystrokes longer than this = "pause"

FEATURE_COLS = [
    "blink_rate_per_min", "avg_ear", "fixation_duration_ms", "gaze_dispersion",
    "saccade_rate_per_min", "pupil_diameter_norm",
    "mouse_speed_mean", "mouse_speed_var", "click_rate_per_min", "mouse_idle_ratio",
    "typing_speed_kpm", "key_interval_var", "backspace_rate", "long_pause_count",
]


class FeatureAggregator:
    def __init__(self, window_seconds: float = 10.0,
                 head_yaw_threshold_deg: float = 25.0,
                 head_pitch_threshold_deg: float = 18.0,
                 head_pose_calibration_frames: int = 30,
                 head_pose_smoothing_window: int = 5):
        self.window_seconds = window_seconds
        self._eye_frame_buffer = []  # list of dicts from EyeTracker.read_frame_features()
        # Separate log kept for EVERY frame attempt (including no-face
        # frames), used only for the head-pose / off-screen override below
        # -- _eye_frame_buffer intentionally never includes no-face frames
        # since the 14 trained ML features can't represent "no face" as a
        # value, only as an absence.
        self._pose_log = []  # list of {timestamp, has_face, yaw, pitch}
        self.head_yaw_threshold_deg = head_yaw_threshold_deg
        self.head_pitch_threshold_deg = head_pitch_threshold_deg
        self.head_pose_smoothing_window = head_pose_smoothing_window

        # ---- per-person/per-camera calibration ----
        # estimate_head_pose() uses a generic, non-per-person 3D face model,
        # not one fitted to this user's actual face geometry or camera
        # angle. That means "facing the screen normally" can genuinely read
        # as e.g. yaw=15 deg for one person's face/webcam setup and yaw=-8
        # for another's -- there's no reason to expect 0 deg = neutral for
        # everyone. Using a fixed absolute threshold against that raw,
        # un-calibrated value is what let SLIGHT movements get flagged: if
        # someone's neutral pose already reads close to the threshold, a
        # small additional turn is enough to cross it, even though it's a
        # small movement in reality. We collect a short baseline sample at
        # the start of each session and threshold against DEVIATION from
        # that baseline instead of the raw angle.
        self.head_pose_calibration_frames = head_pose_calibration_frames
        self._pose_baseline = None       # (yaw0, pitch0) once calibrated
        self.last_face_seen_ts = 0.0     # unbounded, for idle/away detection (daemon.py)
        self._calibration_samples = []   # accumulating (yaw, pitch) until calibrated

    def push_eye_frame(self, frame_feats: dict):
        now = frame_feats["timestamp"] if frame_feats is not None else time.time()

        if frame_feats is None:
            self._pose_log.append({"timestamp": now, "has_face": False, "yaw": None, "pitch": None})
        else:
            self._eye_frame_buffer.append(frame_feats)
            self.last_face_seen_ts = now  # unbounded, for idle/away detection (daemon.py)
            head_pose = frame_feats.get("head_pose")
            yaw, pitch = head_pose if head_pose else (None, None)
            self._pose_log.append({"timestamp": now, "has_face": True, "yaw": yaw, "pitch": pitch})
            if yaw is not None and self._pose_baseline is None:
                self._calibration_samples.append((yaw, pitch))
                if len(self._calibration_samples) >= self.head_pose_calibration_frames:
                    ys = [s[0] for s in self._calibration_samples]
                    ps = [s[1] for s in self._calibration_samples]
                    self._pose_baseline = (float(np.median(ys)), float(np.median(ps)))

        cutoff = now - self.window_seconds
        self._eye_frame_buffer = [f for f in self._eye_frame_buffer if f["timestamp"] >= cutoff]
        self._pose_log = [p for p in self._pose_log if p["timestamp"] >= cutoff]

    def compute_off_screen_ratio(self) -> dict:
        """
        Fraction of frames in the current window where the user's head
        wasn't reasonably facing the screen -- either no face detected at
        all, or yaw/pitch deviating from this session's calibrated neutral
        baseline by more than the configured thresholds (bending your head
        down to look at your phone, or turning to look left/right).
        Deliberately NOT one of the 14 trained ML features (see README) --
        this is a rule-based override ahead of the model (see daemon.py).
        """
        if not self._pose_log:
            return {"off_screen_ratio": 0.0, "no_face_ratio": 0.0, "n_frames": 0, "calibrated": False}

        n = len(self._pose_log)
        no_face = sum(1 for p in self._pose_log if not p["has_face"])

        if self._pose_baseline is None:
            # Still calibrating this session's neutral pose -- don't claim
            # off-screen from head angle yet (we don't know what "normal"
            # looks like for this person/camera yet), only no-face frames
            # count during this brief warm-up.
            return {"off_screen_ratio": no_face / n, "no_face_ratio": no_face / n,
                    "n_frames": n, "calibrated": False}

        base_yaw, base_pitch = self._pose_baseline
        posed = [p for p in self._pose_log if p["has_face"] and p["yaw"] is not None]

        # Rolling-median smoothing over neighboring posed frames dampens
        # single-frame landmark-detection jitter (noise, not real
        # movement) so an isolated noisy reading doesn't get individually
        # counted as off-pose.
        w = self.head_pose_smoothing_window
        off_pose = 0
        for i in range(len(posed)):
            lo, hi = max(0, i - w // 2), min(len(posed), i + w // 2 + 1)
            local_yaw = float(np.median([posed[j]["yaw"] for j in range(lo, hi)]))
            local_pitch = float(np.median([posed[j]["pitch"] for j in range(lo, hi)]))
            if (abs(local_yaw - base_yaw) > self.head_yaw_threshold_deg
                    or abs(local_pitch - base_pitch) > self.head_pitch_threshold_deg):
                off_pose += 1

        return {
            "off_screen_ratio": (no_face + off_pose) / n,
            "no_face_ratio": no_face / n,
            "n_frames": n,
            "calibrated": True,
        }

    def compute_features(self, input_snapshot: dict) -> dict:
        """
        input_snapshot: output of InputTracker.snapshot(window_seconds)
        Returns a dict with all 14 feature_columns needed by the model.
        Falls back to neutral defaults for any signal with insufficient data
        (e.g. no face detected -> no eye features that window).
        """
        window_s = input_snapshot["window_seconds"]
        window_min = window_s / 60.0

        feats = {}

        # ---------------- eye features ----------------
        eye_frames = self._eye_frame_buffer
        if len(eye_frames) >= 3:
            blinks = sum(1 for f in eye_frames if f["is_blink"])
            feats["blink_rate_per_min"] = blinks / window_min if window_min > 0 else 0.0
            feats["avg_ear"] = float(np.mean([f["ear"] for f in eye_frames]))

            gaze_pts = np.array([[f["gaze_x"], f["gaze_y"]] for f in eye_frames])
            feats["gaze_dispersion"] = float(np.mean(np.var(gaze_pts, axis=0)))

            saccades = sum(1 for f in eye_frames if f["is_saccade"])
            feats["saccade_rate_per_min"] = saccades / window_min if window_min > 0 else 0.0

            feats["pupil_diameter_norm"] = float(np.mean([f["pupil_diameter"] for f in eye_frames]))

            # fixation duration: approximate as mean run-length (in ms) of
            # consecutive non-saccade frames
            run_lengths = []
            run = 0
            timestamps = [f["timestamp"] for f in eye_frames]
            for i, f in enumerate(eye_frames):
                if not f["is_saccade"]:
                    run += 1
                else:
                    if run > 0:
                        run_lengths.append(run)
                    run = 0
            if run > 0:
                run_lengths.append(run)
            if run_lengths and len(timestamps) > 1:
                avg_frame_dt = (timestamps[-1] - timestamps[0]) / max(len(timestamps) - 1, 1)
                feats["fixation_duration_ms"] = float(np.mean(run_lengths) * avg_frame_dt * 1000)
            else:
                feats["fixation_duration_ms"] = 250.0  # neutral default
        else:
            # no reliable face/eye data this window -> neutral fallback values
            feats.update({
                "blink_rate_per_min": 18.0,
                "avg_ear": 0.27,
                "gaze_dispersion": 0.06,
                "saccade_rate_per_min": 25.0,
                "pupil_diameter_norm": 0.55,
                "fixation_duration_ms": 250.0,
            })

        # ---------------- mouse features ----------------
        positions = input_snapshot["mouse_positions"]
        if len(positions) >= 2:
            speeds = []
            for (t0, x0, y0), (t1, x1, y1) in zip(positions[:-1], positions[1:]):
                dt = max(t1 - t0, 1e-3)
                dist = np.hypot(x1 - x0, y1 - y0)
                speeds.append(dist / dt)
            feats["mouse_speed_mean"] = float(np.mean(speeds))
            feats["mouse_speed_var"] = float(np.var(speeds))

            moving_time = positions[-1][0] - positions[0][0]
            feats["mouse_idle_ratio"] = float(max(0.0, 1.0 - min(moving_time / window_s, 1.0)))
        else:
            feats["mouse_speed_mean"] = 0.0
            feats["mouse_speed_var"] = 0.0
            feats["mouse_idle_ratio"] = 1.0

        feats["click_rate_per_min"] = len(input_snapshot["mouse_clicks"]) / window_min if window_min > 0 else 0.0

        # ---------------- keyboard features ----------------
        keys = sorted(input_snapshot["key_events"])
        feats["typing_speed_kpm"] = len(keys) / window_min if window_min > 0 else 0.0

        if len(keys) >= 2:
            intervals = np.diff(keys) * 1000.0  # ms
            feats["key_interval_var"] = float(np.var(intervals))
            feats["long_pause_count"] = float(np.sum(np.diff(keys) > PAUSE_THRESHOLD_SECONDS))
        else:
            feats["key_interval_var"] = 8000.0
            feats["long_pause_count"] = 0.0

        n_backspace = len(input_snapshot["backspace_events"])
        feats["backspace_rate"] = float(n_backspace / len(keys)) if keys else 0.0

        return feats

    def compute_sequence_features(self, input_snapshot: dict, t_steps: int = 10,
                                   overlap_factor: float = 1.0) -> np.ndarray:
        """
        Same 14 features as compute_features(), but as a (t_steps, 14) array:
        one row per sub-window of the window, for the CNN model, which
        convolves over time.

        Sub-windows OVERLAP (each is `overlap_factor` times wider than the
        disjoint-bin width and centered at t_steps evenly spaced points)
        rather than being a hard 1/t_steps partition. Disjoint 1-second bins
        starve count-based features (blink/click/keystroke rates) of enough
        raw events to estimate a rate at all -- e.g. a single blink landing
        in a 1s bin reads as a spike to 60/min. Overlapping windows give
        each timestep enough underlying events for a stable estimate while
        still tracking how the signal changes across the window, and it's
        this smoothed trajectory the CNN was trained on (see
        generate_synthetic_data.py's AR(1) process, which is autocorrelated
        step to step, not independent noise per step).

        Reuses compute_features()'s exact per-signal formulas/fallbacks on
        each sub-window.
        """
        window_s = input_snapshot["window_seconds"]
        step = window_s / t_steps
        sub_width = min(window_s, step * overlap_factor)

        # anchor to the latest timestamp we actually have data for
        candidate_ends = [self._eye_frame_buffer[-1]["timestamp"]] if self._eye_frame_buffer else []
        for key in ("mouse_positions",):
            if input_snapshot.get(key):
                candidate_ends.append(input_snapshot[key][-1][0])
        for key in ("mouse_clicks", "key_events"):
            if input_snapshot.get(key):
                candidate_ends.append(max(input_snapshot[key]))
        end_t = max(candidate_ends) if candidate_ends else 0.0
        start_t = end_t - window_s

        full_eye_buffer = self._eye_frame_buffer
        rows = np.empty((t_steps, len(FEATURE_COLS)), dtype=np.float32)

        for i in range(t_steps):
            center = start_t + (i + 0.5) * step
            sub_start = max(start_t, center - sub_width / 2)
            sub_end = min(end_t, center + sub_width / 2)

            self._eye_frame_buffer = [
                f for f in full_eye_buffer if sub_start <= f["timestamp"] < sub_end
            ]
            sub_snapshot = {
                "mouse_positions": [p for p in input_snapshot["mouse_positions"] if sub_start <= p[0] < sub_end],
                "mouse_clicks": [c for c in input_snapshot["mouse_clicks"] if sub_start <= c < sub_end],
                "key_events": [k for k in input_snapshot["key_events"] if sub_start <= k < sub_end],
                "backspace_events": [b for b in input_snapshot["backspace_events"] if sub_start <= b < sub_end],
                "window_seconds": max(sub_end - sub_start, 1e-3),
            }
            bin_feats = self.compute_features(sub_snapshot)
            rows[i, :] = [bin_feats[c] for c in FEATURE_COLS]

        self._eye_frame_buffer = full_eye_buffer  # restore
        return rows


if __name__ == "__main__":
    # minimal self-test with fabricated data (no real devices needed)
    agg = FeatureAggregator(window_seconds=10)
    fake_input_snapshot = {
        "mouse_positions": [(0.0, 100, 100), (1.0, 140, 130), (2.0, 190, 160)],
        "mouse_clicks": [0.5, 3.2],
        "key_events": [0.1, 0.3, 0.6, 4.5, 4.7],
        "backspace_events": [4.5],
        "window_seconds": 10,
    }
    print(agg.compute_features(fake_input_snapshot))
