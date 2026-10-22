"""
real_label_logger.py
----------------------
Persists genuinely human-labeled (feature_vector, true_state) pairs to
logs/real_labels.csv -- the actual training data this project's synthetic
dataset always lacked (see README's "About the training data" section).

Same 14-feature column schema as data/cognitive_state_dataset.csv on
purpose, so retrain_from_real_data.py can load and concatenate both
directly without any column remapping. `source` distinguishes rows (all
"real" here) in case this file is ever merged with something else.
"""

import csv
import os
import time

FEATURE_COLS = [
    "blink_rate_per_min", "avg_ear", "fixation_duration_ms", "gaze_dispersion",
    "saccade_rate_per_min", "pupil_diameter_norm",
    "mouse_speed_mean", "mouse_speed_var", "click_rate_per_min", "mouse_idle_ratio",
    "typing_speed_kpm", "key_interval_var", "backspace_rate", "long_pause_count",
]


class RealLabelLogger:
    def __init__(self, log_dir: str, log_file: str = "real_labels.csv"):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, log_file)
        self._ensure_header()

    def _ensure_header(self):
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp"] + FEATURE_COLS + ["cognitive_state", "source"])

    def log(self, features: dict, state: str):
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            row = [time.strftime("%Y-%m-%d %H:%M:%S")]
            row += [features.get(c, "") for c in FEATURE_COLS]
            row += [state, "real"]
            writer.writerow(row)

    def count(self) -> dict:
        """Per-class counts of real labels collected so far -- used by the
        CLI/dashboard to show progress toward retrain_from_real_data.py's
        minimum-sample threshold."""
        counts = {c: 0 for c in ["focus", "fatigue", "overload", "distracted"]}
        if not os.path.exists(self.path):
            return counts
        with open(self.path, newline="") as f:
            for row in csv.DictReader(f):
                s = row.get("cognitive_state", "")
                if s in counts:
                    counts[s] += 1
        return counts
