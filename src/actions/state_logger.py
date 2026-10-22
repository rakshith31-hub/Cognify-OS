"""
state_logger.py
----------------
Appends every prediction to a local CSV so the user can review their own
state history later (e.g. "when am I actually focused during the day?").
Nothing here leaves the machine.
"""

import csv
import os
import time


class StateLogger:
    def __init__(self, log_dir: str, log_file: str):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, log_file)
        self._ensure_header()

    def _ensure_header(self):
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "state", "confidence",
                                  "p_focus", "p_fatigue", "p_overload", "p_distracted",
                                  "action_fired"])

    def log(self, result: dict, action_fired: str = ""):
        probs = result.get("all_probabilities", {})
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                result.get("state", ""),
                f"{result.get('confidence', 0.0):.4f}",
                f"{probs.get('focus', 0.0):.4f}",
                f"{probs.get('fatigue', 0.0):.4f}",
                f"{probs.get('overload', 0.0):.4f}",
                f"{probs.get('distracted', 0.0):.4f}",
                action_fired,
            ])
