"""
app_usage_logger.py
--------------------
Appends one row per prediction tick summarizing which app(s) were in the
foreground during that window — local-only, same pattern as
state_logger.py. This is the raw data the later app-productivity learner
will train on; it does no classification itself.
"""

import csv
import os
import time


class AppUsageLogger:
    def __init__(self, log_dir: str, log_file: str = "app_usage.csv"):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, log_file)
        self._ensure_header()

    def _ensure_header(self):
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "cognitive_state", "dominant_app",
                    "app_switch_count", "app_seconds_breakdown",
                ])

    def log(self, window_snapshot: dict, cognitive_state: str = ""):
        app_seconds = window_snapshot.get("app_seconds", {})
        breakdown = ";".join(f"{app}:{secs:.0f}" for app, secs in app_seconds.items())
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                cognitive_state,
                window_snapshot.get("dominant_app", ""),
                window_snapshot.get("app_switch_count", 0),
                breakdown,
            ])
