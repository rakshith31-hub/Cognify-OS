"""
session_rating_logger.py
--------------------------
Logs end-of-session ratings the user submits via the dashboard — a
coarser, session-level complement to the per-action bandit feedback.
Local CSV only, same pattern as the other loggers.
"""

import csv
import os
import time


class SessionRatingLogger:
    def __init__(self, log_dir: str, log_file: str = "session_ratings.csv"):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, log_file)
        self._ensure_header()

    def _ensure_header(self):
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "rating", "comment"])

    def log(self, rating: int, comment: str = ""):
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), rating, comment])

    def all_ratings(self):
        if not os.path.exists(self.path):
            return []
        with open(self.path, newline="") as f:
            return list(csv.DictReader(f))
