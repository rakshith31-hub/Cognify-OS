"""
block_attempt_logger.py
------------------------
Logs every connection attempt to a domain the user has blocked — this is
the "log how often you try anyway" data point. Local CSV only, same
pattern as the other loggers in this project.
"""

import csv
import os
import time


class BlockAttemptLogger:
    def __init__(self, log_dir: str, log_file: str = "blocked_attempts.csv", on_attempt=None):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, log_file)
        self._ensure_header()
        # Optional callback fired on every logged attempt (domain, protocol,
        # path) -- used to trigger a real-time deterrent (e.g. flash-dimming
        # brightness) the instant a blocked site is actually visited, not
        # just for passive after-the-fact logging.
        self.on_attempt = on_attempt

    def _ensure_header(self):
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "domain", "protocol", "path"])

    def log(self, domain: str, protocol: str, path: str = ""):
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), domain, protocol, path])
        if self.on_attempt is not None:
            try:
                self.on_attempt(domain, protocol, path)
            except Exception:
                pass  # a deterrent-callback failure should never break logging
