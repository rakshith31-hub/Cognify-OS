"""
notification_logger.py
------------------------
Logs every notification/alarm THIS APP fires — toast, break popup, or
alarm — regardless of whether Windows actually displayed it.

Why this exists: notifications.py already notes there's no public
scripting API for Windows Focus Assist, so we can't actually know whether
DND suppressed a given notification, nor can we intercept/queue other
apps' notifications during DND. What we CAN do is log every notification
attempt WE make, so the end-of-session dashboard can show "here's
everything that fired this session" — a practical proxy for "here's what
you might have missed while notifications were paused", even though it's
strictly our own notifications, not a true system-wide notification
history.
"""

import csv
import os
import time


class NotificationLogger:
    def __init__(self, log_dir: str, log_file: str = "notifications_sent.csv"):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, log_file)
        self._ensure_header()

    def _ensure_header(self):
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "kind", "state", "title", "message"])

    def log(self, kind: str, state: str, title: str, message: str = ""):
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), kind, state or "", title, message])
