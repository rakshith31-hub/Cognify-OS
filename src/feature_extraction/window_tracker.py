"""
window_tracker.py
------------------
NEW capability (not present in the original project): tracks which
application is in the foreground over time, by polling the OS at a fixed
interval on a background thread — the same "listener + buffer + snapshot"
shape as InputTracker, so it plugs into the daemon the same way.

Windows-only, by design decision (see project chat history: platform was
finalized as Windows-only for this feature). Requires `pywin32` and
`psutil`.

WHAT IT CAPTURES: only the foreground process's executable name (e.g.
"chrome.exe", "Code.exe", "notepad.exe") and, optionally, the window
title. Process name is captured by default; window title capture is
OFF by default because titles can contain sensitive content (document
names, browser tab text, email subject lines) — turn it on explicitly
via `capture_titles=True` only if you specifically want per-title
granularity and understand that trade-off.

This module does NOT decide what's "productive" vs "distracting" — it
only produces the raw (timestamp, app_name) stream. Classification is a
separate, later component that learns from user feedback.
"""

import collections
import logging
import platform
import threading
import time

logger = logging.getLogger("cognitive_monitor.window_tracker")

_SYSTEM = platform.system()

POLL_INTERVAL_SECONDS = 2.0
DEFAULT_MAX_BUFFER_SECONDS = 3600  # keep up to 1hr of app-switch history


class WindowTracker:
    """
    Polls the OS every POLL_INTERVAL_SECONDS for the foreground app and
    keeps a rolling, thread-safe buffer of (timestamp, app_name) samples.

    Usage mirrors InputTracker:
        tracker = WindowTracker()
        tracker.start()
        ...
        snap = tracker.snapshot(window_seconds=10)
        tracker.stop()
    """

    def __init__(self, capture_titles: bool = False,
                 poll_interval: float = POLL_INTERVAL_SECONDS,
                 max_buffer_seconds: int = DEFAULT_MAX_BUFFER_SECONDS):
        self.capture_titles = capture_titles
        self.poll_interval = poll_interval
        self.max_buffer_seconds = max_buffer_seconds

        self._lock = threading.Lock()
        self._samples = collections.deque()  # (timestamp, app_name, title_or_None)
        self._running = False
        self._thread = None

        self._backend = self._init_backend()

    # ------------------------------------------------------------------

    def _init_backend(self):
        if _SYSTEM != "Windows":
            logger.warning(
                "WindowTracker is Windows-only in this build (platform was "
                "finalized as Windows-only for this feature). Detected "
                "system: %s. App-usage tracking will be disabled; "
                "everything else keeps working without it.",
                _SYSTEM,
            )
            return None
        try:
            import win32gui
            import win32process
            import psutil
            return {"win32gui": win32gui, "win32process": win32process, "psutil": psutil}
        except ImportError as e:
            logger.warning(
                "WindowTracker requires pywin32 + psutil on Windows "
                "(pip install pywin32 psutil). App-usage tracking disabled "
                "until installed. Error: %s", e,
            )
            return None

    def _read_foreground_app(self):
        """Returns (app_name, title_or_None) for the current foreground
        window, or (None, None) if unavailable (e.g. no window focused,
        or a permissions/API hiccup — never raises, always degrades)."""
        if self._backend is None:
            return None, None
        try:
            win32gui = self._backend["win32gui"]
            win32process = self._backend["win32process"]
            psutil = self._backend["psutil"]

            hwnd = win32gui.GetForegroundWindow()
            if not hwnd:
                return None, None

            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            if not pid:
                return None, None

            try:
                proc = psutil.Process(pid)
                app_name = proc.name()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return None, None

            title = None
            if self.capture_titles:
                title = win32gui.GetWindowText(hwnd) or None

            return app_name, title
        except Exception as e:
            logger.debug("Foreground window read failed (%s): %s", type(e).__name__, e)
            return None, None

    # ------------------------------------------------------------------

    def start(self):
        if self._backend is None:
            logger.info("WindowTracker.start() called but no backend available — no-op.")
            return
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("WindowTracker started (poll every %.1fs, titles=%s).",
                    self.poll_interval, self.capture_titles)

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self.poll_interval + 1.0)

    def _poll_loop(self):
        while self._running:
            app_name, title = self._read_foreground_app()
            if app_name is not None:
                t = time.time()
                with self._lock:
                    self._samples.append((t, app_name, title))
                    self._trim(t)
            time.sleep(self.poll_interval)

    def _trim(self, now):
        cutoff = now - self.max_buffer_seconds
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    # ------------------------------------------------------------------

    def snapshot(self, window_seconds: float) -> dict:
        """
        Returns a summary of foreground-app activity over the last
        `window_seconds`:
          {
            "samples": [(t, app_name, title_or_None), ...],
            "app_seconds": {app_name: approx_seconds_in_foreground, ...},
            "dominant_app": app_name_or_None,   # most time in this window
            "app_switch_count": int,             # distinct app transitions
            "window_seconds": window_seconds,
          }
        `app_seconds` is approximate: each sample represents roughly one
        `poll_interval` of foreground time, so accuracy is bounded by the
        poll rate (2s default -> +/-2s granularity, which is fine for the
        10s prediction windows this feeds).
        """
        now = time.time()
        cutoff = now - window_seconds
        with self._lock:
            samples = [s for s in self._samples if s[0] >= cutoff]

        app_seconds = collections.defaultdict(float)
        for _, app_name, _ in samples:
            app_seconds[app_name] += self.poll_interval

        dominant_app = max(app_seconds, key=app_seconds.get) if app_seconds else None

        switch_count = 0
        prev_app = None
        for _, app_name, _ in samples:
            if prev_app is not None and app_name != prev_app:
                switch_count += 1
            prev_app = app_name

        return {
            "samples": samples,
            "app_seconds": dict(app_seconds),
            "dominant_app": dominant_app,
            "app_switch_count": switch_count,
            "window_seconds": window_seconds,
        }


if __name__ == "__main__":
    # Standalone debug run (Windows only): prints the foreground app every
    # 5 seconds. Press Ctrl+C to quit.
    logging.basicConfig(level=logging.INFO)
    tracker = WindowTracker()
    tracker.start()
    print("Tracking foreground app... press Ctrl+C to quit.")
    try:
        while True:
            time.sleep(5)
            snap = tracker.snapshot(window_seconds=5)
            print(f"dominant_app={snap['dominant_app']}  "
                  f"app_seconds={snap['app_seconds']}  "
                  f"switches={snap['app_switch_count']}")
    except KeyboardInterrupt:
        pass
    finally:
        tracker.stop()
