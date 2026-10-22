"""
input_tracker.py
-----------------
Listens to global mouse and keyboard events using `pynput` and records raw
timestamped events into thread-safe buffers. FeatureAggregator later turns
these raw events into the windowed features the model expects.

Requires: pynput
    pip install pynput

NOTE (platform permissions):
  - macOS: grant "Input Monitoring" and "Accessibility" permissions to your
    terminal / Python in System Settings -> Privacy & Security.
  - Linux (X11): should work out of the box; Wayland may restrict global
    listeners depending on compositor.
  - Windows: works out of the box, run terminal as the same user session.
"""

import time
import threading
import collections

from pynput import mouse, keyboard


class InputTracker:
    def __init__(self, max_buffer_seconds: int = 120):
        self.max_buffer_seconds = max_buffer_seconds
        self._lock = threading.Lock()

        # raw event buffers: each entry is (timestamp, ...)
        self.mouse_positions = collections.deque()   # (t, x, y)
        self.mouse_clicks = collections.deque()      # (t,)
        self.key_events = collections.deque()        # (t, key_name_or_None)
        self.backspace_events = collections.deque()  # (t,)

        # Unbounded (not trimmed like the buffers above) last-activity
        # timestamp, for idle/away detection in daemon.py -- the windowed
        # buffers only cover the last max_buffer_seconds, which is too
        # short a lookback to answer "how long has it been since ANY
        # input happened" once that gap exceeds ~2 minutes.
        self._last_activity_ts = 0.0

        self._mouse_listener = mouse.Listener(
            on_move=self._on_move, on_click=self._on_click
        )
        self._keyboard_listener = keyboard.Listener(
            on_press=self._on_press
        )

    # ---- event handlers (run on listener threads) ----
    def _on_move(self, x, y):
        t = time.time()
        with self._lock:
            self.mouse_positions.append((t, x, y))
            self._trim(self.mouse_positions, t)
            self._last_activity_ts = t

    def _on_click(self, x, y, button, pressed):
        if pressed:
            t = time.time()
            with self._lock:
                self.mouse_clicks.append(t)
                self._trim(self.mouse_clicks, t)
                self._last_activity_ts = t

    def _on_press(self, key):
        t = time.time()
        with self._lock:
            self.key_events.append(t)
            self._trim(self.key_events, t)
            self._last_activity_ts = t
            try:
                is_backspace = key == keyboard.Key.backspace
            except Exception:
                is_backspace = False
            if is_backspace:
                self.backspace_events.append(t)
                self._trim(self.backspace_events, t)

    def last_activity_timestamp(self) -> float:
        """0.0 if no input has ever been seen this process (treated as
        'idle since process start' by callers, not as an error)."""
        with self._lock:
            return self._last_activity_ts

    def _trim(self, buf, now):
        """Drop entries older than max_buffer_seconds to bound memory."""
        cutoff = now - self.max_buffer_seconds
        while buf and (buf[0][0] if isinstance(buf[0], tuple) else buf[0]) < cutoff:
            buf.popleft()

    def start(self):
        self._mouse_listener.start()
        self._keyboard_listener.start()

    def stop(self):
        self._mouse_listener.stop()
        self._keyboard_listener.stop()

    def snapshot(self, window_seconds: float):
        """Return copies of all events within the last `window_seconds`,
        for the aggregator to compute features from."""
        now = time.time()
        cutoff = now - window_seconds
        with self._lock:
            positions = [e for e in self.mouse_positions if e[0] >= cutoff]
            clicks = [e for e in self.mouse_clicks if e >= cutoff]
            keys = [e for e in self.key_events if e >= cutoff]
            backspaces = [e for e in self.backspace_events if e >= cutoff]
        return {
            "mouse_positions": positions,
            "mouse_clicks": clicks,
            "key_events": keys,
            "backspace_events": backspaces,
            "window_seconds": window_seconds,
        }


if __name__ == "__main__":
    # Standalone debug run: prints raw counts every 5 seconds.
    tracker = InputTracker()
    tracker.start()
    print("Listening to mouse/keyboard... press Ctrl+C to quit.")
    try:
        while True:
            time.sleep(5)
            snap = tracker.snapshot(window_seconds=5)
            print(
                f"moves={len(snap['mouse_positions'])}  "
                f"clicks={len(snap['mouse_clicks'])}  "
                f"keys={len(snap['key_events'])}  "
                f"backspaces={len(snap['backspace_events'])}"
            )
    except KeyboardInterrupt:
        pass
    finally:
        tracker.stop()
