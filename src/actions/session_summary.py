"""
session_summary.py
--------------------
Builds the compact SUMMARY dict uploaded to the supervision backend at
session end (see backend_client.upload_session_summary()). Deliberately
the same shape/granularity as what the local dashboard already shows the
student themselves (src/dashboard/app.py's _state_summary /
_app_usage_summary / _blocked_summary) -- state percentages and top apps,
never raw window-by-window predictions or any camera/keystroke content.

Kept as a separate small module (rather than importing dashboard/app.py's
closures) so daemon.py doesn't need to import Flask just to compute a
summary at shutdown.
"""

import csv
import os
from collections import defaultdict
from datetime import datetime


def _read_csv(log_dir: str, filename: str):
    path = os.path.join(log_dir, filename)
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _in_window(timestamp_str: str, start: float, end: float) -> bool:
    try:
        ts = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S").timestamp()
        return start <= ts <= end
    except ValueError:
        return False


def _state_summary(rows):
    counts = defaultdict(int)
    confidence_sums = defaultdict(float)
    for r in rows:
        state = r.get("state", "")
        if not state:
            continue
        counts[state] += 1
        try:
            confidence_sums[state] += float(r.get("confidence", 0))
        except ValueError:
            pass
    idle_count = counts.pop("idle", 0)
    total = sum(counts.values())
    total_for_pct = total or 1
    summary = []
    for state in ["focus", "fatigue", "overload", "distracted"]:
        n = counts.get(state, 0)
        avg_conf = (confidence_sums.get(state, 0) / n) if n else 0
        summary.append({
            "state": state, "count": n,
            "pct": round(100 * n / total_for_pct),
            "avg_confidence": round(100 * avg_conf),
        })
    return summary, total, idle_count


def _app_usage_summary(rows):
    app_totals = defaultdict(float)
    for r in rows:
        breakdown = r.get("app_seconds_breakdown", "")
        if not breakdown:
            continue
        for entry in breakdown.split(";"):
            if ":" not in entry:
                continue
            app, secs = entry.rsplit(":", 1)
            try:
                app_totals[app] += float(secs)
            except ValueError:
                continue
    top_apps = sorted(app_totals.items(), key=lambda kv: -kv[1])[:8]
    return [{"app": a, "seconds": round(s)} for a, s in top_apps]


def _blocked_summary(rows):
    counts = defaultdict(int)
    for r in rows:
        counts[r.get("domain", "unknown")] += 1
    return sorted(counts.items(), key=lambda kv: -kv[1])


def build_session_summary(log_dir: str, session_start: float, session_end: float) -> dict:
    """Returns the payload shape expected by POST /api/sessions."""
    def _filt(rows):
        return [r for r in rows if _in_window(r.get("timestamp", ""), session_start, session_end)]

    state_rows = _filt(_read_csv(log_dir, "state_history.csv"))
    app_rows = _filt(_read_csv(log_dir, "app_usage.csv"))
    blocked_rows = _filt(_read_csv(log_dir, "blocked_attempts.csv"))

    state_summary, total_windows, idle_windows = _state_summary(state_rows)

    return {
        "started_at": session_start,
        "ended_at": session_end,
        "total_windows": total_windows,
        "idle_windows": idle_windows,
        "state_summary": state_summary,
        "top_apps": _app_usage_summary(app_rows),
        "blocked_summary": _blocked_summary(blocked_rows),
    }
