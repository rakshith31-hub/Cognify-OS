"""
app.py (dashboard)
-------------------
A local-only Flask dashboard (binds 127.0.0.1, never 0.0.0.0 — not
reachable from the network) that reads the same CSV logs and SQLite
bandit db the daemon already writes, and presents:

  - Current daemon status (running / not)
  - Today's state distribution + average confidence per state
  - Bandit-learned acceptance rate per (state, action) — same data as
    `cognitive-monitor bandit-stats`, but readable, and with inline
    Accept/Reject buttons for anything still pending feedback (a
    web-based companion to the toast buttons, for whenever you'd rather
    review a batch at once than respond to toasts as they come)
  - App-usage breakdown: top apps by time, and which state was most
    common while each app was active — this is the raw material for
    future app-productivity personalization, surfaced here first as
    plain data before any classification is built on top of it
  - Blocked-site attempt log
  - An end-of-session rating form (1-5, with an optional comment)

Nothing here sends data anywhere; it only reads local files and writes
to local files, same as the rest of the project.
"""

import csv
import os
import sys
import webbrowser
from collections import defaultdict
from datetime import datetime, timedelta

from flask import Flask, render_template, request, redirect, url_for

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from src.actions.bandit import BanditStore
from src.actions.session_rating_logger import SessionRatingLogger

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


def _build_recommendations(state_summary, total_windows, blocked_summary, notification_rows,
                            bandit_stats, site_blocking_enabled=False, daemon_log_tail=""):
    """
    "Changes to be done" — lightweight, rule-based suggestions derived
    directly from this session's own numbers, not a model. Every
    recommendation names the specific number that produced it, so it's
    checkable against the data on the same page rather than a black box.
    """
    recs = []

    # Checked BEFORE the total_windows==0 early-return below: the
    # elevation/permission failure this catches can happen on the very
    # first session (no state predictions logged yet if e.g. the camera
    # also isn't set up), and is exactly the kind of silent failure this
    # dashboard exists to surface -- it shouldn't require a "real" session
    # with logged windows to show up.
    if site_blocking_enabled and not blocked_summary:
        if "Cannot bind port" in daemon_log_tail or "Cannot read/write" in daemon_log_tail:
            recs.append(
                "site_blocking is enabled but nothing was blocked this session, and the "
                "daemon log shows a permission error — this almost always means the daemon "
                "wasn't run elevated (Administrator on Windows, sudo on macOS/Linux). Both "
                "the hosts-file edit and the blocked-page server need that privilege level; "
                "without it the site just loads normally and nothing gets logged."
            )
        elif total_windows > 0:
            # Only suggested once there's been an actual session (state
            # windows logged) -- on a session with zero windows there's
            # nothing meaningful to distinguish "not elevated" from
            # "never actually tried to visit anything yet".
            recs.append(
                "site_blocking is enabled but no blocked-site attempts were logged this "
                "session. If you visited a blocked domain and it loaded normally anyway, "
                "check: (1) the daemon was run elevated, (2) your browser's Secure DNS / "
                "DNS-over-HTTPS is off, (3) the domain in your config is a bare hostname "
                "like 'youtube.com', not a full URL."
            )

    if total_windows == 0:
        return recs

    by_state = {s["state"]: s for s in state_summary}

    distracted = by_state.get("distracted", {"count": 0, "pct": 0})
    if distracted["pct"] >= 30:
        recs.append(
            f"Distracted was your most common state this session ({distracted['pct']}% of windows). "
            "If site blocking isn't enabled yet, turning on `site_blocking.enabled` in "
            "config/default_config.yaml applies it for the whole session, not just when "
            "'distracted' is confidently detected."
        )

    fatigue = by_state.get("fatigue", {"count": 0, "pct": 0})
    if fatigue["pct"] >= 25:
        recs.append(
            f"Fatigue showed up in {fatigue['pct']}% of windows. Consider an earlier/longer "
            "break schedule, or lowering `action_cooldown_seconds` for the fatigue state so "
            "break reminders come more often."
        )

    if blocked_summary:
        top_domain, top_count = blocked_summary[0]
        recs.append(
            f"You visited a blocked site ({top_domain}) {top_count} time(s) anyway this "
            "session. That's not a failure to fix — it's exactly the data this tool is "
            "supposed to surface. If it keeps happening, the block window/domain list "
            "might not match what you actually meant to restrict."
        )

    low_accept = [b for b in bandit_stats if b.get("trials", 0) >= 3 and b.get("learned_acceptance_rate", 1) < 0.3]
    for b in low_accept:
        recs.append(
            f"You've rejected '{b['action_key']}' for {b['state']} most times it's fired "
            f"({round(b['learned_acceptance_rate']*100)}% accepted). Consider removing it from "
            "that state's actions in config/default_config.yaml rather than continuing to dismiss it."
        )

    if not recs:
        recs.append("Nothing stands out from this session's numbers — no specific changes suggested.")
    return recs


def _recent_daemon_log_tail(log_dir: str, filename: str = "daemon.log", max_bytes: int = 20000) -> str:
    """Last chunk of the daemon's own log file — used only to detect the
    specific permission-error strings block_domains()/blocked_site_server.py
    log, so the dashboard can explain a silent site_blocking failure instead
    of just showing an unexplained empty blocked-attempts list."""
    path = os.path.join(log_dir, filename)
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            return f.read().decode("utf-8", errors="ignore")
    except OSError:
        return ""


def create_app(log_dir: str, bandit_db_path: str, session_window=None, site_blocking_enabled=False):
    """
    session_window: optional (start_epoch, end_epoch) tuple. When set, the
    dashboard's default view filters to that window (a single daemon
    session) instead of "today" — used for the end-of-session summary that
    opens automatically when the daemon stops.
    """
    app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
    rating_logger = SessionRatingLogger(log_dir=log_dir)

    # ------------------------------------------------------------------
    # data loading helpers — plain csv module, no pandas dependency here
    # ------------------------------------------------------------------

    def _read_csv(filename):
        path = os.path.join(log_dir, filename)
        if not os.path.exists(path):
            return []
        with open(path, newline="") as f:
            return list(csv.DictReader(f))

    def _is_today(timestamp_str):
        try:
            ts = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            return ts.date() == datetime.now().date()
        except ValueError:
            return False

    def _in_session_window(timestamp_str):
        if session_window is None:
            return True
        try:
            ts = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S").timestamp()
            return session_window[0] <= ts <= session_window[1]
        except ValueError:
            return False

    def _daemon_status():
        pid_file = os.path.join(_PROJECT_ROOT, "logs", "daemon.pid")
        if not os.path.exists(pid_file):
            return False, None
        try:
            with open(pid_file) as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)  # POSIX-style liveness check; on Windows this
            return True, pid  # still raises OSError if the pid is dead
        except Exception:
            return False, None

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
        # "idle" (see daemon.py's idle/away auto-pause) is excluded from
        # the percentage denominator below on purpose -- it's not a
        # cognitive state, it's an absence of one. Mixing it into `total`
        # would deflate every real state's percentage by however long you
        # were away, making "40% focused" secretly mean "40% of a day
        # that includes 3 hours you weren't even at your desk". Shown as
        # its own separate stat instead (see idle_windows in index()).
        idle_count = counts.pop("idle", 0)
        total = sum(counts.values())
        total_for_pct = total or 1  # avoid a division by zero below; the
        # RETURNED total stays the real (possibly 0) count, so callers can
        # still tell "genuinely no data" apart from "1 data point" -- this
        # used to return `total or 1` directly, which meant total_windows
        # was NEVER actually 0 even with zero rows, so the template's
        # "No predictions logged yet" empty-state check could never fire.
        summary = []
        for state in ["focus", "fatigue", "overload", "distracted"]:
            n = counts.get(state, 0)
            avg_conf = (confidence_sums.get(state, 0) / n) if n else 0
            summary.append({
                "state": state,
                "count": n,
                "pct": round(100 * n / total_for_pct),
                "avg_confidence": round(100 * avg_conf),
            })
        return summary, total, idle_count

    def _app_usage_summary(rows):
        app_totals = defaultdict(float)
        state_app_totals = defaultdict(lambda: defaultdict(float))
        for r in rows:
            breakdown = r.get("app_seconds_breakdown", "")
            state = r.get("cognitive_state", "")
            if not breakdown:
                continue
            for entry in breakdown.split(";"):
                if ":" not in entry:
                    continue
                app, secs = entry.rsplit(":", 1)
                try:
                    secs = float(secs)
                except ValueError:
                    continue
                app_totals[app] += secs
                if state:
                    state_app_totals[state][app] += secs

        top_apps = sorted(app_totals.items(), key=lambda kv: -kv[1])[:8]
        max_seconds = top_apps[0][1] if top_apps else 1

        dominant_app_per_state = {}
        for state, apps in state_app_totals.items():
            if apps:
                dominant_app_per_state[state] = max(apps.items(), key=lambda kv: kv[1])[0]

        return (
            [{"app": a, "seconds": round(s), "pct": round(100 * s / max_seconds)} for a, s in top_apps],
            dominant_app_per_state,
        )

    def _blocked_summary(rows):
        counts = defaultdict(int)
        for r in rows:
            counts[r.get("domain", "unknown")] += 1
        return sorted(counts.items(), key=lambda kv: -kv[1])

    def _state_timeline(rows, max_buckets: int = 24):
        """
        Buckets state_history rows (including idle) into up to
        `max_buckets` equal-width time windows spanning the observed
        range, counting each state per bucket -- feeds the "State Over
        Time" stacked bar chart. Bucket width adapts to the actual span
        (a short session gets minute-scale buckets, "all time" gets
        hour/day-scale ones) rather than a fixed unit, so this works
        for both the single-session dashboard and the long-run view.
        """
        parsed = []
        for r in rows:
            try:
                ts = datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S")
            except (ValueError, KeyError):
                continue
            parsed.append((ts, r.get("state", "")))
        if not parsed:
            return {"labels": [], "datasets": {}}

        parsed.sort(key=lambda p: p[0])
        t_min, t_max = parsed[0][0], parsed[-1][0]
        span_seconds = max((t_max - t_min).total_seconds(), 1.0)
        n_buckets = min(max_buckets, max(1, int(span_seconds // 60) + 1))
        bucket_width = span_seconds / n_buckets

        # label format scales with total span so labels stay readable
        # instead of "14:32:07" for a multi-day "all time" view
        label_fmt = "%H:%M" if span_seconds < 6 * 3600 else (
            "%b %d %H:%M" if span_seconds < 3 * 86400 else "%b %d"
        )

        states_order = ["focus", "fatigue", "overload", "distracted", "idle"]
        buckets = [defaultdict(int) for _ in range(n_buckets)]
        labels = []
        for i in range(n_buckets):
            bucket_start = t_min + timedelta(seconds=i * bucket_width)
            labels.append(bucket_start.strftime(label_fmt))

        for ts, state in parsed:
            offset = (ts - t_min).total_seconds()
            idx = min(int(offset // bucket_width), n_buckets - 1)
            if state:
                buckets[idx][state] += 1

        datasets = {s: [buckets[i].get(s, 0) for i in range(n_buckets)] for s in states_order}
        return {"labels": labels, "datasets": datasets}

    # ------------------------------------------------------------------
    # routes
    # ------------------------------------------------------------------

    @app.route("/")
    def index():
        is_session_view = session_window is not None
        show_all = request.args.get("range") == "all"

        state_rows = _read_csv("state_history.csv")
        app_rows = _read_csv("app_usage.csv")
        blocked_rows = _read_csv("blocked_attempts.csv")
        notification_rows = _read_csv("notifications_sent.csv")

        if is_session_view:
            filt = _in_session_window
        elif not show_all:
            filt = _is_today
        else:
            filt = None

        if filt is not None:
            state_rows = [r for r in state_rows if filt(r.get("timestamp", ""))]
            app_rows = [r for r in app_rows if filt(r.get("timestamp", ""))]
            blocked_rows = [r for r in blocked_rows if filt(r.get("timestamp", ""))]
            notification_rows = [r for r in notification_rows if filt(r.get("timestamp", ""))]

        state_summary, total_windows, idle_windows = _state_summary(state_rows)
        top_apps, dominant_app_per_state = _app_usage_summary(app_rows)
        blocked_summary = _blocked_summary(blocked_rows)
        notification_rows = list(reversed(notification_rows))  # most recent first
        timeline = _state_timeline(state_rows)

        bandit_stats = []
        pending = []
        if os.path.exists(bandit_db_path):
            store = BanditStore(bandit_db_path)
            bandit_stats = store.get_stats()
            pending = [
                {"id": eid, "timestamp": ts, "state": state, "action_key": action_key}
                for eid, ts, state, action_key in store.get_pending_events(limit=15)
            ]
            store.close()

        recommendations = _build_recommendations(
            state_summary, total_windows, blocked_summary, notification_rows, bandit_stats,
            site_blocking_enabled=site_blocking_enabled, daemon_log_tail=_recent_daemon_log_tail(log_dir),
        )

        running, pid = _daemon_status()
        ratings = rating_logger.all_ratings()
        avg_rating = (
            round(sum(float(r["rating"]) for r in ratings) / len(ratings), 1)
            if ratings else None
        )

        return render_template(
            "index.html",
            running=running, pid=pid,
            is_session_view=is_session_view,
            state_summary=state_summary, total_windows=total_windows, idle_windows=idle_windows,
            timeline=timeline,
            top_apps=top_apps, dominant_app_per_state=dominant_app_per_state,
            blocked_summary=blocked_summary,
            notification_rows=notification_rows,
            recommendations=recommendations,
            bandit_stats=bandit_stats, pending=pending,
            avg_rating=avg_rating, rating_count=len(ratings),
            show_all=show_all,
        )

    @app.route("/feedback", methods=["POST"])
    def feedback():
        event_id = int(request.form["event_id"])
        accepted = request.form["response"] == "up"
        if os.path.exists(bandit_db_path):
            store = BanditStore(bandit_db_path)
            store.record_feedback(event_id, accepted)
            store.close()
        return redirect(url_for("index", range="all" if request.args.get("range") == "all" else None))

    @app.route("/rating", methods=["POST"])
    def rating():
        rating_value = int(request.form["rating"])
        comment = request.form.get("comment", "").strip()
        rating_logger.log(rating_value, comment)
        return redirect(url_for("index"))

    return app


def run_dashboard(config_path: str, port: int = 5757, open_browser: bool = True):
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    general = cfg.get("general", {})
    log_dir = os.path.join(_PROJECT_ROOT, general.get("log_dir", "logs"))
    bandit_db_path = os.path.join(log_dir, general.get("bandit_db_file", "bandit.db"))

    app = create_app(log_dir, bandit_db_path, site_blocking_enabled=bool(cfg.get("site_blocking", {}).get("enabled")))
    url = f"http://127.0.0.1:{port}"
    print(f"Dashboard running at {url}  (Ctrl+C to stop)")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    app.run(host="127.0.0.1", port=port, debug=False)


def run_dashboard_for_session(config: dict, project_root: str, session_start: float,
                               session_end: float, port: int = 5757):
    """
    Called by daemon.py at the end of a session: opens the dashboard
    pre-filtered to just this session's window (not "today", which could
    span several earlier daemon runs) and opens the browser to it
    automatically, rather than requiring `cognitive-monitor dashboard` to
    be run manually afterward.

    Blocking (app.run()) is intentional here — this runs as the daemon
    process's last step, keeping the summary server alive so the browser
    tab stays usable until the person closes it with Ctrl+C, rather than
    tearing the server down the instant it's opened.
    """
    general = config.get("general", {})
    log_dir = os.path.join(project_root, general.get("log_dir", "logs"))
    bandit_db_path = os.path.join(log_dir, general.get("bandit_db_file", "bandit.db"))

    app = create_app(log_dir, bandit_db_path, session_window=(session_start, session_end),
                      site_blocking_enabled=bool(config.get("site_blocking", {}).get("enabled")))
    url = f"http://127.0.0.1:{port}"
    print(f"Session summary at {url}  (Ctrl+C to close)")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    run_dashboard(os.path.join(_PROJECT_ROOT, "config", "default_config.yaml"))
