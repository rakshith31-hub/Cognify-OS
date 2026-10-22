#!/usr/bin/env python3
"""
cli.py
------
The command the user actually runs. Installed as the `cognitive-monitor`
console script (see pyproject.toml), or usable directly with
`python3 cli.py <command>`.

Commands:
    cognitive-monitor start [--mode full|input-only] [--camera-index N]
                             [--config path] [--foreground]
    cognitive-monitor stop
    cognitive-monitor status
    cognitive-monitor logs [--follow] [--lines N]
    cognitive-monitor login --server URL --email E [--signup --role teacher|student]
    cognitive-monitor link <class_code>          (students only)
    cognitive-monitor logout
    cognitive-monitor whoami

`start` detaches into the background by default (writes stdout/stderr to
logs/daemon.log and returns your shell immediately) — use --foreground to
keep it attached for debugging.
"""

import argparse
import ctypes
import os
import platform
import subprocess
import sys
import time


def _is_elevated() -> bool:
    """True if running as Administrator (Windows) or root (macOS/Linux).
    Site blocking (hosts-file writes + binding ports 80/443) needs this;
    the daemon degrades gracefully without it (everything else still
    works), but it does so SILENTLY into a log file when launched in the
    default detached mode — see cmd_start below for why that's a problem
    worth checking for up front instead."""
    try:
        if platform.system() == "Windows":
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        return os.geteuid() == 0
    except Exception:
        return False

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _resolve_project_relative(path: str, must_exist: bool = True) -> str:
    """
    Resolves a user-supplied path (e.g. --config, --output) sensibly
    regardless of the shell's current working directory.

    A relative path like "config/user_config.yaml" is meant to refer to
    THIS PROJECT's own config folder -- that's the whole point of it being
    a short, memorable relative path in docs/examples. But Python resolves
    relative paths against the process's CWD, not the script's location,
    so running the exact same command from a different directory (very
    normal for a CLI tool -- you shouldn't have to `cd` into the install
    folder to use one) silently breaks with a confusing FileNotFoundError
    instead of doing what was obviously intended.

    must_exist=True (the default, for paths like --config that must
    already exist to be read): a relative path is tried against
    PROJECT_ROOT first, and only falls back to CWD-relative if that
    doesn't exist -- so a genuine CWD-relative path (someone really does
    have their own config file in their current directory) still works.

    must_exist=False (for paths like --output that are about to be
    CREATED, e.g. a fresh user_config.yaml on the very first setup run --
    it's expected NOT to exist yet, so an existence check would always
    fail and silently defeat the whole point): relative paths always
    resolve against PROJECT_ROOT unconditionally.

    Absolute paths pass through untouched either way.
    """
    if os.path.isabs(path):
        return path
    project_relative = os.path.join(PROJECT_ROOT, path)
    if not must_exist or os.path.exists(project_relative):
        return project_relative
    return path


def _require_config_exists(path: str):
    """Every cmd_* function calls this before touching args.config -- turns
    a raw FileNotFoundError traceback into a clear, actionable message,
    since the single most common cause (confirmed from real user reports)
    is simply never having run `cognitive-monitor setup` yet."""
    if not os.path.exists(path):
        print(f"Config file not found: {path}")
        print("If you haven't set one up yet, run 'cognitive-monitor setup' first "
              "(creates config/user_config.yaml), or pass --config pointing at an "
              "existing file (e.g. the bundled config/default_config.yaml).")
        sys.exit(1)


def _default_config_path() -> str:
    """
    BUG FIX: every --config argument used to default to
    config/default_config.yaml unconditionally, on every single command
    (start, feedback, dashboard, validate-config, label, retrain, ...).
    But `cognitive-monitor setup` writes to config/user_config.yaml, not
    default_config.yaml -- so anyone who ran the wizard, answered "yes"
    to site blocking, and typed in domains, would have every one of those
    answers silently ignored the moment they ran a bare
    `cognitive-monitor start` without remembering to also pass
    `--config config/user_config.yaml` by hand, forever, on every future
    command. Nothing errored: default_config.yaml is a fully valid,
    complete config on its own (that's the whole point of it being the
    bundled default) -- it just has site_blocking.enabled: false and an
    empty domain list as its conservative, opt-in-required baseline. So
    the daemon would run perfectly, every other feature would work
    exactly as documented, and the ONE feature that requires the wizard's
    input would just silently never activate. That's this exact bug
    report ("everything is proper - one mistake is that sites are not
    getting blocked").

    Fixed by making the default itself smarter: prefer
    config/user_config.yaml if it exists (i.e. the wizard has been run at
    least once), and only fall back to config/default_config.yaml if it
    hasn't. `--config` still overrides this explicitly when passed.
    """
    user_config = os.path.join(PROJECT_ROOT, "config", "user_config.yaml")
    if os.path.exists(user_config):
        return user_config
    return os.path.join(PROJECT_ROOT, "config", "default_config.yaml")


PID_FILE = os.path.join(PROJECT_ROOT, "logs", "daemon.pid")
LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "daemon.log")
DAEMON_SCRIPT = os.path.join(PROJECT_ROOT, "src", "realtime", "daemon.py")


def _read_pid():
    if not os.path.exists(PID_FILE):
        return None
    try:
        with open(PID_FILE) as f:
            return int(f.read().strip())
    except (ValueError, FileNotFoundError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def cmd_start(args):
    existing = _read_pid()
    if existing and _pid_alive(existing):
        print(f"Already running (pid {existing}). Use 'cognitive-monitor stop' first if you want to restart.")
        return

    # Always say out loud which config is actually in use. This used to be
    # silent -- combined with every --config default being hardcoded to
    # default_config.yaml (see _default_config_path()'s docstring), that
    # silence is exactly what let a wizard-configured user_config.yaml go
    # unused indefinitely without anyone noticing. Even now that the
    # default is smarter, printing this closes the loop for good: if this
    # ever says something other than what you expect, that's the bug,
    # right here, immediately -- not discovered later via "why isn't X
    # working."
    print(f"Using config: {args.config}")

    sys.path.insert(0, PROJECT_ROOT)
    from src.config_validator import validate_config_file
    result = validate_config_file(args.config)
    for w in result.warnings:
        print(f"⚠ {w}")
    if not result.ok:
        print(f"\n{len(result.errors)} config error(s) — not starting:")
        for e in result.errors:
            print(f"  ✗ {e}")
        print(f"\nRun 'cognitive-monitor validate-config' anytime, or "
              f"'cognitive-monitor setup' to regenerate a clean config.")
        return

    import yaml
    with open(args.config) as f:
        _cfg = yaml.safe_load(f)
    _site_blocking_wanted = bool(_cfg.get("site_blocking", {}).get("enabled"))
    _per_state_blocking_wanted = any(
        a.get("type") == "block_domains" and a.get("enabled")
        for s in _cfg.get("states", {}).values()
        for a in s.get("actions", [])
    )
    if (_site_blocking_wanted or _per_state_blocking_wanted) and not _is_elevated():
        elevate_hint = (
            "right-click your terminal and choose 'Run as administrator', then run "
            "'cognitive-monitor start' again"
            if platform.system() == "Windows" else
            "re-run with 'sudo cognitive-monitor start'"
        )
        print(
            "⚠ Site blocking is enabled in your config, but this terminal is NOT "
            "running elevated. Blocking needs the same privilege level as editing "
            "the hosts file and binding ports 80/443 — without it, BOTH fail "
            f"SILENTLY: the site loads completely normally, unblocked, and (since "
            f"the daemon runs detached by default) that failure only ever shows up "
            f"in 'cognitive-monitor logs', never in this terminal. To actually "
            f"block anything this session, {elevate_hint}.\n"
            "Starting anyway — everything else (state detection, notifications, "
            "dashboard) works fine without elevation, just not site blocking."
        )

    cmd = [
        sys.executable, DAEMON_SCRIPT,
        "--mode", args.mode,
        "--camera-index", str(args.camera_index),
        "--config", args.config,
    ]
    if args.verbose:
        cmd.append("--verbose")

    if args.foreground:
        subprocess.run(cmd)
        return

    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    popen_kwargs = dict(
        stdout=open(LOG_FILE, "a"),
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        cwd=PROJECT_ROOT,
    )
    if platform.system() == "Windows":
        # BUG FIX: start_new_session is POSIX-only (silently ignored on
        # Windows) so the daemon previously shared this console's process
        # group -- meaning cmd_stop had no way to signal it alone without
        # also hitting this shell. CREATE_NEW_PROCESS_GROUP gives it its own
        # group so a later CTRL_BREAK_EVENT (see cmd_stop) can target just
        # the daemon and actually trigger its graceful-shutdown handler
        # instead of force-killing it.
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True  # detach from this terminal (POSIX)
    subprocess.Popen(cmd, **popen_kwargs)
    time.sleep(1.0)  # give it a moment to write the pid file
    pid = _read_pid()
    if pid:
        print(f"Started in background (pid {pid}). Logs: {LOG_FILE}")
        print("Run 'cognitive-monitor status' or 'cognitive-monitor stop' anytime.")
    else:
        print("Started, but couldn't confirm pid yet — check logs/daemon.log.")


def cmd_stop(args):
    pid = _read_pid()
    if not pid or not _pid_alive(pid):
        print("Not currently running.")
        return
    if platform.system() == "Windows":
        # BUG FIX: os.kill(pid, 15) on Windows does NOT deliver SIGTERM --
        # there's no cross-process signal delivery on Windows, so this
        # silently fell back to TerminateProcess(), which kills the daemon
        # instantly and skips its try/finally shutdown (DND/brightness/
        # color-filter/high-contrast/taskbar/hosts-block never revert).
        # CTRL_BREAK_EVENT is the one signal Windows actually delivers
        # cross-process, and daemon.py now handles it (via SIGBREAK) the
        # same way it handles SIGTERM on macOS/Linux. Requires the daemon to
        # have been started with CREATE_NEW_PROCESS_GROUP (see cmd_start).
        import signal as _signal
        os.kill(pid, _signal.CTRL_BREAK_EVENT)
    else:
        os.kill(pid, 15)  # SIGTERM -> daemon's own handler reverts DND/brightness/hosts before exiting
    for _ in range(20):
        if not _pid_alive(pid):
            break
        time.sleep(0.5)
    if _pid_alive(pid):
        print(f"pid {pid} did not exit cleanly after 10s; sending SIGKILL (system changes may not revert cleanly).")
        os.kill(pid, 9)
    else:
        print("Stopped. Any active Do Not Disturb / brightness / domain-block changes have been reverted.")


def cmd_status(args):
    pid = _read_pid()
    if pid and _pid_alive(pid):
        print(f"Running (pid {pid}).")
    else:
        print("Not running.")


def cmd_logs(args):
    if not os.path.exists(LOG_FILE):
        print("No log file yet — has the daemon been started?")
        return
    if args.follow:
        subprocess.run(["tail", "-f", "-n", str(args.lines), LOG_FILE])
    else:
        subprocess.run(["tail", "-n", str(args.lines), LOG_FILE])


def _bandit_db_path(config_path: str):
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    general = cfg.get("general", {})
    return os.path.join(PROJECT_ROOT, general.get("log_dir", "logs"),
                         general.get("bandit_db_file", "bandit.db"))


def cmd_feedback(args):
    sys.path.insert(0, PROJECT_ROOT)
    from src.actions.bandit import BanditStore

    db_path = _bandit_db_path(args.config)
    if not os.path.exists(db_path):
        print("No bandit database yet — nothing has fired since the daemon was last started with this feature.")
        return

    store = BanditStore(db_path)
    accepted = args.response == "up"
    ok = store.record_feedback(args.event_id, accepted)
    if ok:
        print(f"Recorded '{args.response}' for event {args.event_id}. "
              f"Run 'cognitive-monitor bandit-stats' to see the updated learned rate.")
    else:
        print(f"No action event with id {args.event_id} — check 'cognitive-monitor logs' "
              f"for the '#<id>' next to a recently fired action.")
    store.close()


def cmd_label(args):
    """`cognitive-monitor label <state>` -- attaches a real ground-truth
    label to the most recently computed feature window (written by the
    running daemon to logs/last_features.json on every real tick, see
    daemon.py's _write_last_features). This is the actual data-collection
    half of closing the synthetic-data gap -- see README."""
    sys.path.insert(0, PROJECT_ROOT)
    import json
    import yaml
    from src.actions.real_label_logger import RealLabelLogger

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    general = cfg.get("general", {})
    log_dir = os.path.join(PROJECT_ROOT, general.get("log_dir", "logs"))
    max_age = general.get("label_max_age_seconds", 120)

    last_features_path = os.path.join(log_dir, "last_features.json")
    if not os.path.exists(last_features_path):
        print("No recent window to label yet — the daemon needs to be running "
              "(cognitive-monitor start) and have completed at least one tick.")
        return

    with open(last_features_path) as f:
        data = json.load(f)

    age = time.time() - data["timestamp"]
    if age > max_age:
        print(f"The most recent computed window is {age:.0f}s old (max age: {max_age}s) — "
              f"too stale to label reliably (you might not have been doing the same thing "
              f"then as now). Wait for the daemon's next tick and try again.")
        return

    logger_ = RealLabelLogger(log_dir=log_dir)
    logger_.log(data["features"], args.state)
    counts = logger_.count()
    total = sum(counts.values())
    print(f"Logged '{args.state}' for a window from {age:.0f}s ago "
          f"(model had predicted: {data.get('predicted_state')}, "
          f"{data.get('confidence', 0)*100:.0f}% confidence).")
    print(f"Real labels collected so far: {counts} ({total} total).")
    if total >= 30 and min(counts.values()) >= 8:
        print("Enough data to retrain — run 'cognitive-monitor retrain' when you're ready.")


def cmd_retrain(args):
    sys.path.insert(0, PROJECT_ROOT)
    import yaml
    from src.model.retrain_from_real_data import retrain

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    log_dir = os.path.join(PROJECT_ROOT, cfg.get("general", {}).get("log_dir", "logs"))

    result = retrain(log_dir, force=args.force, max_regression=args.max_regression)
    print(result["message"])
    if result["status"] == "ok" and "real_holdout_report" in result:
        print()
        print(result["real_holdout_report"])


def cmd_feedback_uri(args):
    """Called by the registered cogmonitor:// protocol handler when a toast
    button is clicked. Dispatches by netloc since ONE registered protocol
    handler serves both feedback buttons (cogmonitor://feedback?...) and
    label buttons (cogmonitor://label?...) -- see request_label()/
    request_feedback() in notifications.py, which both point at this same
    registered command."""
    from urllib.parse import urlparse, parse_qs

    parsed = urlparse(args.uri)
    params = parse_qs(parsed.query)

    if parsed.netloc == "label":
        try:
            state = params["state"][0]
        except (KeyError, IndexError):
            print(f"Could not parse label URI: {args.uri}")
            return
        fake_args = argparse.Namespace(state=state, config=args.config)
        cmd_label(fake_args)
        return

    # default / "feedback" netloc -- preserves pre-existing URIs from
    # before netloc-based dispatch existed (cogmonitor://feedback?...)
    try:
        event_id = int(params["event"][0])
        response = params["response"][0]
    except (KeyError, IndexError, ValueError):
        print(f"Could not parse feedback URI: {args.uri}")
        return

    fake_args = argparse.Namespace(event_id=event_id, response=response, config=args.config)
    cmd_feedback(fake_args)


def cmd_pending(args):
    sys.path.insert(0, PROJECT_ROOT)
    from src.actions.bandit import BanditStore

    db_path = _bandit_db_path(args.config)
    if not os.path.exists(db_path):
        print("No bandit database yet.")
        return
    store = BanditStore(db_path)
    rows = store.get_pending_events(limit=args.limit)
    store.close()
    if not rows:
        print("Nothing awaiting feedback.")
        return
    print(f"{'ID':<6}{'Time':<21}{'State':<14}{'Action'}")
    for event_id, ts, state, action_key in rows:
        print(f"{event_id:<6}{ts:<21}{state:<14}{action_key}")
    print(f"\nRespond with: cognitive-monitor feedback <ID> up|down")


def cmd_dashboard(args):
    sys.path.insert(0, PROJECT_ROOT)
    from src.dashboard.app import run_dashboard
    run_dashboard(args.config, port=args.port, open_browser=not args.no_browser)


def cmd_bandit_stats(args):
    sys.path.insert(0, PROJECT_ROOT)
    from src.actions.bandit import BanditStore

    db_path = _bandit_db_path(args.config)
    if not os.path.exists(db_path):
        print("No bandit database yet — nothing has fired since the daemon was last started with this feature.")
        return

    import yaml
    with open(args.config) as f:
        cold_start_trials = yaml.safe_load(f).get("general", {}).get("bandit_cold_start_trials", 5)

    store = BanditStore(db_path)
    stats = store.get_stats()
    store.close()
    if not stats:
        print("No arms recorded yet.")
        return
    print(f"{'State':<14}{'Action':<18}{'Trials':<8}{'Learned accept rate'}")
    for s in stats:
        cold = " (cold start)" if s["trials"] < cold_start_trials else ""
        print(f"{s['state']:<14}{s['action_key']:<18}{s['trials']:<8}"
              f"{s['learned_acceptance_rate']:.0%}{cold}")


def _backend_client():
    sys.path.insert(0, PROJECT_ROOT)
    from src.actions.backend_client import BackendClient
    return BackendClient(log_dir=os.path.join(PROJECT_ROOT, "logs"))


def cmd_login(args):
    import getpass
    import requests

    client = _backend_client()
    password = args.password or getpass.getpass("Password: ")
    try:
        if args.signup:
            role = args.role or input("Role (teacher/student): ").strip().lower()
            display_name = args.display_name or input("Display name: ").strip()
            data = client.signup(args.server, args.email, password, role, display_name)
            print(f"Account created as {data['role']} ({data['display_name']}).")
            if data.get("class_code"):
                print(f"Your class code (share this with students): {data['class_code']}")
        else:
            data = client.login(args.server, args.email, password)
            print(f"Logged in as {data['role']} ({data['display_name']}) at {args.server}")
            if data.get("class_code"):
                print(f"Your class code (share this with students): {data['class_code']}")
    except requests.exceptions.ConnectionError:
        print(f"Could not reach {args.server} — check the URL and that the backend is running.")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        try:
            msg = e.response.json().get("error", str(e))
        except Exception:
            msg = str(e)
        print(f"Login failed: {msg}")
        sys.exit(1)


def cmd_link(args):
    import requests

    client = _backend_client()
    try:
        data = client.link(args.class_code)
        print(f"Linked to teacher: {data['linked_to']}")
        print("The teacher's blocked-domain list will apply starting your next session.")
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("Could not reach the backend — check your connection.")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        try:
            msg = e.response.json().get("error", str(e))
        except Exception:
            msg = str(e)
        print(f"Linking failed: {msg}")
        sys.exit(1)


def cmd_logout(args):
    client = _backend_client()
    client.logout()
    print("Logged out of the supervision backend (local session history stays on this machine).")


def cmd_whoami(args):
    client = _backend_client()
    who = client.whoami()
    if not who:
        print("Not logged in to a supervision backend. Run `cognitive-monitor login` to set one up.")
        return
    print(f"{who['display_name']} ({who['role']}) at {who['server_url']}")


def cmd_validate_config(args):
    sys.path.insert(0, PROJECT_ROOT)
    from src.config_validator import validate_config_file
    result = validate_config_file(args.config)
    print(result.format_report())
    if not result.ok:
        sys.exit(1)


def cmd_setup(args):
    sys.path.insert(0, PROJECT_ROOT)
    from src.setup_wizard import run_wizard
    run_wizard(
        template_path=os.path.join(PROJECT_ROOT, "config", "default_config.yaml"),
        output_path=args.output,
    )


def main():
    parser = argparse.ArgumentParser(prog="cognitive-monitor")
    sub = parser.add_subparsers(dest="command", required=True)

    p_start = sub.add_parser("start", help="Start the background monitor")
    p_start.add_argument("--mode", choices=["full", "input-only"], default="full",
                          help="'full' = webcam + mouse/keyboard (default). "
                               "'input-only' = mouse/keyboard ONLY, no camera.")
    p_start.add_argument("--camera-index", type=int, default=0)
    p_start.add_argument("--config", default=_default_config_path())
    p_start.add_argument("--foreground", action="store_true", help="Stay attached (for debugging)")
    p_start.add_argument("--verbose", action="store_true")
    p_start.set_defaults(func=cmd_start)

    p_stop = sub.add_parser("stop", help="Stop the background monitor")
    p_stop.set_defaults(func=cmd_stop)

    p_status = sub.add_parser("status", help="Check whether it's running")
    p_status.set_defaults(func=cmd_status)

    p_logs = sub.add_parser("logs", help="View daemon logs")
    p_logs.add_argument("--follow", "-f", action="store_true")
    p_logs.add_argument("--lines", "-n", type=int, default=50)
    p_logs.set_defaults(func=cmd_logs)

    p_feedback = sub.add_parser("feedback", help="Tell the adaptive layer whether a fired action was good")
    p_feedback.add_argument("event_id", type=int, help="The #ID shown next to a fired action in the logs")
    p_feedback.add_argument("response", choices=["up", "down"])
    p_feedback.add_argument("--config", default=_default_config_path())
    p_feedback.set_defaults(func=cmd_feedback)

    p_feedback_uri = sub.add_parser("feedback-uri", help="(internal) called by toast feedback/label buttons")
    p_feedback_uri.add_argument("uri")
    p_feedback_uri.add_argument("--config", default=_default_config_path())
    p_feedback_uri.set_defaults(func=cmd_feedback_uri)

    p_pending = sub.add_parser("pending", help="List recently fired actions still awaiting feedback")
    p_pending.add_argument("--limit", type=int, default=20)
    p_pending.add_argument("--config", default=_default_config_path())
    p_pending.set_defaults(func=cmd_pending)

    p_bandit = sub.add_parser("bandit-stats", help="Show what the adaptive layer has learned so far")
    p_bandit.add_argument("--config", default=_default_config_path())
    p_bandit.set_defaults(func=cmd_bandit_stats)

    p_dashboard = sub.add_parser("dashboard", help="Launch the local web dashboard")
    p_dashboard.add_argument("--port", type=int, default=5757)
    p_dashboard.add_argument("--no-browser", action="store_true", help="Don't auto-open a browser tab")
    p_dashboard.add_argument("--config", default=_default_config_path())
    p_dashboard.set_defaults(func=cmd_dashboard)

    p_validate = sub.add_parser("validate-config", help="Check a config file for errors/warnings without starting")
    p_validate.add_argument("--config", default=_default_config_path())
    p_validate.set_defaults(func=cmd_validate_config)

    p_setup = sub.add_parser("setup", help="Guided wizard to generate a working config from scratch")
    p_setup.add_argument("--output", default=os.path.join(PROJECT_ROOT, "config", "user_config.yaml"))
    p_setup.set_defaults(func=cmd_setup)

    p_label = sub.add_parser("label", help="Attach a real ground-truth label to the most recent window")
    p_label.add_argument("state", choices=["focus", "fatigue", "overload", "distracted"])
    p_label.add_argument("--config", default=_default_config_path())
    p_label.set_defaults(func=cmd_label)

    p_retrain = sub.add_parser("retrain", help="Retrain the classifier on real + synthetic data")
    p_retrain.add_argument("--config", default=_default_config_path())
    p_retrain.add_argument("--force", action="store_true",
                            help="Deploy even if real-holdout accuracy regressed vs. last retrain")
    p_retrain.add_argument("--max-regression", type=float, default=0.10)
    p_retrain.set_defaults(func=cmd_retrain)

    p_login = sub.add_parser("login", help="Log in to a teacher/supervision backend")
    p_login.add_argument("--server", required=True, help="Backend URL, e.g. https://cogmon.myschool.org")
    p_login.add_argument("--email", required=True)
    p_login.add_argument("--password", help="Omit to be prompted (recommended, keeps it out of shell history)")
    p_login.add_argument("--signup", action="store_true", help="Create a new account instead of logging into an existing one")
    p_login.add_argument("--role", choices=["teacher", "student"], help="Only used with --signup")
    p_login.add_argument("--display-name", help="Only used with --signup")
    p_login.set_defaults(func=cmd_login)

    p_link = sub.add_parser("link", help="(students) Link this machine to a teacher via their class code")
    p_link.add_argument("class_code")
    p_link.set_defaults(func=cmd_link)

    p_logout = sub.add_parser("logout", help="Forget stored backend credentials on this machine")
    p_logout.set_defaults(func=cmd_logout)

    p_whoami = sub.add_parser("whoami", help="Show which backend account this machine is logged into")
    p_whoami.set_defaults(func=cmd_whoami)

    args = parser.parse_args()
    if hasattr(args, "config"):
        args.config = _resolve_project_relative(args.config, must_exist=True)
        _require_config_exists(args.config)
    if hasattr(args, "output"):
        args.output = _resolve_project_relative(args.output, must_exist=False)
    args.func(args)


if __name__ == "__main__":
    main()
