"""
setup_wizard.py
----------------
`cognitive-monitor setup` — an interactive first-run wizard that produces
a working config/user_config.yaml instead of hand-editing
config/default_config.yaml (which stays untouched as a documented
reference/template — every field in it has an inline rationale comment
that's easy to lose if you edit it directly and easy to break with a
stray indentation slip, which is exactly how several of this project's
"why isn't this working" bugs happened).

Design choices:
  - Uses ruamel.yaml (round-trip mode) instead of plain PyYAML to edit
    the template, so the output config KEEPS all the inline "Rationale:"
    comments instead of losing them to a plain re-serialize.
  - Every question has a sensible default (just press Enter) so the
    whole wizard is usable in under a minute for someone who just wants
    to get started, while still surfacing the handful of choices that
    materially matter (site blocking domains, mode, dashboard auto-open).
  - Runs config_validator on the result before declaring success, so the
    wizard can never itself hand you back a broken config.
  - All I/O goes through injectable input_func/print_func so this is
    testable without a real terminal (see the __main__ smoke test at the
    bottom, and this module's use from a test harness).
"""

import ctypes
import os
import platform
import subprocess
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _ask_yn(prompt, default, input_func):
    suffix = " [Y/n] " if default else " [y/N] "
    while True:
        raw = input_func(prompt + suffix).strip().lower()
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False


def _ask_choice(prompt, choices, default_index, input_func):
    print_lines = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(choices))
    while True:
        raw = input_func(f"{prompt}\n{print_lines}\n> [{default_index+1}] ").strip()
        if raw == "":
            return default_index
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(choices):
                return idx
        except ValueError:
            pass


def _ask_text(prompt, default, input_func):
    raw = input_func(f"{prompt} [{default}] ").strip()
    return raw if raw else default


def _ask_int(prompt, default, input_func):
    while True:
        raw = input_func(f"{prompt} [{default}] ").strip()
        if raw == "":
            return default
        try:
            return int(raw)
        except ValueError:
            pass


def _ask_domain_list(prompt, input_func):
    raw = input_func(f"{prompt} (comma-separated, e.g. youtube.com, reddit.com; blank for none)\n> ").strip()
    if not raw:
        return []
    return [d.strip() for d in raw.split(",") if d.strip()]


def _is_elevated() -> bool:
    system = platform.system()
    try:
        if system == "Windows":
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        else:
            return os.geteuid() == 0
    except Exception:
        return False


def _check_burnttoast() -> bool:
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-Module -ListAvailable -Name BurntToast | Select-Object -First 1"],
            capture_output=True, text=True, timeout=10,
        )
        return "BurntToast" in out.stdout
    except Exception:
        return False


def run_wizard(template_path: str, output_path: str, input_func=input, print_func=print):
    p = print_func
    p("=" * 60)
    p("Cognitive Monitor — guided setup")
    p("=" * 60)
    p("This writes a working config to:")
    p(f"  {output_path}")
    p(f"({template_path} is left untouched as a reference — every")
    p(" setting in it has an inline comment explaining why it's there.)")
    p("")
    p("Everything below runs locally. No data leaves this machine.")
    p("")

    system = platform.system()
    elevated = _is_elevated()
    p(f"Detected OS: {system}")
    if elevated:
        p("Running with elevated privileges (Administrator/root) — "
          "site blocking and Windows color filters will work.")
    else:
        p("NOT running with elevated privileges. Site blocking (hosts file) "
          "needs Administrator on Windows / root on macOS+Linux, or it will "
          "fail silently later. You can still finish this wizard and elevate "
          "later when you actually run `cognitive-monitor start`.")
    p("")

    # ---- mode ----
    mode_idx = _ask_choice(
        "Which mode?",
        ["full (webcam + mouse/keyboard) — recommended, most accurate",
         "input-only (mouse/keyboard only, no camera) — 8 of 14 features, lower accuracy"],
        default_index=0, input_func=input_func,
    )
    mode = "full" if mode_idx == 0 else "input-only"
    camera_index = 0
    if mode == "full":
        camera_index = _ask_int("Camera index (0 is usually the built-in/default webcam)", 0, input_func)
    p("")

    # ---- site blocking ----
    enable_blocking = _ask_yn(
        "Enable session-wide site blocking (blocks listed domains for the "
        "WHOLE session, regardless of cognitive state)?",
        default=False, input_func=input_func,
    )
    domains, dim_on_attempt, dim_percent, dim_duration = [], True, 5, 4
    if enable_blocking:
        domains = _ask_domain_list("Domains to block", input_func)
        if not domains:
            p("No domains entered — leaving site_blocking disabled (nothing to block).")
            enable_blocking = False
        else:
            dim_on_attempt = _ask_yn(
                "Flash-dim brightness as a deterrent the instant a blocked site is visited?",
                default=True, input_func=input_func,
            )
            if dim_on_attempt:
                dim_percent = _ask_int("Cut brightness TO this percent (5 = ~95% cut)", 5, input_func)
                dim_duration = _ask_int("...for how many seconds", 4, input_func)
    p("")

    # ---- dashboard ----
    open_dashboard = _ask_yn(
        "Open a browser dashboard automatically after each session ends?",
        default=True, input_func=input_func,
    )
    p("")

    # ---- OS-specific notification notes ----
    if system == "Windows":
        if _check_burnttoast():
            p("BurntToast PowerShell module found — you'll get real clickable toast notifications.")
        else:
            p("BurntToast not installed — notifications will use a plainer fallback popup.")
            p("For better notifications (optional, no admin needed):")
            p("  Install-Module -Name BurntToast -Scope CurrentUser")
        p("")
        p("For the grayscale 'distracted' color filter to visibly apply, enable once in:")
        p("  Settings > Accessibility > Color filters > 'Allow the shortcut key to toggle color filters'")
    elif system == "Darwin":
        p("macOS Focus/DND has no public scripting API — dnd_on falls back to this app's own "
          "internal notification suppression only, which is still effective for actions "
          "this app fires, just not system-wide.")
    elif system == "Linux":
        p("Linux/GNOME DND uses `gsettings` and works out of the box, no setup needed.")
    p("")

    # ---- write config ----
    from ruamel.yaml import YAML
    yaml = YAML()
    yaml.preserve_quotes = True
    with open(template_path) as f:
        cfg = yaml.load(f)

    cfg["site_blocking"]["enabled"] = enable_blocking
    cfg["site_blocking"]["domains"] = domains
    cfg["site_blocking"]["dim_on_attempt"] = dim_on_attempt
    cfg["site_blocking"]["dim_percent"] = dim_percent
    cfg["site_blocking"]["dim_duration_seconds"] = dim_duration
    cfg["general"]["open_dashboard_on_exit"] = open_dashboard

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(cfg, f)

    # ---- validate the result before declaring success ----
    from src.config_validator import validate_config
    # ruamel's CommentedMap behaves like a dict for our purposes, but
    # validate_config does isinstance(x, dict) checks in a couple of
    # places -- a plain-dict round-trip via its own .items() keeps that
    # working without needing config_validator to know about ruamel types.
    plain_cfg = {k: v for k, v in cfg.items()}
    result = validate_config(plain_cfg)

    p("=" * 60)
    p(f"Config written to {output_path}")
    p(result.format_report())
    p("=" * 60)
    if result.ok:
        p(f"\nStart it with:\n  cognitive-monitor start --config {output_path}"
          + (" --mode input-only" if mode == "input-only" else "")
          + (f" --camera-index {camera_index}" if mode == "full" and camera_index != 0 else ""))
        if enable_blocking and not elevated:
            p("\nRemember: site blocking needs Administrator (Windows) / root (macOS/Linux) "
              "to actually take effect.")
    return result


if __name__ == "__main__":
    # Non-interactive smoke test: feed canned answers instead of a real
    # terminal, so this can be exercised in CI/sandboxes.
    canned = iter([
        "1",            # mode: full
        "0",            # camera index
        "y",            # enable blocking
        "youtube.com, reddit.com",  # domains
        "y",            # dim on attempt
        "5",            # dim percent
        "4",            # dim seconds
        "y",            # open dashboard
    ])
    run_wizard(
        template_path=os.path.join(_PROJECT_ROOT, "config", "default_config.yaml"),
        output_path=os.path.join(_PROJECT_ROOT, "config", "user_config.yaml"),
        input_func=lambda _: next(canned),
    )
