"""
notifications.py
-----------------
Cross-platform helpers for:
  1. Showing a local desktop notification ("toast").
  2. Toggling OS-level Do Not Disturb / Focus mode.

Every OS handles (2) differently and some require the user to have granted
permissions or pre-created a Focus/Shortcut once. Everything here is
best-effort: if a mechanism isn't available, we log a warning and continue
rather than crash the daemon. Nothing here touches any data or account —
it only flips local OS notification settings.
"""

import os
import platform
import shutil
import subprocess
import sys
import logging

logger = logging.getLogger("cognitive_monitor.notifications")

_SYSTEM = platform.system()  # "Windows", "Darwin", "Linux"

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CLI_PATH = os.path.join(_PROJECT_ROOT, "cli.py")
_PROTOCOL_NAME = "cogmonitor"

_burnttoast_checked = False
_burnttoast_available = False
_protocol_registered = False


def show_notification(title: str, message: str):
    """Best-effort local desktop notification. No network calls."""
    try:
        if _SYSTEM == "Darwin":
            # osascript display notification — built into macOS, no deps.
            script = f'display notification "{_escape(message)}" with title "{_escape(title)}"'
            subprocess.run(["osascript", "-e", script], check=False,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif _SYSTEM == "Linux":
            if shutil.which("notify-send"):
                subprocess.run(["notify-send", title, message], check=False)
            else:
                logger.warning("notify-send not found; install libnotify-bin for desktop toasts.")
        elif _SYSTEM == "Windows":
            _windows_toast(title, message)
        else:
            logger.info("[%s] %s", title, message)
    except Exception as e:
        logger.warning("Notification failed (%s): %s", type(e).__name__, e)


_aumid_registered = False
_AUMID = "CognitiveMonitor.App"


def _ensure_aumid_registered():
    """
    Registers a stable AppUserModelID for this app under HKCU (no admin
    needed) so Windows' native toast API (below) has an app identity to
    attribute notifications to. Idempotent -- safe to call every time,
    just re-writes the same values.
    """
    global _aumid_registered
    if _aumid_registered:
        return
    try:
        subprocess.run(
            ["reg", "add", rf"HKCU\Software\Classes\AppUserModelId\{_AUMID}",
             "/v", "DisplayName", "/t", "REG_SZ", "/d", "Cognitive Monitor", "/f"],
            check=False, capture_output=True,
        )
        _aumid_registered = True
    except Exception as e:
        logger.debug("AUMID registration failed (non-fatal, toast may still work): %s", e)


def _windows_toast(title: str, message: str, scenario: str = "alarm", silent: bool = True):
    """
    Native Windows toast via Windows.UI.Notifications directly (no
    BurntToast dependency for this path) -- built specifically to bypass
    Focus Assist / Do Not Disturb.

    WHY scenario="alarm": Microsoft's own toast documentation defines
    "alarm" as one of four special scenarios (default/reminder/alarm/
    incomingCall) that are explicitly exempt from Focus Assist's Priority
    and Alarms-only quiet-hours filtering -- the same mechanism a phone
    alarm clock app uses to still ring during Do Not Disturb. This is the
    actual, documented fix for "notifications from this app should still
    appear even during DND", not a workaround -- ordinary-priority toasts
    (the previous scenario="default"/BurntToast-basic behavior) are
    exactly what Focus Assist is designed to hide, regardless of how
    they're constructed.

    silent=True by default because callers that want a sound already
    play one explicitly via winsound (play_alarm/show_break_popup) --
    the toast's own default alarm sound would otherwise double up.
    """
    _ensure_aumid_registered()
    try:
        ps_script = (
            "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, "
            "ContentType = WindowsRuntime] | Out-Null;"
            "[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom, ContentType = WindowsRuntime] | Out-Null;"
            "$xml = New-Object Windows.Data.Xml.Dom.XmlDocument;"
            f"$xml.LoadXml('<toast scenario=\"{scenario}\">"
            f"<visual><binding template=\"ToastGeneric\">"
            f"<text>{_escape_xml(title)}</text><text>{_escape_xml(message)}</text>"
            f"</binding></visual><audio silent=\"{'true' if silent else 'false'}\"/></toast>');"
            "$toast = New-Object Windows.UI.Notifications.ToastNotification $xml;"
            f"[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('{_AUMID}').Show($toast);"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            check=False, capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return
        logger.debug("Native alarm-scenario toast failed (rc=%s): %s — falling back.",
                     result.returncode, result.stderr.strip()[:300])
    except Exception as e:
        logger.debug("Native alarm-scenario toast raised %s: %s — falling back.", type(e).__name__, e)

    # ---- fallback chain: BurntToast, then the balloon tip ----
    # Both of these are ORDINARY-priority notifications and Focus Assist
    # WILL still hide them -- this fallback exists only for the rare case
    # where the native WinRT call above fails outright (e.g. very old
    # Windows build), not as an equivalent DND-bypass path.
    if _check_burnttoast_available():
        try:
            ps_script = (
                "Import-Module BurntToast -ErrorAction Stop;"
                f"New-BurntToastNotification -Text '{_escape_ps(title)}', '{_escape_ps(message)}'"
            )
            subprocess.run(["powershell", "-NoProfile", "-Command", ps_script],
                            check=False, capture_output=True, timeout=10)
            return
        except Exception as e:
            logger.debug("BurntToast plain notification failed (falling back): %s", e)

    try:
        ps_script = (
            "[reflection.assembly]::loadwithpartialname('System.Windows.Forms') | Out-Null;"
            "[reflection.assembly]::loadwithpartialname('System.Drawing') | Out-Null;"
            "$notify = New-Object System.Windows.Forms.NotifyIcon;"
            "$notify.Icon = [System.Drawing.SystemIcons]::Information;"
            "$notify.Visible = $true;"
            f"$notify.ShowBalloonTip(6000,'{_escape_ps(title)}','{_escape_ps(message)}',"
            "[System.Windows.Forms.ToolTipIcon]::Info);"
            "Start-Sleep -Seconds 6;"
            "$notify.Dispose();"
        )
        subprocess.Popen(
            ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps_script],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        logger.warning("Windows toast fallback failed: %s", e)


def _escape_xml(s: str) -> str:
    """Escape for embedding inside toast XML text nodes."""
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace("'", "&apos;").replace('"', "&quot;"))


def set_do_not_disturb(enabled: bool):
    """
    Best-effort toggle of OS-level Do Not Disturb / Focus for OTHER apps.

    IMPORTANT: this app's OWN notifications are designed to bypass DND
    entirely (see _windows_toast's scenario="alarm" — Microsoft's
    documented mechanism for exactly this), so they should still appear
    on Windows even while this is "on". This function only concerns
    itself with suppressing notifications from OTHER apps while you're
    in a focus/fatigue/overload/distracted state.

    macOS: Focus modes (Sonoma+) are not directly scriptable via a public
    API; the supported approach is a user-created Shortcut named
    "Cognitive Monitor Focus On/Off" that this triggers via `shortcuts run`.
    If the shortcuts don't exist yet, we no-op. If you set one up AND want
    this app's own notifications to still get through, add this app under
    System Settings > Focus > <your mode> > Apps (the actual macOS-native
    way to allow a specific app through a Focus filter — there's no public
    API to do this from a script).

    Linux (GNOME/most freedesktop DEs): previously toggled
    `show-banners` globally via gsettings, which was a real bug — that
    setting suppresses ALL desktop banners including this app's OWN
    notify-send calls, since GNOME doesn't offer a per-app bypass via
    gsettings. Deliberately NOT doing that anymore: this now no-ops on
    Linux, same as Windows, rather than trading "hide other apps" for
    "also silently hide our own alerts". If you want a true GNOME Do Not
    Disturb toggle and are fine with it affecting our own notifications
    too, you can still call `gsettings set org.gnome.desktop.notifications
    show-banners false` yourself outside this tool.

    Windows: Focus Assist has no supported public API to toggle FROM a
    script; we no-op at the OS level here. This app's own notifications
    use scenario="alarm" (see _windows_toast) specifically so they're
    visible regardless of whatever Focus Assist state you're already in.
    """
    try:
        if _SYSTEM == "Darwin":
            shortcut_name = "Cognitive Monitor Focus On" if enabled else "Cognitive Monitor Focus Off"
            if shutil.which("shortcuts"):
                result = subprocess.run(["shortcuts", "run", shortcut_name],
                                         check=False, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.info(
                        "macOS Focus toggle skipped (create a Shortcut named "
                        "'%s' in Shortcuts.app to enable this — see README).",
                        shortcut_name,
                    )
        elif _SYSTEM == "Linux":
            logger.debug(
                "Linux DND toggle is a no-op by design (see set_do_not_disturb docstring) "
                "— gsettings show-banners is global and would suppress this app's own "
                "notifications too, with no per-app exception available."
            )
        elif _SYSTEM == "Windows":
            logger.debug(
                "Windows Focus Assist has no public scripting API; this app's own "
                "notifications bypass it via scenario=\"alarm\" regardless (see _windows_toast)."
            )
    except Exception as e:
        logger.warning("Do Not Disturb toggle failed (%s): %s", type(e).__name__, e)


def _escape(s: str) -> str:
    """Escape for AppleScript double-quoted strings (osascript path):
    backslash first, then double-quote."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _escape_ps(s: str) -> str:
    """Escape for PowerShell single-quoted strings. PowerShell single-quoted
    strings do NOT treat backslash as an escape character at all — the only
    way to embed a literal ' is to double it (''), per PowerShell's own
    quoting rules. Using '\\'' (as the old shared _escape() did) doesn't
    escape anything: PowerShell sees a literal backslash followed by an
    unescaped quote that closes the string early, corrupting everything
    after it in the script — which is exactly the "Missing ')' in method
    call" cascade this was producing for any title/message containing a
    single quote (e.g. "Was 'dnd_on' helpful?")."""
    return s.replace("'", "''")


# ============================================================================
# Alarm + distinct break popup (Windows-only, per finalized platform scope)
# ============================================================================

def play_alarm(pattern: str = "triple_beep"):
    """
    Audible alert for `distracted` — deliberately louder/more attention-
    grabbing than a silent toast, since the whole point is to interrupt
    scattered attention. Uses `winsound` (Python standard library on
    Windows — no extra install needed).
    """
    if _SYSTEM != "Windows":
        logger.info("Audible alarm is Windows-only; skipping on %s (state still logged normally).", _SYSTEM)
        return
    try:
        import winsound
        import time as _time
        if pattern == "triple_beep":
            for _ in range(3):
                winsound.Beep(880, 180)
                _time.sleep(0.08)
        else:
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
    except Exception as e:
        logger.warning("Alarm playback failed (%s): %s — skipping, no crash.", type(e).__name__, e)


def show_break_popup(title: str, message: str):
    """
    A distinct, more prominent notification for `fatigue` break reminders —
    separate from show_notification() so it can't get lost among routine
    "Do Not Disturb enabled" style notices. Pairs a short attention tone
    with the toast rather than relying on the OS notification alone.
    """
    if _SYSTEM == "Windows":
        try:
            import winsound
            winsound.Beep(660, 150)
        except Exception as e:
            logger.debug("Break-popup tone failed (non-fatal): %s", e)
    show_notification(title, message)


# ============================================================================
# Actionable feedback notifications (the bandit's feedback UI)
# ----------------------------------------------------------------------------
# Uses the BurntToast PowerShell module for real clickable Accept/Reject
# buttons on Windows toasts, via a custom "cogmonitor://" URI protocol that
# routes button clicks back into `cli.py feedback`. Falls back cleanly to
# a plain notification pointing at the CLI if BurntToast isn't installed —
# never a silent failure, never a crash.
# ============================================================================

def _check_burnttoast_available() -> bool:
    global _burnttoast_checked, _burnttoast_available
    if _burnttoast_checked:
        return _burnttoast_available
    _burnttoast_checked = True
    if _SYSTEM != "Windows":
        _burnttoast_available = False
        return False
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "if (Get-Module -ListAvailable -Name BurntToast) { exit 0 } else { exit 1 }"],
            check=False, capture_output=True, timeout=10,
        )
        _burnttoast_available = (result.returncode == 0)
    except Exception as e:
        logger.debug("BurntToast availability check failed (non-fatal): %s", e)
        _burnttoast_available = False

    if not _burnttoast_available:
        logger.info(
            "BurntToast PowerShell module not found — feedback prompts will "
            "fall back to plain notifications pointing at the CLI. To enable "
            "real clickable Accept/Reject buttons, run in PowerShell (no "
            "admin needed): Install-Module -Name BurntToast -Scope CurrentUser"
        )
    return _burnttoast_available


def _ensure_protocol_registered():
    """
    Registers the `cogmonitor://` URI protocol in the current user's
    registry (HKCU — no admin needed) so clicking a toast button runs
    `python cli.py feedback-uri "<uri>"`. Idempotent: `reg add /f`
    overwrites cleanly, but we still cache after the first successful run
    to avoid a subprocess call on every single notification.
    """
    global _protocol_registered
    if _protocol_registered or _SYSTEM != "Windows":
        return
    try:
        python_exe = sys.executable
        command = f'"{python_exe}" "{_CLI_PATH}" feedback-uri "%1"'

        subprocess.run(
            ["reg", "add", fr"HKCU\Software\Classes\{_PROTOCOL_NAME}",
             "/ve", "/t", "REG_SZ", "/d", f"URL:Cognitive Monitor Feedback", "/f"],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["reg", "add", fr"HKCU\Software\Classes\{_PROTOCOL_NAME}",
             "/v", "URL Protocol", "/t", "REG_SZ", "/d", "", "/f"],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["reg", "add", fr"HKCU\Software\Classes\{_PROTOCOL_NAME}\shell\open\command",
             "/ve", "/t", "REG_SZ", "/d", command, "/f"],
            check=True, capture_output=True,
        )
        _protocol_registered = True
        logger.info("Registered '%s://' protocol handler for toast feedback buttons.", _PROTOCOL_NAME)
    except Exception as e:
        logger.warning("Could not register feedback protocol handler (%s): %s — "
                        "toast buttons won't work this session, CLI fallback still available.",
                        type(e).__name__, e)


def request_label(title: str, message: str):
    """
    Fires a quick 4-button label prompt (Focus/Fatigue/Overload/Distracted)
    -- the tap-based alternative to typing `cognitive-monitor label
    <state>`, used during a bounded labeling session (see daemon.py's
    _maybe_run_label_session). Real clickable buttons if BurntToast is
    available; otherwise a plain notification with the CLI command spelled
    out, same fallback pattern as request_feedback.

    No event_id needed (unlike request_feedback) -- clicking a button
    labels whatever the daemon's most recent computed window is AT CLICK
    TIME (via last_features.json, same freshness check as the CLI path),
    not a specific window captured when the prompt was fired. That's
    consistent with the CLI: label attaches to "now", not to a moment in
    the past you have to remember precisely.
    """
    if _SYSTEM != "Windows":
        logger.debug("Tap-to-label prompts are Windows-only in this build; "
                      "use 'cognitive-monitor label <state>' directly.")
        return

    if not _check_burnttoast_available():
        show_notification(
            title,
            f"{message}  Reply: cognitive-monitor label focus|fatigue|overload|distracted",
        )
        return

    _ensure_protocol_registered()

    uris = {s: f"{_PROTOCOL_NAME}://label?state={s}" for s in
            ("focus", "fatigue", "overload", "distracted")}

    ps_script = f"""
Import-Module BurntToast -ErrorAction Stop
$b1 = New-BTButton -Content 'Focus' -Arguments '{uris["focus"]}' -ActivationType Protocol
$b2 = New-BTButton -Content 'Fatigue' -Arguments '{uris["fatigue"]}' -ActivationType Protocol
$b3 = New-BTButton -Content 'Overload' -Arguments '{uris["overload"]}' -ActivationType Protocol
$b4 = New-BTButton -Content 'Distracted' -Arguments '{uris["distracted"]}' -ActivationType Protocol
try {{
    New-BurntToastNotification -Text '{_escape_ps(title)}', '{_escape_ps(message)}' -Button $b1, $b2, $b3, $b4 -Scenario Alarm
}} catch {{
    New-BurntToastNotification -Text '{_escape_ps(title)}', '{_escape_ps(message)}' -Button $b1, $b2, $b3, $b4
}}
"""
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            check=False, capture_output=True, timeout=10,
        )
    except Exception as e:
        logger.warning("BurntToast label prompt failed (%s): %s — falling back to plain notification.",
                        type(e).__name__, e)
        show_notification(title, f"{message}  Reply: cognitive-monitor label focus|fatigue|overload|distracted")


def request_feedback(event_id: int, title: str, message: str):
    """
    Fires a feedback prompt for a specific fired action (identified by
    event_id from BanditStore.record_fired). Real clickable buttons if
    BurntToast is available; otherwise a plain notification with the CLI
    command spelled out — either way, the user always has a way to respond.
    """
    if _SYSTEM != "Windows":
        logger.debug("Feedback prompts are Windows-only in this build; "
                      "use 'cognitive-monitor pending' / 'feedback' directly.")
        return

    if not _check_burnttoast_available():
        show_notification(
            title,
            f"{message}  Reply: cognitive-monitor feedback {event_id} up|down",
        )
        return

    _ensure_protocol_registered()

    up_uri = f"{_PROTOCOL_NAME}://feedback?event={event_id}&response=up"
    down_uri = f"{_PROTOCOL_NAME}://feedback?event={event_id}&response=down"

    ps_script = f"""
Import-Module BurntToast -ErrorAction Stop
$up = New-BTButton -Content '👍 Good' -Arguments '{up_uri}' -ActivationType Protocol
$down = New-BTButton -Content '👎 Needs work' -Arguments '{down_uri}' -ActivationType Protocol
try {{
    New-BurntToastNotification -Text '{_escape_ps(title)}', '{_escape_ps(message)}' -Button $up, $down -Scenario Alarm
}} catch {{
    # Older BurntToast versions (pre ~0.6) don't have -Scenario. Same
    # notification, minus the DND-bypass -- still better than crashing
    # the whole call and losing the buttons entirely.
    New-BurntToastNotification -Text '{_escape_ps(title)}', '{_escape_ps(message)}' -Button $up, $down
}}
"""
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            check=False, capture_output=True, timeout=10,
        )
    except Exception as e:
        logger.warning("BurntToast feedback prompt failed (%s): %s — falling back to plain notification.",
                        type(e).__name__, e)
        show_notification(title, f"{message}  Reply: cognitive-monitor feedback {event_id} up|down")
