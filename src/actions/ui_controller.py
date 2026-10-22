"""
ui_controller.py
-----------------
Two kinds of "UI/system change":

1. Display brightness nudges (fatigue -> reduce eye strain).
2. OPT-IN local hosts-file domain blocking (distracted -> remove a
   pre-approved list of the user's own distraction sources for a fixed
   window of time, then automatically restore the file). Domains route
   to 127.0.0.1, where blocked_site_server.py serves an honest block
   page (HTTP) and logs every attempt including HTTPS (see that file for
   how it reads the domain from HTTPS without forging a certificate).

Design principles:
  - Nothing here ever blocks a domain the user did not explicitly list in
    config/default_config.yaml (distracted.block.domains, default empty).
  - The hosts file is always backed up before editing and restored either
    when the block window ends or when the daemon exits/is interrupted.
  - All OS calls are best-effort and log+continue on failure rather than
    crash the daemon; a laptop without brightness control (e.g. some
    external-monitor setups) should never take down the whole service.
"""

import os
import platform
import re
import shutil
import subprocess
import logging
import time
import threading

logger = logging.getLogger("cognitive_monitor.ui_controller")

_SYSTEM = platform.system()

if _SYSTEM == "Windows":
    HOSTS_PATH = r"C:\Windows\System32\drivers\etc\hosts"
else:
    HOSTS_PATH = "/etc/hosts"

_BLOCK_MARKER_START = "# >>> cognitive-monitor block (auto) >>>"
_BLOCK_MARKER_END = "# <<< cognitive-monitor block (auto) <<<"


# ----------------------------- brightness -----------------------------

_brightness_baseline = None  # remembered so we can revert cleanly


def step_brightness(delta_percent: int, floor: int = 20, ceiling: int = 100):
    """Nudge display brightness by delta_percent (can be negative)."""
    global _brightness_baseline
    try:
        if _SYSTEM == "Darwin":
            _step_brightness_macos(delta_percent, floor, ceiling)
        elif _SYSTEM == "Linux":
            _step_brightness_linux(delta_percent, floor, ceiling)
        elif _SYSTEM == "Windows":
            _step_brightness_windows(delta_percent, floor, ceiling)
    except Exception as e:
        logger.warning("Brightness adjustment failed (%s): %s — skipping, no crash.",
                        type(e).__name__, e)


def restore_brightness():
    """Revert to the brightness recorded before the daemon started nudging it."""
    global _brightness_baseline

    # If a blocked-site flash-dim is mid-flight, resolve it synchronously
    # right now instead of trusting its background watcher thread to fire
    # later. That watcher is a daemon thread -- if the process exits (e.g.
    # `cognitive-monitor stop` lands right as someone hits a blocked site)
    # before its deadline, it's killed with the process and its scheduled
    # restore never runs, which would leave the screen dimmed even after
    # a clean shutdown. restore_brightness() is already called on every
    # revert path and at shutdown, so resolving it here closes that gap
    # for free.
    _resolve_flash_dim_now()

    if _brightness_baseline is None:
        return
    try:
        if _SYSTEM == "Linux" and shutil.which("brightnessctl"):
            subprocess.run(["brightnessctl", "set", f"{_brightness_baseline}%"], check=False)
        elif _SYSTEM == "Darwin" and shutil.which("brightness"):
            subprocess.run(["brightness", str(_brightness_baseline / 100.0)], check=False)
        elif _SYSTEM == "Windows":
            # BUG FIX: this branch was previously missing entirely -- the
            # comment claimed "WMI set below handles restore identically"
            # but nothing actually called it, so brightness never reverted
            # on Windows after a fatigue/overload dim (or a flash_dim, see
            # below); it just stayed dimmed for the rest of the session.
            ps_set = (
                "(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods)."
                f"WmiSetBrightness(1,{_brightness_baseline})"
            )
            subprocess.run(["powershell", "-NoProfile", "-Command", ps_set], check=False)
    except Exception as e:
        logger.warning("Brightness restore failed: %s", e)
    finally:
        # Reset so the NEXT dim cycle re-captures the actual current
        # brightness as its baseline, instead of reusing this stale value
        # across multiple dim/restore cycles in the same session.
        _brightness_baseline = None


def _step_brightness_linux(delta, floor, ceiling):
    global _brightness_baseline
    if not shutil.which("brightnessctl"):
        logger.info("brightnessctl not found (sudo apt install brightnessctl) — skipping brightness step.")
        return
    out = subprocess.run(["brightnessctl", "-m"], check=False, capture_output=True, text=True).stdout
    # format: device,class,current,percent,max
    try:
        current_pct = int(out.strip().split(",")[3].rstrip("%"))
    except Exception:
        current_pct = 70
    if _brightness_baseline is None:
        _brightness_baseline = current_pct
    new_pct = max(floor, min(ceiling, current_pct + delta))
    subprocess.run(["brightnessctl", "set", f"{new_pct}%"], check=False)


def _step_brightness_macos(delta, floor, ceiling):
    global _brightness_baseline
    if not shutil.which("brightness"):
        logger.info("`brightness` CLI not found (brew install brightness) — skipping brightness step.")
        return
    out = subprocess.run(["brightness", "-l"], check=False, capture_output=True, text=True).stdout
    current_pct = 70
    for line in out.splitlines():
        if "brightness" in line.lower():
            try:
                current_pct = int(float(line.strip().split(" ")[-1]) * 100)
            except Exception:
                pass
    if _brightness_baseline is None:
        _brightness_baseline = current_pct
    new_pct = max(floor, min(ceiling, current_pct + delta))
    subprocess.run(["brightness", str(new_pct / 100.0)], check=False)


def _step_brightness_windows(delta, floor, ceiling):
    global _brightness_baseline
    ps_get = (
        "(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightness)."
        "CurrentBrightness"
    )
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps_get],
                          check=False, capture_output=True, text=True).stdout.strip()
    try:
        current_pct = int(out)
    except Exception:
        current_pct = 70
    if _brightness_baseline is None:
        _brightness_baseline = current_pct
    new_pct = max(floor, min(ceiling, current_pct + delta))
    ps_set = (
        "(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods)."
        f"WmiSetBrightness(1,{new_pct})"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps_set], check=False)


def _get_brightness_pct() -> int:
    """Read CURRENT brightness (not the remembered baseline) — used by
    flash_dim_on_block() so it restores to whatever was actually on screen
    right before the flash, even if that was itself a fatigue/overload dim
    rather than the session's very first baseline."""
    try:
        if _SYSTEM == "Linux" and shutil.which("brightnessctl"):
            out = subprocess.run(["brightnessctl", "-m"], check=False,
                                  capture_output=True, text=True).stdout
            return int(out.strip().split(",")[3].rstrip("%"))
        elif _SYSTEM == "Darwin" and shutil.which("brightness"):
            out = subprocess.run(["brightness", "-l"], check=False,
                                  capture_output=True, text=True).stdout
            for line in out.splitlines():
                if "brightness" in line.lower():
                    return int(float(line.strip().split(" ")[-1]) * 100)
        elif _SYSTEM == "Windows":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightness).CurrentBrightness"],
                check=False, capture_output=True, text=True).stdout.strip()
            return int(out)
    except Exception:
        pass
    return 70  # sane fallback if we can't read it


def _set_brightness_abs(pct: int):
    pct = max(0, min(100, pct))
    try:
        if _SYSTEM == "Linux" and shutil.which("brightnessctl"):
            subprocess.run(["brightnessctl", "set", f"{pct}%"], check=False)
        elif _SYSTEM == "Darwin" and shutil.which("brightness"):
            subprocess.run(["brightness", str(pct / 100.0)], check=False)
        elif _SYSTEM == "Windows":
            subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,{pct})"],
                check=False)
    except Exception as e:
        logger.warning("Set-brightness failed: %s", e)


_flash_dim_lock = threading.Lock()
_flash_dim_active = False
_flash_dim_true_baseline = None   # captured ONCE per burst, never per-attempt
_flash_dim_deadline = 0.0         # monotonic time; each new attempt pushes this forward


def _resolve_flash_dim_now():
    """Ends an in-flight flash-dim episode immediately, restoring its true
    baseline. Called from restore_brightness() so state-change/idle/
    shutdown reverts can't be left dark by a flash-dim's own background
    watcher losing the race (or losing the process, on shutdown)."""
    global _flash_dim_active, _flash_dim_true_baseline
    with _flash_dim_lock:
        if not _flash_dim_active:
            return
        baseline = _flash_dim_true_baseline
        _flash_dim_active = False
        _flash_dim_true_baseline = None
    _set_brightness_abs(baseline)


def flash_dim_on_block(percent: int = 5, duration_seconds: float = 4.0):
    """
    Immediate deterrent the moment a blocked site is actually visited (not
    just a general ambient dim for the whole block window): cuts brightness
    to `percent` (default 5, i.e. a ~95% cut), then restores once things
    go quiet -- which may itself be a fatigue/overload dim already in
    progress, not necessarily the session's original baseline, so this
    reads CURRENT brightness fresh rather than reusing the global
    `_brightness_baseline` used elsewhere.

    BUG FIX: a single page load against a blocked domain produces MANY
    connection attempts in quick succession (the main document plus every
    CSS/JS/image/analytics sub-resource, each hitting the loopback block
    server as its own connection -- easily several per second). The old
    version spawned one independent dim/sleep/restore thread PER ATTEMPT,
    each one calling _get_brightness_pct() as its "restore to" baseline.
    Under a burst, a later thread's call to _get_brightness_pct() would
    read the screen WHILE AN EARLIER THREAD HAD IT DIMMED, capturing
    `percent` (5%) as its own "original" brightness -- so its restore just
    re-applied the dim. With enough overlapping attempts this reliably
    ended in the screen staying dark: whichever thread's sleep happened to
    finish last almost always had a stale, already-dimmed baseline.

    Fixed by coalescing an entire burst into one dim episode: only the
    FIRST attempt in a burst captures the true pre-dim baseline and starts
    a watcher thread; every attempt after that just pushes the restore
    deadline forward (debounce) without touching brightness or spawning
    another thread. The single watcher restores exactly once, to the one
    baseline captured before anything was dimmed, `duration_seconds` after
    the LAST attempt rather than the first.

    Runs in a background thread: called from the blocked-site server's
    request-handling thread, which must not block waiting on this.
    """
    global _flash_dim_active, _flash_dim_true_baseline, _flash_dim_deadline

    with _flash_dim_lock:
        _flash_dim_deadline = time.time() + duration_seconds
        if _flash_dim_active:
            # A dim episode is already in progress -- just extend it.
            # Don't re-capture "current" brightness (it's dimmed right
            # now) and don't start a second watcher thread.
            return
        _flash_dim_active = True
        _flash_dim_true_baseline = _get_brightness_pct()  # captured ONCE, before any dimming

    _set_brightness_abs(percent)
    logger.info("Blocked site visited — brightness cut to %d%% as a deterrent "
                "(will restore %.0fs after the last attempt).", percent, duration_seconds)

    def _watch():
        global _flash_dim_active, _flash_dim_true_baseline
        while True:
            with _flash_dim_lock:
                remaining = _flash_dim_deadline - time.time()
                if remaining <= 0:
                    baseline = _flash_dim_true_baseline
                    _flash_dim_active = False
                    _flash_dim_true_baseline = None
                    break
            time.sleep(min(remaining, 0.5))
        _set_brightness_abs(baseline)

    threading.Thread(target=_watch, daemon=True).start()


# --------------------------- domain blocking ---------------------------

_unblock_timer = None


def _normalize_domain(d: str) -> str:
    """
    Turns whatever a person actually typed into a bare hostname a hosts
    file entry can use. Hosts file entries must be a plain hostname --
    no scheme, no path, no port, no trailing slash. Feeding it a full URL
    like "https://www.youtube.com/" (as opposed to "youtube.com")
    silently writes a USELESS line: "127.0.0.1 https://www.youtube.com/"
    is not a hostname anything will ever query, so it doesn't error, it
    just quietly blocks nothing -- easy to miss since nothing looks wrong
    in the logs.
    """
    d = d.strip()
    d = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", "", d)  # strip a leading scheme, e.g. "https://"
    d = d.split("/", 1)[0]                              # strip any path
    d = d.split("?", 1)[0]                               # strip any query string
    d = d.split("#", 1)[0]                                # strip any fragment
    d = d.split(":", 1)[0]                                 # strip a port, e.g. "example.com:8080"
    if d.startswith("www."):
        d = d[4:]
    return d.lower()


def block_domains(domains, duration_minutes: float):
    """
    Append the given domains to the hosts file, routed to localhost, for
    `duration_minutes`. Automatically restores the hosts file afterward.
    Requires admin/root — if the process can't write the hosts file, logs
    a clear message and does nothing else (no partial edits).
    """
    if not domains:
        return
    try:
        with open(HOSTS_PATH, "r") as f:
            content = f.read()
    except PermissionError:
        logger.warning(
            "Cannot read/write %s (need admin/root). Domain blocking skipped. "
            "Re-run the daemon with elevated privileges to enable this feature.",
            HOSTS_PATH,
        )
        return
    except Exception as e:
        logger.warning("Could not read hosts file: %s", e)
        return

    if _BLOCK_MARKER_START in content:
        # BUG FIX: this used to just log "already active" and return here,
        # leaving whatever block was already on disk untouched. That's
        # wrong in two real scenarios: (1) the daemon was killed uncleanly
        # last time (crash, force-close, kill -9 -- anything that skips
        # action_engine.shutdown()'s unblock_domains() call) and the old
        # block never got removed, so a fresh run with an UPDATED domain
        # list would silently keep blocking the OLD list forever; (2) the
        # config changed between runs and the new domains should actually
        # take effect. Always strip whatever's currently there first, then
        # write fresh from the CURRENT domains below -- idempotent either
        # way, and never silently stale.
        pre, _, rest = content.partition(_BLOCK_MARKER_START)
        _, _, post = rest.partition(_BLOCK_MARKER_END)
        content = pre.rstrip("\n") + "\n" + post.lstrip("\n")

    normalized = []
    for d in domains:
        nd = _normalize_domain(d)
        if nd and nd not in normalized:
            normalized.append(nd)
    if not normalized:
        return

    lines = [_BLOCK_MARKER_START]
    for d in normalized:
        lines.append(f"127.0.0.1 {d}")
        lines.append(f"127.0.0.1 www.{d}")
    lines.append(_BLOCK_MARKER_END)
    block_text = "\n".join(lines) + "\n"

    try:
        with open(HOSTS_PATH, "w") as f:
            f.write(content.rstrip("\n") + "\n\n" + block_text)
        logger.info("Blocked %d domain(s) for %.0f min: %s", len(normalized), duration_minutes, normalized)
        _flush_dns()
    except PermissionError:
        logger.warning("Cannot write %s (need admin/root). Domain blocking skipped.", HOSTS_PATH)
        return

    _flush_dns()

    global _unblock_timer
    if _unblock_timer:
        _unblock_timer.cancel()
    _unblock_timer = threading.Timer(duration_minutes * 60, unblock_domains)
    _unblock_timer.daemon = True
    _unblock_timer.start()


def unblock_domains():
    """Remove the auto-inserted block from the hosts file, if present."""
    try:
        with open(HOSTS_PATH, "r") as f:
            content = f.read()
    except Exception as e:
        logger.warning("Could not read hosts file to unblock: %s", e)
        return

    if _BLOCK_MARKER_START not in content:
        return

    pre, _, rest = content.partition(_BLOCK_MARKER_START)
    _, _, post = rest.partition(_BLOCK_MARKER_END)
    new_content = pre.rstrip("\n") + "\n" + post.lstrip("\n")

    try:
        with open(HOSTS_PATH, "w") as f:
            f.write(new_content)
        logger.info("Domain block removed; hosts file restored.")
        _flush_dns()
    except PermissionError:
        logger.warning("Cannot write %s to remove block (need admin/root).", HOSTS_PATH)


def _flush_dns():
    try:
        if _SYSTEM == "Darwin":
            subprocess.run(["dscacheutil", "-flushcache"], check=False)
        elif _SYSTEM == "Windows":
            subprocess.run(["ipconfig", "/flushdns"], check=False)
        # Linux: usually not needed (nsswitch reads hosts live via NSS).
    except Exception:
        pass


# ============================================================================
# Windows-only expanded UI actions
# ----------------------------------------------------------------------------
# All four use PUBLIC, documented Win32 APIs via ctypes (SystemParametersInfo,
# SHAppBarMessage) or Microsoft's own built-in Color Filters registry key
# (the same one Settings > Accessibility > Color Filters writes to) — nothing
# here is an undocumented hack, and every change is reverted on state change
# or daemon shutdown, same guarantee as brightness/DND above.
# ============================================================================

_color_filter_active = False
_color_filter_type = None
_high_contrast_active = False
_taskbar_hidden = False


def _send_color_filter_hotkey():
    """
    Simulates Ctrl+Win+C, the OS-level Color Filters toggle hotkey.

    Writing HKCU\\Software\\Microsoft\\ColorFiltering directly (below) is
    necessary but NOT sufficient on Windows 10/11: the OS doesn't watch
    that key live, so a plain `reg add` silently sits there with no visual
    effect until the filter is toggled through its own hotkey (or a
    logoff/logon). This simulates that same hotkey so the registry values
    we just wrote actually take effect immediately.

    Ctrl+Win+C specifically requires simulating the Windows key, which
    PowerShell's own SendKeys can't do (no Windows-key support) -- so this
    uses a tiny inline P/Invoke of user32's keybd_event instead, a public
    documented Win32 API.

    NOTE: this requires "Allow the shortcut key to toggle color filters"
    to be ON in Settings > Accessibility > Color filters -- a one-time
    manual setup step in Windows itself that no script can flip on your
    behalf. If grayscale still doesn't visibly apply after this, check
    that setting first.
    """
    ps_script = (
        "Add-Type -TypeDefinition '"
        "using System; using System.Runtime.InteropServices;"
        "public class CFKB { [DllImport(\"user32.dll\")] "
        "public static extern void keybd_event(byte b, byte s, uint f, UIntPtr e); }"
        "' -ErrorAction SilentlyContinue;"
        "$VK_LWIN=0x5B; $VK_CTRL=0x11; $VK_C=0x43; $KEYUP=0x2;"
        "[CFKB]::keybd_event($VK_LWIN,0,0,[UIntPtr]::Zero);"
        "[CFKB]::keybd_event($VK_CTRL,0,0,[UIntPtr]::Zero);"
        "Start-Sleep -Milliseconds 30;"
        "[CFKB]::keybd_event($VK_C,0,0,[UIntPtr]::Zero);"
        # BUG FIX: held for 60ms total previously (including release delay
        # below) -- bumped the hold and added a settle pause after release,
        # since a very short simulated key-hold can be dropped by Windows'
        # input processing under system load, and a reported symptom
        # (revert consistently failing while the initial activation
        # consistently succeeds) is consistent with marginal timing rather
        # than a logic error -- the decision logic itself is confirmed
        # correct from the daemon's own log output.
        "Start-Sleep -Milliseconds 120;"
        "[CFKB]::keybd_event($VK_C,0,$KEYUP,[UIntPtr]::Zero);"
        "[CFKB]::keybd_event($VK_CTRL,0,$KEYUP,[UIntPtr]::Zero);"
        "[CFKB]::keybd_event($VK_LWIN,0,$KEYUP,[UIntPtr]::Zero);"
        "Start-Sleep -Milliseconds 100;"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps_script],
        check=False, capture_output=True,
    )


# ---------------------------- color filter (grayscale) ---------------------
# Rationale: grayscale display mode is a widely used digital-wellbeing
# technique (it's a built-in iOS/Android "reduce distraction" toggle) — the
# idea is that removing color reduces the visual reward/salience of apps and
# feeds designed to be colorful and attention-grabbing. This is a plausible,
# widely deployed heuristic, not a rigorously RCT-validated claim — treat it
# as a reasonable default to try, not a proven intervention.

def _query_color_filter_state():
    """
    Reads the REAL current on-screen Active AND FilterType from the
    registry, rather than trusting our own cached flags.

    BUG FIX: this used to only query Active, and compared the desired
    filter_type against `_color_filter_type` -- a plain Python variable
    that starts at None every time the daemon process restarts. On the
    very first color_filter activation of ANY fresh session, `filter_type
    (e.g. 0) != _color_filter_type (None)` is unavoidably True regardless
    of what the real previous type actually was, which forced the
    off-then-on "type changed" double-toggle path to fire on essentially
    every session's first activation -- sending two hotkey presses where
    one was needed, with the net result depending on fragile timing
    rather than a clean single toggle. Querying FilterType from the
    registry alongside Active fixes this at the source: the comparison
    is now against the real previous type, not a cache that resets on
    every process restart.
    """
    try:
        result = subprocess.run(
            ["reg", "query", r"HKCU\Software\Microsoft\ColorFiltering"],
            check=True, capture_output=True, text=True,
        )
        active_match = re.search(r"Active\s+REG_DWORD\s+0x([0-9a-fA-F]+)", result.stdout)
        type_match = re.search(r"FilterType\s+REG_DWORD\s+0x([0-9a-fA-F]+)", result.stdout)
        active = bool(active_match and int(active_match.group(1), 16) != 0)
        filter_type = int(type_match.group(1), 16) if type_match else None
        return active, filter_type
    except Exception:
        # Key doesn't exist yet (filter never touched on this machine) or
        # the query itself failed -- Windows' own default is "off", so
        # treating an unreadable/missing key as "currently off" matches
        # reality far more often than assuming our stale cached flag.
        return False, None


def _query_color_filter_active() -> bool:
    """Backward-compatible wrapper -- see _query_color_filter_state()."""
    return _query_color_filter_state()[0]


def set_color_filter(enabled: bool, filter_type: int = 0):
    """
    Toggles Windows' built-in Color Filters accessibility feature.
    filter_type: 0=Grayscale (default), 1=Invert, 2=GrayscaleInverted,
                 3=Deuteranopia, 4=Protanopia, 5=Tritanopia.
    """
    global _color_filter_active, _color_filter_type
    if _SYSTEM != "Windows":
        logger.info("Color filter toggle is Windows-only; skipping on %s.", _SYSTEM)
        return
    try:
        actual_active, actual_type = _query_color_filter_state()

        subprocess.run(
            ["reg", "add", r"HKCU\Software\Microsoft\ColorFiltering",
             "/v", "Active", "/t", "REG_DWORD", "/d", "1" if enabled else "0", "/f"],
            check=True, capture_output=True,
        )
        if enabled:
            subprocess.run(
                ["reg", "add", r"HKCU\Software\Microsoft\ColorFiltering",
                 "/v", "FilterType", "/t", "REG_DWORD", "/d", str(filter_type), "/f"],
                check=True, capture_output=True,
            )

        # The registry write above only pre-stages the setting -- it needs
        # the hotkey to actually be applied live. If we're switching TYPE
        # while already active, the OS also won't re-read FilterType just
        # from toggling once, so cycle off/on to force a re-read.
        # BUG FIX: compares against `actual_type` (queried live from the
        # registry, above) instead of the cached `_color_filter_type`
        # module variable -- that cache resets to None on every fresh
        # daemon process, which made this comparison spuriously True on
        # a session's very first activation regardless of the real
        # previous type, forcing an unnecessary double-toggle. See
        # _query_color_filter_state()'s docstring.
        type_changed = enabled and actual_active and filter_type != actual_type
        if type_changed:
            _send_color_filter_hotkey()  # off
            time.sleep(0.2)
            _send_color_filter_hotkey()  # on again, now reads the new FilterType
        elif enabled != actual_active:
            _send_color_filter_hotkey()
        # else: OS is already in the requested state, nothing to toggle

        _broadcast_setting_change("ColorFiltering")
        _color_filter_active = enabled
        _color_filter_type = filter_type if enabled else None
        logger.info("Color filter %s (type=%d, was actually %s beforehand). If this didn't "
                    "visibly change anything, check Settings > Accessibility > Color filters > "
                    "'Allow the shortcut key to toggle color filters' is turned ON (one-time "
                    "setup, can't be set by a script).",
                    "enabled" if enabled else "disabled", filter_type,
                    "on" if actual_active else "off")
    except Exception as e:
        logger.warning("Color filter toggle failed (%s): %s — skipping, no crash.", type(e).__name__, e)


def restore_color_filter():
    # BUG FIX: was gated on the cached `_color_filter_active` flag, so if
    # that flag had drifted false (e.g. after the query-based fix above
    # already corrected a prior call) while the OS was still actually on,
    # this early-return would skip reverting entirely. set_color_filter()
    # now does its own live query internally and is a safe no-op if the
    # OS is already off, so it's correct (and cheap) to call
    # unconditionally here rather than gate on a value that's exactly
    # the kind of state this whole fix distrusts.
    set_color_filter(False)


# ---------------------------- high contrast ---------------------------------
# Rationale: overload's top model drivers are backspace/error rate and pupil
# diameter — consistent with visual/processing strain. High-contrast mode is
# a standard, OS-native accessibility feature for reducing visual strain and
# improving legibility, applied here as a low-effort "reduce extraneous load"
# action per Cognitive Load Theory (see docs/RESEARCH_RATIONALE.md).

def set_high_contrast(enabled: bool):
    global _high_contrast_active
    if _SYSTEM != "Windows":
        logger.info("High contrast toggle is Windows-only; skipping on %s.", _SYSTEM)
        return
    try:
        import ctypes
        from ctypes import wintypes

        SPI_GETHIGHCONTRAST = 0x42
        SPI_SETHIGHCONTRAST = 0x43
        HCF_HIGHCONTRASTON = 0x1
        SPIF_UPDATEINIFILE = 0x1
        SPIF_SENDCHANGE = 0x2

        class HIGHCONTRASTW(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.UINT),
                ("dwFlags", wintypes.DWORD),
                ("lpszDefaultScheme", wintypes.LPWSTR),
            ]

        hc = HIGHCONTRASTW()
        hc.cbSize = ctypes.sizeof(HIGHCONTRASTW)
        # read current state first so we don't clobber lpszDefaultScheme
        ctypes.windll.user32.SystemParametersInfoW(SPI_GETHIGHCONTRAST, hc.cbSize, ctypes.byref(hc), 0)

        if enabled:
            hc.dwFlags |= HCF_HIGHCONTRASTON
        else:
            hc.dwFlags &= ~HCF_HIGHCONTRASTON

        ctypes.windll.user32.SystemParametersInfoW(
            SPI_SETHIGHCONTRAST, hc.cbSize, ctypes.byref(hc),
            SPIF_UPDATEINIFILE | SPIF_SENDCHANGE,
        )
        _high_contrast_active = enabled
        logger.info("High contrast %s.", "enabled" if enabled else "disabled")
    except Exception as e:
        logger.warning("High contrast toggle failed (%s): %s — skipping, no crash.", type(e).__name__, e)


def restore_high_contrast():
    # BUG FIX: same class of bug as restore_color_filter() above — this
    # was gated on the cached `_high_contrast_active` flag, so if that
    # flag was ever wrong (skipped due to a bandit decision, or simply
    # never synced after an external change) while high contrast was
    # actually on, this would skip reverting it entirely. set_high_contrast()
    # already reads the live OS state via SystemParametersInfoW before
    # deciding what to write, so it's a cheap, safe no-op when already off
    # — call it unconditionally rather than trusting our own cache to gate it.
    set_high_contrast(False)


# ---------------------------- taskbar auto-hide -----------------------------
# Rationale: same "strip extraneous load, don't add more prompts" logic as
# high contrast — the taskbar (pinned apps, tray icons, clock) is a constant
# low-grade source of visual competition for attention during focus/overload.

def set_taskbar_autohide(enabled: bool):
    global _taskbar_hidden
    if _SYSTEM != "Windows":
        logger.info("Taskbar auto-hide is Windows-only; skipping on %s.", _SYSTEM)
        return
    try:
        import ctypes
        from ctypes import wintypes

        ABM_SETSTATE = 0xA
        ABS_AUTOHIDE = 0x1

        class APPBARDATA(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.DWORD),
                ("hWnd", wintypes.HWND),
                ("uCallbackMessage", wintypes.UINT),
                ("uEdge", wintypes.UINT),
                ("rc", wintypes.RECT),
                ("lParam", wintypes.LPARAM),
            ]

        hwnd = ctypes.windll.user32.FindWindowW("Shell_TrayWnd", None)
        if not hwnd:
            logger.warning("Could not find taskbar window handle — skipping.")
            return

        data = APPBARDATA()
        data.cbSize = ctypes.sizeof(APPBARDATA)
        data.hWnd = hwnd
        data.lParam = ABS_AUTOHIDE if enabled else 0

        ctypes.windll.shell32.SHAppBarMessage(ABM_SETSTATE, ctypes.byref(data))
        _taskbar_hidden = enabled
        logger.info("Taskbar auto-hide %s.", "enabled" if enabled else "disabled")
    except Exception as e:
        logger.warning("Taskbar auto-hide toggle failed (%s): %s — skipping, no crash.", type(e).__name__, e)


def restore_taskbar():
    # BUG FIX: same class of bug as restore_color_filter()/restore_high_contrast()
    # above — gating on our own cached `_taskbar_hidden` flag meant a stale
    # flag could skip the revert. set_taskbar_autohide(False) does an
    # ABSOLUTE set (SHAppBarMessage with ABS_AUTOHIDE cleared), not a
    # relative toggle, so it's always safe and cheap to call unconditionally.
    set_taskbar_autohide(False)


def _broadcast_setting_change(setting_name: str):
    """Tells running apps (and the OS) a setting changed, so color-filter
    changes apply live without logging off."""
    try:
        import ctypes
        HWND_BROADCAST = 0xFFFF
        WM_SETTINGCHANGE = 0x1A
        SMTO_ABORTIFHUNG = 0x2
        result = ctypes.c_long()
        ctypes.windll.user32.SendMessageTimeoutW(
            HWND_BROADCAST, WM_SETTINGCHANGE, 0, setting_name,
            SMTO_ABORTIFHUNG, 5000, ctypes.byref(result),
        )
    except Exception as e:
        logger.debug("Setting-change broadcast failed (non-fatal): %s", e)


def restore_all_expanded_ui():
    """Called on daemon shutdown / state revert: undo every expanded UI
    action regardless of which ones are currently active."""
    restore_color_filter()
    restore_high_contrast()
    restore_taskbar()
