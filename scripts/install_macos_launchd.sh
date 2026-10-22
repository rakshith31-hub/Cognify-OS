#!/usr/bin/env bash
# Installs Cognitive Monitor as a launchd user agent so it starts
# automatically at login.
#
# Usage:
#   ./scripts/install_macos_launchd.sh [--mode full|input-only]
#
# IMPORTANT (macOS permissions): grant your terminal / python3 both
# "Camera" and "Input Monitoring" + "Accessibility" access in
# System Settings -> Privacy & Security BEFORE installing, or the agent
# will silently fail to capture signals.
set -euo pipefail

MODE="full"
if [[ "${1:-}" == "--mode" ]]; then
  MODE="${2:-full}"
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$(command -v python3)"
LABEL="com.local.cognitive-monitor"
PLIST_DIR="${HOME}/Library/LaunchAgents"
PLIST_FILE="${PLIST_DIR}/${LABEL}.plist"

mkdir -p "${PLIST_DIR}" "${PROJECT_ROOT}/logs"

cat > "${PLIST_FILE}" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>${LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${PYTHON_BIN}</string>
        <string>${PROJECT_ROOT}/src/realtime/daemon.py</string>
        <string>--mode</string>
        <string>${MODE}</string>
    </array>
    <key>WorkingDirectory</key><string>${PROJECT_ROOT}</string>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>${PROJECT_ROOT}/logs/daemon.log</string>
    <key>StandardErrorPath</key><string>${PROJECT_ROOT}/logs/daemon.log</string>
</dict>
</plist>
EOF

launchctl unload "${PLIST_FILE}" 2>/dev/null || true
launchctl load "${PLIST_FILE}"

echo "Installed and started. Manage it with:"
echo "  launchctl list | grep cognitive-monitor"
echo "  launchctl unload ${PLIST_FILE}   # to stop + remove autostart"
echo ""
echo "Reminder: for Do Not Disturb toggling to work, create two Shortcuts"
echo "in Shortcuts.app named 'Cognitive Monitor Focus On' and"
echo "'Cognitive Monitor Focus Off' that turn your macOS Focus mode on/off"
echo "(Apple does not expose a scriptable Focus API, this is the workaround)."
