#!/usr/bin/env bash
# Installs Cognitive Monitor as a systemd --user service so it starts
# automatically at login and restarts if it crashes.
#
# Usage:
#   ./scripts/install_linux_systemd.sh [--mode full|input-only]
#
set -euo pipefail

MODE="full"
if [[ "${1:-}" == "--mode" ]]; then
  MODE="${2:-full}"
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$(command -v python3)"
UNIT_DIR="${HOME}/.config/systemd/user"
UNIT_FILE="${UNIT_DIR}/cognitive-monitor.service"

mkdir -p "${UNIT_DIR}"

cat > "${UNIT_FILE}" <<EOF
[Unit]
Description=Cognitive State Monitor (focus/fatigue/overload/distraction detection)
After=graphical-session.target

[Service]
Type=simple
WorkingDirectory=${PROJECT_ROOT}
ExecStart=${PYTHON_BIN} ${PROJECT_ROOT}/src/realtime/daemon.py --mode ${MODE}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now cognitive-monitor.service

echo "Installed and started. Manage it with:"
echo "  systemctl --user status cognitive-monitor"
echo "  systemctl --user stop cognitive-monitor"
echo "  systemctl --user disable cognitive-monitor   # to remove autostart"
echo ""
echo "Note: mouse/keyboard capture (pynput) needs X11, or a Wayland"
echo "compositor that allows global input listeners; camera capture needs"
echo "the service to have access to /dev/video0 (usually fine for a user session)."
