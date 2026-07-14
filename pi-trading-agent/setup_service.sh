#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="trading-agent.service"
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}"
RUN_USER="${SUDO_USER:-$(id -un)}"
RUN_GROUP="$(id -gn "${RUN_USER}")"

if [[ "${EUID}" -ne 0 ]]; then
    echo "Run this installer with sudo: sudo ./setup_service.sh" >&2
    exit 1
fi

if [[ ! -f "${PROJECT_DIR}/main.py" || ! -f "${PROJECT_DIR}/strategy.py" || \
      ! -f "${PROJECT_DIR}/adaptive_news_model.py" || \
      ! -f "${PROJECT_DIR}/news_context.py" || ! -f "${PROJECT_DIR}/config.json" ]]; then
    echo "Required project files are missing from ${PROJECT_DIR}." >&2
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required but was not found." >&2
    exit 1
fi

python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install --requirement "${PROJECT_DIR}/requirements.txt"

chown -R "${RUN_USER}:${RUN_GROUP}" "${PROJECT_DIR}"
chmod 600 "${PROJECT_DIR}/config.json"
chmod 755 "${PROJECT_DIR}/main.py"

install -o root -g root -m 0644 /dev/null "${SERVICE_FILE}"
tee "${SERVICE_FILE}" >/dev/null <<EOF
[Unit]
Description=Lumibot Alpaca Trading Agent
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=${RUN_USER}
Group=${RUN_GROUP}
WorkingDirectory=${PROJECT_DIR}
ExecStart=${VENV_DIR}/bin/python ${PROJECT_DIR}/main.py
Restart=always
RestartSec=30
TimeoutStopSec=90
KillSignal=SIGINT
Environment=PYTHONUNBUFFERED=1
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ProtectHome=read-only
ReadWritePaths=${PROJECT_DIR}

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now "${SERVICE_NAME}"

echo "${SERVICE_NAME} is installed and running."
echo "View status with: sudo systemctl status ${SERVICE_NAME}"
echo "Follow logs with: sudo journalctl -u ${SERVICE_NAME} -f"
