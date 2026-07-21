#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="trading-agent.service"
CRYPTO_SERVICE_NAME="trading-agent-crypto.service"
WATCHDOG_SERVICE_NAME="trading-agent-cpu-watchdog.service"
WATCHDOG_TIMER_NAME="trading-agent-cpu-watchdog.timer"
OLLAMA_SERVICE_NAME="ollama.service"
OLLAMA_WARMUP_SERVICE_NAME="ollama-warmup.service"
OLLAMA_WARMUP_TIMER_NAME="ollama-warmup.timer"
NIGHTLY_PREEVAL_SERVICE_NAME="trading-agent-nightly-preeval.service"
NIGHTLY_PREEVAL_TIMER_NAME="trading-agent-nightly-preeval.timer"
DASHBOARD_SERVICE_NAME="trading-agent-dashboard.service"
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}"
CRYPTO_SERVICE_FILE="/etc/systemd/system/${CRYPTO_SERVICE_NAME}"
WATCHDOG_SERVICE_FILE="/etc/systemd/system/${WATCHDOG_SERVICE_NAME}"
WATCHDOG_TIMER_FILE="/etc/systemd/system/${WATCHDOG_TIMER_NAME}"
OLLAMA_SERVICE_FILE="/etc/systemd/system/${OLLAMA_SERVICE_NAME}"
OLLAMA_WARMUP_SERVICE_FILE="/etc/systemd/system/${OLLAMA_WARMUP_SERVICE_NAME}"
OLLAMA_WARMUP_TIMER_FILE="/etc/systemd/system/${OLLAMA_WARMUP_TIMER_NAME}"
NIGHTLY_PREEVAL_SERVICE_FILE="/etc/systemd/system/${NIGHTLY_PREEVAL_SERVICE_NAME}"
NIGHTLY_PREEVAL_TIMER_FILE="/etc/systemd/system/${NIGHTLY_PREEVAL_TIMER_NAME}"
DASHBOARD_SERVICE_FILE="/etc/systemd/system/${DASHBOARD_SERVICE_NAME}"
RUN_USER="${SUDO_USER:-$(id -un)}"
RUN_GROUP="$(id -gn "${RUN_USER}")"

if [[ "${EUID}" -ne 0 ]]; then
    echo "Run this installer with sudo: sudo ./setup_service.sh" >&2
    exit 1
fi

if [[ ! -f "${PROJECT_DIR}/config.json" && -f "${PROJECT_DIR}/config.example.json" ]]; then
    cp "${PROJECT_DIR}/config.example.json" "${PROJECT_DIR}/config.json"
    chown "${RUN_USER}:${RUN_GROUP}" "${PROJECT_DIR}/config.json"
    chmod 600 "${PROJECT_DIR}/config.json"
    echo "Created config.json from config.example.json."
    echo "Edit ${PROJECT_DIR}/config.json with your Alpaca credentials, then rerun this installer."
    exit 1
fi

if [[ ! -f "${PROJECT_DIR}/main.py" || ! -f "${PROJECT_DIR}/strategy.py" || \
      ! -f "${PROJECT_DIR}/main_crypto.py" || ! -f "${PROJECT_DIR}/crypto_strategy.py" || \
      ! -f "${PROJECT_DIR}/news_context.py" || ! -f "${PROJECT_DIR}/config.json" || \
      ! -f "${PROJECT_DIR}/requirements.lock" ]]; then
    echo "Required project files are missing from ${PROJECT_DIR}." >&2
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required but was not found." >&2
    exit 1
fi

python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --requirement "${PROJECT_DIR}/requirements.lock"

chown -R "${RUN_USER}:${RUN_GROUP}" "${PROJECT_DIR}"
chmod 600 "${PROJECT_DIR}/config.json"
chmod 755 "${PROJECT_DIR}/main.py"
chmod 755 "${PROJECT_DIR}/main_crypto.py"
chmod 755 "${PROJECT_DIR}/scripts/cpu_watchdog.sh"
chmod 755 "${PROJECT_DIR}/scripts/ollama_warmup.sh"
chmod 755 "${PROJECT_DIR}/scripts/nightly_preeval.py"
chmod 755 "${PROJECT_DIR}/scripts/web_dashboard.py"

# The optional LLM news assessment (llm_news.py) only ever talks to a local
# Ollama server -- never an outside API -- so Ollama is provisioned here too.
if ! command -v ollama >/dev/null 2>&1; then
    echo "Ollama is required but was not found." >&2
    echo "Install a verified Ollama package from https://ollama.com/download, then rerun this installer." >&2
    exit 1
fi

# The official installer's own ollama.service has no loopback restriction and
# no keep-alive; overwrite it so the server is never reachable off the device
# and the model stays resident between daily runs instead of reloading. The
# strategy now evaluates up to twice a trading day (market open, plus
# PORTFOLIO_SECOND_ITERATION_OFFSET_MINUTES later, default 210 = ~3.5h) --
# 8h keeps the model loaded from the 09:00 warm-up through both possible
# calls regardless of that offset, still unloading well before the next
# morning's warm-up instead of holding ~2GB of RAM around the clock.
# OLLAMA_MAX_LOADED_MODELS=1 is a hard cap, not a reaction to any current
# multi-model use -- LLM_NEWS_MODEL and article_filter.py's MODEL constant
# are already the same tag -- but this Pi has no RAM to spare if that ever
# changes, so the limit is made explicit rather than implicit.
install -o root -g root -m 0644 /dev/null "${OLLAMA_SERVICE_FILE}"
tee "${OLLAMA_SERVICE_FILE}" >/dev/null <<EOF
[Unit]
Description=Ollama local LLM server
After=network.target

[Service]
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment=OLLAMA_HOST=127.0.0.1:11434
Environment=OLLAMA_KEEP_ALIVE=8h
Environment=OLLAMA_MAX_LOADED_MODELS=1

[Install]
WantedBy=multi-user.target
EOF

install -o root -g root -m 0644 /dev/null "${OLLAMA_WARMUP_SERVICE_FILE}"
tee "${OLLAMA_WARMUP_SERVICE_FILE}" >/dev/null <<EOF
[Unit]
Description=Warm up the local Ollama news-assessment model before market open
After=${OLLAMA_SERVICE_NAME}
Requires=${OLLAMA_SERVICE_NAME}

[Service]
Type=oneshot
User=${RUN_USER}
Group=${RUN_GROUP}
ExecStart=${PROJECT_DIR}/scripts/ollama_warmup.sh
EOF

install -o root -g root -m 0644 /dev/null "${OLLAMA_WARMUP_TIMER_FILE}"
tee "${OLLAMA_WARMUP_TIMER_FILE}" >/dev/null <<EOF
[Unit]
Description=Trigger the Ollama news-model warm-up before market open

[Timer]
# Assumes the system timezone is America/New_York (the market's own
# timezone); adjust if the host's timezone differs. 30 minutes ahead of the
# 9:30 ET market open the strategy trades at.
OnCalendar=Mon..Fri 09:00:00
Persistent=false

[Install]
WantedBy=timers.target
EOF

install -o root -g root -m 0644 /dev/null "${NIGHTLY_PREEVAL_SERVICE_FILE}"
tee "${NIGHTLY_PREEVAL_SERVICE_FILE}" >/dev/null <<EOF
[Unit]
Description=Pre-evaluate every candidate symbol's LLM news verdict before the trading day
After=${OLLAMA_SERVICE_NAME}
Requires=${OLLAMA_SERVICE_NAME}

[Service]
Type=oneshot
User=${RUN_USER}
Group=${RUN_GROUP}
WorkingDirectory=${PROJECT_DIR}
ExecStart=${VENV_DIR}/bin/python ${PROJECT_DIR}/scripts/nightly_preeval.py
EOF

install -o root -g root -m 0644 /dev/null "${NIGHTLY_PREEVAL_TIMER_FILE}"
tee "${NIGHTLY_PREEVAL_TIMER_FILE}" >/dev/null <<EOF
[Unit]
Description=Trigger the nightly LLM pre-evaluation pass before the trading day

[Timer]
# Assumes the system timezone is America/New_York, same as the warmup timer.
# Deliberately well after midnight -- article_filter.py's per-symbol verdict
# cache is keyed by calendar day, so running before midnight would cache
# under yesterday's date and never be read by the trading day it was meant
# for -- and well before the 09:00 warmup so the two never overlap.
OnCalendar=Mon..Fri 03:00:00
Persistent=false

[Install]
WantedBy=timers.target
EOF

install -o root -g root -m 0644 /dev/null "${DASHBOARD_SERVICE_FILE}"
tee "${DASHBOARD_SERVICE_FILE}" >/dev/null <<EOF
[Unit]
Description=Browser dashboard for the trading agent's per-symbol signal snapshot
After=network.target awg-quick@awg0.service
Requires=awg-quick@awg0.service

[Service]
Type=simple
User=${RUN_USER}
Group=${RUN_GROUP}
WorkingDirectory=${PROJECT_DIR}
ExecStart=${VENV_DIR}/bin/python ${PROJECT_DIR}/scripts/web_dashboard.py
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1
# The dashboard has no login and contains position/signal information. Bind only
# to AmneziaWG's server address so neither Ethernet nor Wi-Fi can reach it.
Environment=DASHBOARD_HOST=10.29.70.1
Environment=DASHBOARD_PORT=8765
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ProtectHome=read-only
# Read-only view of the two snapshot JSON files -- no ReadWritePaths needed.

[Install]
WantedBy=multi-user.target
EOF

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
TimeoutStopSec=15
KillSignal=SIGINT
# control-group (the default) signals every thread in the cgroup individually, which
# this multi-threaded Lumibot process never cleanly recovers from -- every stop needed
# a SIGKILL after the full TimeoutStopSec. Signaling only the main PID lets Python's
# normal SIGINT handling run and the process exits gracefully in under a second.
KillMode=process
Environment=PYTHONUNBUFFERED=1
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ProtectHome=read-only
ReadWritePaths=${PROJECT_DIR}

[Install]
WantedBy=multi-user.target
EOF

install -o root -g root -m 0644 /dev/null "${CRYPTO_SERVICE_FILE}"
tee "${CRYPTO_SERVICE_FILE}" >/dev/null <<EOF
[Unit]
Description=Lumibot Alpaca Crypto Trading Agent (active only while NYSE is closed)
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=${RUN_USER}
Group=${RUN_GROUP}
WorkingDirectory=${PROJECT_DIR}
ExecStart=${VENV_DIR}/bin/python ${PROJECT_DIR}/main_crypto.py
Restart=always
RestartSec=30
# 30s, not equity's 15s: once this process has run its first iteration (Alpaca
# websocket connect, per-symbol historical-bar fetches), a SIGINT stop
# reliably needs a few extra seconds before the process actually exits --
# py-spy showed every thread idle throughout that stall, so it isn't a slow
# iteration blocking shutdown, just the signal being noticed late. 30s gives
# it room to exit cleanly instead of always falling through to SIGKILL.
TimeoutStopSec=30
KillSignal=SIGINT
# Same rationale as trading-agent.service's KillMode -- process (not the
# default control-group) lets Python's own SIGINT handling exit this
# multi-threaded Lumibot process cleanly instead of needing a SIGKILL.
KillMode=process
Environment=PYTHONUNBUFFERED=1
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ProtectHome=read-only
ReadWritePaths=${PROJECT_DIR}

[Install]
WantedBy=multi-user.target
EOF

install -o root -g root -m 0644 /dev/null "${WATCHDOG_SERVICE_FILE}"
tee "${WATCHDOG_SERVICE_FILE}" >/dev/null <<EOF
[Unit]
Description=Sample trading-agent.service CPU usage for the watchdog log

[Service]
Type=oneshot
User=${RUN_USER}
Group=${RUN_GROUP}
ExecStart=${PROJECT_DIR}/scripts/cpu_watchdog.sh
EOF

install -o root -g root -m 0644 /dev/null "${WATCHDOG_TIMER_FILE}"
tee "${WATCHDOG_TIMER_FILE}" >/dev/null <<EOF
[Unit]
Description=Periodically sample trading-agent.service CPU usage

[Timer]
OnBootSec=2min
OnUnitActiveSec=5min
Persistent=true

[Install]
WantedBy=timers.target
EOF

systemctl daemon-reload
systemctl enable --now "${SERVICE_NAME}"
systemctl enable --now "${CRYPTO_SERVICE_NAME}"
systemctl enable --now "${WATCHDOG_TIMER_NAME}"
systemctl enable --now "${OLLAMA_SERVICE_NAME}"
systemctl enable --now "${OLLAMA_WARMUP_TIMER_NAME}"
systemctl enable --now "${NIGHTLY_PREEVAL_TIMER_NAME}"
systemctl enable --now "${DASHBOARD_SERVICE_NAME}"

echo "${SERVICE_NAME} is installed and running."
echo "View status with: sudo systemctl status ${SERVICE_NAME}"
echo "Follow logs with: sudo journalctl -u ${SERVICE_NAME} -f"
echo "${CRYPTO_SERVICE_NAME} is also installed and running, but idles until CRYPTO_ENABLED is true in config.json (and only trades while NYSE is closed)."
echo "Follow crypto logs with: sudo journalctl -u ${CRYPTO_SERVICE_NAME} -f"
echo "CPU usage is sampled every 5 minutes into .cpu_watchdog.log (warnings also go to the journal, tag trading-agent-cpu-watchdog)."
echo "${DASHBOARD_SERVICE_NAME} is installed on AmneziaWG at http://10.29.70.1:8765."
if [[ ! $(ollama list 2>/dev/null | grep -c .) -gt 1 ]]; then
    echo "Ollama is running but has no model yet. If LLM_NEWS_ENABLED is true, pull the model named in LLM_NEWS_MODEL, e.g.: ollama pull hf.co/ibm-granite/granite-4.1-3b-GGUF:Q4_K_M"
fi
