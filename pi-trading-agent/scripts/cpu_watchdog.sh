#!/usr/bin/env bash
# Samples trading-agent.service's cgroup CPU usage and appends one line to
# .cpu_watchdog.log, so a future CPU spike (like the ~33%-for-26-hours one a
# stale market-hours cache once caused) leaves a timestamped trail instead of
# only being visible in the aggregate CPU total at process exit. Run every
# few minutes from trading-agent-cpu-watchdog.timer.
set -euo pipefail

SERVICE_NAME="trading-agent.service"
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_FILE="${PROJECT_DIR}/.cpu_watchdog_state"
LOG_FILE="${PROJECT_DIR}/.cpu_watchdog.log"
MAX_LOG_LINES=2000
WARN_THRESHOLD_PCT=10

cgroup_path="$(systemctl show -p ControlGroup --value "${SERVICE_NAME}" 2>/dev/null || true)"
if [[ -z "${cgroup_path}" ]]; then
    exit 0
fi

cpu_stat_file="/sys/fs/cgroup${cgroup_path}/cpu.stat"
if [[ ! -r "${cpu_stat_file}" ]]; then
    # Service isn't running (or cgroup v1 host) -- nothing to sample.
    exit 0
fi

usage_usec="$(awk '$1 == "usage_usec" { print $2 }' "${cpu_stat_file}")"
now_usec="$(date +%s%N)"
now_usec="$((now_usec / 1000))"
now_iso="$(date -Is)"

if [[ -f "${STATE_FILE}" ]]; then
    read -r prev_usage_usec prev_now_usec < "${STATE_FILE}"
else
    prev_usage_usec=""
    prev_now_usec=""
fi

echo "${usage_usec} ${now_usec}" > "${STATE_FILE}"

if [[ -z "${prev_usage_usec}" || -z "${prev_now_usec}" ]]; then
    # First sample since the watchdog started (or a restart) -- no interval yet.
    exit 0
fi

delta_usage_usec="$((usage_usec - prev_usage_usec))"
delta_wall_usec="$((now_usec - prev_now_usec))"
if [[ "${delta_wall_usec}" -le 0 ]]; then
    exit 0
fi

cpu_pct="$(awk -v u="${delta_usage_usec}" -v w="${delta_wall_usec}" 'BEGIN { printf "%.2f", (u / w) * 100 }')"

echo "${now_iso},${cpu_pct}" >> "${LOG_FILE}"
tail -n "${MAX_LOG_LINES}" "${LOG_FILE}" > "${LOG_FILE}.tmp" && mv "${LOG_FILE}.tmp" "${LOG_FILE}"

if awk -v p="${cpu_pct}" -v t="${WARN_THRESHOLD_PCT}" 'BEGIN { exit !(p >= t) }'; then
    logger -t trading-agent-cpu-watchdog "trading-agent.service CPU at ${cpu_pct}% over the last sample interval (threshold ${WARN_THRESHOLD_PCT}%)"
fi
