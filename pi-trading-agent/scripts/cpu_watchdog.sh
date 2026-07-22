#!/usr/bin/env bash
# Record cgroup CPU for every resident trading component plus the host's
# memory headroom and CPU temperature. Run from trading-agent-cpu-watchdog.timer.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_FILE="${PROJECT_DIR}/.cpu_watchdog_state"
LOG_FILE="${PROJECT_DIR}/.cpu_watchdog.log"
MAX_LOG_LINES=2000
WARN_CPU_PCT=10
WARN_AVAILABLE_MEMORY_MIB=256
WARN_TEMPERATURE_C=80
CGROUP_ROOT="${CGROUP_ROOT:-/sys/fs/cgroup}"
MEMINFO_FILE="${MEMINFO_FILE:-/proc/meminfo}"
THERMAL_FILE="${THERMAL_FILE:-/sys/class/thermal/thermal_zone0/temp}"
SERVICES=(
    "equity:trading-agent.service"
    "crypto:trading-agent-crypto.service"
    "ollama:ollama.service"
)

declare -A previous_usage=()
declare -A previous_time=()
if [[ -f "${STATE_FILE}" ]]; then
    while read -r label usage_usec sampled_at_usec; do
        [[ -n "${label:-}" && -n "${usage_usec:-}" && -n "${sampled_at_usec:-}" ]] || continue
        previous_usage["${label}"]="${usage_usec}"
        previous_time["${label}"]="${sampled_at_usec}"
    done < "${STATE_FILE}"
fi

now_usec="$(date +%s%N)"
now_usec="$((now_usec / 1000))"
now_iso="$(date -Is)"
state_tmp="${STATE_FILE}.tmp"
: > "${state_tmp}"

samples=()
warnings=()
valid_cpu_sample=false
for entry in "${SERVICES[@]}"; do
    label="${entry%%:*}"
    service="${entry#*:}"
    cpu_pct="na"
    cgroup_path="$(systemctl show -p ControlGroup --value "${service}" 2>/dev/null || true)"
    cpu_stat_file="${CGROUP_ROOT}${cgroup_path}/cpu.stat"
    if [[ -n "${cgroup_path}" && -r "${cpu_stat_file}" ]]; then
        usage_usec="$(awk '$1 == "usage_usec" { print $2 }' "${cpu_stat_file}")"
        echo "${label} ${usage_usec} ${now_usec}" >> "${state_tmp}"
        prior_usage="${previous_usage[${label}]:-}"
        prior_time="${previous_time[${label}]:-}"
        if [[ -n "${prior_usage}" && -n "${prior_time}" && "${usage_usec}" -ge "${prior_usage}" ]]; then
            delta_usage_usec="$((usage_usec - prior_usage))"
            delta_wall_usec="$((now_usec - prior_time))"
            if [[ "${delta_wall_usec}" -gt 0 ]]; then
                cpu_pct="$(awk -v u="${delta_usage_usec}" -v w="${delta_wall_usec}" 'BEGIN { printf "%.2f", (u / w) * 100 }')"
                valid_cpu_sample=true
                if awk -v p="${cpu_pct}" -v t="${WARN_CPU_PCT}" 'BEGIN { exit !(p >= t) }'; then
                    warnings+=("${service} CPU ${cpu_pct}% (threshold ${WARN_CPU_PCT}%)")
                fi
            fi
        fi
    fi
    samples+=("${label}_cpu_pct=${cpu_pct}")
done
mv "${state_tmp}" "${STATE_FILE}"

# Preserve the original first-sample behavior: establish cgroup baselines
# without writing an incomplete log record or generating startup warnings.
if [[ "${valid_cpu_sample}" != true ]]; then
    exit 0
fi

available_memory_mib="na"
if [[ -r "${MEMINFO_FILE}" ]]; then
    available_memory_kib="$(awk '$1 == "MemAvailable:" { print $2 }' "${MEMINFO_FILE}")"
    if [[ -n "${available_memory_kib}" ]]; then
        available_memory_mib="$((available_memory_kib / 1024))"
        if [[ "${available_memory_mib}" -lt "${WARN_AVAILABLE_MEMORY_MIB}" ]]; then
            warnings+=("available memory ${available_memory_mib} MiB (threshold ${WARN_AVAILABLE_MEMORY_MIB} MiB)")
        fi
    fi
fi

temperature_c="na"
if [[ -r "${THERMAL_FILE}" ]]; then
    raw_temperature="$(tr -d '[:space:]' < "${THERMAL_FILE}")"
    if [[ "${raw_temperature}" =~ ^[0-9]+$ ]]; then
        temperature_c="$(awk -v t="${raw_temperature}" 'BEGIN { printf "%.1f", t / 1000 }')"
        if awk -v t="${temperature_c}" -v w="${WARN_TEMPERATURE_C}" 'BEGIN { exit !(t >= w) }'; then
            warnings+=("CPU temperature ${temperature_c}C (threshold ${WARN_TEMPERATURE_C}C)")
        fi
    fi
fi

samples+=("available_memory_mib=${available_memory_mib}")
samples+=("temperature_c=${temperature_c}")
sample_csv="$(IFS=,; echo "${samples[*]}")"
echo "${now_iso},${sample_csv}" >> "${LOG_FILE}"
tail -n "${MAX_LOG_LINES}" "${LOG_FILE}" > "${LOG_FILE}.tmp" && mv "${LOG_FILE}.tmp" "${LOG_FILE}"

if [[ "${#warnings[@]}" -gt 0 ]]; then
    warning_text="$(IFS='; '; echo "${warnings[*]}")"
    logger -t trading-agent-cpu-watchdog "${warning_text}" || true
fi
