#!/bin/bash
#
# launchd wrapper for the Fixed ADX Hybrid position manager (every 1 min,
# 9:30-16:00 ET, trading days only). Swapped from execution_engine.py
# --manage to hybrid_engine.py --manage after compare_strategies.py showed
# the Hybrid strategy winning on both Sharpe and Calmar; this now manages
# data/hybrid_positions.json, not open_positions.json.
#
# Cheap bash checks (weekday, clock) run on every 60s tick; the market-holiday
# check only spawns Python when those already pass, so a 24/7 StartInterval
# does not mean a Python interpreter every minute of every day.
#
# PID lock mirrors run_intraday.sh: launchd already declines to start a second
# copy of a job that is still running, but the lock also covers manual runs.
# `flock` does not exist on macOS — hence the PID-file fallback.

set -uo pipefail
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin"

PY="/opt/anaconda3/bin/python3"
[[ -x "$PY" ]] || PY="$(command -v python3)"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${DIR}/logs/execution_manager.log"
mkdir -p "${DIR}/logs" "${DIR}/.state"

log()  { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$1" >> "$LOG"; }
skip() { log "SKIP: $1"; exit 0; }

cd "$DIR" || exit 1

# --- clock/weekday gate (bash-only, no Python spawn) -----------------------
NY_DOW=$(TZ="America/New_York" date '+%u')     # 1=Mon .. 7=Sun
NY_HHMM_RAW=$(TZ="America/New_York" date '+%H%M')
NY_HHMM=$((10#$NY_HHMM_RAW))                   # force base-10 (strip leading 0)
NY_HM_STR=$(TZ="America/New_York" date '+%H:%M')

if (( NY_DOW > 5 )); then
  skip "weekend (NY day $NY_DOW)"
fi
if (( NY_HHMM < 930 || NY_HHMM > 1600 )); then
  skip "outside 9:30-16:00 ET (now $NY_HM_STR ET)"
fi

# --- market-holiday gate (one source of truth: calendar_util.HOLIDAYS) -----
HOLIDAY_REASON="$("$PY" - <<'PYEOF'
from calendar_util import now_ny, is_trading_day
now = now_ny()
if not is_trading_day(now.date()):
    print(f"holiday/non-trading day ({now.date()})")
PYEOF
)"
[[ -n "$HOLIDAY_REASON" ]] && skip "$HOLIDAY_REASON"

# --- serialize passes --------------------------------------------------------
PIDFILE="${DIR}/.state/manage.pid"
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
  skip "a management pass is already running (pid $(cat "$PIDFILE"))"
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

"$PY" hybrid_engine.py --manage 2>&1 | tail -20 >> "$LOG"
exit "${PIPESTATUS[0]}"
