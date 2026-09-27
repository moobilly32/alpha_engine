#!/bin/bash
#
# launchd wrapper for the Fixed ADX Hybrid entry scanner (every 15 min,
# 10:00-14:00 ET — see com.billy.alpha-intraday.plist's StartCalendarInterval
# for the schedule, unchanged from before). Swapped from intraday.py to
# hybrid_engine.py --execute after compare_strategies.py showed the Hybrid
# strategy winning on both Sharpe and Calmar.
#
# hybrid_engine.py, unlike the old intraday.py, has no internal calendar
# gate of its own. This wrapper now carries its own weekday/holiday/time-
# window gate (mirroring run_manage.sh's calendar_util.is_trading_day()
# check) so a weekday market holiday (Thanksgiving, Christmas, etc.) does
# not fire a real --execute pass against closed-market data — the plist's
# Weekday-only StartCalendarInterval was never sufficient for that on its
# own. Added after --execute went live against a real Robinhood account.
#
# Runs are serialised by a PID lock. launchd already declines to start a second
# copy of a job that is still running, but the lock also covers manual runs, and
# two concurrent passes would double-submit the same signal.
#
# `flock` does not exist on macOS — hence the PID-file fallback.

set -uo pipefail
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin"

PY="/opt/anaconda3/bin/python3"
[[ -x "$PY" ]] || PY="$(command -v python3)"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE="${DIR}/.state"
LOG="${DIR}/logs/intraday.log"
mkdir -p "$STATE" "${DIR}/logs"

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
if (( NY_HHMM < 1000 || NY_HHMM > 1400 )); then
  skip "outside 10:00-14:00 ET (now $NY_HM_STR ET)"
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

PIDFILE="${STATE}/intraday.pid"
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
  skip "a pass is already running (pid $(cat "$PIDFILE"))"
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

"$PY" hybrid_engine.py --execute 2>&1 | tail -20 >> "$LOG"
exit "${PIPESTATUS[0]}"
