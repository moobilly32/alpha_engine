#!/bin/bash
#
# launchd wrapper for the Fixed ADX Hybrid screen (08:30 ET, Mon-Fri).
# Swapped from screener.py (build data/watchlist.json + Discord notify) to
# hybrid_engine.py --screen after compare_strategies.py showed the Hybrid
# strategy winning on both Sharpe and Calmar. --screen is a stateless,
# read-only DIP/BREAKOUT snapshot over the fixed Mixed Universe — it
# produces no watchlist file and needs no "already ran today" marker or
# custom notification, so both are dropped along with screener.py's
# fundamentals-based selection.
#
# All gating lives HERE, not in the plist. launchd re-fires a missed
# StartCalendarInterval job when the Mac wakes, regardless of what the clock
# then says — that is the catch-up behaviour we want, but it means the wrapper
# must decide for itself whether the run is still valid.
#
# Guards: weekend · US holiday · time window.

set -uo pipefail
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin"

# launchd gets a minimal PATH and cannot find the Anaconda interpreter that
# actually holds yfinance. Pin it, and fall back to the system one.
PY="/opt/anaconda3/bin/python3"
[[ -x "$PY" ]] || PY="$(command -v python3)"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${DIR}/logs/screener.log"
mkdir -p "${DIR}/logs"

log()  { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$1" >> "$LOG"; }
skip() { log "SKIP: $1"; exit 0; }

read -r NY_DATE NY_HHMM NY_DOW <<<"$("$PY" - <<'PY'
import datetime
from zoneinfo import ZoneInfo
print(datetime.datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H%M %u"))
PY
)"

# ---------------------------------------------------------------- gates
[[ "$NY_DOW" -ge 6 ]] && skip "weekend (NY ${NY_DATE})"

HOLIDAYS="
2026-01-01 2026-01-19 2026-02-16 2026-04-03 2026-05-25 2026-06-19
2026-07-03 2026-09-07 2026-11-26 2026-12-25
2027-01-01 2027-01-18 2027-02-15 2027-03-26 2027-05-31 2027-06-18
2027-07-05 2027-09-06 2027-11-25 2027-12-24
"
[[ "$HOLIDAYS" == *"$NY_DATE"* ]] && skip "US market holiday (${NY_DATE})"

[[ "$NY_HHMM" < "0400" ]] && skip "too early (now ${NY_HHMM} NY)"
[[ "$NY_HHMM" > "1400" ]] && skip "too late to be useful (now ${NY_HHMM} NY)"

# ---------------------------------------------------------------- run
log "RUN: NY ${NY_DATE} ${NY_HHMM}"
cd "$DIR" || exit 1

"$PY" hybrid_engine.py --screen 2>&1 | tail -40 >> "$LOG"
exit "${PIPESTATUS[0]}"
