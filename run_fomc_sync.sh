#!/bin/bash
#
# launchd wrapper for the monthly FOMC calendar sync (update_fomc_calendar.py).
# Keeps hybrid_engine.py's live FOMC_DATES macro-blackout gate in sync with
# the official Fed calendar (federalreserve.gov) without a manual check-in.
# Network-only maintenance task, not part of the trading path — no
# market-hours/holiday gate needed, unlike the other three jobs.
#
# All gating lives HERE, not in the plist. StartCalendarInterval fires once
# for the 1st of the month, but launchd re-fires a MISSED interval (the Mac
# was asleep) on wake regardless of what day it then is — a monthly
# "already ran this cycle" marker keeps a late wake-catch-up from re-running
# more than once for the same month.

set -uo pipefail
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin"

# launchd hands jobs a minimal PATH; the Anaconda interpreter that holds
# requests/bs4 is not on it unless pinned here.
PY="/opt/anaconda3/bin/python3"
[[ -x "$PY" ]] || PY="$(command -v python3)"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${DIR}/logs/fomc_sync.log"
STATE_DIR="${DIR}/.state"
mkdir -p "${DIR}/logs" "${STATE_DIR}"

log()  { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$1" >> "$LOG"; }
skip() { log "SKIP: $1"; exit 0; }

read -r NY_YM <<<"$("$PY" - <<'PY'
import datetime
from zoneinfo import ZoneInfo
print(datetime.datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m"))
PY
)"

# ---------------------------------------------------------------- gates
MARKER="${STATE_DIR}/fomc_sync_${NY_YM}.done"
[[ -f "$MARKER" ]] && skip "already ran this month (${NY_YM})"

# ---------------------------------------------------------------- run
log "RUN: month ${NY_YM}"
cd "$DIR" || exit 1

"$PY" update_fomc_calendar.py 2>&1 | tail -60 >> "$LOG"
rc="${PIPESTATUS[0]}"

if [[ "$rc" -eq 0 ]]; then
    touch "$MARKER"
else
    log "update_fomc_calendar.py exited ${rc} — marker NOT written, will retry next fire"
fi
exit "$rc"
