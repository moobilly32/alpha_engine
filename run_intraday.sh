#!/bin/bash
#
# launchd wrapper for the Fixed ADX Hybrid entry scanner (every 15 min,
# 10:00-14:00 ET — see com.billy.alpha-intraday.plist's StartCalendarInterval
# for the schedule, unchanged from before). Swapped from intraday.py to
# hybrid_engine.py --execute after compare_strategies.py showed the Hybrid
# strategy winning on both Sharpe and Calmar.
#
# hybrid_engine.py, unlike the old intraday.py, has no internal calendar
# gate of its own, so this wrapper's PID lock is the only guard against a
# manual run outside the scheduled window — the plist's own schedule is
# what actually restricts when this fires unattended.
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

PIDFILE="${STATE}/intraday.pid"
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
  skip "a pass is already running (pid $(cat "$PIDFILE"))"
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

cd "$DIR" || exit 1

"$PY" hybrid_engine.py --execute 2>&1 | tail -20 >> "$LOG"
exit "${PIPESTATUS[0]}"
