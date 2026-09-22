"""
New York clock and market-calendar helpers.

Deliberately dependency-free (no pandas_market_calendars): the holiday list is
short, auditable, and does not add an install step to a launchd job that runs
with a minimal PATH.

Why this module exists at all: on a closed market day the strategy cannot
produce a hit, for a mechanical reason. A quote on a closed day returns the last
COMPLETED daily bar, so current price equals the previous daily close — and a
close can never exceed its own bar's high, making `price > prev_daily_high`
arithmetically impossible. Every ticker fails for a reason that has nothing to
do with the setup. Gate on the calendar, not just the clock.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

from config import TZ, WINDOW_START, WINDOW_END, FLAT_HHMM

NY = ZoneInfo(TZ)

# US equity market holidays. Extend as needed.
HOLIDAYS = {
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}

# Early closes (13:00 ET). Trades must be flat well before the bell on these.
HALF_DAYS = {
    "2025-07-03", "2025-11-28", "2025-12-24",
    "2026-11-27", "2026-12-24",
    "2027-11-26",
}


def now_ny() -> dt.datetime:
    return dt.datetime.now(NY)


def is_trading_day(d: dt.date) -> bool:
    return d.isoweekday() <= 5 and d.isoformat() not in HOLIDAYS


def is_half_day(d: dt.date) -> bool:
    return d.isoformat() in HALF_DAYS


def flat_time(d: dt.date) -> dt.time:
    """Flatten time for a given session — 25 minutes earlier on a half day."""
    if is_half_day(d):
        return dt.time(12, 55)
    return dt.time(*FLAT_HHMM)


def in_scan_window(ts: dt.datetime) -> bool:
    """True inside the 10:00-14:00 ET scanning window."""
    if not is_trading_day(ts.date()):
        return False
    hm = (ts.hour, ts.minute)
    end = (12, 0) if is_half_day(ts.date()) else WINDOW_END
    return WINDOW_START <= hm <= end


def window_reason(ts: dt.datetime) -> str | None:
    """Human-readable reason the engine should not run, or None if it should."""
    d = ts.date()
    if d.isoweekday() >= 6:
        return f"weekend ({d.isoformat()})"
    if d.isoformat() in HOLIDAYS:
        return f"US market holiday ({d.isoformat()})"
    hm = (ts.hour, ts.minute)
    end = (12, 0) if is_half_day(d) else WINDOW_END
    if hm < WINDOW_START:
        return f"before the {WINDOW_START[0]:02d}:{WINDOW_START[1]:02d} ET window (now {ts:%H:%M} ET)"
    if hm > end:
        return f"after the {end[0]:02d}:{end[1]:02d} ET window (now {ts:%H:%M} ET)"
    return None


def is_cadence_bar(ts: dt.datetime, interval_min: int) -> bool:
    """
    True when a bar timestamp lines up with the live scan cadence.

    The live engine only looks at the market every `interval_min` minutes. A
    backtest that evaluates entries on every 5-minute bar is testing a system
    you do not run — it sees three times as many entry opportunities and picks
    better prices than the real thing ever could.
    """
    return ts.minute % interval_min == 0
