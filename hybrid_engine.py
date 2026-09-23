"""
Steps 1-3 of the Fixed ADX Hybrid Strategy — screening, sizing, order
execution, and position management, built in COMPLETE ISOLATION from the
live paper-trading engine.

This file does not import intraday.py or execution_engine.py, and never
will — including for the broker connection. Steps 2/3 need a live equity
figure and real order submission (entries under --execute, exits under
--manage), so it connects to the broker directly via alpaca_client.
AlpacaBroker / robinhood_client.RobinhoodBroker (the underlying broker
implementations, not the live engine that wraps them) through its own tiny
make_broker() factory below, and tracks whatever it opens in its OWN state
files, data/hybrid_positions.json (open) and
data/hybrid_closed_positions.jsonl (closed) — never open_positions.json.
Two fully independent connections to the same paper account is fine:
alpaca-py authenticates per-instance over HTTP with no shared local session,
so this and execution_engine.py's own connection never collide.

PRODUCTION RISK MANAGEMENT — WIDENED SPREAD REGIME FILTER (permanent config,
deployed after hypothetical_backtester.py's regime-filter research this
session; best isolated backtest: ~+115% total return / ~9.8% max drawdown /
Sharpe ~1.2 over 2022-present, $100k. DISCLOSED CAVEAT: unlike the ORIGINAL
0.625%/0.25% baseline this replaced — which went through a live dry-run and
fill-verification pass before being cleared for forward paper testing —
this config's validation is backtest-only; it has not been through that
same live-testing cycle at these risk levels):
  1. Dynamic ATR/stop-distance position sizing, regime-gated asymmetric
     risk — implemented in hybrid_indicators.size_order() /
     current_spy_regime(): whenever SPY's last completed daily close is
     above its own rolling 50-day SMA ("BULL"), DIP risks 1.50% of live
     equity off a fixed 2.5% stop and BREAKOUT risks 1.00% off a 2.0x
     ATR14 stop; otherwise ("BEAR"), DIP risks 0.50% and BREAKOUT risks
     0.25% — the SAME asymmetric DIP-heavier-than-BREAKOUT shape as
     before, just doubled in the bull regime. Recomputed every cycle from
     broker.get_equity() (never a fixed capital figure), the live regime
     read, and the live ATR-based stop distance. Sized as a PRECISE
     FRACTIONAL share quantity (no whole-share floor), which is why
     execute_signals() submits entries via Broker.buy_market() rather than
     buy_limit() — neither Alpaca nor Robinhood accepts a fractional
     quantity on a limit order, only on market/notional orders. There is
     no resting limit price on entry as a result; see execute_signals()'s
     own docstring for how that's handled (stop/target always recomputed
     off the ACTUAL fill price).
  2. FOMC macro blackout (is_fomc_blackout / FOMC_DATES below) — blocks new
     entries 1:45-3:30 PM ET on a scheduled Fed day.
  3. SPY circuit breaker (spy_circuit_breaker_status below) — blocks new
     entries for 60 minutes after any EXACT rolling 10-minute SPY decline
     > 0.75%. This was left explicitly DISABLED in every multi-year daily-bar
     backtest (hypothetical_backtester.py's main_daily(), including the
     official production-baseline run with the synced FOMC calendar), per
     the established finding that daily bars can only express this rule via
     an Open->Low proxy that over-blocks and distorts results at that
     resolution — carrying an unreliable proxy into a long-term "will this
     break production" number would have defeated the purpose of that test.
     It is fully enabled here, on this file's own live 5-minute SPY data,
     since this IS the intraday engine the rule was designed for and no
     daily-bar proxy is needed.
  4. HYBRID_MAX_CONCURRENT (config.py, currently 6) caps simultaneously
     OPEN/PENDING positions — enforced in execute_signals(), a live gap
     that did not exist before this deployment (the book had no
     concurrency cap at all until now).
Both gates (2/3) block ONLY new entries (--execute / --dry-run, via
check_entry_gates()) — an already-OPEN position keeps managing off its own
stop/target/trailing-stop in --manage regardless of either gate.

Usage:
    python3 hybrid_engine.py --screen      # Step 1: read-only DIP/BREAKOUT scan
    python3 hybrid_engine.py --dry-run     # Step 2: size + print orders, no broker order calls
    python3 hybrid_engine.py --execute     # Step 2: submit real paper limit orders
    python3 hybrid_engine.py --manage      # Step 3: check stops/targets, exit if breached
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import tempfile
import time

from dotenv import load_dotenv

import data
from calendar_util import now_ny
from config import BASE, DATA as DATA_DIR, HYBRID_MAX_CONCURRENT
from hybrid_indicators import (
    MIXED_UNIVERSE, RISK_PCT_BREAKOUT_BEAR, RISK_PCT_BREAKOUT_BULL,
    RISK_PCT_DIP_BEAR, RISK_PCT_DIP_BULL, STOP_ATR_MULT, STOP_PCT, TARGET_R,
    current_atr, size_order, snapshot_universe,
)

load_dotenv(BASE / ".env")

HYBRID_POSITIONS_FILE = DATA_DIR / "hybrid_positions.json"
HYBRID_CLOSED_LOG = DATA_DIR / "hybrid_closed_positions.jsonl"


# ---------------------------------------------------------------- risk gates
# Merged from hypothetical_backtester.py's validated 4-year stress test
# (asymmetric 0.625%/0.25% risk sizing + FOMC blackout, circuit breaker
# disabled only because Yahoo's daily bars can't express a real 10-minute
# rule). Live here, the circuit breaker IS enabled — this is the intraday
# engine it was always meant to run on. Both gates block ONLY new entries
# (execute_signals / print_dry_run); an already-OPEN position keeps
# managing off its own stop/target/trailing-stop in --manage regardless of
# either gate being active, per the original spec.
FOMC_START_TIME = dt.time(13, 45)
FOMC_END_TIME = dt.time(15, 30)

# 2022-2026 Q3 compiled from memory (unverified against the source at the
# time); 2026 Q4 and all of 2027 fetched live from
# federalreserve.gov/monetarypolicy/fomccalendars.htm on 2026-09-19 and
# verified against that page directly. Every date is the SECOND day of each
# two-day meeting (statement + press conference), matching this list's own
# existing convention. Every FOMC meeting has held a press conference since
# 2019, so all 8 meetings/year are included regardless of whether the Fed's
# site had yet posted post-meeting documents for a future date.
# PRODUCTION CAVEAT #1: the 2022-2026 Q3 portion has still not been
# independently re-verified against federalreserve.gov (unchanged from the
# prior version of this list).
# PRODUCTION CAVEAT #2: federalreserve.gov itself states 2027 meeting dates
# are "tentative until confirmed at the meeting immediately preceding it" —
# i.e. the Fed can and does move future dates. Re-check this list against
# the live calendar periodically; a live gate silently stops protecting (or
# blocks on a date that got moved) if this goes stale, unlike a backtest.
FOMC_DATES: set[dt.date] = {
    dt.date(2022, 1, 26), dt.date(2022, 3, 16), dt.date(2022, 5, 4), dt.date(2022, 6, 15),
    dt.date(2022, 7, 27), dt.date(2022, 9, 21), dt.date(2022, 11, 2), dt.date(2022, 12, 14),
    dt.date(2023, 2, 1), dt.date(2023, 3, 22), dt.date(2023, 5, 3), dt.date(2023, 6, 14),
    dt.date(2023, 7, 26), dt.date(2023, 9, 20), dt.date(2023, 11, 1), dt.date(2023, 12, 13),
    dt.date(2024, 1, 31), dt.date(2024, 3, 20), dt.date(2024, 5, 1), dt.date(2024, 6, 12),
    dt.date(2024, 7, 31), dt.date(2024, 9, 18), dt.date(2024, 11, 7), dt.date(2024, 12, 18),
    dt.date(2025, 1, 29), dt.date(2025, 3, 19), dt.date(2025, 5, 7), dt.date(2025, 6, 18),
    dt.date(2025, 7, 30), dt.date(2025, 9, 17), dt.date(2025, 10, 29), dt.date(2025, 12, 10),
    dt.date(2026, 1, 28), dt.date(2026, 3, 18), dt.date(2026, 4, 29), dt.date(2026, 6, 17),
    dt.date(2026, 7, 29), dt.date(2026, 9, 16), dt.date(2026, 10, 28), dt.date(2026, 12, 9),
    dt.date(2027, 1, 27), dt.date(2027, 3, 17), dt.date(2027, 4, 28), dt.date(2027, 6, 9),
    dt.date(2027, 7, 28), dt.date(2027, 9, 15), dt.date(2027, 10, 27), dt.date(2027, 12, 8),
}

CB_LOOKBACK_PERIOD = "5d"    # native 5-min SPY bars to scan for a recent trip
CB_WINDOW_BARS = 2           # 2 x 5-minute bars = the EXACT 10-minute window
CB_DROP_PCT = 0.0075         # SPY decline over that window that trips the breaker
CB_BLOCK_MINUTES = 60        # new entries stay blocked for 60 min after a trip
RTH_START_MIN, RTH_END_MIN = 570, 960   # 9:30-16:00 ET, minutes past midnight


def is_fomc_blackout(now: dt.datetime) -> bool:
    """Exact-timestamp FOMC blackout: 1:45-3:30 PM ET on a scheduled Fed day."""
    return now.date() in FOMC_DATES and FOMC_START_TIME <= now.time() <= FOMC_END_TIME


def spy_circuit_breaker_status(now: dt.datetime) -> tuple[bool, str]:
    """
    Live version of hypothetical_backtester.compute_spy_circuit_breaker_events()
    + is_circuit_breaker_active(): an EXACT rolling 10-minute (2 x 5-min bar)
    SPY decline > CB_DROP_PCT, grouped per trading day so the rolling return
    never diffs across an overnight gap (that gap-vs-real-move bug was caught
    and fixed in the backtest before this was merged in). Returns
    (blocked, reason) — blocked if any such trip occurred within the
    trailing CB_BLOCK_MINUTES minutes of `now`.
    """
    try:
        raw = data.intraday_5m("SPY", period=CB_LOOKBACK_PERIOD)
    except Exception as e:                                          # noqa: BLE001
        return False, f"circuit breaker check failed ({type(e).__name__}: {e}) — not blocking"
    if raw.empty:
        return False, "no recent SPY data — not blocking"

    mins = raw.index.hour * 60 + raw.index.minute
    rth = raw[(mins >= RTH_START_MIN) & (mins < RTH_END_MIN)]
    if rth.empty:
        return False, "no recent RTH SPY data — not blocking"

    roll_ret = rth.groupby(rth.index.date)["Close"].pct_change(periods=CB_WINDOW_BARS)
    trips = rth.index[roll_ret < -CB_DROP_PCT]
    if len(trips) == 0:
        return False, "no 10-minute SPY drop > 0.75% detected"

    window_start = now - dt.timedelta(minutes=CB_BLOCK_MINUTES)
    recent = trips[(trips > window_start) & (trips <= now)]
    if len(recent) == 0:
        return False, "no 10-minute SPY drop > 0.75% in the trailing 60 minutes"

    last_trip = recent[-1]
    drop_pct = float(-roll_ret.loc[last_trip] * 100.0)
    return True, (f"SPY dropped {drop_pct:.2f}% in 10 minutes at {last_trip} "
                  f"— new entries blocked until {last_trip + dt.timedelta(minutes=CB_BLOCK_MINUTES)}")


def check_entry_gates() -> tuple[bool, list[str]]:
    """Returns (blocked, reasons) for whether NEW entries should be queued
    right now. Never affects --manage; existing positions always keep
    managing off their own stop/target/trailing-stop."""
    now = now_ny()
    reasons = []
    if is_fomc_blackout(now):
        reasons.append(f"FOMC blackout active (1:45-3:30 PM ET on a scheduled "
                       f"Fed day, now {now.strftime('%H:%M')} ET)")
    cb_blocked, cb_reason = spy_circuit_breaker_status(now)
    if cb_blocked:
        reasons.append(f"SPY circuit breaker: {cb_reason}")
    return bool(reasons), reasons


# ---------------------------------------------------------------- isolated broker
def make_broker():
    """
    Independent BROKER_MODE-driven factory — mirrors
    execution_engine.make_broker()'s selection logic but does not import
    execution_engine.py itself, per the strict isolation rule this file is
    built under.
    """
    mode = os.getenv("BROKER_MODE", "").strip().lower()
    if mode == "alpaca_paper":
        from alpaca_client import AlpacaBroker
        return AlpacaBroker()
    if mode == "robinhood_live":
        from robinhood_client import RobinhoodBroker
        return RobinhoodBroker()
    raise ValueError(f"BROKER_MODE must be 'alpaca_paper' or 'robinhood_live', "
                     f"got {mode!r}. Set it in {BASE / '.env'}")


# ---------------------------------------------------------------- state file
def _load_positions() -> dict:
    if not HYBRID_POSITIONS_FILE.exists():
        return {}
    try:
        return json.loads(HYBRID_POSITIONS_FILE.read_text()) or {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_positions(state: dict) -> None:
    """Atomic write, same pattern execution_engine.py uses for its own state
    file — applied independently here, not imported from there."""
    HYBRID_POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=HYBRID_POSITIONS_FILE.parent,
                               prefix=".hybrid_positions.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(state, fh, indent=2, sort_keys=True)
        os.replace(tmp, HYBRID_POSITIONS_FILE)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------- Step 1: screen
def print_screen() -> None:
    """
    Read-only terminal snapshot of the Mixed universe under the Fixed ADX
    Hybrid's DIP/BREAKOUT regime-routing rules. Makes ZERO broker calls,
    submits no orders, and writes no state files — the only disk write
    anywhere in this call path is data.py's existing market-data cache
    (.cache/*.pkl), the same cache every other read-only tool in this repo
    already relies on.
    """
    snaps = snapshot_universe()
    if not snaps:
        print("hybrid_engine: no symbols returned usable data")
        return

    order = {"DIP": 0, "BREAKOUT": 1, "NONE": 2}
    snaps.sort(key=lambda s: (order[s.mode],
                              s.rsi if s.rsi == s.rsi else 999.0))

    print(f"Fixed ADX Hybrid screen — {len(snaps)}/{len(MIXED_UNIVERSE)} "
          f"symbols (Mixed universe)")
    print("RSI/ADX/Vol/ATR from the last completed daily bar; PRICE is live")
    header = (f"{'SYMBOL':<8}{'PRICE':>10}{'MODE':>10}{'RSI':>8}"
             f"{'ADX/80th':>12}{'VOL/1.5xSMA':>13}{'ATR14':>9}")
    print(header)
    print("-" * len(header))
    for s in snaps:
        adx_col = (f"{s.adx:.1f}/{s.adx_80th:.1f}" if s.adx_80th == s.adx_80th
                  else f"{s.adx:.1f}/n-a")
        vol_col = f"{s.vol_ratio:.2f}x" if s.vol_ratio == s.vol_ratio else "n/a"
        print(f"{s.symbol:<8}{s.price:>10.2f}{s.mode:>10}{s.rsi:>8.1f}"
              f"{adx_col:>12}{vol_col:>13}{s.atr:>9.2f}")
    print("-" * len(header))

    n_dip = sum(1 for s in snaps if s.mode == "DIP")
    n_brk = sum(1 for s in snaps if s.mode == "BREAKOUT")
    print(f"DIP: {n_dip}   BREAKOUT: {n_brk}   NONE: {len(snaps) - n_dip - n_brk}")
    print()
    print("DIP: ADX < 20 (ranging) AND (RSI < 35 OR price <= lower BB).")
    print("BREAKOUT: ADX > this symbol's own rolling 50-session 80th-")
    print("percentile ADX (floored at 25) AND close > prior 20-day high AND")
    print("volume > 1.5x its 20-day average.")
    print("Read-only screen: no orders placed, no state written. "
          "intraday.py / execution_engine.py are untouched by this file.")


# ---------------------------------------------------------------- Step 2: sizing/orders
def _order_payload(order, broker_mode: str) -> dict:
    return {
        "symbol": order.symbol,
        "trade_type": order.trade_type,
        # NOT a limit price — market orders take no price. This is the
        # snapshot price size_order() sized/computed stop-target off of;
        # the ACTUAL fill price is confirmed after submission (see
        # execute_signals()) and may differ.
        "sizing_snapshot_price": order.entry_price,
        "atr14": order.atr14,
        "initial_stop": order.stop_price,
        "risk_pct": f"{order.risk_pct:.3%}",
        "risk_dollars": order.risk_dollars,
        "shares": order.shares,
        "notional": order.notional,
        "side": "buy",
        "order_type": "market",
        "broker_mode": broker_mode,
    }


def print_dry_run() -> None:
    """
    Simulates Step 2's position sizing for every currently active DIP/
    BREAKOUT signal, using REAL current equity (broker.get_equity() — the
    ONLY broker call this makes; buy_market() is never called) and real
    live prices. Prints the complete order payload for each and exits. No
    orders placed, no state written anywhere, including
    hybrid_positions.json.
    """
    snaps = [s for s in snapshot_universe() if s.mode in ("DIP", "BREAKOUT")]
    broker_mode = os.getenv("BROKER_MODE", "")

    try:
        b = make_broker()
        equity = b.get_equity()
    except Exception as e:                                          # noqa: BLE001
        print(f"hybrid_engine --dry-run: could not fetch live equity "
              f"({type(e).__name__}: {e}) — aborting; no simulated numbers "
              f"without a real balance")
        return

    print(f"DRY RUN — equity ${equity:,.2f} (live, read-only get_equity() "
          f"call; no order will be placed)")

    blocked, reasons = check_entry_gates()
    if blocked:
        print("NEW ENTRIES WOULD BE BLOCKED right now:")
        for r in reasons:
            print(f"  - {r}")
        print("(existing positions are unaffected by this — see --manage)\n")
        return

    if not snaps:
        print("No active DIP/BREAKOUT signals right now — nothing to size.")
        return

    print(f"{len(snaps)} active signal(s):\n")
    for s in snaps:
        order = size_order(s, equity)
        if order is None:
            print(f"--- {s.symbol} ({s.mode}) ---")
            print("  no valid order (ATR undefined, or risk/share count "
                  "rounds to 0 shares)\n")
            continue
        print(f"--- {order.symbol} ({order.trade_type}) ---")
        for k, v in _order_payload(order, broker_mode).items():
            print(f"  {k:<12}: {v}")
        print()


# ---------------------------------------------------------------- fill verification
# Root cause of the JPM incident (2026-09-17): execute_signals() used to
# write status="OPEN" to hybrid_positions.json the INSTANT buy_limit()
# returned an order_id, with no check that the order ever actually filled.
# JPM's limit order expired unfilled at end of day; local state kept
# tracking a position the broker never held. Fixed below: a submitted
# order is polled briefly for a fill; only a CONFIRMED fill (filled_qty >
# 0) is written as OPEN, using the ACTUAL fill price/quantity. Anything
# still working after the poll window is written as PENDING instead — not
# treated as an open position, not re-submitted next cycle (still blocks
# the per-symbol idempotency check), and resolved on a later --manage
# cycle by reconcile_pending_entries() below. An order already in a
# terminal non-fill state (rejected immediately, etc.) writes nothing at
# all, since there's nothing to track.
#
# Uses order.status.value (plain strings, e.g. "filled", "expired") rather
# than importing alpaca.trading.enums.OrderStatus at module level, so this
# file keeps its existing pattern of only loading a broker SDK lazily
# inside make_broker() — a Robinhood-only setup still never needs
# alpaca-py installed just to import this module.
_TERMINAL_UNFILLED_STATUS_VALUES = {
    "canceled", "expired", "rejected", "done_for_day", "stopped", "suspended",
}
FILL_POLL_ATTEMPTS = 3
FILL_POLL_INTERVAL_SEC = 1.5


def _poll_order_fill(b, order_id: str, attempts: int = FILL_POLL_ATTEMPTS,
                     interval: float = FILL_POLL_INTERVAL_SEC):
    """
    Briefly polls a just-submitted order to catch a fast fill (common for a
    limit order placed at/through the current price) before falling back to
    PENDING. Returns the last Order object seen (or None if every lookup
    attempt raised) — NOT a guarantee of a terminal state; a still-working
    order after this window is legitimately PENDING, not an error.
    Alpaca-specific (b._trading.get_order_by_id), matching the same
    already-accepted scope limit as reconcile_local_state() below.
    """
    order = None
    for attempt in range(1, attempts + 1):
        try:
            order = b._trading.get_order_by_id(order_id)
        except Exception as e:                                      # noqa: BLE001
            print(f"  fill-check {attempt}/{attempts} failed "
                 f"({type(e).__name__}: {e})")
            order = None
        else:
            filled_qty = float(order.filled_qty or 0)
            if filled_qty > 0 or order.status.value in _TERMINAL_UNFILLED_STATUS_VALUES:
                return order
        if attempt < attempts:
            time.sleep(interval)
    return order


def _recompute_after_fill(pos: dict, fill_price: float, filled_qty: float) -> dict:
    """
    Recomputes entry/stop/target off the ACTUAL fill price and quantity —
    never the originally planned limit price/size — so a filled position's
    risk parameters reflect what really happened at the broker.

    NOTE on partial fills: this snapshots whatever filled_qty is true at
    the moment it's checked. If the remainder of a still-working
    partially-filled order fills later, local `shares` can undercount vs.
    the broker's true position size — a real but separate limitation from
    the phantom-position bug this reconciliation path exists to fix (a
    share-count-too-low position is still a real, correctly-protected
    position, not a phantom).
    """
    trade_type = pos["trade_type"]
    highest_high = round(fill_price, 4)
    if trade_type == "DIP":
        stop = fill_price * (1.0 - STOP_PCT)
        r_unit = fill_price - stop
        target = fill_price + r_unit * TARGET_R if r_unit > 0 else None
        return dict(entry_price=round(fill_price, 4), shares=filled_qty,
                   current_stop=round(stop, 4),
                   target_price=round(target, 4) if target is not None else None,
                   highest_high=highest_high)
    atr = pos.get("atr14_at_entry")
    stop = (fill_price - STOP_ATR_MULT * atr) if atr else pos.get("current_stop")
    return dict(entry_price=round(fill_price, 4), shares=filled_qty,
               current_stop=round(stop, 4) if stop is not None else pos.get("current_stop"),
               target_price=None, highest_high=highest_high)


def _build_position_record(order, order_id: str, broker_order, broker_mode: str) -> dict:
    """
    Builds the hybrid_positions.json record for a just-submitted order,
    after _poll_order_fill()'s short verification window. `order` is the
    planned HybridOrder from size_order(); `broker_order` is the last Order
    _poll_order_fill() saw (None if every lookup attempt failed).
    """
    base = {
        "symbol": order.symbol,
        "trade_type": order.trade_type,
        "atr14_at_entry": order.atr14,
        "order_id": order_id,
        "broker_mode": broker_mode,
        "submitted_at": now_ny().isoformat(timespec="seconds"),
    }

    filled_qty = float(broker_order.filled_qty or 0) if broker_order is not None else 0.0
    if filled_qty > 0:
        fill_price = float(broker_order.filled_avg_price)
        base.update(_recompute_after_fill(base, fill_price, filled_qty))
        base["status"] = "OPEN"
        base["broker_order_status"] = broker_order.status.value
        return base

    # Not filled within the poll window (or every poll attempt failed) —
    # keep the ORIGINALLY PLANNED sizing as a placeholder; PENDING blocks a
    # resubmit next cycle, and reconcile_pending_entries() overwrites
    # entry_price/stop/target/shares with the real fill once one happens.
    base.update(dict(
        entry_price=order.entry_price, current_stop=order.stop_price,
        highest_high=order.entry_price, shares=order.shares,
        target_price=order.target_price, status="PENDING",
        broker_order_status=(broker_order.status.value if broker_order is not None
                             else "lookup_failed"),
    ))
    return base


def execute_signals() -> None:
    """
    Live path: submits a real buy_market() for every active DIP/BREAKOUT
    signal not already tracked in data/hybrid_positions.json — idempotent
    per symbol (an existing entry in ANY status, OPEN or PENDING, blocks a
    resubmit), in the same spirit as execution_engine.execute_new_trade(),
    but against an entirely separate state file/book. Never touches
    open_positions.json.

    MARKET, NOT LIMIT: size_order() sizes a precise FRACTIONAL share
    quantity (hybrid_indicators.py no longer floors to a whole share), and
    neither Alpaca nor Robinhood accepts a fractional quantity on a LIMIT
    order — only on MARKET/notional orders (Broker.buy_market(), see
    broker_interface.py). Entries submit via buy_market() accordingly.
    This is a real execution-mechanics change, not just a sizing tweak:
    there is no resting limit price protecting the entry from slippage —
    the fill price is whatever the market prints at submission, which is
    exactly why _build_position_record() below always recomputes
    stop/target off the ACTUAL fill price, never the signal snapshot price
    size_order() computed against.

    FILL VERIFICATION: see the module comment above _poll_order_fill() for
    the full incident this replaces (JPM, 2026-09-17) — in short, a
    position is only ever written as status="OPEN" after a CONFIRMED fill;
    an order still working after a short poll is written as PENDING
    instead and resolved later by manage_positions()'s
    reconcile_pending_entries(), never left silently marked OPEN on the
    strength of order SUBMISSION alone. A market order for a liquid symbol
    normally fills within the first poll attempt, so PENDING should be
    the rare case here, not the common one.
    """
    snaps = [s for s in snapshot_universe() if s.mode in ("DIP", "BREAKOUT")]

    try:
        b = make_broker()
        equity = b.get_equity()
    except Exception as e:                                          # noqa: BLE001
        print(f"hybrid_engine --execute: could not connect/fetch equity "
              f"({type(e).__name__}: {e}) — aborting")
        return

    broker_mode = os.getenv("BROKER_MODE", "")
    print(f"EXECUTE — equity ${equity:,.2f} ({broker_mode})")

    blocked, reasons = check_entry_gates()
    if blocked:
        print("NEW ENTRIES BLOCKED this cycle:")
        for r in reasons:
            print(f"  - {r}")
        print("Existing open positions are unaffected — --manage still runs "
              "their stops/targets/trailing-stops normally.")
        return

    if not snaps:
        print("No active DIP/BREAKOUT signals right now — nothing to submit.")
        return

    positions = _load_positions()
    active_count = sum(1 for p in positions.values()
                       if p.get("status") in ("OPEN", "PENDING"))

    for s in snaps:
        if s.symbol in positions:
            print(f"{s.symbol}: already tracked in hybrid_positions.json "
                 f"(status={positions[s.symbol].get('status')}) — skipping")
            continue
        if active_count >= HYBRID_MAX_CONCURRENT:
            print(f"{s.symbol}: {s.mode} signal active but book is full "
                 f"({active_count}/{HYBRID_MAX_CONCURRENT} OPEN/PENDING) — skipping")
            continue
        order = size_order(s, equity)
        if order is None:
            print(f"{s.symbol}: {s.mode} signal active but no valid order — skipping")
            continue
        try:
            order_id = b.buy_market(order.symbol, order.shares)
        except Exception as e:                                      # noqa: BLE001
            print(f"{s.symbol}: buy_market failed — {type(e).__name__}: {e}")
            continue

        print(f"{s.symbol}: market order submitted -> {order_id} "
             f"({order.shares} sh, sized off ~{order.entry_price} snapshot "
             f"price) — verifying fill...")
        broker_order = _poll_order_fill(b, order_id)
        record = _build_position_record(order, order_id, broker_order, broker_mode)
        positions[order.symbol] = record
        _save_positions(positions)
        active_count += 1   # this cycle's new OPEN/PENDING entry now occupies a slot too

        if record["status"] == "OPEN":
            print(f"{s.symbol}: FILLED -> OPEN ({record['shares']} sh @ "
                 f"{record['entry_price']})")
        else:
            print(f"{s.symbol}: not yet filled (order status="
                 f"{record.get('broker_order_status', 'unknown')}) -> PENDING, "
                 f"will confirm on a later --manage cycle")


# ---------------------------------------------------------------- Step 3: manage
def _archive_closed(record: dict) -> None:
    try:
        HYBRID_CLOSED_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(HYBRID_CLOSED_LOG, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------- broker reconciliation
# Found live on this account: an entry buy_limit() for JPM (2026-09-17) was
# submitted, immediately recorded as status="OPEN" in hybrid_positions.json,
# and then EXPIRED unfilled at the broker end-of-day. Nothing ever checked
# the fill, so local state kept tracking a position the broker never
# actually held -- --manage would eventually have called sell_market() on
# shares the account was never long, which on a shortable symbol risks
# opening an unintended short. The fix: broker.list_positions() is now the
# source of truth for "is this actually open," never local JSON alone.
def _broker_positions_by_symbol(b) -> dict[str, object]:
    """Live ground truth, keyed by symbol. Every status/management path
    below treats THIS as authoritative for what's actually held and how
    many shares — data/hybrid_positions.json only supplements a
    broker-confirmed position with this engine's own bookkeeping
    (trade_type, stop, target, atr14_at_entry), never decides membership."""
    return {p.symbol: p for p in b.list_positions()}


def reconcile_local_state(b, positions: dict) -> list[dict]:
    """
    For every symbol marked OPEN in local state but NOT held at the broker,
    looks up its entry order directly. If the order never filled (still
    pending/expired/canceled, filled_qty == 0) it's archived as
    ORDER_NOT_FILLED — a phantom entry, exactly the JPM case found live on
    this account. If it DID fill at some point (filled_qty > 0) but the
    broker shows no shares now, it was closed by something outside this
    engine (manual intervention, etc.) and is archived as CLOSED_EXTERNALLY
    instead — same cleanup, more honest reason code. If the order lookup
    itself fails, the entry is left OPEN (retried next cycle) rather than
    guessed away. Returns the records it archived, for logging.
    """
    live = _broker_positions_by_symbol(b)
    archived = []
    for sym, pos in list(positions.items()):
        if pos.get("status") != "OPEN" or sym in live:
            continue
        try:
            order = b._trading.get_order_by_id(pos["order_id"])
            filled_qty = float(order.filled_qty or 0)
            order_status = str(order.status)
        except Exception as e:                                      # noqa: BLE001
            print(f"{sym}: local state says OPEN but broker holds no shares, and "
                  f"the entry order lookup failed ({type(e).__name__}: {e}) — "
                  f"leaving it OPEN, will retry reconciliation next cycle")
            continue

        reason = "CLOSED_EXTERNALLY" if filled_qty > 0 else "ORDER_NOT_FILLED"
        print(f"{sym}: RECONCILE — local state says OPEN ({pos.get('shares')} sh) "
              f"but broker holds none (entry order status={order_status}, "
              f"filled_qty={filled_qty}) — archiving as {reason}")
        pos["status"] = reason
        pos["exit_reason"] = reason
        pos["broker_order_status_at_reconcile"] = order_status
        pos["realized_pnl"] = 0.0
        pos["closed_at"] = now_ny().isoformat(timespec="seconds")
        _archive_closed(pos)
        del positions[sym]
        archived.append(pos)
    return archived


def reconcile_pending_entries(b, positions: dict) -> tuple[list[dict], list[dict]]:
    """
    Resolves every PENDING entry — an order execute_signals() submitted
    whose fill wasn't confirmed within its short _poll_order_fill() window
    — by re-checking the order directly:
      - filled_qty > 0  -> PROMOTED to OPEN, using the ACTUAL fill price and
        quantity (see _recompute_after_fill()) rather than the originally
        planned limit price/size.
      - filled_qty == 0 and a terminal non-fill status (canceled/expired/
        rejected/done_for_day/stopped/suspended) -> archived as
        ORDER_NOT_FILLED, same reason code reconcile_local_state() uses for
        a position that was already (wrongly) marked OPEN before this fix.
      - otherwise (still genuinely working) -> left PENDING, retried next
        cycle; this is the expected, non-error common case for an order
        that simply hasn't filled yet.
    Returns (promoted, archived), both as lists of the updated records, for
    logging by the caller.
    """
    promoted, archived = [], []
    for sym, pos in list(positions.items()):
        if pos.get("status") != "PENDING":
            continue
        try:
            order = b._trading.get_order_by_id(pos["order_id"])
        except Exception as e:                                      # noqa: BLE001
            print(f"{sym}: PENDING order lookup failed ({type(e).__name__}: {e}) "
                  f"— leaving PENDING, will retry next cycle")
            continue

        filled_qty = float(order.filled_qty or 0)
        order_status = order.status.value

        if filled_qty > 0:
            fill_price = float(order.filled_avg_price)
            pos.update(_recompute_after_fill(pos, fill_price, filled_qty))
            pos["status"] = "OPEN"
            pos["broker_order_status"] = order_status
            print(f"{sym}: PENDING -> OPEN (filled {filled_qty} sh @ "
                  f"{fill_price:.4f}, order status={order_status})")
            promoted.append(pos)
        elif order_status in _TERMINAL_UNFILLED_STATUS_VALUES:
            pos["status"] = "ORDER_NOT_FILLED"
            pos["exit_reason"] = "ORDER_NOT_FILLED"
            pos["broker_order_status_at_reconcile"] = order_status
            pos["realized_pnl"] = 0.0
            pos["closed_at"] = now_ny().isoformat(timespec="seconds")
            _archive_closed(pos)
            del positions[sym]
            print(f"{sym}: PENDING order never filled (status={order_status}) "
                  f"— archived as ORDER_NOT_FILLED")
            archived.append(pos)
        else:
            print(f"{sym}: still PENDING (order status={order_status}, "
                  f"filled_qty=0) — retrying next cycle")
    return promoted, archived


def print_status() -> None:
    """
    Broker-authoritative status: reads broker.list_positions() directly, not
    data/hybrid_positions.json, so the reported position count can never
    drift from what the broker actually holds the way local state did for
    JPM. Local state is used only to ANNOTATE a broker-confirmed position
    with this engine's own trade_type/stop/target bookkeeping — never to
    decide what counts as a position. Also flags (read-only, no writes) any
    local OPEN entry not backed by a real broker position, and any broker
    position this engine isn't tracking locally.
    """
    try:
        b = make_broker()
        equity = b.get_equity()
    except Exception as e:                                          # noqa: BLE001
        print(f"hybrid_engine --status: could not connect to broker "
              f"({type(e).__name__}: {e}) — aborting")
        return

    live = _broker_positions_by_symbol(b)
    positions = _load_positions()
    local_open = {s: p for s, p in positions.items() if p.get("status") == "OPEN"}

    print(f"STATUS — equity ${equity:,.2f} ({os.getenv('BROKER_MODE', '')})")
    print(f"{len(live)} live position(s) at the broker (source of truth: "
          f"list_positions(), not {HYBRID_POSITIONS_FILE.name})")

    if not live:
        print("  (none)")
    else:
        header = (f"{'SYMBOL':<8}{'QTY':>10}{'ENTRY':>10}{'CURRENT':>10}"
                 f"{'UPNL':>12}{'TYPE':>10}{'STOP':>10}{'TARGET':>10}")
        print(header)
        print("-" * len(header))
        total_pnl = 0.0
        for sym, p in sorted(live.items()):
            meta = local_open.get(sym, {})
            total_pnl += p.unrealized_pnl
            stop = meta.get("current_stop")
            target = meta.get("target_price")
            stop_txt = f"{float(stop):.2f}" if stop is not None else "n/a"
            target_txt = f"{float(target):.2f}" if target is not None else "n/a"
            # .6f, not .0f: qty can be fractional now (buy_market() entries) —
            # truncating it here would misreport a real fractional holding.
            print(f"{sym:<8}{p.qty:>10.6f}{p.avg_entry_price:>10.2f}"
                  f"{p.current_price:>10.2f}{p.unrealized_pnl:>+12.2f}"
                  f"{meta.get('trade_type', 'n/a'):>10}{stop_txt:>10}{target_txt:>10}")
        print("-" * len(header))
        print(f"Total unrealized PnL: {total_pnl:+.2f}")

    ghosts = sorted(s for s in local_open if s not in live)
    if ghosts:
        print()
        print(f"WARNING: {len(ghosts)} symbol(s) marked OPEN in "
              f"{HYBRID_POSITIONS_FILE.name} but NOT held at the broker — local "
              f"state is stale: {', '.join(ghosts)}")
        print("Run --manage to reconcile (it does this automatically every cycle).")

    untracked = sorted(s for s in live if s not in local_open)
    if untracked:
        print()
        print(f"NOTE: {len(untracked)} broker position(s) not tracked in "
              f"{HYBRID_POSITIONS_FILE.name} (opened outside this engine?): "
              f"{', '.join(untracked)}")

    local_pending = sorted(s for s, p in positions.items() if p.get("status") == "PENDING")
    if local_pending:
        print()
        print(f"PENDING (order submitted, fill not yet confirmed — not counted "
              f"as a position above): {', '.join(local_pending)}")
        print("Run --manage to check for a fill (it does this automatically every cycle).")


def manage_positions() -> None:
    """
    Step 3 position manager: for every OPEN position in
    data/hybrid_positions.json, fetches a live price and today's updated
    ATR14, then applies each trade_type's exit rule:

      DIP      — current_stop is fixed (2.5%, set once at entry, never
                 moves). Exit (market sell) if price <= current_stop or
                 price >= target_price.
      BREAKOUT — trailing: highest_high = max(highest_high, live price);
                 candidate_stop = highest_high - STOP_ATR_MULT x today's
                 ATR14; if candidate_stop > current_stop, ratchet
                 current_stop up to it (never lowered). Exit (market sell)
                 if price <= current_stop.

    A breached stop/target submits a REAL sell_market() and moves the
    position from data/hybrid_positions.json to
    data/hybrid_closed_positions.jsonl. This is the live command, not a
    simulation — it WILL place a real paper market order if a position needs
    to close. With no positions open, it safely reports that and exits.

    BEFORE any of that, reconciles local state against the broker in two
    passes:
      1. reconcile_pending_entries() — resolves every PENDING entry (an
         order execute_signals() submitted but hadn't confirmed filled
         yet): promoted to OPEN on a confirmed fill, archived as
         ORDER_NOT_FILLED if it never will. This runs even when there are
         no OPEN positions at all, so a PENDING-only pass still gets
         resolved instead of short-circuiting on "nothing to manage."
      2. reconcile_local_state() — a second safety net for any symbol
         marked OPEN but not actually held at the broker (e.g. closed
         outside this engine). Archived, not managed.
    Together these are what stop a phantom entry from ever reaching the
    sell_market() call below against shares the account doesn't hold.
    """
    positions = _load_positions()
    initial_open = {s: p for s, p in positions.items() if p.get("status") == "OPEN"}
    initial_pending = {s: p for s, p in positions.items() if p.get("status") == "PENDING"}

    if not initial_open and not initial_pending:
        print(f"hybrid_engine --manage: no active or pending hybrid positions to "
              f"manage ({HYBRID_POSITIONS_FILE})")
        return

    try:
        b = make_broker()
    except Exception as e:                                      # noqa: BLE001
        print(f"hybrid_engine --manage: could not connect to broker "
              f"({type(e).__name__}: {e}) — aborting")
        return

    promoted, pending_archived = reconcile_pending_entries(b, positions)
    ghost_archived = reconcile_local_state(b, positions)
    archived = pending_archived + ghost_archived
    if promoted or archived:
        _save_positions(positions)

    open_positions = {s: p for s, p in positions.items() if p.get("status") == "OPEN"}
    if not open_positions:
        print(f"hybrid_engine --manage: no open hybrid positions to manage after "
              f"reconciling ({len(promoted)} promoted from PENDING, "
              f"{len(archived)} archived) ({HYBRID_POSITIONS_FILE})")
        return

    note = []
    if promoted:
        note.append(f"{len(promoted)} promoted from PENDING")
    if archived:
        note.append(f"{len(archived)} archived")
    print(f"MANAGE — {len(open_positions)} open hybrid position(s)"
         + (f" ({', '.join(note)})" if note else ""))
    closed_any = bool(archived)

    for sym, pos in open_positions.items():
        price = data.last_price(sym)
        if price is None or price <= 0:
            print(f"{sym}: could not fetch a live price — skipping this pass")
            continue

        trade_type = pos["trade_type"]
        current_stop = float(pos["current_stop"])
        exit_reason = None

        if trade_type == "DIP":
            target = pos.get("target_price")
            if price <= current_stop:
                exit_reason = "STOP"
            elif target is not None and price >= float(target):
                exit_reason = "TARGET"
            tgt_txt = f"{float(target):.2f}" if target is not None else "n/a"
            print(f"{sym}: DIP       price={price:.2f}  stop={current_stop:.2f}  "
                  f"target={tgt_txt}"
                  + (f"  -> {exit_reason}" if exit_reason else "  -> holding"))

        else:  # BREAKOUT
            atr_now = current_atr(sym)
            highest_high = max(float(pos.get("highest_high", pos["entry_price"])),
                               price)
            new_stop = current_stop
            if atr_now is not None:
                candidate = highest_high - STOP_ATR_MULT * atr_now
                if candidate > current_stop:
                    new_stop = candidate

            pos["highest_high"] = highest_high
            if new_stop != current_stop:
                print(f"{sym}: BREAKOUT stop ratcheted {current_stop:.2f} -> "
                      f"{new_stop:.2f} (highest_high={highest_high:.2f}, "
                      f"ATR14={atr_now:.2f})")
                pos["current_stop"] = new_stop
                current_stop = new_stop
                _save_positions(positions)

            if price <= current_stop:
                exit_reason = "TRAIL_STOP"
            print(f"{sym}: BREAKOUT  price={price:.2f}  stop={current_stop:.2f}  "
                  f"highest_high={highest_high:.2f}"
                  + (f"  -> {exit_reason}" if exit_reason else "  -> holding"))

        if exit_reason:
            try:
                order_id = b.sell_market(sym, pos["shares"])
            except Exception as e:                                  # noqa: BLE001
                print(f"{sym}: sell_market failed — {type(e).__name__}: {e} "
                      f"— position left OPEN, will retry next cycle")
                continue
            pnl = (price - float(pos["entry_price"])) * pos["shares"]
            pos["status"] = "CLOSED"
            pos["exit_price"] = price
            pos["exit_reason"] = exit_reason
            pos["exit_order_id"] = order_id
            pos["closed_at"] = now_ny().isoformat(timespec="seconds")
            pos["realized_pnl"] = round(pnl, 2)
            _archive_closed(pos)
            del positions[sym]
            closed_any = True
            print(f"{sym}: EXIT {exit_reason} -> sell_market order {order_id}, "
                  f"realized_pnl={pnl:+.2f}")

    _save_positions(positions)
    if not closed_any:
        print("No stops/targets breached this pass — all positions remain open.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fixed ADX Hybrid — isolated screener + sizing/execution")
    ap.add_argument("--screen", action="store_true",
                    help="read-only Mixed-universe DIP/BREAKOUT screen")
    ap.add_argument("--dry-run", action="store_true",
                    help="size + print orders for active signals; no broker "
                         "order calls, no state written")
    ap.add_argument("--execute", action="store_true",
                    help="submit real paper limit orders for active signals; "
                         "saves to data/hybrid_positions.json")
    ap.add_argument("--manage", action="store_true",
                    help="check open hybrid positions' stops/targets and "
                         "exit (real sell_market) if breached")
    ap.add_argument("--status", action="store_true",
                    help="broker-authoritative position/PnL status (reads "
                         "list_positions() directly, not local state; "
                         "read-only, flags any local/broker mismatch)")
    args = ap.parse_args()

    print(f"[SYSTEM ACTIVE] Widened Spread Engine | Bull "
         f"({RISK_PCT_BREAKOUT_BULL*100:.2f}%/{RISK_PCT_DIP_BULL*100:.2f}%) "
         f"Bear ({RISK_PCT_BREAKOUT_BEAR*100:.2f}%/{RISK_PCT_DIP_BEAR*100:.2f}%) "
         f"| Max Concurrency: {HYBRID_MAX_CONCURRENT}")

    if args.screen:
        print_screen()
        return 0
    if args.dry_run:
        print_dry_run()
        return 0
    if args.execute:
        execute_signals()
        return 0
    if args.manage:
        manage_positions()
        return 0
    if args.status:
        print_status()
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
