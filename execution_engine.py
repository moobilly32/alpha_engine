"""
Stateless execution manager — broker-agnostic order lifecycle over a JSON file.

WHAT "STATELESS" MEANS HERE
    The process holds nothing between calls. Everything needed to resume — every
    open position, which order ids are working, whether the first half has been
    scaled — lives in `data/open_positions.json`. Kill the process at any point
    and the next `manage_active_trades()` picks up exactly where it left off.
    That is what makes it safe to run from cron / launchd every few minutes.

BROKER-BLINDNESS
    This module imports `broker_interface` and a factory. It never imports
    `alpaca_client` or `robinhood_client` directly and never branches on which
    broker is live. It calls five methods:
        broker.get_price / buy_limit / sell_market / place_stop_loss / cancel_order
    `BROKER_MODE` in .env ("alpaca_paper" | "robinhood_live") picks the class.

THE STATE MACHINE  (status field on each position)
    PENDING_ENTRY  buy limit is working. Each pass CONFIRMS the real fill via
                   Broker.get_order_status() (never inferred from price — see
                   FILL VERIFICATION below), records the actual fill price/
                   quantity, and places the protective stop only once a real
                   fill is confirmed.
    OPEN           entry filled, a resting SELL stop is at the broker.
                     price >= scale_price  and not yet scaled
                         -> cancel stop, sell SCALE_FRACTION at market,
                            re-place a smaller stop at breakeven, mark scaled.
                     price >= target_price
                         -> cancel stop, sell the remainder at market, CLOSED.
                     price <= stop_price
                         -> the resting stop has done its job; reconcile to
                            CLOSED (reason STOP / BE_STOP).
                     age >= MAX_HOLD_DAYS (if configured)
                         -> cancel stop, sell remainder at market, CLOSED.
    CLOSING        a market sell was in flight when the process last stopped.
                   On restart we re-send a flat-everything market order and
                   close — at worst this is a no-op, never a double position.
    CLOSED         flattened; moved to data/closed_positions.jsonl and dropped.

FILL VERIFICATION (entry side) — resolves the former LIMITATION note
    _handle_pending() used to infer a fill purely from "price traded at/
    through the limit," with no check against the broker at all — a
    "blind-fill assumption." That is the same class of phantom-position
    bug found and fixed live on hybrid_engine.py's side of this project (a
    limit order that expired unfilled stayed tracked as a real position
    indefinitely). Fixed here by adding `Broker.get_order_status(order_id)`
    to the shared interface (broker_interface.py) — implemented for both
    Alpaca and Robinhood — and having _handle_pending() call it directly:
    a position only moves PENDING_ENTRY -> OPEN on a CONFIRMED fill
    (filled_qty > 0), using the broker's actual fill price/quantity, not
    the planned limit. A confirmed dead order (canceled/rejected) closes
    immediately as ENTRY_NOT_FILLED instead of silently waiting out
    MAX_HOLD_DAYS.

REMAINING LIMITATION — the OPEN state's exit side (stop/scale/target) still
    infers fills from price alone, the same way swing_backtest.py models a
    resting stop/limit (fills when price reaches it) — this file's
    five-method state-machine contract was kept for that side; only the
    entry side (the one that can create a phantom OPEN position) was
    rewired to Broker.get_order_status(). `Broker.list_positions()` also
    exists (see broker_interface.py) but is wired only to
    `--broker-positions`, a read-only human-facing view — the state
    machine's exit side still never reconciles against it. Run this often
    enough that exit-side price inference and reality do not drift (every
    1-5 min during RTH).

Usage:
    python3 execution_engine.py --status             # print the book
    python3 execution_engine.py --manage             # one management pass
    python3 execution_engine.py --selftest           # connect + quote SPY
    python3 execution_engine.py --broker-positions   # live broker positions
    python3 execution_engine.py --screen             # read-only Mixed-universe
                                                      # RSI/BB screen, no trades
    # entries are normally driven by the strategy layer:
    from execution_engine import execute_new_trade, manage_active_trades
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import tempfile
from typing import Any

from dotenv import load_dotenv

from broker_interface import Broker, BrokerError, TransientError, log
from calendar_util import now_ny
from config import (
    BASE, DATA, MAX_CONCURRENT, MAX_HOLD_DAYS,
    SCALE_ENABLED, SCALE_FRACTION, SCALE_R, TARGET_R,
)

load_dotenv(BASE / ".env")

POSITIONS_FILE = DATA / "open_positions.json"
CLOSED_LOG = DATA / "closed_positions.jsonl"

_ACTIVE = ("PENDING_ENTRY", "OPEN", "CLOSING")


# ================================================================= broker factory
def make_broker(mode: str | None = None) -> Broker:
    """Instantiate the broker named by BROKER_MODE (or the `mode` argument)."""
    mode = (mode or os.getenv("BROKER_MODE", "")).strip().lower()
    if mode == "alpaca_paper":
        from alpaca_client import AlpacaBroker
        return AlpacaBroker()
    if mode == "robinhood_live":
        from robinhood_client import RobinhoodBroker
        return RobinhoodBroker()
    raise ValueError(
        f"BROKER_MODE must be 'alpaca_paper' or 'robinhood_live', got {mode!r}. "
        f"Set it in {BASE / '.env'}")


_BROKER: Broker | None = None


def broker() -> Broker:
    """Lazy singleton so importing this module does not open a broker session."""
    global _BROKER
    if _BROKER is None:
        _BROKER = make_broker()
        log(f"execution_engine: broker = {_BROKER!r} "
            f"(BROKER_MODE={os.getenv('BROKER_MODE')!r})")
    return _BROKER


# ================================================================= state file
def _load() -> dict[str, dict]:
    if not POSITIONS_FILE.exists():
        return {}
    try:
        return json.loads(POSITIONS_FILE.read_text()) or {}
    except (json.JSONDecodeError, OSError) as e:
        log(f"execution_engine: {POSITIONS_FILE.name} unreadable ({e}); "
            f"treating as empty — inspect the file before trading")
        return {}


def _save(state: dict[str, dict]) -> None:
    """Atomic write: a crash mid-save must not truncate the book."""
    POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=POSITIONS_FILE.parent,
                               prefix=".open_positions.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(state, fh, indent=2, sort_keys=True)
        os.replace(tmp, POSITIONS_FILE)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _now() -> str:
    return now_ny().isoformat(timespec="seconds")


def _event(pos: dict, event: str, **fields: Any) -> None:
    pos.setdefault("history", []).append({"ts": _now(), "event": event, **fields})
    log(f"  {pos['ticker']}: {event}"
        + (f"  {fields}" if fields else ""))


def _age_days(pos: dict) -> float:
    ref = pos.get("opened_at") or pos.get("submitted_at")
    if not ref:
        return 0.0
    try:
        opened = dt.datetime.fromisoformat(ref)
        return (now_ny() - opened).total_seconds() / 86400.0
    except (ValueError, TypeError):
        return 0.0


def _archive(ticker: str, pos: dict) -> None:
    try:
        with open(CLOSED_LOG, "a") as fh:
            fh.write(json.dumps({"ticker": ticker, "archived_at": _now(), **pos})
                     + "\n")
    except OSError as e:
        log(f"execution_engine: could not archive {ticker} ({e})")


# ================================================================= entries
def execute_new_trade(ticker: str, shares: float, entry_price: float,
                      stop_price: float, target_price: float | None = None,
                      *, meta: dict | None = None) -> dict | None:
    """
    Submit a new BUY limit and start tracking it. Idempotent per ticker: calling
    it again while a position is active returns the existing record untouched.

    `entry_price` is the limit. `stop_price` is the initial protective stop
    (placed only once the entry is judged filled). The scale level and the final
    target are derived from the risk unit using the locked strategy constants
    (SCALE_R, TARGET_R) unless `target_price` is passed explicitly.
    """
    ticker = ticker.upper()
    state = _load()

    existing = state.get(ticker)
    if existing and existing.get("status") in _ACTIVE:
        log(f"execute_new_trade: {ticker} already {existing['status']} — no-op")
        return existing

    active = sum(1 for p in state.values() if p.get("status") in _ACTIVE)
    if active >= MAX_CONCURRENT:
        log(f"execute_new_trade: book full ({active}/{MAX_CONCURRENT}) — "
            f"{ticker} skipped")
        return None

    r_unit = float(entry_price) - float(stop_price)
    if r_unit <= 0:
        log(f"execute_new_trade: {ticker} stop {stop_price} not below entry "
            f"{entry_price} — rejected")
        return None

    if target_price is None:
        target_price = entry_price + r_unit * TARGET_R
    scale_price = (entry_price + r_unit * SCALE_R
                   if SCALE_ENABLED and 0 < SCALE_FRACTION < 1 else None)

    try:
        entry_oid = broker().buy_limit(ticker, shares, entry_price)
    except BrokerError as e:
        log(f"execute_new_trade: {ticker} buy_limit failed — {e}")
        return None

    pos = {
        "ticker": ticker,
        "broker_mode": os.getenv("BROKER_MODE"),
        "status": "PENDING_ENTRY",
        "shares_total": float(shares),
        "shares_open": float(shares),
        "entry_limit": round(float(entry_price), 4),
        "entry_order_id": entry_oid,
        "entry_fill": None,
        "stop_price": round(float(stop_price), 4),
        "stop_order_id": None,
        "scale_price": round(scale_price, 4) if scale_price else None,
        "scaled": False,
        "realized_pnl": 0.0,
        "target_price": round(float(target_price), 4),
        "r_unit": round(r_unit, 4),
        "submitted_at": _now(),
        "opened_at": None,
        "closed_at": None,
        "reason": None,
        "meta": meta or {},
        "history": [],
    }
    _event(pos, "ENTRY_SUBMITTED", order_id=entry_oid,
           limit=pos["entry_limit"], shares=pos["shares_total"])
    state[ticker] = pos
    _save(state)
    return pos


# ================================================================= management
def manage_active_trades() -> dict[str, str]:
    """
    One pass over every active position. Returns {ticker: status_after}.

    Broker calls are wrapped: a TransientError on one position logs and leaves
    that position for the next pass (the state machine is safe to re-enter);
    it does not abort the sweep over the others.
    """
    state = _load()
    if not state:
        return {}

    out: dict[str, str] = {}
    for ticker in list(state):
        pos = state[ticker]
        status = pos.get("status")

        if status not in _ACTIVE:
            _archive(ticker, pos)
            del state[ticker]
            _save(state)
            continue

        try:
            price = broker().get_price(ticker)
        except TransientError as e:
            log(f"  {ticker}: price unavailable ({e}) — deferring to next pass")
            out[ticker] = status
            continue
        except BrokerError as e:
            log(f"  {ticker}: get_price hard error ({e}) — deferring")
            out[ticker] = status
            continue

        try:
            if status == "PENDING_ENTRY":
                _handle_pending(pos)
            elif status == "OPEN":
                _handle_open(pos, price)
            elif status == "CLOSING":
                _handle_closing(pos, price)
        except BrokerError as e:
            # Leave the position where it is; next pass retries. The only
            # mutations above that precede a broker call are status flips to
            # CLOSING, which _handle_closing is built to recover from.
            log(f"  {ticker}: management step failed ({e}) — state preserved")

        out[ticker] = pos.get("status", status)
        _save(state)                       # persist after every ticker

    # sweep out anything that closed this pass
    for ticker in [t for t, p in state.items() if p.get("status") == "CLOSED"]:
        _archive(ticker, state[ticker])
        del state[ticker]
    _save(state)
    return out


def _handle_pending(pos: dict) -> None:
    """
    Entry limit working: confirm its REAL status at the broker via
    Broker.get_order_status() — never inferred from price the way this
    function used to (see the module docstring's former LIMITATION note).
    A confirmed fill (filled_qty > 0, full or partial — see
    get_order_status()'s docstring on the partial-fill quantity-snapshot
    caveat this shares with hybrid_engine.py's equivalent fix) moves the
    position to OPEN and places the protective stop off the ACTUAL fill
    price, not the planned limit. A confirmed dead order (canceled/
    rejected) closes immediately instead of waiting out MAX_HOLD_DAYS. An
    order that's still genuinely working ("new") or in an unrecognized
    state ("unknown") falls through to the pre-existing age-based expiry,
    unchanged.
    """
    try:
        info = broker().get_order_status(pos["entry_order_id"])
    except BrokerError as e:
        log(f"  {pos['ticker']}: get_order_status failed ({e}) — leaving "
            f"PENDING_ENTRY, will retry next pass")
        return

    if info.filled_qty > 0:
        pos["entry_fill"] = round(float(info.filled_avg_price or pos["entry_limit"]), 4)
        pos["shares_open"] = info.filled_qty
        pos["shares_total"] = info.filled_qty
        pos["opened_at"] = _now()
        _event(pos, "ENTRY_FILLED", price=pos["entry_fill"], shares=pos["shares_open"],
               order_status=info.status,
               note="confirmed via get_order_status, not inferred from price")
        pos["status"] = "OPEN"
        stop_oid = broker().place_stop_loss(
            pos["ticker"], pos["shares_open"], pos["stop_price"])
        pos["stop_order_id"] = stop_oid
        _event(pos, "STOP_PLACED", order_id=stop_oid, stop=pos["stop_price"])
        return

    if info.status in ("canceled", "rejected"):
        pos["status"] = "CLOSED"
        pos["closed_at"] = _now()
        pos["reason"] = "ENTRY_NOT_FILLED"
        _event(pos, "ENTRY_NOT_FILLED", order_status=info.status)
        return

    if _age_days(pos) >= max(1.0, float(MAX_HOLD_DAYS or 1)):
        broker().cancel_order(pos["entry_order_id"])
        pos["status"] = "CLOSED"
        pos["closed_at"] = _now()
        pos["reason"] = "ENTRY_EXPIRED"
        _event(pos, "ENTRY_EXPIRED", age_days=round(_age_days(pos), 2))


def _handle_open(pos: dict, price: float) -> None:
    """Filled position with a resting stop: stop / scale / target / time."""
    tk = pos["ticker"]

    # --- protective stop reached: the broker's resting order handled the sell.
    if price <= pos["stop_price"]:
        _best_effort_cancel(pos.get("stop_order_id"), tk)
        realized = (pos["stop_price"] - pos["entry_fill"]) * pos["shares_open"]
        pos["realized_pnl"] = round(pos.get("realized_pnl", 0.0) + realized, 2)
        pos["shares_open"] = 0.0
        pos["status"] = "CLOSED"
        pos["closed_at"] = _now()
        pos["reason"] = "BE_STOP" if pos["scaled"] else "STOP"
        _event(pos, pos["reason"], price=pos["stop_price"],
               realized_pnl=pos["realized_pnl"])
        return

    # --- scale the first tranche and lift the stop to breakeven.
    if pos["scale_price"] and not pos["scaled"] and price >= pos["scale_price"]:
        qty = round(pos["shares_total"] * SCALE_FRACTION, 6)
        pos["status"] = "CLOSING"                    # crash marker
        _event(pos, "SCALE_START", qty=qty, at=pos["scale_price"])
        _best_effort_cancel(pos.get("stop_order_id"), tk)
        broker().sell_market(tk, qty)
        realized = (pos["scale_price"] - pos["entry_fill"]) * qty
        pos["realized_pnl"] = round(pos.get("realized_pnl", 0.0) + realized, 2)
        pos["shares_open"] = round(pos["shares_total"] - qty, 6)
        pos["scaled"] = True
        pos["stop_price"] = round(pos["entry_fill"], 4)     # breakeven
        new_stop = broker().place_stop_loss(tk, pos["shares_open"],
                                            pos["stop_price"])
        pos["stop_order_id"] = new_stop
        pos["status"] = "OPEN"
        _event(pos, "SCALED", sold=qty, remaining=pos["shares_open"],
               stop_to=pos["stop_price"], stop_order_id=new_stop,
               realized_pnl=pos["realized_pnl"])
        return

    # --- final target: flatten the runner.
    if price >= pos["target_price"]:
        pos["status"] = "CLOSING"
        _event(pos, "TARGET_HIT", price=price, target=pos["target_price"])
        _best_effort_cancel(pos.get("stop_order_id"), tk)
        broker().sell_market(tk, pos["shares_open"])
        realized = (pos["target_price"] - pos["entry_fill"]) * pos["shares_open"]
        pos["realized_pnl"] = round(pos.get("realized_pnl", 0.0) + realized, 2)
        pos["shares_open"] = 0.0
        pos["status"] = "CLOSED"
        pos["closed_at"] = _now()
        pos["reason"] = "TARGET"
        _event(pos, "TARGET", realized_pnl=pos["realized_pnl"])
        return

    # --- time stop.
    if MAX_HOLD_DAYS and _age_days(pos) >= float(MAX_HOLD_DAYS):
        pos["status"] = "CLOSING"
        _event(pos, "TIME_STOP_START", age_days=round(_age_days(pos), 2))
        _best_effort_cancel(pos.get("stop_order_id"), tk)
        broker().sell_market(tk, pos["shares_open"])
        realized = (price - pos["entry_fill"]) * pos["shares_open"]
        pos["realized_pnl"] = round(pos.get("realized_pnl", 0.0) + realized, 2)
        pos["shares_open"] = 0.0
        pos["status"] = "CLOSED"
        pos["closed_at"] = _now()
        pos["reason"] = "TIME_STOP"
        _event(pos, "TIME_STOP", realized_pnl=pos["realized_pnl"])


def _handle_closing(pos: dict, price: float) -> None:
    """
    Recovery: the process died with a market sell in flight. Re-flatten whatever
    the book still thinks is open. sell_market is safe to repeat — if the first
    one landed, the second is for 0 shares and is rejected harmlessly (we guard
    on shares_open) or fills nothing material.
    """
    tk = pos["ticker"]
    log(f"  {tk}: recovering from CLOSING (shares_open={pos['shares_open']})")
    _best_effort_cancel(pos.get("stop_order_id"), tk)
    if pos.get("shares_open", 0) > 0:
        broker().sell_market(tk, pos["shares_open"])
        realized = (price - (pos.get("entry_fill") or price)) * pos["shares_open"]
        pos["realized_pnl"] = round(pos.get("realized_pnl", 0.0) + realized, 2)
        pos["shares_open"] = 0.0
    pos["status"] = "CLOSED"
    pos["closed_at"] = _now()
    pos["reason"] = pos.get("reason") or "CLOSING_RECOVERED"
    _event(pos, "CLOSING_RECOVERED", realized_pnl=pos["realized_pnl"])


def _best_effort_cancel(order_id: str | None, ticker: str) -> None:
    if not order_id:
        return
    try:
        broker().cancel_order(order_id)
    except BrokerError as e:
        log(f"  {ticker}: cancel {order_id} failed ({e}) — continuing; "
            f"a stale resting order may need a manual check")


# ================================================================= introspection
def status() -> None:
    state = _load()
    if not state:
        print("open_positions.json: empty")
        return
    print(f"{'TICKER':<8}{'STATUS':<14}{'SH_OPEN':>9}{'ENTRY':>10}"
          f"{'STOP':>10}{'SCALE':>10}{'TARGET':>10}{'R$':>9}  SCALED")
    for tk, p in sorted(state.items()):
        print(f"{tk:<8}{p.get('status',''):<14}{p.get('shares_open',0):>9.4f}"
              f"{(p.get('entry_fill') or p.get('entry_limit') or 0):>10.2f}"
              f"{p.get('stop_price') or 0:>10.2f}"
              f"{(p.get('scale_price') or 0):>10.2f}"
              f"{p.get('target_price') or 0:>10.2f}"
              f"{p.get('realized_pnl',0):>9.2f}  {p.get('scaled')}")


def broker_positions() -> None:
    """
    Query BROKER_MODE directly for whatever it actually holds right now —
    bypassing open_positions.json entirely. Useful as a sanity check against
    the local book, or when a position was opened/closed outside this engine.
    """
    b = broker()
    try:
        positions = b.list_positions()
    except BrokerError as e:
        print(f"broker-positions: failed to fetch from {b!r} — {e}")
        return

    print(f"{b!r} — {len(positions)} open position(s)")
    if not positions:
        return

    print(f"{'TICKER':<8}{'QTY':>10}{'AVG ENTRY':>12}{'CURRENT':>12}"
          f"{'UNREAL PNL':>14}")
    total_pnl = 0.0
    for p in sorted(positions, key=lambda p: p.symbol):
        print(f"{p.symbol:<8}{p.qty:>10.4f}{p.avg_entry_price:>12.2f}"
              f"{p.current_price:>12.2f}{p.unrealized_pnl:>14.2f}")
        total_pnl += p.unrealized_pnl
    print("-" * 56)
    print(f"{'TOTAL':<42}{total_pnl:>14.2f}")


def screen_watchlist() -> None:
    """
    Read-only live screen of the Mixed universe, kept byte-for-byte in sync
    with intraday.py's live oversold path — this calls the exact same
    backtest.daily_signal_frame(daily, variant="A") on the exact same
    backtest.MIXED_UNIVERSE, reads the SAME "last fully completed daily bar"
    (never a still-forming "today" bar from yfinance, which would let this
    repaint through the session) that _oversold_levels_daily() in
    intraday.py reads, and flags a row with SIGNAL straight off that frame
    rather than recomputing the RSI/BB condition by hand — a hand-rolled
    copy is exactly how this drifted from the live signal before (it used to
    check RSI/BB without the SMA200 trend gate the real signal requires).
    "PRICE" is a genuinely live quote (data.last_price), decoupled from the
    signal basis, since a daily close is never a tradable "right now" price
    — this mirrors intraday.py's own signal/price split exactly.

    This queries market data only. It never calls broker(), never touches
    open_positions.json, and places no orders — safe to run at any time,
    including with a live BROKER_MODE.
    """
    import backtest as bt   # local: keeps this research-module import out of
                            # every other CLI path's startup cost
    import data
    from calendar_util import now_ny

    today = now_ny().date()
    rows: list[tuple[str, float, float, float, float, bool]] = []
    for sym in bt.MIXED_UNIVERSE:
        try:
            daily = data.daily_bars(sym, period="max")
            if daily.empty or len(daily) < 210:
                log(f"screen: {sym} skipped — insufficient history")
                continue
            frame = bt.daily_signal_frame(daily, variant="A")
            hist = frame[frame.index.date < today]
            if hist.empty:
                log(f"screen: {sym} skipped — no completed daily bar yet")
                continue
            last = hist.iloc[-1]
            rsi_v = float(last["RSI"])
            bb_low = float(last["BB_LOW"])
            fired = bool(last["SIGNAL"])
            if rsi_v != rsi_v or bb_low != bb_low:   # NaN guard (still warming up)
                log(f"screen: {sym} skipped — RSI/BB not yet defined")
                continue
            price = data.last_price(sym)
            if price is None or price <= 0:
                price = float(last["Close"])         # fall back to last completed close
            dist_pct = (price - bb_low) / price * 100.0
            rows.append((sym, price, rsi_v, bb_low, dist_pct, fired))
        except Exception as e:                                     # noqa: BLE001
            log(f"screen: {sym} failed — {type(e).__name__}: {e}")
            continue

    if not rows:
        print("screen: no symbols returned usable data")
        return

    rows.sort(key=lambda r: r[2])    # lowest RSI (most oversold) first

    print(f"Mixed universe screen — {len(rows)}/{len(bt.MIXED_UNIVERSE)} symbols, "
          f"RSI/BB from the last completed daily bar, sorted by RSI ascending")
    print(f"{'SYMBOL':<8}{'PRICE':>10}{'RSI':>8}{'LOWER BB':>11}{'% TO BB':>10}")
    print("-" * 47)
    for sym, price, rsi_v, bb_low, dist_pct, fired in rows:
        flag = " *" if fired else ""
        print(f"{sym:<8}{price:>10.2f}{rsi_v:>8.1f}{bb_low:>11.2f}"
              f"{dist_pct:>9.2f}%{flag}")
    print("-" * 47)
    print("* SIGNAL fired on the last completed bar — this is EXACTLY what "
          "intraday.py's oversold path currently sees for this symbol.")
    print("Note: SIGNAL's BB leg checks that bar's own LOW touching the band; "
          "% TO BB compares the LIVE price to it instead (forward-looking), so "
          "a negative % TO BB without a flag means price has since fallen "
          "through the band but hasn't been confirmed by a completed bar yet.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Broker-agnostic execution manager")
    ap.add_argument("--status", action="store_true", help="print the book and exit")
    ap.add_argument("--manage", action="store_true", help="run one management pass")
    ap.add_argument("--selftest", action="store_true",
                    help="connect to BROKER_MODE and quote SPY")
    ap.add_argument("--broker-positions", action="store_true",
                    help="query BROKER_MODE directly and print live positions")
    ap.add_argument("--screen", action="store_true",
                    help="read-only live RSI/BB screen of the Mixed universe, "
                         "sorted by RSI ascending — no trades, no state file")
    args = ap.parse_args()

    if args.selftest:
        b = broker()
        print(f"broker   : {b!r}")
        print(f"SPY price: {b.get_price('SPY')}")
        return 0
    if args.status:
        status()
        return 0
    if args.manage:
        result = manage_active_trades()
        print("manage_active_trades ->", result or "{}")
        return 0
    if args.broker_positions:
        broker_positions()
        return 0
    if args.screen:
        screen_watchlist()
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
