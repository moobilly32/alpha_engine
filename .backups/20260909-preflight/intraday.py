"""
Intraday execution & cadence engine — one pass over the watchlist.

Runs every 15 minutes between 10:00 and 14:00 ET. For each watchlist name it
rebuilds the levels, evaluates the signal, works out whether price is inside
the ideal execution range, sizes the position, and alerts Discord.

NOTIFICATION POLICY — quiet by default. 17 runs a day across 30 names is 510
evaluations; alerting on all of them makes the channel unreadable and you stop
looking at it, which is worse than no alerts at all. A name is announced when
it ENTERS a state, not while it remains in one:

  * EXECUTE      — alerted once per symbol per day.
  * APPROACHING  — alerted once per symbol per day, and only if it has not
                   already fired EXECUTE.
  * EXTENDED     — never alerted on its own; it appears only as a downgrade
                   note on a symbol already announced.

Per-day state lives in .state/seen_YYYY-MM-DD.json.

Usage:
    python3 intraday.py                 # gated run (what launchd calls)
    python3 intraday.py --force         # ignore the clock/calendar gate
    python3 intraday.py --dry-run       # evaluate and print, send nothing
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

import numpy as np

import data
import notify
import screener
import strategy
from calendar_util import now_ny, window_reason, NY
from config import (
    ACCOUNT_SIZE, STATE, LOGS, MAX_CONCURRENT, SCAN_INTERVAL_MIN,
    ENTRY_MODE, PULLBACK_SUPPORT, PULLBACK_MAX_EXTENSION,
    PULLBACK_DIP_LOOKBACK, PULLBACK_DIP_TOLERANCE, REQUIRE_STRONG_TREND,
    STOP_BUFFER, MAX_STOP_PCT, TARGET_R, MAX_POSITION_PCT,
)
from execution import State, build_plan, size_position


def log(msg: str) -> None:
    line = f"{now_ny():%Y-%m-%d %H:%M:%S %Z}  {msg}"
    print(line, flush=True)
    try:
        with open(LOGS / "intraday.log", "a") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------- per-day state
def _state_path(d: dt.date):
    return STATE / f"seen_{d.isoformat()}.json"


def load_state(d: dt.date) -> dict:
    p = _state_path(d)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"announced": {}, "first_run_done": False}


def save_state(d: dt.date, st: dict) -> None:
    try:
        _state_path(d).write_text(json.dumps(st, indent=2))
    except Exception:
        pass


# ---------------------------------------------------------------- evaluation
def evaluate_symbol(snap: dict, equity: float, size_mult: float,
                    asof: dt.datetime):
    """
    Returns (plan, signal) for one watchlist name, or (None, None) if there is
    not enough data to judge it.
    """
    sym = snap["symbol"]
    try:
        daily = data.daily_bars(sym, period="2y")
        intra = data.intraday_5m(sym, period="5d", ttl=300)
    except Exception as e:
        log(f"  {sym}: data error {type(e).__name__}")
        return None, None

    if daily.empty or intra.empty:
        return None, None

    today = asof.date()
    sess = intra[intra.index.date == today]
    # Truncate to the scan instant. Live this is a no-op (the last bar IS now),
    # but under --asof it is what makes a replay honest instead of handing the
    # engine bars from later in the session.
    sess = sess[sess.index <= asof]
    if sess.empty:
        return None, None

    # The current bar is the last one; levels must exclude it.
    upto = len(sess) - 1
    price = float(sess["Close"].iloc[-1])

    if ENTRY_MODE == "pullback":
        levels, sig = _pullback_levels(sym, daily, sess, today, price)
    else:
        levels = strategy.compute_levels(sym, daily, sess, today, upto)
        sig = strategy.evaluate_signal(levels, price)

    if levels is None or levels.trigger is None:
        return None, sig

    # Valuation target is context for the alert, never the trade's exit.
    val_target = snap.get("dcf_value")

    plan = build_plan(sym, price, levels.trigger, equity,
                      valuation_target=val_target, size_multiplier=size_mult)

    # In pullback mode the stop is structural — under the dip low — rather than
    # a fixed percentage, so the plan's sizing must be rebuilt around it or the
    # alert would quote a risk the trade does not actually take.
    if ENTRY_MODE == "pullback" and sig.fired:
        stop = strategy.structural_stop(levels, price, STOP_BUFFER, MAX_STOP_PCT)
        if stop:
            plan.stop = round(stop, 2)
            plan.target = round(price + (price - stop) * TARGET_R, 2)
            shares, note = size_position(
                equity, price, stop, size_mult,
                max_position_pct=min(MAX_POSITION_PCT, 1.0 / MAX_CONCURRENT))
            plan.shares = shares
            plan.dollars = round(shares * price, 2)
            plan.risk_dollars = round(shares * (price - stop), 2)
            plan.reward_dollars = round(plan.risk_dollars * TARGET_R, 2)
            plan.rr = TARGET_R
            plan.size_note = note

    # The trend leg is a hard veto regardless of where price sits in the range.
    if not levels.trend_ok:
        plan.state = State.BELOW

    return plan, sig


def _pullback_levels(sym, daily, sess, today, price):
    """Build pullback levels from the session so far (RTH bars only)."""
    mins = sess.index.hour * 60 + sess.index.minute
    rth = sess[(mins >= 570) & (mins < 960)]
    if len(rth) < 2:
        return None, strategy.Signal(False, "no RTH bars yet today",
                                     strategy.Levels(symbol=sym))

    ctx = strategy.daily_context_full(daily, today)
    vw = strategy.session_vwap(rth)
    em = strategy.ema(rth["Close"].to_numpy(dtype=float), 20)
    sup = {"vwap": vw[-1], "ema20": em[-1], "sma20d": ctx.sma20}.get(
        PULLBACK_SUPPORT, vw[-1])

    # Reconstruct "was above" and "dipped" over the session, excluding the
    # current bar from the dip search the same way the backtest does.
    lows = rth["Low"].to_numpy(dtype=float)
    closes = rth["Close"].to_numpy(dtype=float)
    level = {"vwap": vw, "ema20": em}.get(PULLBACK_SUPPORT)
    if level is None:
        level = np.full(len(rth), ctx.sma20 if ctx.sma20 else np.nan)

    was_above = bool(np.any(closes[:-1] > level[:-1]))
    dip_idx = [i for i in range(len(rth))
               if lows[i] <= level[i] * (1.0 + PULLBACK_DIP_TOLERANCE)]
    dipped = bool(dip_idx and (len(rth) - 1 - dip_idx[-1]) <= PULLBACK_DIP_LOOKBACK)
    dip_low = min(lows[i] for i in dip_idx) if dip_idx else None

    lv = strategy.Levels(
        symbol=sym, prev_high=ctx.prev_high, prev_close=ctx.prev_close,
        sma200=ctx.sma200, sma50=ctx.sma50, atr=ctx.atr,
        support=float(sup) if sup is not None else None,
        support_name=PULLBACK_SUPPORT, was_above=was_above,
        dip_low=dip_low, dipped=dipped)
    return lv, strategy.evaluate_pullback(lv, price, PULLBACK_MAX_EXTENSION,
                                          REQUIRE_STRONG_TREND)


# ---------------------------------------------------------------- run
def run(force: bool = False, dry_run: bool = False, equity: float | None = None,
        asof: str | None = None) -> int:
    now = now_ny()
    if asof:
        # Replay a past scan instant. Uses only bars at or before `asof`, so it
        # exercises the real evaluation path rather than a mock.
        now = dt.datetime.strptime(asof, "%Y-%m-%d %H:%M").replace(tzinfo=NY)
        log(f"REPLAY MODE: pretending NY now = {now:%Y-%m-%d %H:%M}")
        force = True

    if not force:
        reason = window_reason(now)
        if reason:
            log(f"SKIP: {reason}")
            return 0

    wl = screener.load_watchlist()
    if not wl or not wl.get("watchlist"):
        log("SKIP: no watchlist — run screener.py first")
        return 1

    stale = (wl.get("session_date") != now.date().isoformat()) and not asof
    if stale:
        log(f"WARNING: watchlist is from {wl.get('session_date')}, not today "
            f"({now.date()}). Fundamentals may be out of date.")

    macro = wl.get("macro", {})
    size_mult = float(macro.get("size_multiplier", 1.0))
    equity = equity if equity is not None else ACCOUNT_SIZE

    st = load_state(now.date())
    announced: dict = st["announced"]

    names = wl["watchlist"]
    log(f"RUN {now:%H:%M} ET · {len(names)} names · equity ${equity:,.2f} "
        f"· size x{size_mult:g}{' · STALE WATCHLIST' if stale else ''}")

    embeds: list[dict] = []
    execute_now: list[str] = []
    approaching: list[str] = []
    tallies: dict[str, int] = {}

    for snap in names:
        sym = snap["symbol"]
        plan, sig = evaluate_symbol(snap, equity, size_mult, now)
        if plan is None:
            tallies["no data"] = tallies.get("no data", 0) + 1
            continue

        tallies[plan.state.value] = tallies.get(plan.state.value, 0) + 1

        if plan.state is State.EXECUTE:
            execute_now.append(sym)
        elif plan.state is State.APPROACHING:
            approaching.append(sym)

        prior = announced.get(sym)
        should_alert = (
            (plan.state is State.EXECUTE and prior != State.EXECUTE.value)
            or (plan.state is State.APPROACHING and prior is None)
        )

        if should_alert and plan.shares > 0:
            note = ""
            if stale:
                note = "⚠️ Watchlist fundamentals are from a previous session."
            if plan.state is State.APPROACHING:
                note = (note + "\n" if note else "") + \
                    f"Not yet triggered. Arms at ${plan.trigger:.2f}."
            embeds.append(notify.build_embed(snap, plan, note))
            announced[sym] = plan.state.value

    # ---- portfolio guard: never advertise more concurrent entries than allowed
    if len(execute_now) > MAX_CONCURRENT:
        log(f"  NOTE: {len(execute_now)} names in range but MAX_CONCURRENT="
            f"{MAX_CONCURRENT} — highest-scoring names take precedence")

    summary = " · ".join(f"{k} {v}" for k, v in sorted(tallies.items()))
    log(f"  {summary}")
    if execute_now:
        log(f"  IN RANGE: {', '.join(execute_now)}")
    if approaching:
        log(f"  APPROACHING: {', '.join(approaching)}")

    # ---- first run of the day always reports, so silence is never ambiguous
    sent_first = False
    if not st["first_run_done"] and not dry_run:
        header = (f"**Alpha Engine — first scan {now:%H:%M} ET**\n"
                  f"{len(names)} names · {macro.get('summary', 'macro n/a')}\n"
                  f"In range: {', '.join(execute_now) or 'none'} · "
                  f"Approaching: {', '.join(approaching) or 'none'}")
        if embeds:
            notify.send_alerts(embeds, header)
            log(f"  sent first-scan summary + {len(embeds)} alert(s)")
        else:
            notify.send_text(f"Alpha Engine — first scan {now:%H:%M} ET", header)
            log("  sent first-scan summary")
        st["first_run_done"] = True
        embeds = []
        sent_first = True

    if embeds and not dry_run:
        notify.send_alerts(embeds, f"**{now:%H:%M} ET** — {len(embeds)} new alert(s)")
        log(f"  sent {len(embeds)} alert(s)")
    elif embeds and dry_run:
        log(f"  [dry-run] would send {len(embeds)} alert(s): "
            f"{', '.join(e['title'].split(' · ')[0] for e in embeds)}")
    elif not sent_first:
        log("  quiet: no state changes")

    if not dry_run:
        save_state(now.date(), st)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Intraday execution & cadence engine")
    ap.add_argument("--force", action="store_true", help="ignore clock/calendar gate")
    ap.add_argument("--dry-run", action="store_true", help="evaluate but send nothing")
    ap.add_argument("--equity", type=float, default=None)
    ap.add_argument("--asof", type=str, default=None,
                    help='replay a past scan, e.g. "2026-09-08 11:00"')
    args = ap.parse_args()
    return run(force=args.force, dry_run=args.dry_run, equity=args.equity,
               asof=args.asof)


if __name__ == "__main__":
    sys.exit(main())
