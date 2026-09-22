"""
Multi-day swing backtest — continuous positions, gap-aware exits.

WHAT CHANGED FROM THE DAY-TRADING ENGINE, and why each change matters:

  * POSITIONS PERSIST ACROSS SESSIONS. The loop is one continuous timeline per
    symbol rather than a fresh book each morning. No 15:55 flatten.

  * OVERNIGHT GAPS ARE MODELLED EXPLICITLY. This is the single most important
    difference. Intraday, a stop fills at the stop. Held overnight it does not:
    if a stock closes at 100 with a stop at 98 and opens at 95, you are filled
    at 95, not 98. At the first bar of every session the OPEN is checked against
    the stop and the target before anything else, and the fill is the open. A
    swing backtest without this systematically understates losses, because every
    gap-through is quietly repaired to the stop price.

  * UNRESOLVED TRADES ARE NOT COUNTED AS WINS. Holding "until stop or target"
    means trades opened late in a 59-day window may never finish. Those are
    marked to market, reported separately, and excluded from win rate and
    profit factor. Folding them in at a favourable mark is how a swing backtest
    invents a track record.

  * EQUITY IS MARKED TO MARKET DAILY. With positions open for days, a cash-only
    equity curve reports no drawdown at all until something closes, which would
    make max drawdown meaningless.

  * ENTRIES ARE STILL ONLY EVALUATED ON THE 15-MINUTE CADENCE, because that is
    when the live engine looks. Exits are checked on every 5-minute bar, since
    a stop and a target are resting orders that do not wait for a scan.

Reproduces the day-trading configuration with --eod-flat --entry breakout, which
is used as a regression check that the rewrite did not change the old answer.

Usage:
    python3 swing_backtest.py
    python3 swing_backtest.py --target-r 3.0 --stop-method structure
    python3 swing_backtest.py --eod-flat --entry breakout     # legacy check
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import data
import screener
import strategy
from backtest import wilson_ci, max_drawdown, sharpe, sortino
from calendar_util import is_cadence_bar, is_trading_day, flat_time
from config import (
    MAX_POSITION_PCT, BACKTEST_DAYS, BACKTEST_CAPITAL, SCAN_INTERVAL_MIN, WINDOW_START, WINDOW_END,
    MAX_CONCURRENT, COMMISSION_PER_TRADE, DATA, ENTRY_MODE, PULLBACK_SUPPORT,
    PULLBACK_MAX_EXTENSION, PULLBACK_DIP_LOOKBACK, PULLBACK_DIP_TOLERANCE,
    REQUIRE_STRONG_TREND, STOP_METHOD, STOP_PCT, STOP_ATR_MULT, STOP_BUFFER,
    MAX_STOP_PCT, TARGET_R, EOD_FLAT, MAX_HOLD_DAYS,
    RSI_PERIOD, RSI_OVERSOLD, BB_PERIOD, BB_K, OVERSOLD_REQUIRE_ABOVE_SMA200,
    OVERSOLD_STOP_LOOKBACK, SCALE_ENABLED, SCALE_R, SCALE_FRACTION,
    BREAKEVEN_AFTER_SCALE, EARNINGS_BLOCK_DAYS, EARNINGS_EXIT,
    RUNNER_EXIT, TRAIL_ATR_MULT, TRAIL_EMA_SPAN, TRAIL_PCT, RISK_PCT,
)
from execution import State, classify, size_position, apply_slippage


# ---------------------------------------------------------------- records
@dataclass
class Trade:
    symbol: str
    entry_ts: pd.Timestamp
    entry_px: float
    shares: float               # INITIAL size
    stop: float                 # live stop (moves to breakeven after a scale)
    target: float               # final target for the runner
    support: float = 0.0
    orig_stop: float = 0.0      # stop at entry — the R unit is measured off this
    r_unit: float = 0.0         # entry_px - orig_stop, in price units

    # partial profit scaling
    scale_px: float | None = None
    scaled: bool = False
    scale_ts: pd.Timestamp | None = None
    scale_fill: float = 0.0
    scale_shares: float = 0.0
    realized: float = 0.0       # P&L already banked from the partial
    shares_open: float = 0.0    # what is still at risk
    high_water: float = 0.0     # highest high since entry (ATR trail)
    trailing: bool = False      # runner is on a trailing stop

    exit_ts: pd.Timestamp | None = None
    exit_px: float | None = None
    reason: str = ""
    pnl: float = 0.0            # realized partial + final exit
    pnl_pct: float = 0.0        # return on the ORIGINAL capital committed
    hold_days: int = 0
    bars_held: int = 0
    gapped: bool = False        # exit filled through a gap, not at the level

    @property
    def closed(self) -> bool:
        return self.exit_ts is not None

    @property
    def is_win(self) -> bool:
        return self.pnl > 0

    @property
    def r_multiple(self) -> float:
        """P&L in units of initial risk — the scale-invariant way to compare."""
        risk = self.r_unit * self.shares
        return self.pnl / risk if risk > 0 else 0.0


@dataclass
class Result:
    trades: list[Trade] = field(default_factory=list)      # closed only
    open_trades: list[Trade] = field(default_factory=list)  # unresolved at end
    equity: pd.Series = field(default_factory=pd.Series)
    signals_seen: int = 0
    skipped_extended: int = 0
    skipped_concurrency: int = 0
    skipped_size: int = 0
    skipped_earnings: int = 0
    sessions: int = 0
    symbols: list[str] = field(default_factory=list)
    start: dt.date | None = None
    end: dt.date | None = None


# ---------------------------------------------------------------- prep
def _load(sym: str):
    try:
        return sym, data.daily_bars(sym, period="2y"), data.intraday_5m(sym, period="60d")
    except Exception:
        return sym, pd.DataFrame(), pd.DataFrame()


def prepare(symbols: list[str], days: int, support_kind: str, verbose: bool = True):
    """
    Per-symbol continuous RTH series plus per-session daily context.

    VWAP is anchored per session (that is what "the VWAP" means intraday). The
    EMA is computed on the CONTINUOUS series and then sliced, because an EMA
    that resets every morning is not the line anyone is watching.
    """
    if verbose:
        print(f"Loading {len(symbols)} symbols ...", flush=True)

    loaded = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for sym, d, i in ex.map(_load, symbols):
            if d.empty or i.empty or len(d) < 210:
                if verbose:
                    print(f"  skip {sym}: insufficient history", flush=True)
                continue
            loaded[sym] = (d, i)
    if not loaded:
        return {}, []

    all_dates = sorted({ts.date() for _, i in loaded.values() for ts in i.index})
    all_dates = [d for d in all_dates if is_trading_day(d)][-days:]
    keep = set(all_dates)

    prepped: dict[str, dict] = {}
    for sym, (daily, intra) in loaded.items():
        m = intra.index.hour * 60 + intra.index.minute
        rth = intra[(m >= 570) & (m < 960)]
        rth = rth[[d in keep for d in rth.index.date]]
        if rth.empty:
            continue

        ema20 = strategy.ema(rth["Close"].to_numpy(dtype=float), 20)

        vwap = np.empty(len(rth))
        pos = 0
        sessions: dict[dt.date, tuple[int, int]] = {}
        for sdate, grp in rth.groupby(rth.index.date):
            n = len(grp)
            vwap[pos:pos + n] = strategy.session_vwap(grp)
            sessions[sdate] = (pos, pos + n)
            pos += n

        # Premarket highs, computed from the PRE-RTH bars before they are
        # dropped. The pullback strategy does not use them, but breakout mode
        # does, and without them the legacy regression check silently compares
        # two different strategies.
        pm = intra[(m >= 240) & (m < 570)]
        pmh_by_date: dict[dt.date, float] = {}
        if not pm.empty:
            for sdate, grp in pm.groupby(pm.index.date):
                pmh_by_date[sdate] = float(grp["High"].max())

        # Hourly RSI and lower Bollinger band, stamped onto the 5-minute index
        # using only hourly bars that have actually CLOSED (see strategy.py).
        rsi_h, bb_low_h, _hc = strategy.hourly_signals_at_5m(
            rth, RSI_PERIOD, BB_PERIOD, BB_K)
        ema_h = strategy.hourly_ema_at_5m(rth, TRAIL_EMA_SPAN)

        # Earnings dates are announced weeks ahead, so consulting the calendar
        # inside a backtest is NOT look-ahead — a real trader knows them too.
        try:
            earn = set(data.earnings_dates(sym))
        except Exception:
            earn = set()

        ctx = {}
        for sdate in sessions:
            c = strategy.daily_context_full(daily, sdate)
            if c.ok():
                ctx[sdate] = c
        if not ctx:
            continue

        prepped[sym] = {"rth": rth, "vwap": vwap, "ema20": ema20,
                        "sessions": sessions, "ctx": ctx, "pmh": pmh_by_date,
                        "rsi": rsi_h, "bb_low": bb_low_h, "earnings": earn,
                        "ema_h": ema_h}

    if verbose:
        tot = sum(len(p["ctx"]) for p in prepped.values())
        print(f"  {len(prepped)} symbols · {len(all_dates)} sessions · "
              f"{tot} symbol-sessions with complete context", flush=True)
    return prepped, all_dates


# ---------------------------------------------------------------- stops
def compute_stop(method: str, entry: float, lv: strategy.Levels,
                 stop_pct: float, atr_mult: float) -> float | None:
    if method == "structure":
        s = strategy.structural_stop(lv, entry, STOP_BUFFER, MAX_STOP_PCT)
        if s is None:                      # no dip low (breakout mode) — fall back
            s = entry * (1.0 - stop_pct)
        return s
    if method == "atr":
        if lv.atr is None or lv.atr <= 0:
            return entry * (1.0 - stop_pct)
        s = entry - atr_mult * lv.atr
        floor = entry * (1.0 - MAX_STOP_PCT)
        return max(s, floor) if s < entry else None
    return entry * (1.0 - stop_pct)


# ---------------------------------------------------------------- engine
def run(symbols: list[str], days: int = BACKTEST_DAYS,
        capital: float = BACKTEST_CAPITAL, scores: dict[str, float] | None = None,
        entry_mode: str = ENTRY_MODE, support_kind: str = PULLBACK_SUPPORT,
        stop_method: str = STOP_METHOD, stop_pct: float = STOP_PCT,
        atr_mult: float = STOP_ATR_MULT, target_r: float = TARGET_R,
        max_concurrent: int = MAX_CONCURRENT, eod_flat: bool = EOD_FLAT,
        max_hold_days: int | None = MAX_HOLD_DAYS, require_range: bool = True,
        random_p: float = 0.05, seed: int = 0,
        scale_enabled: bool = SCALE_ENABLED, scale_r: float = SCALE_R,
        scale_fraction: float = SCALE_FRACTION,
        breakeven_after_scale: bool = BREAKEVEN_AFTER_SCALE,
        earnings_block_days: int = EARNINGS_BLOCK_DAYS,
        earnings_exit: bool = EARNINGS_EXIT,
        rsi_threshold: float = RSI_OVERSOLD,
        runner_exit: str = RUNNER_EXIT,
        trail_atr_mult: float = TRAIL_ATR_MULT, trail_pct: float = TRAIL_PCT,
        risk_pct: float = RISK_PCT, max_position_pct: float | None = None,
        prepped=None, dates=None, verbose: bool = True) -> Result:

    if prepped is None:
        prepped, dates = prepare(symbols, days, support_kind, verbose)
    res = Result(symbols=sorted(prepped.keys()), sessions=len(dates or []))
    if not prepped:
        return res

    scores = scores or {}
    _rng = np.random.default_rng(seed)
    res.start, res.end = dates[0], dates[-1]
    cash = capital
    open_pos: dict[str, Trade] = {}
    equity_points: list[tuple[dt.date, float]] = []
    last_px: dict[str, float] = {}

    win_start = WINDOW_START[0] * 60 + WINDOW_START[1]
    win_end = WINDOW_END[0] * 60 + WINDOW_END[1]

    def close_trade(t: Trade, ts, px, reason, gapped=False, bars=0):
        """Close whatever is still open and fold in any P&L already banked."""
        nonlocal cash
        fill = apply_slippage(px, "sell")
        qty = t.shares_open
        cash += fill * qty - COMMISSION_PER_TRADE
        t.exit_ts, t.exit_px, t.reason, t.gapped = ts, fill, reason, gapped
        t.pnl = t.realized + (fill - t.entry_px) * qty
        # Return on the ORIGINAL capital committed, so a scaled trade and an
        # unscaled one are directly comparable.
        t.pnl_pct = t.pnl / (t.entry_px * t.shares) if t.shares else 0.0
        t.hold_days = max(0, (ts.date() - t.entry_ts.date()).days)
        t.bars_held = bars
        t.shares_open = 0.0
        res.trades.append(t)

    def _earnings_imminent(sym, sdate):
        """True if a report lands today or tomorrow for this name."""
        e = prepped.get(sym, {}).get("earnings") or ()
        return any(0 <= (d - sdate).days <= 1 for d in e)

    def _earnings_blocks_entry(sym, sdate):
        """True if a report lands within the block window after entry."""
        if earnings_block_days <= 0:
            return False
        e = prepped.get(sym, {}).get("earnings") or ()
        return any(0 <= (d - sdate).days <= earnings_block_days for d in e)

    def do_scale(t: Trade, ts, px):
        """
        Sell `scale_fraction` and move the stop on the remainder to breakeven.

        The runner's target stays measured off the ORIGINAL risk unit, so
        "2.5R" keeps meaning 2.5x the risk actually taken at entry rather than
        2.5x some new, smaller distance.
        """
        nonlocal cash
        fill = apply_slippage(px, "sell")
        qty = t.shares * scale_fraction
        cash += fill * qty - COMMISSION_PER_TRADE
        t.realized += (fill - t.entry_px) * qty
        t.scaled, t.scale_ts, t.scale_fill, t.scale_shares = True, ts, fill, qty
        t.shares_open = t.shares - qty
        if breakeven_after_scale:
            t.stop = t.entry_px
        if runner_exit in ("ema20h", "atr", "pct"):
            # Uncapped upside: drop the fixed target and let the trail decide.
            t.trailing = True
            t.target = float("inf")

    for sdate in dates:
        todays = {s: p for s, p in prepped.items()
                  if sdate in p["sessions"] and sdate in p["ctx"]}
        if not todays:
            if equity_points:
                equity_points.append((sdate, equity_points[-1][1]))
            continue

        frames, idx_of = {}, {}
        for sym, p in todays.items():
            a, b = p["sessions"][sdate]
            frames[sym] = (p["rth"].iloc[a:b], p["vwap"][a:b], p["ema20"][a:b])
            p.setdefault("gidx", {})[sdate] = (a, b)
            idx_of[sym] = {ts: k for k, ts in enumerate(frames[sym][0].index)}

        timestamps = sorted({ts for f, _, _ in frames.values() for ts in f.index})
        first_ts = timestamps[0] if timestamps else None

        # per-session pullback state
        was_above = {s: False for s in frames}
        dip_low: dict[str, float | None] = {s: None for s in frames}
        dip_bar: dict[str, int] = {s: -999 for s in frames}
        entered_today: set[str] = set()
        hod: dict[str, float] = {}
        bar_count: dict[str, int] = {s: 0 for s in open_pos}

        flat_at = flat_time(sdate)
        flat_min = flat_at.hour * 60 + flat_at.minute

        for ts in timestamps:
            tmin = ts.hour * 60 + ts.minute
            live = {}
            for sym, (f, vw, em) in frames.items():
                k = idx_of[sym].get(ts)
                if k is None:
                    continue
                b = f.iloc[k]
                live[sym] = (k, float(b["Open"]), float(b["High"]),
                             float(b["Low"]), float(b["Close"]), vw[k], em[k])
                last_px[sym] = live[sym][4]

            # ---------- phase 1: exits
            for sym, (k, o, h, l, c, vw, em) in live.items():
                t = open_pos.get(sym)
                if not t:
                    continue
                bar_count[sym] = bar_count.get(sym, 0) + 1

                # Advance the trail using the PREVIOUS bar's high-water mark and
                # the last completed hourly EMA. Updating the high-water first
                # would let this bar's own high pull the stop up and then stop
                # out on the same bar's low — a level that never existed.
                if t.trailing:
                    gi2 = todays[sym]["gidx"][sdate][0] + k
                    ev = todays[sym]["ema_h"][gi2]
                    t.stop = strategy.trail_stop(
                        runner_exit, t.stop, t.high_water, ev,
                        todays[sym]["ctx"][sdate].atr, trail_atr_mult,
                        t.entry_px, trail_pct)

                # OVERNIGHT GAP. On the first bar of a session the open is the
                # first tradable print; if it is already through the stop or the
                # target, that open IS the fill. Checking the intrabar range
                # first would silently repair the gap to the level.
                if ts == first_ts and t.entry_ts.date() < sdate:
                    if o <= t.stop:
                        close_trade(t, ts, o, "GAP_BE_STOP" if t.scaled
                                    else "GAP_STOP", True, bar_count[sym])
                        del open_pos[sym]
                        continue
                    if np.isfinite(t.target) and o >= t.target:
                        # Gapping above the final target fills BOTH resting
                        # sells at the open, so the whole position exits there —
                        # scaling first would be fiction, no trade happened in
                        # between.
                        close_trade(t, ts, o, "GAP_TARGET", True, bar_count[sym])
                        del open_pos[sym]
                        continue
                    if (t.scale_px is not None and not t.scaled
                            and o >= t.scale_px):
                        do_scale(t, ts, o)

                if l <= t.stop:
                    # Pessimistic: a bar touching both stop and target is
                    # assumed to have hit the stop first. 5-minute OHLC cannot
                    # order them.
                    _r = ("TRAIL_STOP" if (t.scaled and t.trailing
                                           and t.stop > t.entry_px)
                          else "BE_STOP" if t.scaled else "STOP")
                    close_trade(t, ts, t.stop, _r, False, bar_count[sym])
                    del open_pos[sym]
                    continue

                # Scale before the target check: a bar that runs through both
                # the scale level and the target fills both resting orders.
                if t.scale_px is not None and not t.scaled and h >= t.scale_px:
                    do_scale(t, ts, t.scale_px)
                    # The stop has just moved up to breakeven. If this same bar
                    # also traded down to it, we cannot know the order, so the
                    # runner is assumed stopped — conservative, and it still
                    # leaves the trade net positive on the banked half.
                    if breakeven_after_scale and l <= t.stop:
                        close_trade(t, ts, t.stop, "BE_STOP", False, bar_count[sym])
                        del open_pos[sym]
                        continue

                if np.isfinite(t.target) and h >= t.target:
                    close_trade(t, ts, t.target, "TARGET", False, bar_count[sym])
                    del open_pos[sym]
                    continue

                t.high_water = max(t.high_water, h)

                # Never hold through an earnings print. It is the fattest gap in
                # the book and the date is known weeks ahead.
                if earnings_exit and _earnings_imminent(sym, sdate) and \
                        ts == timestamps[-1]:
                    close_trade(t, ts, c, "PRE_EARNINGS", False, bar_count[sym])
                    del open_pos[sym]
                    continue
                if max_hold_days is not None and \
                        (sdate - t.entry_ts.date()).days >= max_hold_days:
                    close_trade(t, ts, c, "TIME_STOP", False, bar_count[sym])
                    del open_pos[sym]
                    continue
                if eod_flat and tmin >= flat_min:
                    close_trade(t, ts, c, "EOD_FLAT", False, bar_count[sym])
                    del open_pos[sym]
                    continue

            # ---------- phase 2: entries, on the scan cadence only
            if win_start <= tmin <= win_end and is_cadence_bar(ts, SCAN_INTERVAL_MIN):
                candidates = []
                for sym, (k, o, h, l, c, vw, em) in live.items():
                    if sym in open_pos or sym in entered_today:
                        continue
                    ctx = todays[sym]["ctx"][sdate]

                    if entry_mode == "oversold":
                        gi = todays[sym]["gidx"][sdate][0] + k
                        rv = todays[sym]["rsi"][gi]
                        bl = todays[sym]["bb_low"][gi]
                        # Structural stop for a dip entry: under the low of the
                        # last hour. There is no "defended level" here the way a
                        # pullback has one — price is still falling — so the
                        # recent swing low is the nearest thing to structure.
                        lo0 = max(0, k - OVERSOLD_STOP_LOOKBACK)
                        recent_low = float(frames[sym][0]["Low"]
                                           .iloc[lo0:k + 1].min())
                        lv = strategy.Levels(
                            symbol=sym, prev_high=ctx.prev_high,
                            prev_close=ctx.prev_close, sma200=ctx.sma200,
                            sma50=ctx.sma50, atr=ctx.atr,
                            support=float(bl) if bl == bl else float(recent_low),
                            support_name="oversold", dip_low=recent_low,
                            was_above=True, dipped=True)
                        sig = strategy.evaluate_oversold(
                            lv, c, l, rv, bl, rsi_threshold,
                            OVERSOLD_REQUIRE_ABOVE_SMA200)
                    elif entry_mode == "random":
                        # CONTROL ARM. Same trend filter, same exits, same
                        # sizing and concurrency — only the pullback TIMING is
                        # replaced by a coin flip. If the real strategy cannot
                        # beat this, the pullback trigger is decoration and the
                        # result belongs to the trend filter and the exits.
                        lv = strategy.Levels(
                            symbol=sym, prev_high=ctx.prev_high,
                            prev_close=ctx.prev_close, sma200=ctx.sma200,
                            sma50=ctx.sma50, atr=ctx.atr, support=float(vw))
                        ok = (lv.strong_trend_ok if REQUIRE_STRONG_TREND
                              else lv.trend_ok)
                        sig = strategy.Signal(
                            bool(ok and _rng.random() < random_p), "random", lv)
                    elif entry_mode == "pullback":
                        sup = {"vwap": vw, "ema20": em,
                               "sma20d": ctx.sma20}.get(support_kind, vw)
                        if sup is None or not np.isfinite(sup):
                            continue
                        lv = strategy.Levels(
                            symbol=sym, prev_high=ctx.prev_high,
                            prev_close=ctx.prev_close, sma200=ctx.sma200,
                            sma50=ctx.sma50, atr=ctx.atr, support=float(sup),
                            support_name=support_kind, was_above=was_above[sym],
                            dip_low=dip_low[sym],
                            dipped=(k - dip_bar[sym]) <= PULLBACK_DIP_LOOKBACK)
                        sig = strategy.evaluate_pullback(
                            lv, c, PULLBACK_MAX_EXTENSION, REQUIRE_STRONG_TREND)
                    else:
                        hod_prev = hod.get(sym)
                        pmh = todays[sym]["pmh"].get(sdate)
                        if hod_prev is None or pmh is None:
                            continue
                        lv = strategy.Levels(
                            symbol=sym, prev_high=ctx.prev_high,
                            prev_close=ctx.prev_close, sma200=ctx.sma200,
                            sma50=ctx.sma50, atr=ctx.atr,
                            pmh=pmh, hod_prev=hod_prev)
                        sig = strategy.evaluate_signal(lv, c)

                    if not sig.fired:
                        continue
                    res.signals_seen += 1

                    # Execution-range gate. In pullback mode this is already
                    # enforced inside evaluate_pullback (the extension cap above
                    # support IS the range), so applying it twice would be
                    # redundant. In breakout mode it is a separate check, and
                    # omitting it is what made the legacy regression disagree.
                    if entry_mode == "breakout" and require_range:
                        state, _ext = classify(c, lv.trigger)
                        if state is not State.EXECUTE:
                            res.skipped_extended += 1
                            continue

                    candidates.append((sym, c, lv))

                candidates.sort(key=lambda x: -scores.get(x[0], 0.0))

                for sym, c, lv in candidates:
                    if len(open_pos) >= max_concurrent:
                        res.skipped_concurrency += 1
                        continue
                    if _earnings_blocks_entry(sym, sdate):
                        res.skipped_earnings += 1
                        continue
                    fill = apply_slippage(c, "buy")
                    stop = compute_stop(stop_method, fill, lv, stop_pct, atr_mult)
                    if stop is None or stop >= fill:
                        res.skipped_size += 1
                        continue
                    eq = cash + sum(p.shares_open * last_px.get(s, p.entry_px)
                                    for s, p in open_pos.items())
                    shares, _n = size_position(
                        eq, fill, stop, risk_pct=risk_pct,
                        max_position_pct=(max_position_pct
                                          if max_position_pct is not None
                                          else min(MAX_POSITION_PCT,
                                                   1.0 / max_concurrent)))
                    cost = shares * fill + COMMISSION_PER_TRADE
                    if shares <= 0 or cost > cash:
                        res.skipped_size += 1
                        continue
                    cash -= cost
                    r_unit = fill - stop
                    t = Trade(
                        symbol=sym, entry_ts=ts, entry_px=fill, shares=shares,
                        stop=stop, target=fill + r_unit * target_r,
                        support=float(lv.support or 0.0),
                        orig_stop=stop, r_unit=r_unit, shares_open=shares,
                        high_water=fill)
                    if scale_enabled and scale_fraction > 0:
                        t.scale_px = fill + r_unit * scale_r
                    open_pos[sym] = t
                    entered_today.add(sym)
                    bar_count[sym] = 0

            # ---------- phase 3: state updates, deliberately last so nothing
            # ---------- above can be satisfied by the current bar's own extreme
            for sym, (k, o, h, l, c, vw, em) in live.items():
                hod[sym] = max(hod.get(sym, h), h)
                sup = {"vwap": vw, "ema20": em,
                       "sma20d": todays[sym]["ctx"][sdate].sma20}.get(support_kind, vw)
                if sup is None or not np.isfinite(sup):
                    continue
                if c > sup:
                    was_above[sym] = True
                if l <= sup * (1.0 + PULLBACK_DIP_TOLERANCE):
                    dip_bar[sym] = k
                    dip_low[sym] = l if dip_low[sym] is None else min(dip_low[sym], l)

        # mark to market at the session close
        mtm = cash + sum(p.shares_open * last_px.get(s, p.entry_px)
                         for s, p in open_pos.items())
        equity_points.append((sdate, mtm))

    # Unresolved at the end of the window: marked, reported, NOT counted as wins.
    for sym, t in open_pos.items():
        px = last_px.get(sym, t.entry_px)
        t.exit_px = px
        t.pnl = t.realized + (px - t.entry_px) * t.shares_open
        t.pnl_pct = t.pnl / (t.entry_px * t.shares) if t.shares else 0.0
        t.reason = "OPEN_AT_END"
        t.hold_days = max(0, (dates[-1] - t.entry_ts.date()).days)
        res.open_trades.append(t)

    res.equity = pd.Series(dict(equity_points)).sort_index()
    return res


# ---------------------------------------------------------------- reporting
def summarise(res: Result, capital: float) -> dict:
    t = res.trades
    n = len(t)
    wins = [x for x in t if x.is_win]
    gw = sum(x.pnl for x in wins)
    gl = -sum(x.pnl for x in t if not x.is_win)
    net = sum(x.pnl for x in t)
    open_pnl = sum(x.pnl for x in res.open_trades)
    dd_d, dd_p = max_drawdown(res.equity) if not res.equity.empty else (0.0, 0.0)
    lo, hi = wilson_ci(len(wins), n)
    return {
        "trades": n, "wins": len(wins),
        "win_rate": len(wins) / n if n else 0.0, "ci_lo": lo, "ci_hi": hi,
        "net": net, "net_pct": net / capital,
        "open_n": len(res.open_trades), "open_pnl": open_pnl,
        "pf": (gw / gl) if gl > 0 else (float("inf") if gw > 0 else 0.0),
        "dd_d": dd_d, "dd_p": dd_p,
        "sharpe": sharpe(res.equity), "sortino": sortino(res.equity),
        "avg_hold": float(np.mean([x.hold_days for x in t])) if n else 0.0,
        "gapped": sum(1 for x in t if x.gapped),
        "signals": res.signals_seen, "skip_conc": res.skipped_concurrency,
        "skip_size": res.skipped_size, "skip_earn": res.skipped_earnings,
        "scaled_n": sum(1 for x in t if x.scaled),
        "expectancy_r": float(np.mean([x.r_multiple for x in t])) if n else 0.0,
        "avg_win": (gw / len(wins)) if wins else 0.0,
        "avg_loss": (gl / max(n - len(wins), 1)) if n > len(wins) else 0.0,
    }


def report(res: Result, capital: float, cfg: dict) -> str:
    s = summarise(res, capital)
    L, A = [], None
    A = L.append
    A("=" * 78)
    A("  59-DAY SWING BACKTEST — multi-day holds, gap-aware exits")
    A("=" * 78)
    A(f"  Window          {res.start} → {res.end}  ({res.sessions} sessions)")
    A(f"  Universe        {len(res.symbols)} symbols")
    A(f"  Entry           {cfg['entry']}" +
      (f" to {cfg['support']}" if cfg['entry'] == 'pullback' else ""))
    A(f"  Stop            {cfg['stop_desc']}")
    A(f"  Target          {cfg['target_r']:.1f}R")
    A(f"  Hold            until stop or target — NO end-of-day flatten"
      if not cfg["eod_flat"] else "  Hold            intraday, flat 15:55")
    A(f"  Max concurrent  {cfg['max_concurrent']}")
    A("")
    A("  " + "-" * 74)
    A(f"  {'METRIC':<28} {'VALUE':>20}   {'NOTE':<20}")
    A("  " + "-" * 74)
    A(f"  {'Closed trades':<28} {s['trades']:>20}")
    A(f"  {'Win rate':<28} {s['win_rate']*100:>19.1f}%   "
      f"95% CI {s['ci_lo']*100:.0f}–{s['ci_hi']*100:.0f}%")
    A(f"  {'Net P&L ($)':<28} {s['net']:>+20,.2f}")
    A(f"  {'Net P&L (%)':<28} {s['net_pct']*100:>+19.2f}%   on starting capital")
    A(f"  {'Max drawdown ($)':<28} {s['dd_d']:>20,.2f}   marked to market")
    A(f"  {'Max drawdown (%)':<28} {s['dd_p']*100:>19.2f}%")
    A(f"  {'Sharpe ratio':<28} {s['sharpe']:>20.3f}   annualised, daily")
    A(f"  {'Sortino ratio':<28} {s['sortino']:>20.3f}")
    A(f"  {'Profit factor':<28} {s['pf']:>20.3f}")
    A(f"  {'Avg hold (days)':<28} {s['avg_hold']:>20.1f}")
    A("  " + "-" * 74)
    A(f"  {'Expectancy per trade (R)':<28} {s['expectancy_r']:>20.3f}   P&L / risk taken")
    A(f"  {'Partial scales taken':<28} {s['scaled_n']:>20}   "
      f"{s['scaled_n']/max(s['trades'],1)*100:.0f}% of closed trades")
    A(f"  {'Blocked — earnings':<28} {s['skip_earn']:>20}   entries skipped")
    A("  " + "-" * 74)
    A(f"  {'Unresolved at end':<28} {s['open_n']:>20}   "
      f"MTM {s['open_pnl']:+,.2f}, excluded")
    A(f"  {'Exits filled via gap':<28} {s['gapped']:>20}   "
      f"{s['gapped']/max(s['trades'],1)*100:.0f}% of closed trades")
    A("  " + "-" * 74)

    for bench in ("SPY", "QQQ"):
        try:
            b = data.daily_bars(bench, period="6mo")
            w = b[(b.index.date >= res.start) & (b.index.date <= res.end)]
            if len(w) > 1:
                r = float(w["Close"].iloc[-1] / w["Close"].iloc[0] - 1.0)
                A(f"  {bench + ' buy & hold':<28} {r*100:>19.2f}%   same window")
        except Exception:
            pass
    A("  " + "-" * 74)

    A("")
    A("  FUNNEL")
    A(f"    signals fired          {s['signals']}")
    A(f"    skipped — book full    {s['skip_conc']}")
    A(f"    skipped — earnings     {s['skip_earn']}")
    A(f"    skipped — sizing/cash  {s['skip_size']}")
    A(f"    trades taken           {s['trades'] + s['open_n']}")
    _tot = (s['skip_conc'] + s['skip_earn'] + s['skip_size']
            + s['trades'] + s['open_n'])
    A(f"    reconciles" if _tot == s['signals']
      else f"    WARNING: funnel does not reconcile ({_tot} vs {s['signals']})")

    if res.trades:
        by: dict[str, list] = {}
        for x in res.trades:
            by.setdefault(x.reason, []).append(x)
        A("")
        A("  EXITS")
        for r, xs in sorted(by.items(), key=lambda kv: -len(kv[1])):
            A(f"    {r:<13} {len(xs):>3} ({len(xs)/len(res.trades)*100:>4.0f}%)  "
              f"net {sum(x.pnl for x in xs):>+10,.2f}")
        A("")
        A("  GAP RISK — the cost of holding overnight")
        gs = [x for x in res.trades if x.reason == "GAP_STOP"]
        if gs:
            slip = np.mean([(x.stop - x.exit_px) / x.stop for x in gs])
            A(f"    stops gapped through   {len(gs)}")
            A(f"    average overshoot      {slip*100:.2f}% worse than the stop")
        else:
            A("    no stop was gapped through in this window")
    return "\n".join(L)


# ---------------------------------------------------------------- cli
def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-day swing backtest")
    ap.add_argument("--symbols", type=str, default=None)
    ap.add_argument("--days", type=int, default=BACKTEST_DAYS)
    ap.add_argument("--capital", type=float, default=BACKTEST_CAPITAL)
    ap.add_argument("--entry", type=str, default=ENTRY_MODE,
                    choices=["pullback", "breakout", "oversold"])
    ap.add_argument("--support", type=str, default=PULLBACK_SUPPORT,
                    choices=["vwap", "ema20", "sma20d"])
    ap.add_argument("--stop-method", type=str, default=STOP_METHOD,
                    choices=["structure", "pct", "atr"])
    ap.add_argument("--stop-pct", type=float, default=STOP_PCT)
    ap.add_argument("--atr-mult", type=float, default=STOP_ATR_MULT)
    ap.add_argument("--target-r", type=float, default=TARGET_R)
    ap.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    ap.add_argument("--eod-flat", action="store_true")
    ap.add_argument("--max-hold-days", type=int, default=None)
    ap.add_argument("--scale-r", type=float, default=None,
                    help="sell 50%% at this R multiple, then stop to breakeven")
    ap.add_argument("--no-scale", action="store_true")
    ap.add_argument("--scale-fraction", type=float, default=None)
    ap.add_argument("--no-earnings", action="store_true")
    args = ap.parse_args()

    scores = {}
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        wl = screener.load_watchlist()
        if not wl:
            print("No watchlist. Run: python3 screener.py")
            return 1
        symbols = [s["symbol"] for s in wl["watchlist"]]
        scores = {s["symbol"]: s.get("score", 0.0) for s in wl["watchlist"]}

    res = run(symbols, days=args.days, capital=args.capital, scores=scores,
              entry_mode=args.entry, support_kind=args.support,
              stop_method=args.stop_method, stop_pct=args.stop_pct,
              atr_mult=args.atr_mult, target_r=args.target_r,
              max_concurrent=args.max_concurrent, eod_flat=args.eod_flat,
              max_hold_days=args.max_hold_days,
              scale_enabled=(not args.no_scale) and args.scale_r is not None,
              scale_r=args.scale_r or 1.0,
              scale_fraction=(args.scale_fraction
                              if args.scale_fraction is not None else 0.5),
              earnings_block_days=0 if args.no_earnings else 5,
              earnings_exit=not args.no_earnings)

    desc = {"structure": f"structural (under dip low, cap {MAX_STOP_PCT*100:.0f}%)",
            "pct": f"{args.stop_pct*100:.1f}% fixed",
            "atr": f"{args.atr_mult:.1f} x ATR(14)"}[args.stop_method]
    cfg = {"entry": args.entry, "support": args.support, "stop_desc": desc,
           "target_r": args.target_r, "eod_flat": args.eod_flat,
           "max_concurrent": args.max_concurrent}

    out = report(res, args.capital, cfg)
    print("\n" + out)
    (DATA / "swing_backtest.txt").write_text(out)
    if res.trades:
        pd.DataFrame([{
            "symbol": t.symbol, "entry_ts": t.entry_ts.isoformat(),
            "entry_px": round(t.entry_px, 4), "support": round(t.support, 4),
            "exit_ts": t.exit_ts.isoformat() if t.exit_ts else None,
            "exit_px": round(t.exit_px, 4) if t.exit_px else None,
            "shares": t.shares, "stop": round(t.stop, 4),
            "target": round(t.target, 4), "reason": t.reason,
            "gapped": t.gapped, "pnl": round(t.pnl, 2),
            "pnl_pct": round(t.pnl_pct * 100, 3), "hold_days": t.hold_days,
        } for t in res.trades]).to_csv(DATA / "swing_trades.csv", index=False)
    print(f"\nwrote {DATA}/swing_backtest.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
