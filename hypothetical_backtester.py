"""
hypothetical_backtester.py — standalone, isolated backtest of three proposed
risk-management overlays on top of the Fixed ADX Hybrid's entry/exit rules.

DOES NOT IMPORT OR MODIFY hybrid_engine.py. That file has no historical
event loop at all — it is a live, single-point-in-time screening/execution
tool (--screen/--dry-run/--execute/--manage), not something that can "run
over 2022-present." So "cloning its core entry and exit logic" means
reproducing the SAME DIP/BREAKOUT signal rules and exit mechanics it (and
backtest.py's already-validated run_adx_hybrid_fixed_backtest()) implement,
not literally importing that file. This script reuses backtest.py's
daily_hybrid_fixed_frame() for indicator/signal computation — the exact
validated RSI/Bollinger/ADX/ADX-percentile/ATR/volume-gate math, not a
third hand-typed copy of it — and data.py for the shared data layer, but
implements its OWN portfolio event loop (closely modeled on, but separate
from, run_adx_hybrid_fixed_backtest()) so the two new entry-blocking
overlays below can be added without growing that shared function's
parameter list for a one-off experiment. backtest.py itself is untouched.

TWO METHODOLOGY CAVEATS — READ BEFORE TRUSTING THE CIRCUIT-BREAKER OR
MACRO-BLACKOUT NUMBERS SPECIFICALLY. Both are forced by data availability,
not a shortcut taken for convenience:

1. CIRCUIT BREAKER IS A DAILY-BAR APPROXIMATION, NOT THE LITERAL 10-MINUTE
   RULE. Yahoo Finance (data.py's only source) caps 5-minute/intraday
   history at 60 calendar days (see data.intraday_5m's own docstring) — no
   argument gets minute-level SPY data back to 2022. "A drop of >0.75%
   within a rolling 10-minute window" cannot be evaluated over a multi-year
   backtest with the data actually available here. This implements the
   closest daily-bar analogue instead: a session where SPY's own
   (Open -> Low) decline exceeds CIRCUIT_BREAKER_DROP_PCT blocks NEW entries
   from being queued out of THAT session — the finest "pause new entries for
   a while" unit expressible when a day is the smallest tradable unit. It is
   a disclosed proxy for the intent (pause after a fast, sharp SPY selloff),
   not the specified sub-day mechanism, and it will trip on fewer, coarser
   occasions than a true 10-minute/60-minute rule would.

2. MACRO BLACKOUT is similarly coarsened to whole trading days: there is no
   intraday timestamp in a daily-bar backtest to represent "1:45 PM-3:30 PM
   ET," so this blocks new entries for the ENTIRE calendar date of each
   known FOMC announcement day instead. The FOMC_DATES list below is
   compiled from public Federal Reserve meeting calendars (memory, not a
   live feed) for 2022 through today — verify against federalreserve.gov
   before relying on this for anything beyond this exploratory comparison.

In both cases, EXISTING open positions are exempt — they keep managing off
their own stop/scale/target/trailing-stop regardless of a blackout or
breaker being active that day, exactly as requested; only the "queue a new
entry" step is skipped.

THE COMPARISON: "Original Baseline" reuses backtest.run_adx_hybrid_fixed_
backtest() completely unchanged (asymmetric 0.625%/0.25% risk sizing, no
breaker, no blackout — the same Strategy B validated in every prior
comparison this session). "New Risk-Managed Strategy" is this file's engine:
flat 1% risk sizing (a real, disclosed formula change, not just the two new
gates) PLUS the breaker PLUS the blackout. Three things differ at once by
design (per the request) — this is a bundled-feature comparison, not a
single-variable A/B test.

Usage:
    python3 hypothetical_backtester.py
"""

from __future__ import annotations

import datetime as dt
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import backtest as bt
import data
import strategy
from compare_strategies import compute_metrics
from config import (
    BB_K, BB_PERIOD, COMMISSION_PER_TRADE, RSI_OVERSOLD, RSI_PERIOD,
    SCALE_ENABLED, SCALE_FRACTION, SCALE_R, STOP_PCT, TARGET_R,
)
from execution import apply_slippage

START = dt.date(2022, 1, 1)
END = dt.date.today()
STARTING_CAPITAL = 1_500.0          # small-account stress test — see the
                                     # capital diagnostics printed by main_daily()
UNIVERSE_NAME = "Mixed"
MAX_CONCURRENT = 4                  # matches the validated-best config from the last sweep

# ---- Feature 1: dynamic risk sizing ----------------------------------------
FLAT_RISK_PCT = 0.01                # flat 1% of equity, BOTH DIP and BREAKOUT
                                     # (a real formula change from the Original
                                     # Baseline's asymmetric 0.625%/0.25% split)
                                     # -- still used by the 15-min intraday engine
                                     # (run_intraday_backtest); NOT used by
                                     # run_risk_managed_backtest's default path
                                     # anymore, see RISK_PCT_MR/RISK_PCT_MOM below.

RISK_PCT_MR = 0.00625                # DIP/Pullback entries: 0.625% of equity —
                                     # matches the Original Baseline's own
                                     # risk_pct_mr exactly (see main_daily()).
RISK_PCT_MOM = 0.0025                # BREAKOUT entries: 0.25% of equity —
                                     # matches the Original Baseline's own
                                     # risk_pct_mom exactly.

# ---- Feature 2: circuit breaker (daily-bar proxy — see module docstring) ---
CIRCUIT_BREAKER_DROP_PCT = 0.0075   # SPY intraday (Open -> Low) decline

# ---- Feature 3: macro blackout (daily-bar proxy — see module docstring) ---
# FOMC announcement dates, 2022-2027. Synced to match the live hybrid_engine.py
# copy exactly: 2022-2026 Q3 compiled from memory (unverified against the
# source), 2026 Q4 and all of 2027 fetched live from
# federalreserve.gov/monetarypolicy/fomccalendars.htm on 2026-09-19. Every
# date is the SECOND day of each two-day meeting. Fed's own site states 2027
# dates are "tentative until confirmed at the meeting immediately preceding
# it" — re-verify before relying on this beyond exploratory use.
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


# ---------------------------------------------------------------- data loading
def _load_frames(symbols: list[str], start: dt.date, end: dt.date) -> dict[str, pd.DataFrame]:
    """Same fetch/seed/trim/indicator pattern as backtest.py's own hybrid
    loaders, calling the SAME daily_hybrid_fixed_frame() — not a rewritten
    copy of its RSI/ADX/ATR/volume-gate math."""
    def _load(sym: str):
        try:
            return sym, data.daily_bars(sym, period="max")
        except Exception:
            return sym, pd.DataFrame()

    seed_start = start - dt.timedelta(days=400)
    frames: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for sym, d in ex.map(_load, symbols):
            if d.empty:
                continue
            d = d[(d.index.date >= seed_start) & (d.index.date <= end)]
            if len(d) < 210:
                continue
            frames[sym] = bt.daily_hybrid_fixed_frame(d)
    return frames


def compute_circuit_breaker_days(start: dt.date, end: dt.date) -> set[dt.date]:
    """See Feature 2's module-level caveat: a daily-bar proxy, not the
    literal 10-minute rule."""
    spy = data.daily_bars("SPY", period="max")
    w = spy[(spy.index.date >= start) & (spy.index.date <= end)]
    if w.empty:
        return set()
    drop = (w["Open"] - w["Low"]) / w["Open"]
    return {ts.date() for ts in w.index[drop > CIRCUIT_BREAKER_DROP_PCT]}


@dataclass
class BlockStats:
    breaker_days: int = 0
    macro_days: int = 0
    signals_blocked: int = 0
    circuit_breaker_enabled: bool = True
    # Capital diagnostics (see print_capital_diagnostics()): zero_share_signals
    # counts a candidate whose sizing formula floored to 0 shares -- couldn't
    # afford even 1 whole share at the target risk_pct, distinct from a
    # signal skipped for lack of CASH after other positions consumed it.
    # realized_risk_pct_mr/mom record, per FILLED trade, (shares x r_unit) /
    # equity_at_entry -- the ACTUAL risk taken, vs. the target RISK_PCT_MR/MOM
    # -- so whole-share rounding's impact on the target can be measured
    # directly instead of assumed.
    zero_share_signals: int = 0
    realized_risk_pct_mr: list[float] = field(default_factory=list)
    realized_risk_pct_mom: list[float] = field(default_factory=list)


# ---------------------------------------------------------------- the new engine
def run_risk_managed_backtest(symbols: list[str], start: dt.date, end: dt.date,
                              capital: float = STARTING_CAPITAL,
                              use_circuit_breaker: bool = True,
                              risk_pct_mr: float = RISK_PCT_MR,
                              risk_pct_mom: float = RISK_PCT_MOM,
                              fractional_shares: bool = False,
                              ) -> tuple["bt.HybridResult", BlockStats]:
    """
    Portfolio loop identical in shape/exit-mechanics to
    backtest.run_adx_hybrid_fixed_backtest(mode="HYBRID") — copied from
    there deliberately (not re-derived) to avoid subtly drifting from the
    validated exit logic — with one remaining change from the Original
    Baseline: a combined circuit-breaker/macro-blackout gate (Features 2+3)
    that skips queuing ANY new entry on a blocked day. Existing positions
    still exit normally on a blocked day — only phase 3 (scanning for NEW
    signals) is skipped.

    Position sizing (Feature 1) defaults to RISK_PCT_MR/RISK_PCT_MOM —
    the SAME asymmetric 0.625%/0.25% split as the Original Baseline's own
    risk_pct_mr/risk_pct_mom — so a clean run isolates the effect of the
    blackout/breaker gates alone, with no sizing-formula confound. Pass a
    different risk_pct_mr/risk_pct_mom (e.g. the earlier FLAT_RISK_PCT) to
    reintroduce a sizing-formula change deliberately.

    `use_circuit_breaker=False` disables Feature 2 entirely for this run
    (FOMC blackout, Feature 3, stays active): compute_circuit_breaker_days()
    is a daily Open->Low proxy for a genuinely intraday 10-minute rule (see
    its own docstring), and it was already established, in the 15-minute
    validation run, that this proxy over-blocks and distorts results at
    daily resolution — so it's left out of this multi-year stress test
    rather than mixing an unreliable daily proxy into the production
    go/no-go number.

    `fractional_shares=False` (default — every existing caller keeps this
    behavior unchanged) sizes with `math.floor(...)`, whole shares only —
    this matches Alpaca/Robinhood's actual requirement for LIMIT/STOP
    orders (Broker._whole_shares() in broker_interface.py) AND matches
    what hybrid_engine.py's live entries actually submit today
    (buy_limit()). `fractional_shares=True` removes the floor and sizes
    EXACTLY to the target risk_pct with a floating-point share count —
    what a live broker WOULD allow on a fractional-capable order type
    (both Alpaca and Robinhood support fractional shares on MARKET/
    notional orders, per alpaca_client.py's own module docstring), but
    NOT what the current LIMIT-order-based live engine can execute without
    also changing its order type. This flag is a backtest research tool
    for measuring that gap's impact — not, by itself, a description of
    what's deployed today.
    """
    frames = _load_frames(symbols, start, end)
    res = bt.HybridResult(universe_name=UNIVERSE_NAME, mode="HYBRID", fixed=True,
                          start=start, end=end)
    stats = BlockStats(circuit_breaker_enabled=use_circuit_breaker)
    if not frames:
        return res, stats

    all_dates = sorted({ts.date() for d in frames.values()
                        for ts in d.index if start <= ts.date() <= end})
    if not all_dates:
        return res, stats

    breaker_days = compute_circuit_breaker_days(start, end) if use_circuit_breaker else set()
    blocked_days = breaker_days | FOMC_DATES
    stats.breaker_days = len(breaker_days & set(all_dates))
    stats.macro_days = len(FOMC_DATES & set(all_dates))

    cash = capital
    open_pos: dict[str, "bt.HybridTrade"] = {}
    pending_entries: list[tuple[str, str, dict | None]] = []
    equity_points: list[tuple[dt.date, float]] = []

    def _mark(todays: dict) -> float:
        return cash + sum(
            p.shares_open * float(todays[s]["Close"]) if s in todays
            else p.shares_open * p.entry_px
            for s, p in open_pos.items())

    for sdate in all_dates:
        todays = {}
        for sym, d in frames.items():
            row = d[d.index.date == sdate]
            if not row.empty:
                todays[sym] = row.iloc[0]

        # ---- phase 1: same-day exits — UNCHANGED regardless of any gate
        # today. Existing positions rely on their own structural stop, per
        # the request.
        for sym, t in list(open_pos.items()):
            row = todays.get(sym)
            if row is None:
                continue
            h, l = float(row["High"]), float(row["Low"])

            if t.style == "MR":
                if l <= t.stop:
                    fill = apply_slippage(t.stop, "sell")
                    proceeds = fill * t.shares_open - COMMISSION_PER_TRADE
                    cash += proceeds
                    t.realized_pnl += proceeds - t.entry_px * t.shares_open
                    t.shares_open = 0.0
                    t.exit_date, t.reason = sdate, ("BE_STOP" if t.scaled else "STOP")
                    res.trades.append(t)
                    del open_pos[sym]
                    continue

                if (SCALE_ENABLED and not t.scaled
                        and h >= t.entry_px + t.r_unit * SCALE_R):
                    scale_price = t.entry_px + t.r_unit * SCALE_R
                    qty = round(t.shares_total * SCALE_FRACTION, 6)
                    fill = apply_slippage(scale_price, "sell")
                    proceeds = fill * qty - COMMISSION_PER_TRADE
                    cash += proceeds
                    t.realized_pnl += proceeds - t.entry_px * qty
                    t.shares_open = round(t.shares_open - qty, 6)
                    t.scaled = True
                    t.stop = t.entry_px

                if t.shares_open > 0 and h >= t.target:
                    fill = apply_slippage(t.target, "sell")
                    proceeds = fill * t.shares_open - COMMISSION_PER_TRADE
                    cash += proceeds
                    t.realized_pnl += proceeds - t.entry_px * t.shares_open
                    t.shares_open = 0.0
                    t.exit_date, t.reason = sdate, "TARGET"
                    res.trades.append(t)
                    del open_pos[sym]

            else:  # MOM: 2.0x ATR trailing stop, ratchet-up only
                if l <= t.stop:
                    fill = apply_slippage(t.stop, "sell")
                    proceeds = fill * t.shares_open - COMMISSION_PER_TRADE
                    cash += proceeds
                    t.realized_pnl += proceeds - t.entry_px * t.shares_open
                    t.shares_open = 0.0
                    t.exit_date, t.reason = sdate, "ATR_TRAIL_STOP"
                    res.trades.append(t)
                    del open_pos[sym]
                    continue

                atr_today = float(row["ATR14"])
                if atr_today == atr_today:
                    t.high_water = max(t.high_water, h)
                    new_stop = t.high_water - bt.ATR_TRAIL_MULT * atr_today
                    if new_stop > t.stop:
                        t.stop = new_stop

        # ---- phase 2: fill yesterday's queued entries at TODAY's open —
        # FEATURE 1: asymmetric risk_pct_mr/risk_pct_mom sizing (matches the
        # Original Baseline's own split by default; see the risk_pct_mr/mom
        # docstring above).
        for sym, style, basis in pending_entries:
            if sym in open_pos or len(open_pos) >= MAX_CONCURRENT:
                continue
            row = todays.get(sym)
            if row is None:
                continue
            open_px = float(row["Open"])
            fill = apply_slippage(open_px, "buy")

            if style == "MR":
                stop = fill * (1.0 - STOP_PCT)
                high_water = 0.0
            else:
                high_water = basis["high"]
                stop = high_water - bt.ATR_TRAIL_MULT * basis["atr"]

            if fill <= stop:
                continue
            eq = _mark(todays)
            risk_pct = risk_pct_mr if style == "MR" else risk_pct_mom
            r_unit = fill - stop
            raw_shares = (eq * risk_pct) / r_unit
            shares = raw_shares if fractional_shares else math.floor(raw_shares)
            if shares <= 0:
                stats.zero_share_signals += 1
                continue
            cost = shares * fill + COMMISSION_PER_TRADE
            if cost > cash:
                continue
            cash -= cost
            realized_risk_pct = (shares * r_unit) / eq
            (stats.realized_risk_pct_mr if style == "MR"
            else stats.realized_risk_pct_mom).append(realized_risk_pct)
            target = fill + r_unit * TARGET_R if style == "MR" else None
            open_pos[sym] = bt.HybridTrade(
                symbol=sym, style=style, entry_date=sdate, entry_px=fill,
                shares_total=shares, shares_open=shares, stop=stop,
                r_unit=r_unit, target=target, high_water=high_water)
        pending_entries = []

        # ---- phase 3: scan today's closes -> queue tomorrow's entries.
        # FEATURES 2+3: a blocked day (circuit breaker OR macro blackout)
        # queues NOTHING at all this session — existing positions above
        # were already exited/managed unaffected. Candidates are evaluated
        # regardless of block status (same loop either way) so
        # stats.signals_blocked reflects what was actually suppressed, not
        # just a placeholder that always reads zero.
        today_blocked = sdate in blocked_days
        candidates: list[tuple[str, str, dict | None]] = []
        for sym, row in todays.items():
            if sym in open_pos:
                continue
            style = None
            basis = None
            adx_v = float(row["ADX"])
            if adx_v == adx_v:
                if adx_v < bt.ADX_RANGE_MAX and bool(row["SIGNAL_MR"]):
                    style = "MR"
                else:
                    dyn = float(row["ADX_DYN_THRESH"])
                    if dyn == dyn and adx_v > dyn and bool(row["SIGNAL_MOM"]):
                        style = "MOM"
            if style == "MOM":
                atr_v = float(row["ATR14"])
                if atr_v != atr_v:
                    style = None
                else:
                    basis = {"high": float(row["High"]), "atr": atr_v}
            if style:
                candidates.append((sym, style, basis))

        if today_blocked:
            stats.signals_blocked += len(candidates)
            pending_entries = []
        else:
            mr_c = [x for x in candidates if x[1] == "MR"]
            mom_c = [x for x in candidates if x[1] == "MOM"]
            mr_c.sort(key=lambda x: (lambda v: v if v == v else 999.0)(float(todays[x[0]]["RSI"])))
            mom_c.sort(key=lambda x: (lambda v: -v if v == v else 0.0)(float(todays[x[0]]["ADX"])))
            pending_entries = mr_c + mom_c

        equity_points.append((sdate, _mark(todays)))

    # force-close anything still open at the end of the window
    for sym, t in list(open_pos.items()):
        last_close = float(frames[sym]["Close"].iloc[-1])
        fill = apply_slippage(last_close, "sell")
        proceeds = fill * t.shares_open - COMMISSION_PER_TRADE
        cash += proceeds
        t.realized_pnl += proceeds - t.entry_px * t.shares_open
        t.shares_open = 0.0
        t.exit_date, t.reason = all_dates[-1], "WINDOW_END"
        res.trades.append(t)
    if equity_points:
        equity_points[-1] = (equity_points[-1][0], cash)

    res.equity = pd.Series(dict(equity_points)).sort_index()
    return res, stats


# ---------------------------------------------------------------- reporting
def _fmt_ratio(x: float) -> str:
    return "inf" if x == float("inf") else f"{x:.2f}"


def print_comparison(baseline_metrics: dict, managed_metrics: dict,
                     stats: BlockStats) -> None:
    width = 78
    print("=" * width)
    print("ORIGINAL BASELINE vs NEW RISK-MANAGED STRATEGY")
    print(f"Mixed universe (30 symbols), MAX_CONCURRENT=4, {START} -> {END}, "
          f"${STARTING_CAPITAL:,.0f} each")
    print("=" * width)
    print(f"{'Metric':<28}{'Original Baseline':>24}{'Risk-Managed':>24}")
    print("-" * width)
    print(f"{'Win Rate (%)':<28}{baseline_metrics['win_rate_pct']:>23.1f}%"
          f"{managed_metrics['win_rate_pct']:>23.1f}%")
    print(f"{'Total Return (%) — Upside':<28}{baseline_metrics['total_return_pct']:>+23.2f}%"
          f"{managed_metrics['total_return_pct']:>+23.2f}%")
    print(f"{'Max Drawdown (%)':<28}{baseline_metrics['max_dd_pct']:>23.2f}%"
          f"{managed_metrics['max_dd_pct']:>23.2f}%")
    print("-" * width)
    print(f"{'Sharpe Ratio':<28}{baseline_metrics['sharpe']:>24.3f}"
          f"{managed_metrics['sharpe']:>24.3f}")
    print(f"{'Calmar Ratio':<28}{_fmt_ratio(baseline_metrics['calmar']):>24}"
          f"{_fmt_ratio(managed_metrics['calmar']):>24}")
    print(f"{'Total Trades':<28}{baseline_metrics['n_trades']:>24}"
          f"{managed_metrics['n_trades']:>24}")
    print("=" * width)
    if stats.circuit_breaker_enabled:
        print(f"Risk-Managed strategy: {stats.breaker_days} circuit-breaker day(s), "
              f"{stats.macro_days} FOMC blackout day(s) in-window, "
              f"{stats.signals_blocked} signal(s) blocked from queuing as a result.")
    else:
        print(f"Risk-Managed strategy: SPY circuit breaker DISABLED for this run "
              f"(the daily Open->Low proxy over-blocks at this resolution — see "
              f"run_risk_managed_backtest's use_circuit_breaker docstring). "
              f"{stats.macro_days} FOMC blackout day(s) in-window, "
              f"{stats.signals_blocked} signal(s) blocked from queuing (FOMC only).")
    print()

    dd_improved = managed_metrics["max_dd_pct"] < baseline_metrics["max_dd_pct"]
    upside_retained_pct = (managed_metrics["total_return_pct"]
                          / baseline_metrics["total_return_pct"] * 100.0
                          if baseline_metrics["total_return_pct"] > 0 else float("nan"))
    print("=" * width)
    if dd_improved:
        print(f"Max Drawdown improved: {baseline_metrics['max_dd_pct']:.2f}% -> "
              f"{managed_metrics['max_dd_pct']:.2f}%.")
    else:
        print(f"Max Drawdown did NOT improve: {baseline_metrics['max_dd_pct']:.2f}% -> "
              f"{managed_metrics['max_dd_pct']:.2f}%.")
    if upside_retained_pct == upside_retained_pct:  # not NaN
        print(f"Upside retained: {upside_retained_pct:.1f}% of the Original "
              f"Baseline's total return ({baseline_metrics['total_return_pct']:+.2f}% "
              f"-> {managed_metrics['total_return_pct']:+.2f}%).")
    print("=" * width)


def print_capital_diagnostics(stats: BlockStats, capital: float) -> None:
    """
    Answers two questions the standard comparison table doesn't: (1) how
    often was a real, otherwise-valid signal skipped because
    floor(equity x risk_pct / risk_per_share) rounded to 0 shares — capital
    too small to buy even 1 share at the target risk; (2) for trades that
    DID fill, how far did whole-share rounding push the REALIZED risk
    (shares x r_unit / equity_at_entry) away from the 0.625%/0.25% target.
    Measured directly from run_risk_managed_backtest()'s New Risk-Managed
    Strategy run — the Original Baseline (backtest.py) uses the textually
    identical `math.floor(eq * risk_pct / (fill - stop))` sizing formula,
    so this finding applies equally to it without needing separate
    instrumentation there.
    """
    width = 78
    print("=" * width)
    print(f"CAPITAL DIAGNOSTICS — ${capital:,.0f} starting capital")
    print("=" * width)
    print(f"Q1: Signals skipped because sizing floored to 0 shares (capital "
          f"insufficient to buy even 1 whole share at the target risk): "
          f"{stats.zero_share_signals}")
    print()
    print("Q2: Whole-share-rounding impact on REALIZED risk-per-trade, "
          "vs. the 0.625% (DIP) / 0.25% (BREAKOUT) target:")

    def _summarize(label: str, target_pct: float, values: list[float]) -> None:
        if not values:
            print(f"  {label}: no filled trades to measure")
            return
        arr = np.array(values) * 100.0
        target_pp = target_pct * 100.0
        mean_dev_pct = (arr.mean() / target_pp - 1.0) * 100.0
        print(f"  {label} (target {target_pp:.3f}%): {len(values)} filled "
              f"trade(s) — realized risk mean={arr.mean():.4f}%  "
              f"min={arr.min():.4f}%  max={arr.max():.4f}%  "
              f"(mean deviation from target: {mean_dev_pct:+.1f}%)")

    _summarize("DIP     ", RISK_PCT_MR, stats.realized_risk_pct_mr)
    _summarize("BREAKOUT", RISK_PCT_MOM, stats.realized_risk_pct_mom)
    print("=" * width)


def print_fractional_comparison(metrics_a: dict, stats_a: BlockStats,
                                metrics_b: dict, stats_b: BlockStats,
                                capital: float) -> None:
    width = 84

    def _final_value(m: dict) -> float:
        return capital * (1.0 + m["total_return_pct"] / 100.0)

    print("=" * width)
    print("TEST A (Whole Shares) vs TEST B (Fractional Shares)")
    print(f"${capital:,.0f} starting capital, Mixed universe (30 symbols), "
         f"MAX_CONCURRENT={MAX_CONCURRENT}, {START} -> {END}")
    print("=" * width)
    print(f"{'Metric':<28}{'Test A: Whole Shares':>28}{'Test B: Fractional':>28}")
    print("-" * width)
    print(f"{'Total Return (%)':<28}{metrics_a['total_return_pct']:>+27.2f}%"
         f"{metrics_b['total_return_pct']:>+27.2f}%")
    print(f"{'Final Account Value ($)':<28}{_final_value(metrics_a):>28,.2f}"
         f"{_final_value(metrics_b):>28,.2f}")
    print(f"{'Max Drawdown (%)':<28}{metrics_a['max_dd_pct']:>27.2f}%"
         f"{metrics_b['max_dd_pct']:>27.2f}%")
    print(f"{'Win Rate (%)':<28}{metrics_a['win_rate_pct']:>27.1f}%"
         f"{metrics_b['win_rate_pct']:>27.1f}%")
    print(f"{'Sharpe Ratio':<28}{metrics_a['sharpe']:>28.3f}{metrics_b['sharpe']:>28.3f}")
    print(f"{'Total Trades Executed':<28}{metrics_a['n_trades']:>28}{metrics_b['n_trades']:>28}")
    print(f"{'Skipped Trades (0-share)':<28}{stats_a.zero_share_signals:>28}"
         f"{stats_b.zero_share_signals:>28}")
    print("=" * width)


def run_fractional_share_comparison(capital: float = 1_900.0) -> None:
    """
    Test A vs Test B: identical production config (asymmetric 0.625%
    DIP / 0.25% BREAKOUT risk, FOMC blackout active, circuit breaker
    disabled per the established daily-bar over-blocking finding), same
    ${capital} starting capital, same 2022-present window — differing
    ONLY in run_risk_managed_backtest()'s fractional_shares flag. See that
    parameter's docstring for the real caveat: Test B models what a
    fractional-capable MARKET/notional order type would allow, NOT what
    hybrid_engine.py's live LIMIT-order entries can execute today.
    """
    symbols = bt.UNIVERSES[UNIVERSE_NAME]

    print(f"Test A: ${capital:,.0f}, WHOLE shares (baseline constraint): "
         f"{START} -> {END} ...", flush=True)
    res_a, stats_a = run_risk_managed_backtest(
        symbols, START, END, capital, use_circuit_breaker=False,
        risk_pct_mr=RISK_PCT_MR, risk_pct_mom=RISK_PCT_MOM,
        fractional_shares=False)

    print(f"Test B: ${capital:,.0f}, FRACTIONAL shares (unlocked execution): "
         f"{START} -> {END} ...", flush=True)
    res_b, stats_b = run_risk_managed_backtest(
        symbols, START, END, capital, use_circuit_breaker=False,
        risk_pct_mr=RISK_PCT_MR, risk_pct_mom=RISK_PCT_MOM,
        fractional_shares=True)

    metrics_a = compute_metrics(res_a, capital)
    metrics_b = compute_metrics(res_b, capital)

    print_fractional_comparison(metrics_a, stats_a, metrics_b, stats_b, capital)


def main_daily() -> int:
    """
    Multi-year DAILY-bar, 2022-present stress test (not called by default —
    see main() below for the 60-day 15-minute-bar version this file runs
    when executed directly). This is the production go/no-go run: Feature 1
    (dynamic ATR/stop-distance sizing) now uses the SAME asymmetric
    0.625% (DIP) / 0.25% (BREAKOUT) risk split as the Original Baseline
    itself, so this run isolates the effect of Feature 3 (FOMC macro
    blackout) alone, with no sizing-formula confound. Feature 2 (the SPY
    circuit breaker) is explicitly DISABLED here, per the established
    finding that its only available daily-bar form (Open->Low proxy)
    over-blocks and distorts results at this resolution — carrying an
    unreliable proxy into a long-term "will this break production" number
    would defeat the purpose of the test.
    """
    symbols = bt.UNIVERSES[UNIVERSE_NAME]

    print(f"Original Baseline (Strategy B, unmodified): {START} -> {END} ...",
          flush=True)
    baseline_res = bt.run_adx_hybrid_fixed_backtest(
        symbols, START, END, mode="HYBRID", capital=STARTING_CAPITAL,
        universe_name=UNIVERSE_NAME, risk_pct_mr=RISK_PCT_MR, risk_pct_mom=RISK_PCT_MOM,
        max_concurrent=MAX_CONCURRENT)

    print(f"New Risk-Managed Strategy (asymmetric 0.625%/0.25% risk sizing, "
          f"matches baseline; FOMC blackout only; circuit breaker DISABLED): "
          f"{START} -> {END} ...", flush=True)
    managed_res, stats = run_risk_managed_backtest(symbols, START, END,
                                                   STARTING_CAPITAL,
                                                   use_circuit_breaker=False,
                                                   risk_pct_mr=RISK_PCT_MR,
                                                   risk_pct_mom=RISK_PCT_MOM)

    baseline_metrics = compute_metrics(baseline_res, STARTING_CAPITAL)
    managed_metrics = compute_metrics(managed_res, STARTING_CAPITAL)

    print_comparison(baseline_metrics, managed_metrics, stats)
    print_capital_diagnostics(stats, STARTING_CAPITAL)
    return 0


# =============================================================================
# INTRADAY MODE — last 60 days, 15-minute bars, EXACT 10-minute SPY windows
# and precise 1:45-3:30 PM ET FOMC timestamps.
#
# This is now DATA-FEASIBLE where the daily-bar version above was not: Yahoo
# Finance caps 15-minute/5-minute history at 60 calendar days, and this
# request's window fits inside that cap — unlike the 2022-present window
# above, which is why THAT version had to fall back to daily-bar proxies for
# the breaker/blackout. Here both are implemented literally: the circuit
# breaker is computed on native 5-minute SPY bars (a true rolling 2-bar/
# 10-minute return) and the FOMC blackout checks the actual bar timestamp
# against 13:45-15:30 ET, not a whole-day block.
#
# MULTI-TIMEFRAME FIX (superseding an earlier bug in this section): RSI(14),
# Bollinger(20), the 50-period ADX-percentile window, and the 200-period SMA
# trend gate were originally applied DIRECTLY to 15-minute bars. A "200-period
# SMA" on 15-minute bars is only ~7.7 trading days (200 bars x 15min / 390min
# RTH) — not a macro trend filter by any reasonable reading, and a 50-BAR ADX
# percentile window was under 2 trading days. Fixed via multi-timeframe
# resampling:
#   1. SMA-200, the daily RSI-14, and daily ADX-14 (+ its 50-DAY 80th
#      percentile threshold) are computed on genuine DAILY bars — see
#      _daily_macro_frame(). These are real macro reads: SMA-200 is ~10
#      months, the ADX percentile window is ~2.5 months.
#   2. Each 15-minute bar is forward-filled with the LAST COMPLETED daily
#      bar's macro values via merge_asof (direction="backward"), never the
#      still-forming current day's — the same last-completed-bar discipline
#      used everywhere else in this codebase. A day's 15-minute bars all see
#      the identical macro snapshot from the prior day's close; it updates
#      once per day, not once per bar.
#   3. The 15-minute bars are used ONLY for what's genuinely intraday: the
#      DIP entry trigger (price touching the 15-min Bollinger lower band),
#      the BREAKOUT entry trigger (price > the prior 20-BAR 15-min high with
#      a 15-min volume surge), the 2.0x-ATR14 trailing stop (ATR14 computed
#      on 15-min bars), the exact 10-minute SPY circuit breaker, and the
#      exact FOMC blackout timestamp check.
# NOTE: "Daily resampled bars" is read here as genuine daily-frequency OHLC
# data (data.daily_bars, ~2 years of history so SMA-200/the 50-day ADX
# percentile are fully warmed up before this 60-day window even starts) —
# NOT a resample of the 60-day intraday dataset itself, which is far too
# short to support a 200-day or 50-day rolling calculation. Flagging this
# interpretation explicitly since the two readings would give very different
# numbers.
#
# SAMPLE SIZE: because the macro filters are now warmed up on ~2 years of
# real daily history rather than burning ~200-250 BARS of the 60-day intraday
# window itself, virtually the entire window is usable — a meaningful
# improvement over the previous single-timeframe version, though ~60
# calendar days (~42 trading days) is still a modest sample; treat every
# metric below as exploratory, not conclusive.
# =============================================================================

INTRADAY_DAYS = 60
INTRADAY_BAR = "15min"
RTH_START_MIN, RTH_END_MIN = 570, 960     # 9:30-16:00 ET, minutes past midnight
INTRADAY_MIN_BARS = 30                    # just needs BB20/ATR14/VolSMA20/PriorHigh20 warm

DAILY_HISTORY_PERIOD = "2y"               # for SMA-200 / 50-day ADX percentile warmup
DAILY_MACRO_MIN_BARS = 260                # ~200 (SMA) + 50 (ADX pctile) + buffer

CB_WINDOW_BARS = 2          # 2 x 5-minute bars = the EXACT 10-minute window
CB_DROP_PCT = 0.0075        # unchanged threshold, now on a genuinely 10-min window
CB_BLOCK_MINUTES = 60       # block new entries for 60 minutes after a trip

FOMC_START_TIME = dt.time(13, 45)
FOMC_END_TIME = dt.time(15, 30)


@dataclass
class IntradayTrade:
    symbol: str
    style: str
    entry_ts: pd.Timestamp
    entry_px: float
    shares_total: float
    shares_open: float
    stop: float
    r_unit: float = 0.0
    target: float | None = None
    scaled: bool = False
    high_water: float = 0.0
    realized_pnl: float = 0.0
    exit_ts: pd.Timestamp | None = None
    reason: str = ""

    @property
    def is_win(self) -> bool:
        return self.realized_pnl > 0


@dataclass
class IntradayResult:
    trades: list[IntradayTrade] = field(default_factory=list)
    equity: pd.Series = field(default_factory=pd.Series)
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    blocked_signals: int = 0


def _rth_15m_bars(symbol: str) -> pd.DataFrame:
    """5-minute RTH bars (data.intraday_5m's 60-day max) resampled to 15-min
    OHLCV. 9:30/16:00 are exact 15-minute boundaries from midnight (570/15=38,
    960/15=64), so bins never straddle the overnight gap once non-RTH rows
    are dropped first — no special resample offset needed (unlike the
    60-minute case elsewhere in this codebase, which does need one)."""
    raw = data.intraday_5m(symbol, period=f"{INTRADAY_DAYS}d")
    if raw.empty:
        return pd.DataFrame()
    mins = raw.index.hour * 60 + raw.index.minute
    rth = raw[(mins >= RTH_START_MIN) & (mins < RTH_END_MIN)]
    if rth.empty:
        return pd.DataFrame()
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    bars = rth.resample(INTRADAY_BAR).agg(agg).dropna(subset=["Close"])
    return bars


def _naive_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Normalize a (possibly tz-aware) DatetimeIndex to midnight, tz-naive —
    the common calendar-date key used to align daily macro data onto 15-min
    bars via merge_asof."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def _daily_macro_frame(symbol: str) -> pd.DataFrame:
    """
    Macro trend/regime filters computed on genuine DAILY bars: SMA-200,
    daily RSI-14, and daily ADX-14 with its 50-DAY 80th-percentile threshold
    (bt.ADX_PERCENTILE_WINDOW/bt.ADX_PERCENTILE_Q/bt.ADX_TREND_MIN, the same
    constants backtest.py's daily engines use). Fetches DAILY_HISTORY_PERIOD
    (2y) of real daily bars so these are fully warmed up long before the
    60-day intraday window starts.

    Every macro column is shift(1)'d: a row dated D holds D-1's COMPLETED
    daily values, never D's own still-forming bar. When this frame is
    forward-filled onto 15-minute bars by calendar date, every 15-min bar on
    trading day D therefore sees the last COMPLETED (D-1) daily reading —
    the same non-repainting discipline used throughout this codebase.
    """
    daily = data.daily_bars(symbol, period=DAILY_HISTORY_PERIOD)
    if daily.empty or len(daily) < DAILY_MACRO_MIN_BARS:
        return pd.DataFrame()

    close = daily["Close"].to_numpy(dtype=float)
    high = daily["High"].to_numpy(dtype=float)
    low = daily["Low"].to_numpy(dtype=float)

    rsi_d = strategy.rsi(close, RSI_PERIOD)
    adx_d = strategy.adx(high, low, close, bt.ATR_PERIOD)
    sma200_d = pd.Series(close).rolling(200).mean().to_numpy()
    adx_d_thresh = (pd.Series(adx_d).shift(1).rolling(bt.ADX_PERCENTILE_WINDOW)
                    .quantile(bt.ADX_PERCENTILE_Q).clip(lower=bt.ADX_TREND_MIN).to_numpy())

    out = pd.DataFrame({
        "DATE": _naive_dates(daily.index),
        "CLOSE_D": close,
        "SMA200_D": sma200_d,
        "RSI_D": rsi_d,
        "ADX_D": adx_d,
        "ADX_D_THRESH": adx_d_thresh,
    })
    macro_cols = ["CLOSE_D", "SMA200_D", "RSI_D", "ADX_D", "ADX_D_THRESH"]
    out[macro_cols] = out[macro_cols].shift(1)
    return out.dropna(subset=["SMA200_D", "ADX_D_THRESH"]).reset_index(drop=True)


def build_intraday_frame(bars: pd.DataFrame, daily_macro: pd.DataFrame) -> pd.DataFrame:
    """
    Intraday-native columns — Bollinger(20), ATR14, the 20-bar breakout
    lookback, and the 20-bar volume SMA — are computed on the 15-minute
    bars themselves (per instruction #3: entry triggers, ATR sizing, all
    intraday-native). The macro trend/regime columns (ADX/ADX_DYN_THRESH
    here ARE the daily ADX-14 and its 50-DAY percentile threshold; the
    SIGNAL_MR trend/oversold legs use the daily SMA-200/RSI-14) are
    forward-filled from daily_macro (see _daily_macro_frame) by calendar
    date — fixing the earlier bug where these were computed directly on
    15-minute bars.
    """
    close = bars["Close"].to_numpy(dtype=float)
    high = bars["High"].to_numpy(dtype=float)
    low = bars["Low"].to_numpy(dtype=float)
    volume = bars["Volume"].to_numpy(dtype=float)

    _, bb_low, _ = strategy.bollinger(close, BB_PERIOD, BB_K)
    atr_v = strategy.atr(high, low, close, bt.ATR_PERIOD)
    vol_sma20 = pd.Series(volume).rolling(bt.VOLUME_SMA_WINDOW).mean().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_ok = np.where(vol_sma20 > 0, volume > bt.VOLUME_MULT * vol_sma20, False)
    prior_high = pd.Series(high).shift(1).rolling(bt.BREAKOUT_LOOKBACK).max().to_numpy()
    breakout_price_ok = np.where(np.isnan(prior_high), False, close > prior_high)
    signal_mom = breakout_price_ok & vol_ok

    bar_dates = pd.DataFrame({"DATE": _naive_dates(bars.index)})
    aligned = pd.merge_asof(bar_dates, daily_macro, on="DATE", direction="backward")

    sma200_d = aligned["SMA200_D"].to_numpy()
    close_d = aligned["CLOSE_D"].to_numpy()
    rsi_d = aligned["RSI_D"].to_numpy()
    adx_d = aligned["ADX_D"].to_numpy()
    adx_d_thresh = aligned["ADX_D_THRESH"].to_numpy()

    # trend_ok: was YESTERDAY's daily close above yesterday's SMA-200?
    trend_ok = np.where(np.isnan(sma200_d) | np.isnan(close_d), False,
                        close_d > sma200_d)
    # above_now: is the LIVE 15-min price still above that (once-per-day) level?
    above_now = np.where(np.isnan(sma200_d), False, close > sma200_d)
    rsi_ok = np.where(np.isnan(rsi_d), False, rsi_d < RSI_OVERSOLD)
    bb_ok = np.where(np.isnan(bb_low), False, low <= bb_low)
    signal_mr = trend_ok & above_now & (rsi_ok | bb_ok)

    out = bars.copy()
    out["RSI"] = rsi_d
    out["BB_LOW"] = bb_low
    out["ADX"] = adx_d
    out["ADX_DYN_THRESH"] = adx_d_thresh
    out["ATR14"] = atr_v
    out["SIGNAL_MR"] = signal_mr
    out["SIGNAL_MOM"] = signal_mom
    return out


def compute_spy_circuit_breaker_events() -> pd.DatetimeIndex:
    """EXACT rolling-10-minute SPY decline > CB_DROP_PCT, on native 5-minute
    bars (2-bar lookback = 10 minutes) — the literal rule from the request,
    not a daily-bar proxy, now that the 60-day window fits Yahoo's cap.

    The 2-bar pct_change is computed PER TRADING DAY (grouped by calendar
    date), not on the raw concatenated RTH series: without that, the first
    two bars of each day diff against the prior day's last close/bars, which
    turns an ordinary overnight gap into a spurious "10-minute -X% crash" at
    09:30/09:35 every day the market gapped down — confirmed empirically (5
    of 5 raw trips landed at 09:30/09:35). A circuit breaker on a real
    10-minute INTRADAY move should never fire on bar 1 or 2 of the session."""
    raw = data.intraday_5m("SPY", period=f"{INTRADAY_DAYS}d")
    if raw.empty:
        return pd.DatetimeIndex([])
    mins = raw.index.hour * 60 + raw.index.minute
    rth = raw[(mins >= RTH_START_MIN) & (mins < RTH_END_MIN)]
    if rth.empty:
        return pd.DatetimeIndex([])
    roll_ret = rth.groupby(rth.index.date)["Close"].pct_change(periods=CB_WINDOW_BARS)
    return rth.index[roll_ret < -CB_DROP_PCT]


def is_circuit_breaker_active(ts: pd.Timestamp, trip_events: pd.DatetimeIndex) -> bool:
    if len(trip_events) == 0:
        return False
    window_start = ts - pd.Timedelta(minutes=CB_BLOCK_MINUTES)
    return bool(((trip_events > window_start) & (trip_events <= ts)).any())


def is_fomc_blackout(ts: pd.Timestamp) -> bool:
    return ts.date() in FOMC_DATES and FOMC_START_TIME <= ts.time() <= FOMC_END_TIME


def run_intraday_backtest(symbols: list[str], capital: float, use_new_features: bool,
                          breaker_events: pd.DatetimeIndex | None = None
                          ) -> IntradayResult:
    """
    Same phase structure (exits, fill queued entries, scan+queue new
    signals) as the daily engines above, at 15-minute-bar granularity.
    `use_new_features=False` reproduces the Original Baseline's asymmetric
    0.625%/0.25% risk sizing with no gates; `True` applies FLAT_RISK_PCT
    (Feature 1) plus the circuit-breaker/FOMC gate (Features 2+3) — mirroring
    the daily version's two-mode comparison exactly, just on finer bars.
    """
    frames: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        bars = _rth_15m_bars(sym)
        if bars.empty or len(bars) < INTRADAY_MIN_BARS:
            continue
        daily_macro = _daily_macro_frame(sym)
        if daily_macro.empty:
            continue
        frames[sym] = build_intraday_frame(bars, daily_macro)

    res = IntradayResult()
    if not frames:
        return res

    all_ts = sorted({ts for d in frames.values() for ts in d.index})
    if not all_ts:
        return res
    res.start, res.end = all_ts[0], all_ts[-1]

    cash = capital
    open_pos: dict[str, IntradayTrade] = {}
    pending_entries: list[tuple[str, str, dict | None]] = []
    equity_points: list[tuple[pd.Timestamp, float]] = []

    def _mark(nowbars: dict) -> float:
        return cash + sum(
            p.shares_open * float(nowbars[s]["Close"]) if s in nowbars
            else p.shares_open * p.entry_px
            for s, p in open_pos.items())

    for ts in all_ts:
        nowbars = {sym: d.loc[ts] for sym, d in frames.items() if ts in d.index}

        # ---- phase 1: same-bar exits, unaffected by any gate (existing
        # positions rely on their own structural stop, per the request)
        for sym, t in list(open_pos.items()):
            row = nowbars.get(sym)
            if row is None:
                continue
            h, l = float(row["High"]), float(row["Low"])

            if t.style == "MR":
                if l <= t.stop:
                    fill = apply_slippage(t.stop, "sell")
                    proceeds = fill * t.shares_open - COMMISSION_PER_TRADE
                    cash += proceeds
                    t.realized_pnl += proceeds - t.entry_px * t.shares_open
                    t.shares_open = 0.0
                    t.exit_ts, t.reason = ts, ("BE_STOP" if t.scaled else "STOP")
                    res.trades.append(t)
                    del open_pos[sym]
                    continue

                if (SCALE_ENABLED and not t.scaled
                        and h >= t.entry_px + t.r_unit * SCALE_R):
                    scale_price = t.entry_px + t.r_unit * SCALE_R
                    qty = round(t.shares_total * SCALE_FRACTION, 6)
                    fill = apply_slippage(scale_price, "sell")
                    proceeds = fill * qty - COMMISSION_PER_TRADE
                    cash += proceeds
                    t.realized_pnl += proceeds - t.entry_px * qty
                    t.shares_open = round(t.shares_open - qty, 6)
                    t.scaled = True
                    t.stop = t.entry_px

                if t.shares_open > 0 and h >= t.target:
                    fill = apply_slippage(t.target, "sell")
                    proceeds = fill * t.shares_open - COMMISSION_PER_TRADE
                    cash += proceeds
                    t.realized_pnl += proceeds - t.entry_px * t.shares_open
                    t.shares_open = 0.0
                    t.exit_ts, t.reason = ts, "TARGET"
                    res.trades.append(t)
                    del open_pos[sym]

            else:  # MOM: 2.0x ATR trailing stop, ratchet-up only
                if l <= t.stop:
                    fill = apply_slippage(t.stop, "sell")
                    proceeds = fill * t.shares_open - COMMISSION_PER_TRADE
                    cash += proceeds
                    t.realized_pnl += proceeds - t.entry_px * t.shares_open
                    t.shares_open = 0.0
                    t.exit_ts, t.reason = ts, "ATR_TRAIL_STOP"
                    res.trades.append(t)
                    del open_pos[sym]
                    continue

                atr_now = float(row["ATR14"])
                if atr_now == atr_now:
                    t.high_water = max(t.high_water, h)
                    new_stop = t.high_water - bt.ATR_TRAIL_MULT * atr_now
                    if new_stop > t.stop:
                        t.stop = new_stop

        # ---- phase 2: fill previously-queued entries at THIS bar's open
        for sym, style, basis in pending_entries:
            if sym in open_pos or len(open_pos) >= MAX_CONCURRENT:
                continue
            row = nowbars.get(sym)
            if row is None:
                continue
            open_px = float(row["Open"])
            fill = apply_slippage(open_px, "buy")

            if style == "MR":
                stop = fill * (1.0 - STOP_PCT)
                high_water = 0.0
            else:
                high_water = basis["high"]
                stop = high_water - bt.ATR_TRAIL_MULT * basis["atr"]

            if fill <= stop:
                continue
            eq = _mark(nowbars)
            if use_new_features:
                shares = math.floor((eq * FLAT_RISK_PCT) / (fill - stop))
            else:
                risk_pct = 0.00625 if style == "MR" else 0.0025
                shares = math.floor((eq * risk_pct) / (fill - stop))
            cost = shares * fill + COMMISSION_PER_TRADE
            if shares <= 0 or cost > cash:
                continue
            cash -= cost
            r_unit = fill - stop
            target = fill + r_unit * TARGET_R if style == "MR" else None
            open_pos[sym] = IntradayTrade(
                symbol=sym, style=style, entry_ts=ts, entry_px=fill,
                shares_total=shares, shares_open=shares, stop=stop,
                r_unit=r_unit, target=target, high_water=high_water)
        pending_entries = []

        # ---- phase 3: scan this bar's closes -> queue the NEXT bar's entries
        blocked_now = (use_new_features and breaker_events is not None
                       and (is_circuit_breaker_active(ts, breaker_events)
                            or is_fomc_blackout(ts)))

        candidates: list[tuple[str, str, dict | None]] = []
        for sym, row in nowbars.items():
            if sym in open_pos:
                continue
            style = None
            basis = None
            adx_v = float(row["ADX"])
            if adx_v == adx_v:
                if adx_v < bt.ADX_RANGE_MAX and bool(row["SIGNAL_MR"]):
                    style = "MR"
                else:
                    dyn = float(row["ADX_DYN_THRESH"])
                    if dyn == dyn and adx_v > dyn and bool(row["SIGNAL_MOM"]):
                        style = "MOM"
            if style == "MOM":
                atr_v = float(row["ATR14"])
                if atr_v != atr_v:
                    style = None
                else:
                    basis = {"high": float(row["High"]), "atr": atr_v}
            if style:
                candidates.append((sym, style, basis))

        if blocked_now:
            res.blocked_signals += len(candidates)
            pending_entries = []
        else:
            mr_c = [x for x in candidates if x[1] == "MR"]
            mom_c = [x for x in candidates if x[1] == "MOM"]
            mr_c.sort(key=lambda x: (lambda v: v if v == v else 999.0)(float(nowbars[x[0]]["RSI"])))
            mom_c.sort(key=lambda x: (lambda v: -v if v == v else 0.0)(float(nowbars[x[0]]["ADX"])))
            pending_entries = mr_c + mom_c

        equity_points.append((ts, _mark(nowbars)))

    # force-close anything still open at the end of the window
    for sym, t in list(open_pos.items()):
        last_close = float(frames[sym]["Close"].iloc[-1])
        fill = apply_slippage(last_close, "sell")
        proceeds = fill * t.shares_open - COMMISSION_PER_TRADE
        cash += proceeds
        t.realized_pnl += proceeds - t.entry_px * t.shares_open
        t.shares_open = 0.0
        t.exit_ts, t.reason = all_ts[-1], "WINDOW_END"
        res.trades.append(t)
    if equity_points:
        equity_points[-1] = (equity_points[-1][0], cash)

    res.equity = pd.Series(dict(equity_points)).sort_index()
    return res


def compute_intraday_metrics(res: IntradayResult, capital: float) -> dict:
    trades = res.trades
    n = len(trades)
    wins = [t for t in trades if t.realized_pnl > 0]
    losses = [t for t in trades if t.realized_pnl <= 0]
    gross_win = sum(t.realized_pnl for t in wins)
    gross_loss = -sum(t.realized_pnl for t in losses)
    net = sum(t.realized_pnl for t in trades)
    win_rate = len(wins) / n if n else 0.0
    profit_factor = (gross_win / gross_loss if gross_loss > 0
                    else (float("inf") if gross_win > 0 else 0.0))

    eq = res.equity
    _, dd_frac = bt.max_drawdown(eq) if not eq.empty else (0.0, 0.0)
    max_dd_pct = dd_frac * 100.0
    total_return_pct = net / capital * 100.0
    # ~26 fifteen-minute RTH bars/day x 252 trading days/year
    sharpe = bt.sharpe(eq, periods=252 * 26) if not eq.empty else 0.0
    calmar = (total_return_pct / max_dd_pct if max_dd_pct > 0
             else (float("inf") if total_return_pct > 0 else 0.0))

    hold_min = [(t.exit_ts - t.entry_ts).total_seconds() / 60.0
               for t in trades if t.exit_ts]
    avg_hold_min = float(np.mean(hold_min)) if hold_min else 0.0

    return dict(
        win_rate_pct=win_rate * 100.0, total_return_pct=total_return_pct,
        max_dd_pct=max_dd_pct, sharpe=sharpe, calmar=calmar,
        profit_factor=profit_factor, n_trades=n, avg_hold_min=avg_hold_min,
    )


def print_intraday_comparison(baseline_metrics: dict, managed_metrics: dict,
                              baseline_res: IntradayResult,
                              managed_res: IntradayResult,
                              n_breaker_events: int) -> None:
    width = 78
    print("=" * width)
    print("ORIGINAL BASELINE vs NEW RISK-MANAGED STRATEGY — INTRADAY (15-MIN BARS)")
    print(f"Mixed universe (30 symbols), MAX_CONCURRENT={MAX_CONCURRENT}, "
          f"last {INTRADAY_DAYS} days")
    if baseline_res.start is not None:
        print(f"Window: {baseline_res.start} -> {baseline_res.end}")
    print(f"${STARTING_CAPITAL:,.0f} starting capital each")
    print("=" * width)
    print(f"{'Metric':<28}{'Original Baseline':>24}{'Risk-Managed':>24}")
    print("-" * width)
    print(f"{'Win Rate (%)':<28}{baseline_metrics['win_rate_pct']:>23.1f}%"
          f"{managed_metrics['win_rate_pct']:>23.1f}%")
    print(f"{'Total Return (%) — Upside':<28}{baseline_metrics['total_return_pct']:>+23.2f}%"
          f"{managed_metrics['total_return_pct']:>+23.2f}%")
    print(f"{'Max Drawdown (%)':<28}{baseline_metrics['max_dd_pct']:>23.2f}%"
          f"{managed_metrics['max_dd_pct']:>23.2f}%")
    print("-" * width)
    print(f"{'Sharpe Ratio':<28}{baseline_metrics['sharpe']:>24.3f}"
          f"{managed_metrics['sharpe']:>24.3f}")
    print(f"{'Calmar Ratio':<28}{_fmt_ratio(baseline_metrics['calmar']):>24}"
          f"{_fmt_ratio(managed_metrics['calmar']):>24}")
    print(f"{'Total Trades':<28}{baseline_metrics['n_trades']:>24}"
          f"{managed_metrics['n_trades']:>24}")
    print(f"{'Avg Hold (minutes)':<28}{baseline_metrics['avg_hold_min']:>24.1f}"
          f"{managed_metrics['avg_hold_min']:>24.1f}")
    print("=" * width)
    print(f"Risk-Managed strategy: {n_breaker_events} exact 10-minute circuit-"
          f"breaker trip(s) detected in-window, {managed_res.blocked_signals} "
          f"signal(s) blocked from queuing (breaker OR the 1:45-3:30 PM ET "
          f"FOMC window) as a result.")
    print()
    print("SAMPLE SIZE WARNING: macro trend/regime filters (SMA-200, daily "
          "RSI, daily ADX-50) are warmed up on ~2 years of real daily "
          "history and forward-filled in, so they no longer burn the "
          "60-day intraday window's own bars — but ~60 calendar days is "
          "still a modest sample; treat every number here as exploratory, "
          "not conclusive.")
    print("=" * width)


def main() -> int:
    """60-day, 15-minute-bar comparison with EXACT 10-minute SPY windows and
    precise 1:45-3:30 PM ET FOMC timestamps — see the INTRADAY MODE section
    docstring above for the two things that did NOT change (period counts)
    and why that matters. Call main_daily() for the original 2022-present,
    daily-bar version this file used to run by default."""
    symbols = bt.UNIVERSES[UNIVERSE_NAME]

    print(f"Computing exact 10-minute SPY circuit-breaker events over the "
          f"last {INTRADAY_DAYS} days ...", flush=True)
    breaker_events = compute_spy_circuit_breaker_events()

    print(f"Original Baseline (Strategy B, unmodified, 15-min bars, last "
          f"{INTRADAY_DAYS} days) ...", flush=True)
    baseline_res = run_intraday_backtest(symbols, STARTING_CAPITAL,
                                         use_new_features=False)

    print(f"New Risk-Managed Strategy (15-min bars, last {INTRADAY_DAYS} "
          f"days) ...", flush=True)
    managed_res = run_intraday_backtest(symbols, STARTING_CAPITAL,
                                        use_new_features=True,
                                        breaker_events=breaker_events)

    baseline_metrics = compute_intraday_metrics(baseline_res, STARTING_CAPITAL)
    managed_metrics = compute_intraday_metrics(managed_res, STARTING_CAPITAL)

    print_intraday_comparison(baseline_metrics, managed_metrics, baseline_res,
                              managed_res, len(breaker_events))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
