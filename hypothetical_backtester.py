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


def _compute_spy_trending_dates(start: dt.date, end: dt.date,
                                sma_window: int = 200) -> set[dt.date]:
    """
    Simple, standard trend proxy, generalized by `sma_window`: a day
    counts as "trending" if SPY's close is above its own rolling
    `sma_window`-day SMA (computed on the full daily history, so the SMA
    at day T never looks past day T — no forward-looking bias). Not a
    sophisticated regime classifier, just a defensible standard reading of
    "the broad market is in an established uptrend." Used two ways in this
    file: the capital-velocity/cash-drag diagnostic (BlockStats.
    avg_cash_pct_trending, sma_window=200, the default) and the
    `regime_filter` risk-sizing gate (run_risk_managed_backtest,
    sma_window=50 per that feature's spec).
    """
    spy = data.daily_bars("SPY", period="max")
    if spy.empty:
        return set()
    close = spy["Close"].to_numpy(dtype=float)
    sma = pd.Series(close).rolling(sma_window).mean().to_numpy()
    is_trending = np.where(np.isnan(sma), False, close > sma)
    dates = spy.index.date
    return {d for d, t, in zip(dates, is_trending)
           if t and start <= d <= end}


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
    # Capital-velocity diagnostics (see run_risk_managed_backtest's
    # max_concurrent docstring section and print_concurrency_sweep()):
    # avg_open_positions is the mean count of simultaneously-held positions
    # across the whole window, vs. max_concurrent_used (the cap actually in
    # effect for this run). avg_cash_pct/_trending/_non_trending are the
    # mean fraction of equity sitting in cash (not deployed), overall and
    # split by a simple SPY-close-vs-200-day-SMA trend proxy.
    avg_open_positions: float = 0.0
    max_concurrent_used: int = 0
    avg_cash_pct: float = 0.0
    avg_cash_pct_trending: float = 0.0
    avg_cash_pct_non_trending: float = 0.0


# ---------------------------------------------------------------- the new engine
def run_risk_managed_backtest(symbols: list[str], start: dt.date, end: dt.date,
                              capital: float = STARTING_CAPITAL,
                              use_circuit_breaker: bool = True,
                              risk_pct_mr: float = RISK_PCT_MR,
                              risk_pct_mom: float = RISK_PCT_MOM,
                              fractional_shares: bool = False,
                              max_concurrent: int = MAX_CONCURRENT,
                              max_hold_days_stagnant: int | None = None,
                              stagnant_min_progress_r: float = 0.5,
                              regime_filter: dict | None = None,
                              breakout_scale_out: dict | None = None,
                              breakout_gate_symbols: set[str] | None = None,
                              pyramid_config: dict | None = None,
                              margin_multiplier: float = 1.0,
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

    `max_concurrent` (default MAX_CONCURRENT — every existing caller keeps
    this behavior unchanged) is the hard cap on simultaneously open
    positions. Also drives two cash-utilization diagnostics recorded on
    the returned BlockStats (see its fields): `avg_open_positions` (mean
    concurrent positions actually held, vs. this cap) and
    `avg_cash_pct`/`avg_cash_pct_trending`/`avg_cash_pct_non_trending`
    (mean idle-cash fraction of equity, overall and split by whether SPY's
    close was above its own 200-day SMA that day — a simple, standard
    trend proxy, not a sophisticated regime classifier).

    `max_hold_days_stagnant` (default None — off, every existing caller
    unchanged): if set, ANY open position (MR or MOM) held
    `max_hold_days_stagnant` CALENDAR days or more (matching this
    codebase's existing `_age_days` convention in execution_engine.py,
    not a trading-day count) that has NOT reached at least
    `stagnant_min_progress_r` R-multiples of unrealized gain is
    force-exited at that day's close, reason "STAGNANT_TIME_EXIT" — freeing
    the capital/slot for a new signal rather than letting a going-nowhere
    position sit. This is a genuinely new exit path, checked after the
    existing stop/target/scale checks each day (a position that already
    exited via stop/target/scale this same day is not double-counted).

    `regime_filter` (default None — off, every existing caller unchanged):
    a dict `{"sma_window": int, "aggressive": (risk_pct_mr, risk_pct_mom),
    "base": (risk_pct_mr, risk_pct_mom)}`. When set, this REPLACES the
    plain `risk_pct_mr`/`risk_pct_mom` arguments for sizing purposes: each
    day is classified by whether SPY's close is above its own rolling
    `sma_window`-day SMA (computed on real daily history, no
    forward-looking bias) — the `aggressive` pair applies on days SPY is
    above that SMA, the `base` pair otherwise. Existing positions are
    unaffected by a regime change after entry; only NEW entries use the
    day's regime-selected risk_pct.

    `breakout_scale_out` (default None — off, every existing caller
    unchanged): a dict `{"tranche_atr_mult": float, "trail_atr_mult_after_
    scale": float}`. When set, MOM (BREAKOUT) trades gain a two-tranche
    exit in place of the plain single ATR_TRAIL_MULT trailing stop:
      Tranche 1 (SCALE_FRACTION, i.e. 50%, of the position): exits at a
      FIXED price set once at entry — entry_px + tranche_atr_mult x the
      ATR AT ENTRY (not recomputed daily) — and the stop on the remainder
      moves to breakeven (entry_px), $0 risk, at that moment.
      Tranche 2 (the remaining position): trails using
      trail_atr_mult_after_scale (typically wider than ATR_TRAIL_MULT)
      instead of ATR_TRAIL_MULT from that point on, ratcheting up-only
      from the breakeven floor exactly like the existing trailing-stop
      code (a wider multiplier can never ratchet the stop below wherever
      it already is).
    Reuses HybridTrade.target/.scaled (elsewhere documented "MR only") for
    MOM bookkeeping when this feature is active — those fields are never
    otherwise touched for MOM trades, so this can't collide with default
    behavior when the feature is off.

    `breakout_gate_symbols` (default None — off, every existing caller
    unchanged): a set of symbols whose MOM/BREAKOUT candidates are ONLY
    queued on a day SPY is in `regime_filter`'s bull/aggressive regime
    (`sdate in regime_aggressive_dates`) — otherwise that symbol's
    BREAKOUT signal is suppressed for the day, even if its own technicals
    would normally qualify. DIP/MR signals for these symbols are
    unaffected. Requires `regime_filter` to be set to mean anything (with
    no regime_filter, `regime_aggressive_dates` is empty, so a gated
    symbol's BREAKOUT would always be suppressed — a safe, if unhelpful,
    default rather than an error). Built for testing whether admitting
    leveraged benchmark ETFs (QLD/SSO) into the universe, but confining
    them to bull-only BREAKOUT entries, adds edge without adding tail
    risk during a bear/chop regime.

    `pyramid_config` (default None — off, every existing caller unchanged):
    a dict `{"trigger_atr_mult": float, "add_fraction": float}`. When a MOM
    (BREAKOUT) trade's unrealized gain reaches entry_px + trigger_atr_mult
    x ATR-AT-ENTRY (a fixed level, set once at entry, same convention as
    breakout_scale_out's tranche target), a SECOND tranche of
    `add_fraction` x the original share count is bought at that trigger
    price (cash permitting — skipped, not forced, if unaffordable), and
    the stop moves to the ORIGINAL (pre-add) entry price — the literal
    "1st tranche to breakeven" request.

    MODELING SIMPLIFICATION, disclosed rather than silently assumed: this
    trade model has ONE stop for the WHOLE position, not independent stops
    per tranche. After the add, `entry_px` is updated to the blended
    (volume-weighted average) cost basis so that a LATER exit's P&L is
    computed correctly across both tranches — but the single stop set to
    the original entry price is not exactly "$0 risk" on the blended
    position (tranche 2 was bought above the original entry, so a stop at
    the original entry is a small loss on the blended total, not
    breakeven on it). This is the closest defensible single-stop reading
    of the request, not a perfect two-tranche simulation.

    MUTUALLY EXCLUSIVE with `breakout_scale_out`: both reuse
    HybridTrade.target/.scaled (elsewhere "MR only") for different
    purposes and would conflict if both were set on the same call — this
    function raises ValueError if both are given.

    `margin_multiplier` (default 1.0 — no leverage, every existing caller
    unchanged): relaxes the CASH affordability check on every new entry
    (and a pyramid add) from `cost > cash` to
    `cost > cash + equity x (margin_multiplier - 1.0)` — i.e. buying power
    is equity x margin_multiplier, not cash alone, letting `cash` go
    negative (a margin loan balance) up to that limit. Position SIZING
    (risk_dollars = equity x risk_pct) is NOT affected by this — margin
    only relaxes what the strategy can afford to buy with a given sizing
    result, standard/prudent practice (risking a % of equity, not of
    leveraged buying power). MODELING SIMPLIFICATION: margin interest/
    financing cost on a negative cash balance is not modeled — this is
    free leverage, which overstates real returns net of borrowing cost.
    """
    if breakout_scale_out is not None and pyramid_config is not None:
        raise ValueError("breakout_scale_out and pyramid_config cannot both be "
                         "set — they reuse the same trade-level bookkeeping "
                         "fields (HybridTrade.target/.scaled) for different, "
                         "conflicting purposes")

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

    regime_aggressive_dates: set[dt.date] = set()
    if regime_filter is not None:
        regime_aggressive_dates = _compute_spy_trending_dates(
            start, end, sma_window=regime_filter.get("sma_window", 50))

    cash = capital
    open_pos: dict[str, "bt.HybridTrade"] = {}
    pending_entries: list[tuple[str, str, dict | None]] = []
    equity_points: list[tuple[dt.date, float]] = []
    cash_pct_points: list[tuple[dt.date, float]] = []   # cash / equity, per date
    open_count_points: list[int] = []                   # concurrent open positions, per date

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

            else:  # MOM: ATR trailing stop, ratchet-up only (+ optional
                    # two-tranche scale-out, see breakout_scale_out docstring)
                if l <= t.stop:
                    fill = apply_slippage(t.stop, "sell")
                    proceeds = fill * t.shares_open - COMMISSION_PER_TRADE
                    cash += proceeds
                    t.realized_pnl += proceeds - t.entry_px * t.shares_open
                    t.shares_open = 0.0
                    t.exit_date, t.reason = sdate, ("BE_STOP" if t.scaled else "ATR_TRAIL_STOP")
                    res.trades.append(t)
                    del open_pos[sym]
                    continue

                trail_mult = bt.ATR_TRAIL_MULT

                if (breakout_scale_out is not None and not t.scaled
                        and t.target is not None and h >= t.target):
                    scale_price = t.target
                    qty = round(t.shares_total * SCALE_FRACTION, 6)
                    fill = apply_slippage(scale_price, "sell")
                    proceeds = fill * qty - COMMISSION_PER_TRADE
                    cash += proceeds
                    t.realized_pnl += proceeds - t.entry_px * qty
                    t.shares_open = round(t.shares_open - qty, 6)
                    t.scaled = True
                    t.stop = t.entry_px    # breakeven on the remainder, $0 risk

                if (pyramid_config is not None and not t.scaled
                        and t.target is not None and h >= t.target):
                    add_raw = t.shares_total * pyramid_config["add_fraction"]
                    add_shares = add_raw if fractional_shares else math.floor(add_raw)
                    if add_shares > 0:
                        add_price = apply_slippage(t.target, "buy")
                        add_cost = add_shares * add_price + COMMISSION_PER_TRADE
                        buying_power = cash + _mark(todays) * (margin_multiplier - 1.0)
                        if add_cost <= buying_power:
                            cash -= add_cost
                            original_entry_px = t.entry_px   # "1st tranche" breakeven level
                            new_total = t.shares_open + add_shares
                            # Blended cost basis so a LATER exit's realized_pnl
                            # is computed correctly across both tranches — see
                            # pyramid_config's MODELING SIMPLIFICATION note.
                            t.entry_px = ((t.entry_px * t.shares_open + add_price * add_shares)
                                         / new_total)
                            t.shares_open = new_total
                            t.shares_total = new_total
                            t.stop = max(t.stop, original_entry_px)
                    t.scaled = True   # pyramid opportunity used (or attempted) once, never again

                if breakout_scale_out is not None and t.scaled:
                    trail_mult = breakout_scale_out["trail_atr_mult_after_scale"]

                atr_today = float(row["ATR14"])
                if atr_today == atr_today and t.shares_open > 0:
                    t.high_water = max(t.high_water, h)
                    new_stop = t.high_water - trail_mult * atr_today
                    if new_stop > t.stop:
                        t.stop = new_stop

            # ---- Feature: 5-day (etc.) stagnant-time exit — checked AFTER
            # the style-specific stop/target/scale logic above, so a
            # position that already exited this same day is skipped (not
            # in open_pos any more) rather than double-counted.
            if sym in open_pos and max_hold_days_stagnant is not None:
                days_held = (sdate - t.entry_date).days
                if days_held >= max_hold_days_stagnant and t.r_unit > 0:
                    current_close = float(row["Close"])
                    unrealized_r = (current_close - t.entry_px) / t.r_unit
                    if unrealized_r < stagnant_min_progress_r:
                        fill = apply_slippage(current_close, "sell")
                        proceeds = fill * t.shares_open - COMMISSION_PER_TRADE
                        cash += proceeds
                        t.realized_pnl += proceeds - t.entry_px * t.shares_open
                        t.shares_open = 0.0
                        t.exit_date, t.reason = sdate, "STAGNANT_TIME_EXIT"
                        res.trades.append(t)
                        del open_pos[sym]

        # ---- phase 2: fill yesterday's queued entries at TODAY's open —
        # FEATURE 1: asymmetric risk_pct_mr/risk_pct_mom sizing (matches the
        # Original Baseline's own split by default; see the risk_pct_mr/mom
        # docstring above).
        for sym, style, basis in pending_entries:
            if sym in open_pos or len(open_pos) >= max_concurrent:
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
            if regime_filter is not None:
                in_bull = sdate in regime_aggressive_dates
                pair = regime_filter["aggressive"] if in_bull else regime_filter["base"]
                risk_pct = pair[0] if style == "MR" else pair[1]
                # ADX-conviction tiering: MOM (BREAKOUT) only, and only in the
                # bull/aggressive regime — a DIP/MR entry's ADX is always
                # < ADX_RANGE_MAX (20) by construction (that's the ranging-
                # regime eligibility rule), so it can never reach a >35/>45
                # conviction threshold; these tiers are structurally
                # unreachable for MR and are not applied to it.
                tiers = regime_filter.get("conviction_tiers_mom")
                if tiers and in_bull and style == "MOM" and basis is not None:
                    entry_adx = basis.get("adx")
                    if entry_adx is not None:
                        for threshold, multiplier in tiers:   # highest threshold first
                            if entry_adx > threshold:
                                risk_pct = pair[1] * multiplier
                                break
            else:
                risk_pct = risk_pct_mr if style == "MR" else risk_pct_mom
            r_unit = fill - stop
            raw_shares = (eq * risk_pct) / r_unit
            shares = raw_shares if fractional_shares else math.floor(raw_shares)
            if shares <= 0:
                stats.zero_share_signals += 1
                continue
            cost = shares * fill + COMMISSION_PER_TRADE
            buying_power = cash + eq * (margin_multiplier - 1.0)
            if cost > buying_power:
                continue
            cash -= cost
            realized_risk_pct = (shares * r_unit) / eq
            (stats.realized_risk_pct_mr if style == "MR"
            else stats.realized_risk_pct_mom).append(realized_risk_pct)
            if style == "MR":
                target = fill + r_unit * TARGET_R
            elif breakout_scale_out is not None:
                # Tranche 1's FIXED exit level, set once at entry off the
                # ATR AT ENTRY (basis["atr"]) — not recomputed daily, matching
                # "fixed +1.5x ATR gain." Reuses HybridTrade.target/.scaled
                # (documented "MR only" elsewhere in this file) since MOM
                # never otherwise sets them — see run_risk_managed_backtest's
                # breakout_scale_out docstring.
                target = fill + breakout_scale_out["tranche_atr_mult"] * basis["atr"]
            elif pyramid_config is not None:
                # Pyramid-add trigger price, FIXED at entry off the ATR AT
                # ENTRY — see pyramid_config's docstring. Same field reuse as
                # breakout_scale_out, mutually exclusive with it (enforced
                # above).
                target = fill + pyramid_config["trigger_atr_mult"] * basis["atr"]
            else:
                target = None
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
                elif (breakout_gate_symbols and sym in breakout_gate_symbols
                     and sdate not in regime_aggressive_dates):
                    # This symbol's BREAKOUT signals are confined to bull-
                    # regime days only (breakout_gate_symbols) — a normally-
                    # qualifying setup outside the bull regime is suppressed,
                    # not queued. DIP/MR for this symbol is untouched.
                    style = None
                else:
                    # adx_v carried through so phase 2 can apply ADX-conviction
                    # risk tiering (regime_filter["conviction_tiers_mom"]) at
                    # fill time — the entry's own ADX reading, from the signal
                    # day, not looked up again at fill.
                    basis = {"high": float(row["High"]), "atr": atr_v, "adx": adx_v}
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

        today_equity = _mark(todays)
        equity_points.append((sdate, today_equity))
        cash_pct_points.append((sdate, cash / today_equity if today_equity > 0 else 0.0))
        open_count_points.append(len(open_pos))

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

    stats.max_concurrent_used = max_concurrent
    stats.avg_open_positions = (sum(open_count_points) / len(open_count_points)
                                if open_count_points else 0.0)
    if cash_pct_points:
        trending_dates = _compute_spy_trending_dates(start, end)
        stats.avg_cash_pct = sum(p for _, p in cash_pct_points) / len(cash_pct_points)
        trend_vals = [p for d, p in cash_pct_points if d in trending_dates]
        non_trend_vals = [p for d, p in cash_pct_points if d not in trending_dates]
        stats.avg_cash_pct_trending = (sum(trend_vals) / len(trend_vals)
                                       if trend_vals else float("nan"))
        stats.avg_cash_pct_non_trending = (sum(non_trend_vals) / len(non_trend_vals)
                                           if non_trend_vals else float("nan"))

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


AGGRESSIVE_RISK_TIERS: list[tuple[str, float, float]] = [
    # (label, risk_pct_mr [DIP], risk_pct_mom [BREAKOUT])
    ("Tier 1: 0.50% BREAKOUT / 1.00% DIP", 0.0100, 0.0050),
    ("Tier 2: 0.75% BREAKOUT / 1.50% DIP", 0.0150, 0.0075),
    ("Tier 3: 1.00% BREAKOUT / 2.00% DIP", 0.0200, 0.0100),
]


def print_aggressive_risk_sweep(results: list[tuple[str, dict]], capital: float) -> None:
    width = 90
    print("=" * width)
    print("AGGRESSIVE LINEAR RISK SCALING SWEEP")
    print(f"${capital:,.0f} starting capital, Mixed universe (30 symbols), "
         f"MAX_CONCURRENT={MAX_CONCURRENT}, {START} -> {END}")
    print("Whole-share sizing, FOMC blackout active, circuit breaker disabled "
         "(daily-bar proxy over-blocks — established finding)")
    print(f"Reference — current production tiers: DIP {RISK_PCT_MR*100:.3f}% / "
         f"BREAKOUT {RISK_PCT_MOM*100:.3f}%")
    print("=" * width)
    header = (f"{'Tier':<38}{'Total Ret':>12}{'CAGR':>10}{'Max DD':>10}"
             f"{'Sharpe':>9}{'Trades':>9}")
    print(header)
    print("-" * width)
    for label, m in results:
        print(f"{label:<38}{m['total_return_pct']:>+11.2f}%{m['ann_return_pct']:>+9.2f}%"
             f"{m['max_dd_pct']:>9.2f}%{m['sharpe']:>9.3f}{m['n_trades']:>9}")
    print("=" * width)


def run_aggressive_risk_sweep(capital: float = 100_000.0) -> list[tuple[str, dict]]:
    """
    Parameter sweep: three "aggressive linear risk scaling" tiers of
    risk_pct_mr (DIP) / risk_pct_mom (BREAKOUT), each run through
    run_risk_managed_backtest() with the production config otherwise
    unchanged (FOMC blackout active, circuit breaker disabled per the
    established daily-bar over-blocking finding, whole-share sizing —
    fractional_shares defaults to False), full 2022-present window.

    Uses capital=$100,000 by default, NOT this file's current
    STARTING_CAPITAL module constant (currently $1,500, left over from an
    earlier small-account diagnostic task) — deliberately, so this sweep
    isolates the effect of risk_pct scaling alone rather than being
    confounded by the whole-share-rounding/zero-share-skip effects a
    small account introduces (already characterized separately).
    """
    symbols = bt.UNIVERSES[UNIVERSE_NAME]
    results: list[tuple[str, dict]] = []
    for label, risk_pct_mr, risk_pct_mom in AGGRESSIVE_RISK_TIERS:
        print(f"{label}: {START} -> {END} ...", flush=True)
        res, _stats = run_risk_managed_backtest(
            symbols, START, END, capital, use_circuit_breaker=False,
            risk_pct_mr=risk_pct_mr, risk_pct_mom=risk_pct_mom,
            fractional_shares=False)
        results.append((label, compute_metrics(res, capital)))

    print_aggressive_risk_sweep(results, capital)
    return results


def print_universe_expansion_comparison(baseline_metrics: dict, tmt_metrics: dict,
                                        capital: float) -> None:
    width = 78
    print("=" * width)
    print("30-SYMBOL BASELINE (Mixed) vs 100-SYMBOL TMT-HEAVY UNIVERSE")
    print(f"${capital:,.0f} starting capital, risk UNCHANGED (DIP {RISK_PCT_MR*100:.3f}% / "
         f"BREAKOUT {RISK_PCT_MOM*100:.3f}%), {START} -> {END}")
    print("=" * width)
    print(f"{'Metric':<28}{'30-Symbol Baseline':>24}{'100-Symbol TMT-Heavy':>26}")
    print("-" * width)
    print(f"{'Total Return (%)':<28}{baseline_metrics['total_return_pct']:>+23.2f}%"
         f"{tmt_metrics['total_return_pct']:>+25.2f}%")
    print(f"{'Max Drawdown (%)':<28}{baseline_metrics['max_dd_pct']:>23.2f}%"
         f"{tmt_metrics['max_dd_pct']:>25.2f}%")
    print(f"{'Sharpe Ratio':<28}{baseline_metrics['sharpe']:>24.3f}"
         f"{tmt_metrics['sharpe']:>26.3f}")
    print(f"{'Total Trades Executed':<28}{baseline_metrics['n_trades']:>24}"
         f"{tmt_metrics['n_trades']:>26}")
    print("=" * width)


def run_universe_expansion_comparison(capital: float = 100_000.0) -> None:
    """
    Compares the original 30-symbol Mixed universe against the new
    100-symbol TMT-heavy universe (backtest.TMT_HEAVY_UNIVERSE_100, 62%
    Technology/Media/Telecom), same production risk config held
    UNCHANGED (RISK_PCT_MR/RISK_PCT_MOM, FOMC blackout active, circuit
    breaker disabled per the established daily-bar over-blocking finding,
    whole-share sizing), same $capital, same 2022-present window — the
    only variable is which symbols are in the tradeable universe.
    """
    print(f"30-symbol baseline (Mixed): {START} -> {END} ...", flush=True)
    baseline_res, _ = run_risk_managed_backtest(
        bt.UNIVERSES["Mixed"], START, END, capital, use_circuit_breaker=False,
        risk_pct_mr=RISK_PCT_MR, risk_pct_mom=RISK_PCT_MOM,
        fractional_shares=False)

    print(f"100-symbol TMT-heavy universe: {START} -> {END} ...", flush=True)
    tmt_res, _ = run_risk_managed_backtest(
        bt.UNIVERSES["TMT100"], START, END, capital, use_circuit_breaker=False,
        risk_pct_mr=RISK_PCT_MR, risk_pct_mom=RISK_PCT_MOM,
        fractional_shares=False)

    baseline_metrics = compute_metrics(baseline_res, capital)
    tmt_metrics = compute_metrics(tmt_res, capital)

    print_universe_expansion_comparison(baseline_metrics, tmt_metrics, capital)


CONCURRENCY_TIERS: list[int] = [6, 8, 10]


def print_concurrency_sweep(results: list[tuple[int, dict, BlockStats]],
                            baseline_metrics: dict, baseline_stats: BlockStats,
                            capital: float) -> None:
    width = 100
    print("=" * width)
    print("CAPITAL VELOCITY SWEEP — MAX_CONCURRENT at 4 (baseline), 6, 8, 10")
    print(f"${capital:,.0f} starting capital, Mixed universe (30 symbols), risk UNCHANGED "
         f"(DIP {RISK_PCT_MR*100:.3f}% / BREAKOUT {RISK_PCT_MOM*100:.3f}%), {START} -> {END}")
    print("Idle-cash % is the mean fraction of equity NOT deployed in an open position; "
         "'trending' = SPY close > its own 200-day SMA that day")
    print("=" * width)
    header = (f"{'MAX_CONCURRENT':<16}{'Total Ret':>11}{'Max DD':>9}{'Sharpe':>8}"
             f"{'Trades':>8}{'Avg Open':>10}{'Idle$ All':>10}{'Idle$ Trend':>12}"
             f"{'Idle$ Non-T':>12}")
    print(header)
    print("-" * width)

    def _row(tag: str, m: dict, s: BlockStats):
        tr = "n/a" if s.avg_cash_pct_trending != s.avg_cash_pct_trending else f"{s.avg_cash_pct_trending*100:.1f}%"
        nt = "n/a" if s.avg_cash_pct_non_trending != s.avg_cash_pct_non_trending else f"{s.avg_cash_pct_non_trending*100:.1f}%"
        print(f"{tag:<16}{m['total_return_pct']:>+10.2f}%{m['max_dd_pct']:>8.2f}%"
             f"{m['sharpe']:>8.3f}{m['n_trades']:>8}{s.avg_open_positions:>10.2f}"
             f"{s.avg_cash_pct*100:>9.1f}%{tr:>12}{nt:>12}")

    _row(f"4 (baseline)", baseline_metrics, baseline_stats)
    for mc, m, s in results:
        _row(str(mc), m, s)
    print("=" * width)


def run_concurrency_sweep(capital: float = 100_000.0) -> list[tuple[int, dict, BlockStats]]:
    """
    Capital-velocity sweep: MAX_CONCURRENT at 6, 8, and 10, against the
    MAX_CONCURRENT=4 baseline, with risk parameters and universe held
    UNCHANGED (RISK_PCT_MR/RISK_PCT_MOM, Mixed 30-symbol universe, FOMC
    blackout active, circuit breaker disabled per the established
    daily-bar over-blocking finding, whole-share sizing). Full
    2022-present window, same $capital throughout.

    Reports, per tier: Total Return, Max Drawdown, Sharpe, trade count,
    average concurrent open positions (vs. the cap), and average idle-cash
    percentage of equity — overall, and split into SPY-trending vs.
    non-trending days (see _compute_spy_trending_dates()) to answer
    specifically whether loosening the concurrency cap reduces unused
    cash drag during trending regimes, or whether the extra slots simply
    go unfilled.
    """
    symbols = bt.UNIVERSES["Mixed"]

    print(f"Baseline (MAX_CONCURRENT=4): {START} -> {END} ...", flush=True)
    baseline_res, baseline_stats = run_risk_managed_backtest(
        symbols, START, END, capital, use_circuit_breaker=False,
        risk_pct_mr=RISK_PCT_MR, risk_pct_mom=RISK_PCT_MOM,
        fractional_shares=False, max_concurrent=MAX_CONCURRENT)
    baseline_metrics = compute_metrics(baseline_res, capital)

    results: list[tuple[int, dict, BlockStats]] = []
    for mc in CONCURRENCY_TIERS:
        print(f"MAX_CONCURRENT={mc}: {START} -> {END} ...", flush=True)
        res, stats = run_risk_managed_backtest(
            symbols, START, END, capital, use_circuit_breaker=False,
            risk_pct_mr=RISK_PCT_MR, risk_pct_mom=RISK_PCT_MOM,
            fractional_shares=False, max_concurrent=mc)
        results.append((mc, compute_metrics(res, capital), stats))

    print_concurrency_sweep(results, baseline_metrics, baseline_stats, capital)
    return results


def _time_to_reach(equity: pd.Series, target: float) -> tuple[str, dt.date | None]:
    """
    Finds the first date the ACTUAL simulated equity curve closes at or
    above `target` — not a CAGR-based extrapolation, which would smooth
    over the real path's drawdowns and trade-timing variance and could
    over- or understate how long this specific run actually took. Returns
    (human-readable description, the crossing date — or None if `target`
    is never reached within the simulated window at all).
    """
    if equity.empty:
        return "n/a (no equity data)", None
    start_date = equity.index[0]
    hits = equity[equity >= target]
    if hits.empty:
        elapsed_days = (equity.index[-1] - start_date).days
        return (f"NOT REACHED within the simulated window (final equity "
               f"${equity.iloc[-1]:,.2f} after {elapsed_days} days / "
               f"{elapsed_days / 365.25:.2f} years simulated)"), None
    hit_date = hits.index[0]
    elapsed_days = (hit_date - start_date).days
    years = elapsed_days / 365.25
    months = elapsed_days / 30.4368
    return (f"{hit_date} — {elapsed_days} days ({months:.1f} months / "
           f"{years:.2f} years) after {start_date}"), hit_date


def run_sprint_phase_test(capital: float = 1_900.0, target: float = 5_000.0) -> dict:
    """
    Targeted single-configuration backtest: the "Sprint Phase" combination
    of Tier-1 aggressive risk (1.00% DIP / 0.50% BREAKOUT — the one tier
    from the earlier aggressive-risk sweep that beat the production
    baseline on both Sharpe and total return, at the cost of higher
    drawdown), MAX_CONCURRENT=6 (the best-performing tier from the earlier
    concurrency sweep), fractional-share sizing, and the standard
    30-symbol Mixed universe. FOMC blackout active; circuit breaker
    disabled per the established daily-bar over-blocking finding. Full
    2022-present window, $capital starting capital.

    Also reports the time to grow from $capital to $target, read directly
    off the equity curve's first crossing (see _time_to_reach()) — a real
    simulated result, not a smoothed CAGR projection.
    """
    symbols = bt.UNIVERSES["Mixed"]
    risk_pct_mom = 0.0050   # 0.50% BREAKOUT
    risk_pct_mr = 0.0100    # 1.00% DIP

    print(f"Sprint Phase config: {START} -> {END} ...", flush=True)
    res, stats = run_risk_managed_backtest(
        symbols, START, END, capital, use_circuit_breaker=False,
        risk_pct_mr=risk_pct_mr, risk_pct_mom=risk_pct_mom,
        fractional_shares=True, max_concurrent=6)
    metrics = compute_metrics(res, capital)
    time_desc, _hit_date = _time_to_reach(res.equity, target)

    width = 78
    print("=" * width)
    print("SPRINT PHASE CONFIGURATION — TARGETED BACKTEST")
    print(f"${capital:,.0f} starting capital, Mixed universe (30 symbols), "
         f"MAX_CONCURRENT=6, fractional shares, {START} -> {END}")
    print(f"Risk: DIP {risk_pct_mr*100:.2f}% / BREAKOUT {risk_pct_mom*100:.2f}%  "
         f"(FOMC blackout active, circuit breaker disabled)")
    print("=" * width)
    print(f"{'Total Return':<24}{metrics['total_return_pct']:>+10.2f}%")
    print(f"{'CAGR (Annualized)':<24}{metrics['ann_return_pct']:>+10.2f}%")
    print(f"{'Max Drawdown':<24}{metrics['max_dd_pct']:>10.2f}%")
    print(f"{'Sharpe Ratio':<24}{metrics['sharpe']:>10.3f}")
    print(f"{'Total Trades':<24}{metrics['n_trades']:>10}")
    print("-" * width)
    print(f"Time to grow ${capital:,.0f} -> ${target:,.0f}: {time_desc}")
    print("=" * width)
    return metrics


def print_velocity_modifications(results: list[tuple[str, dict]], capital: float) -> None:
    width = 96
    print("=" * width)
    print("VELOCITY MODIFICATIONS — 30-symbol Mixed universe, MAX_CONCURRENT=6")
    print(f"${capital:,.0f} starting capital, {START} -> {END}, FOMC blackout active, "
         f"circuit breaker disabled")
    print("Goal: CAGR > 20% while Max Drawdown stays under 15%")
    print("=" * width)
    header = f"{'Test':<44}{'Total Ret':>11}{'CAGR':>9}{'Max DD':>9}{'Sharpe':>8}{'Trades':>8}"
    print(header)
    print("-" * width)
    for label, m in results:
        goal_hit = " <=20%CAGR/<15%DD" if (m['ann_return_pct'] > 20 and m['max_dd_pct'] < 15) else ""
        print(f"{label:<44}{m['total_return_pct']:>+10.2f}%{m['ann_return_pct']:>+8.2f}%"
             f"{m['max_dd_pct']:>8.2f}%{m['sharpe']:>8.3f}{m['n_trades']:>8}")
    print("=" * width)


def run_velocity_modifications(capital: float = 100_000.0) -> list[tuple[str, dict]]:
    """
    Tests three proposed "velocity" mechanisms against a fresh
    MAX_CONCURRENT=6 baseline, all in the SAME run (so all four rows share
    one data-fetch generation and are internally comparable, even though
    yfinance's own historical data can drift slightly between sessions —
    see this feature's regression check for why that matters).

    1. A 5-day stagnant-time exit (max_hold_days_stagnant=5,
       stagnant_min_progress_r=0.5 default): force-exits any position held
       >=5 calendar days that hasn't reached +0.5R of unrealized gain.
       Base production risk (RISK_PCT_MR/RISK_PCT_MOM) held constant so
       this isolates the mechanism's own effect.
    2. ATR trailing stop on BREAKOUT "instead of fixed targets" — ALREADY
       this engine's existing, unconditional BREAKOUT behavior (2.0x
       ATR14 trailing stop, target=None for MOM style; see
       run_risk_managed_backtest's phase-1 code). No code change applies
       here; this row reproduces the plain MAX_CONCURRENT=6 baseline
       exactly, flagged rather than silently re-labeled as something new.
    3. A 50-day-SMA market regime filter: 1.25% DIP / 0.75% BREAKOUT risk
       when SPY closes above its own 50-day SMA, reverting to 0.50% DIP /
       0.25% BREAKOUT otherwise (regime_filter param).

    All four rows: Mixed universe (30 symbols), MAX_CONCURRENT=6, FOMC
    blackout active, circuit breaker disabled (established daily-bar
    over-blocking finding), whole-share sizing, full 2022-present window.
    """
    symbols = bt.UNIVERSES["Mixed"]
    results: list[tuple[str, dict]] = []

    print(f"Baseline (MAX_CONCURRENT=6, no new features): {START} -> {END} ...", flush=True)
    res, _ = run_risk_managed_backtest(
        symbols, START, END, capital, use_circuit_breaker=False,
        risk_pct_mr=RISK_PCT_MR, risk_pct_mom=RISK_PCT_MOM,
        fractional_shares=False, max_concurrent=6)
    results.append(("Baseline (MC=6, production risk)", compute_metrics(res, capital)))

    print(f"Test 1: 5-day stagnant-time exit: {START} -> {END} ...", flush=True)
    res, _ = run_risk_managed_backtest(
        symbols, START, END, capital, use_circuit_breaker=False,
        risk_pct_mr=RISK_PCT_MR, risk_pct_mom=RISK_PCT_MOM,
        fractional_shares=False, max_concurrent=6,
        max_hold_days_stagnant=5, stagnant_min_progress_r=0.5)
    results.append(("Test 1: 5-day stagnant-time exit", compute_metrics(res, capital)))

    print(f"Test 2: BREAKOUT ATR trailing (already default): {START} -> {END} ...", flush=True)
    res, _ = run_risk_managed_backtest(
        symbols, START, END, capital, use_circuit_breaker=False,
        risk_pct_mr=RISK_PCT_MR, risk_pct_mom=RISK_PCT_MOM,
        fractional_shares=False, max_concurrent=6)
    results.append(("Test 2: BREAKOUT ATR trail (no change — already default)",
                    compute_metrics(res, capital)))

    print(f"Test 3: 50-day SMA regime filter: {START} -> {END} ...", flush=True)
    res, _ = run_risk_managed_backtest(
        symbols, START, END, capital, use_circuit_breaker=False,
        fractional_shares=False, max_concurrent=6,
        regime_filter={"sma_window": 50, "aggressive": (0.0125, 0.0075),
                      "base": (0.0050, 0.0025)})
    results.append(("Test 3: 50-day SMA regime filter", compute_metrics(res, capital)))

    print_velocity_modifications(results, capital)
    return results


def run_tier1_regime_combo(capital: float = 100_000.0) -> dict:
    """
    Iterates on Test 3 (the 50-day SMA regime filter — the one velocity
    mechanism that improved on baseline without adding drawdown) by
    replacing its two risk pairs with the Tier-1 aggressive risk
    (bull regime: 1.00% DIP / 0.50% BREAKOUT, the best-performing tier
    from the earlier aggressive-risk sweep) and the actual production
    baseline (bear/chop regime: RISK_PCT_MR/RISK_PCT_MOM = 0.625% DIP /
    0.25% BREAKOUT), rather than Test 3's own milder 1.25%/0.50% pair.
    MAX_CONCURRENT=6, Mixed 30-symbol universe, FOMC blackout active,
    circuit breaker disabled (established daily-bar over-blocking
    finding), whole-share sizing, full 2022-present window.
    """
    symbols = bt.UNIVERSES["Mixed"]
    print(f"Tier-1/Regime combo: {START} -> {END} ...", flush=True)
    res, _ = run_risk_managed_backtest(
        symbols, START, END, capital, use_circuit_breaker=False,
        fractional_shares=False, max_concurrent=6,
        regime_filter={"sma_window": 50,
                      "aggressive": (0.0100, 0.0050),   # bull: 1.00% DIP / 0.50% BREAKOUT
                      "base": (RISK_PCT_MR, RISK_PCT_MOM)})  # bear/chop: 0.625% DIP / 0.25% BREAKOUT
    metrics = compute_metrics(res, capital)

    goal_hit = metrics["ann_return_pct"] > 20 and metrics["max_dd_pct"] < 15
    width = 78
    print("=" * width)
    print("TIER-1 RISK x 50-DAY SMA REGIME FILTER — COMBINED TEST")
    print(f"${capital:,.0f} starting capital, Mixed universe (30 symbols), "
         f"MAX_CONCURRENT=6, {START} -> {END}")
    print("Bull (SPY > 50d SMA): DIP 1.00% / BREAKOUT 0.50%   "
         "Bear/Chop (SPY <= 50d SMA): DIP 0.625% / BREAKOUT 0.25%")
    print("=" * width)
    print(f"{'Total Return':<24}{metrics['total_return_pct']:>+10.2f}%")
    print(f"{'CAGR (Annualized)':<24}{metrics['ann_return_pct']:>+10.2f}%")
    print(f"{'Max Drawdown':<24}{metrics['max_dd_pct']:>10.2f}%")
    print(f"{'Sharpe Ratio':<24}{metrics['sharpe']:>10.3f}")
    print(f"{'Total Trades':<24}{metrics['n_trades']:>10}")
    print("-" * width)
    print(f"Goal (CAGR>20% AND MaxDD<15%): {'MET' if goal_hit else 'NOT MET'}")
    print("=" * width)
    return metrics


def run_widened_regime_spread(capital: float = 100_000.0) -> dict:
    """
    Widens Test 3's regime spread: bull-regime risk raised from Test 3's
    1.25% DIP / 0.75% BREAKOUT to 1.50% DIP / 1.00% BREAKOUT; bear/chop
    risk held at Test 3's own 0.50% DIP / 0.25% BREAKOUT, unchanged (the
    prior combo's mistake was raising bear/chop risk above Test 3's level
    while LOWERING bull risk below it — this run only pushes the bull
    side further, per the explicit request to widen the spread, not
    narrow it). Re-runs Test 3's own original config fresh in the same
    process/data-fetch generation for a true apples-to-apples comparison
    (yfinance's own historical data can drift slightly session to
    session — see this feature's earlier regression-check note).
    MAX_CONCURRENT=6, Mixed 30-symbol universe, FOMC blackout active,
    circuit breaker disabled, whole-share sizing, full 2022-present
    window.
    """
    symbols = bt.UNIVERSES["Mixed"]

    print(f"Test 3 baseline (re-run fresh for a fair comparison): {START} -> {END} ...",
         flush=True)
    test3_res, _ = run_risk_managed_backtest(
        symbols, START, END, capital, use_circuit_breaker=False,
        fractional_shares=False, max_concurrent=6,
        regime_filter={"sma_window": 50, "aggressive": (0.0125, 0.0075),
                      "base": (0.0050, 0.0025)})
    test3_metrics = compute_metrics(test3_res, capital)

    print(f"Widened spread (1.50%/1.00% bull, 0.50%/0.25% bear/chop): "
         f"{START} -> {END} ...", flush=True)
    wide_res, _ = run_risk_managed_backtest(
        symbols, START, END, capital, use_circuit_breaker=False,
        fractional_shares=False, max_concurrent=6,
        regime_filter={"sma_window": 50,
                      "aggressive": (0.0150, 0.0100),   # bull: 1.50% DIP / 1.00% BREAKOUT
                      "base": (0.0050, 0.0025)})        # bear/chop: 0.50% DIP / 0.25% BREAKOUT
    wide_metrics = compute_metrics(wide_res, capital)

    goal_hit = wide_metrics["ann_return_pct"] > 20 and wide_metrics["max_dd_pct"] < 15
    width = 84
    print("=" * width)
    print("WIDENED REGIME SPREAD vs TEST 3 BASELINE")
    print(f"${capital:,.0f} starting capital, Mixed universe (30 symbols), "
         f"MAX_CONCURRENT=6, {START} -> {END}")
    print("Test 3:  bull 1.25% DIP/0.75% BREAKOUT | bear/chop 0.50% DIP/0.25% BREAKOUT")
    print("Widened: bull 1.50% DIP/1.00% BREAKOUT | bear/chop 0.50% DIP/0.25% BREAKOUT")
    print("=" * width)
    print(f"{'Metric':<24}{'Test 3 Baseline':>22}{'Widened Spread':>22}")
    print("-" * width)
    print(f"{'Total Return':<24}{test3_metrics['total_return_pct']:>+21.2f}%"
         f"{wide_metrics['total_return_pct']:>+21.2f}%")
    print(f"{'CAGR (Annualized)':<24}{test3_metrics['ann_return_pct']:>+21.2f}%"
         f"{wide_metrics['ann_return_pct']:>+21.2f}%")
    print(f"{'Max Drawdown':<24}{test3_metrics['max_dd_pct']:>21.2f}%"
         f"{wide_metrics['max_dd_pct']:>21.2f}%")
    print(f"{'Sharpe Ratio':<24}{test3_metrics['sharpe']:>22.3f}"
         f"{wide_metrics['sharpe']:>22.3f}")
    print(f"{'Total Trades':<24}{test3_metrics['n_trades']:>22}{wide_metrics['n_trades']:>22}")
    print("-" * width)
    print(f"Goal (CAGR>20% AND MaxDD<15%) — Widened Spread: "
         f"{'MET' if goal_hit else 'NOT MET'}")
    print("=" * width)
    return wide_metrics


def run_strategy_a_adx_conviction(capital: float = 1_900.0,
                                  targets: tuple[float, ...] = (4_000.0, 5_000.0)
                                  ) -> dict:
    """
    Strategy A — ADX-Scaled Conviction Sizing: layers ADX-based conviction
    tiers on top of the 50-day SMA regime filter's bull-regime risk.

    IMPORTANT STRUCTURAL NOTE, confirmed from this engine's own signal-
    routing rule (see the candidate-scan phase 3 code): a DIP/MR entry's
    ADX is ALWAYS < ADX_RANGE_MAX (20) — that IS the ranging-regime
    eligibility test. A ">35" or ">45" ADX conviction tier is therefore
    structurally unreachable for DIP trades; conviction tiering only ever
    applies to BREAKOUT (MOM) trades in practice, which typically already
    print ADX comfortably above 25 (BREAKOUT eligibility requires ADX
    above the symbol's own dynamic 80th-percentile threshold, floored at
    that level). DIP trades in the bull regime always size at the
    standard 1.25% tier here — never 1.875% or 2.50%, no matter how the
    request's tiers are written, because they can never qualify.

    Bull regime (SPY > 50-day SMA):
      Standard  (any BREAKOUT ADX not in a higher tier): 0.75% BREAKOUT / 1.25% DIP
      High conviction  (BREAKOUT ADX > 35): 1.5x -> 1.125% BREAKOUT (DIP unreachable)
      Extreme conviction (BREAKOUT ADX > 45): 2.0x -> 1.50% BREAKOUT (DIP unreachable)
    Bear/chop regime (SPY <= 50-day SMA): flat 0.25% BREAKOUT / 0.50% DIP,
    regardless of ADX.

    MAX_CONCURRENT=6, Mixed 30-symbol universe, FOMC blackout active,
    circuit breaker disabled (established daily-bar over-blocking
    finding), whole-share sizing, full 2022-present window, $capital
    starting capital. Reports time-to-target for each value in `targets`
    read directly off the actual simulated equity curve's first crossing
    (see _time_to_reach()) — not a CAGR extrapolation.
    """
    symbols = bt.UNIVERSES["Mixed"]
    print(f"Strategy A (ADX-scaled conviction sizing): {START} -> {END} ...", flush=True)
    res, stats = run_risk_managed_backtest(
        symbols, START, END, capital, use_circuit_breaker=False,
        fractional_shares=False, max_concurrent=6,
        regime_filter={
            "sma_window": 50,
            "aggressive": (0.0125, 0.0075),   # bull standard: 1.25% DIP / 0.75% BREAKOUT
            "base": (0.0050, 0.0025),         # bear/chop: 0.50% DIP / 0.25% BREAKOUT
            "conviction_tiers_mom": [(45, 2.0), (35, 1.5)],  # (ADX threshold, multiplier)
        })
    metrics = compute_metrics(res, capital)
    goal_hit = metrics["ann_return_pct"] > 20 and metrics["max_dd_pct"] < 15

    width = 82
    print("=" * width)
    print("STRATEGY A — ADX-SCALED CONVICTION SIZING")
    print(f"${capital:,.0f} starting capital, Mixed universe (30 symbols), "
         f"MAX_CONCURRENT=6, {START} -> {END}")
    print("Bull: standard 1.25% DIP/0.75% BRK | ADX>35 -> 1.5x BRK | ADX>45 -> 2.0x BRK")
    print("      (DIP conviction tiers unreachable — DIP entries always have ADX<20)")
    print("Bear/Chop: flat 0.50% DIP / 0.25% BREAKOUT")
    print("=" * width)
    print(f"{'Total Return':<24}{metrics['total_return_pct']:>+10.2f}%")
    print(f"{'CAGR (Annualized)':<24}{metrics['ann_return_pct']:>+10.2f}%")
    print(f"{'Max Drawdown':<24}{metrics['max_dd_pct']:>10.2f}%")
    print(f"{'Sharpe Ratio':<24}{metrics['sharpe']:>10.3f}")
    print(f"{'Total Trades':<24}{metrics['n_trades']:>10}")
    print("-" * width)
    for target in targets:
        time_desc, _ = _time_to_reach(res.equity, target)
        print(f"Time to grow ${capital:,.0f} -> ${target:,.0f}: {time_desc}")
    print("-" * width)
    print(f"Goal (CAGR>20% AND MaxDD<15%): {'MET' if goal_hit else 'NOT MET'}")
    print("=" * width)
    return metrics


def run_strategy_b_scale_out(capital: float = 1_900.0,
                             targets: tuple[float, ...] = (4_000.0, 5_000.0)
                             ) -> dict:
    """
    Strategy B — Multi-Stage Scale-Out Execution: Test 3's regime
    parameters (1.25% DIP / 0.75% BREAKOUT bull, 0.50% DIP / 0.25%
    BREAKOUT bear/chop — no ADX conviction tiering this time, per the
    request), with BREAKOUT's exit mechanic replaced by a two-tranche
    scale-out (breakout_scale_out): tranche 1 (50% of the position) exits
    at a FIXED entry_px + 1.5x ATR-at-entry level, moving the remainder's
    stop to breakeven; tranche 2 (the remaining 50%) trails with a wide
    3.5x ATR stop from there until stopped out. DIP trades are entirely
    unaffected — this mechanic only touches MOM/BREAKOUT exits.

    MAX_CONCURRENT=6, Mixed 30-symbol universe, FOMC blackout active,
    circuit breaker disabled (established daily-bar over-blocking
    finding), whole-share sizing, full 2022-present window, $capital
    starting capital. Reports time-to-target for each value in `targets`
    off the actual simulated equity curve (see _time_to_reach()).
    """
    symbols = bt.UNIVERSES["Mixed"]
    print(f"Strategy B (multi-stage scale-out): {START} -> {END} ...", flush=True)
    res, stats = run_risk_managed_backtest(
        symbols, START, END, capital, use_circuit_breaker=False,
        fractional_shares=False, max_concurrent=6,
        regime_filter={"sma_window": 50,
                      "aggressive": (0.0125, 0.0075),   # bull: 1.25% DIP / 0.75% BREAKOUT
                      "base": (0.0050, 0.0025)},         # bear/chop: 0.50% DIP / 0.25% BREAKOUT
        breakout_scale_out={"tranche_atr_mult": 1.5, "trail_atr_mult_after_scale": 3.5})
    metrics = compute_metrics(res, capital)
    goal_hit = metrics["ann_return_pct"] > 20 and metrics["max_dd_pct"] < 15

    mom_trades = [t for t in res.trades if t.style == "MOM"]
    scaled_out = sum(1 for t in mom_trades if t.scaled)

    width = 82
    print("=" * width)
    print("STRATEGY B — MULTI-STAGE SCALE-OUT EXECUTION")
    print(f"${capital:,.0f} starting capital, Mixed universe (30 symbols), "
         f"MAX_CONCURRENT=6, {START} -> {END}")
    print("Regime: bull 1.25% DIP/0.75% BRK | bear/chop 0.50% DIP/0.25% BRK "
         "(Test 3 parameters, no ADX conviction tiers)")
    print("BREAKOUT exits: Tranche 1 (50%) @ +1.5x ATR-at-entry -> breakeven stop "
         "| Tranche 2 (50%) trails @ 3.5x ATR")
    print("=" * width)
    print(f"{'Total Return':<24}{metrics['total_return_pct']:>+10.2f}%")
    print(f"{'CAGR (Annualized)':<24}{metrics['ann_return_pct']:>+10.2f}%")
    print(f"{'Max Drawdown':<24}{metrics['max_dd_pct']:>10.2f}%")
    print(f"{'Sharpe Ratio':<24}{metrics['sharpe']:>10.3f}")
    print(f"{'Total Trades':<24}{metrics['n_trades']:>10}")
    print(f"{'BREAKOUT trades':<24}{len(mom_trades):>10}")
    print(f"{'  -> reached Tranche 1':<24}{scaled_out:>10}")
    print("-" * width)
    for target in targets:
        time_desc, _ = _time_to_reach(res.equity, target)
        print(f"Time to grow ${capital:,.0f} -> ${target:,.0f}: {time_desc}")
    print("-" * width)
    print(f"Goal (CAGR>20% AND MaxDD<15%): {'MET' if goal_hit else 'NOT MET'}")
    print("=" * width)
    return metrics


def run_strategy_c_leveraged_overlay(capital: float = 1_900.0,
                                     targets: tuple[float, ...] = (4_000.0, 5_000.0)
                                     ) -> dict:
    """
    Strategy C — Leveraged Benchmark Overlay: the Mixed 30-symbol universe
    plus QLD (2x Nasdaq) and SSO (2x S&P 500), under the Widened Regime
    Filter (bull: 1.50% DIP / 1.00% BREAKOUT; bear/chop: 0.50% DIP / 0.25%
    BREAKOUT — same as run_widened_regime_spread()'s "Widened Spread" row),
    with QLD/SSO's BREAKOUT signals additionally confined to bull-regime
    days only via breakout_gate_symbols — their DIP signals are
    unrestricted, same as any other symbol. Verified (not assumed): both
    tickers have full daily history back well before 2022.

    MAX_CONCURRENT=6, FOMC blackout active, circuit breaker disabled
    (established daily-bar over-blocking finding), whole-share sizing,
    full 2022-present window, $capital starting capital. Reports
    time-to-target for each value in `targets` off the actual simulated
    equity curve (see _time_to_reach()).
    """
    symbols = bt.UNIVERSES["Mixed"] + ["QLD", "SSO"]
    print(f"Strategy C (leveraged benchmark overlay, {len(symbols)} symbols): "
         f"{START} -> {END} ...", flush=True)
    res, stats = run_risk_managed_backtest(
        symbols, START, END, capital, use_circuit_breaker=False,
        fractional_shares=False, max_concurrent=6,
        regime_filter={"sma_window": 50,
                      "aggressive": (0.0150, 0.0100),   # bull: 1.50% DIP / 1.00% BREAKOUT
                      "base": (0.0050, 0.0025)},         # bear/chop: 0.50% DIP / 0.25% BREAKOUT
        breakout_gate_symbols={"QLD", "SSO"})
    metrics = compute_metrics(res, capital)
    goal_hit = metrics["ann_return_pct"] > 20 and metrics["max_dd_pct"] < 15

    lev_trades = [t for t in res.trades if t.symbol in ("QLD", "SSO")]
    lev_mom = sum(1 for t in lev_trades if t.style == "MOM")
    lev_mr = sum(1 for t in lev_trades if t.style == "MR")

    width = 82
    print("=" * width)
    print("STRATEGY C — LEVERAGED BENCHMARK OVERLAY")
    print(f"${capital:,.0f} starting capital, {len(symbols)}-symbol universe "
         f"(Mixed 30 + QLD + SSO), MAX_CONCURRENT=6, {START} -> {END}")
    print("Widened regime: bull 1.50% DIP/1.00% BRK | bear/chop 0.50% DIP/0.25% BRK")
    print("QLD/SSO: BREAKOUT signals gated to bull-regime days only; DIP unrestricted")
    print("=" * width)
    print(f"{'Total Return':<24}{metrics['total_return_pct']:>+10.2f}%")
    print(f"{'CAGR (Annualized)':<24}{metrics['ann_return_pct']:>+10.2f}%")
    print(f"{'Max Drawdown':<24}{metrics['max_dd_pct']:>10.2f}%")
    print(f"{'Sharpe Ratio':<24}{metrics['sharpe']:>10.3f}")
    print(f"{'Total Trades':<24}{metrics['n_trades']:>10}")
    print(f"{'  QLD/SSO trades':<24}{len(lev_trades):>10}  ({lev_mr} DIP, {lev_mom} BREAKOUT)")
    print("-" * width)
    for target in targets:
        time_desc, _ = _time_to_reach(res.equity, target)
        print(f"Time to grow ${capital:,.0f} -> ${target:,.0f}: {time_desc}")
    print("-" * width)
    print(f"Goal (CAGR>20% AND MaxDD<15%): {'MET' if goal_hit else 'NOT MET'}")
    print("=" * width)
    return metrics


EXTREME_SPRINT_UNIVERSE: list[str] = [
    "NVDA", "AMD", "TSLA", "PLTR", "SMCI", "AVGO", "META", "AMZN", "TQQQ", "SOXL",
]


def run_extreme_sprint_test(capital: float = 1_900.0, target: float = 4_000.0
                            ) -> dict:
    """
    Aggressive Sprint Phase test: a 10-symbol high-volatility TMT-momentum
    + 3x-leverage universe (verified — not assumed — to all have current,
    continuous daily data), regime-gated UNCONSTRAINED risk sizing (bull:
    3.00% DIP / 2.00% BREAKOUT; bear: 0.50% DIP / 0.50% BREAKOUT — both far
    beyond anything else tested in this file), and BREAKOUT pyramiding
    (pyramid_config: +1.0x ATR trigger, 50% add, breakeven stop on the
    original tranche — see that parameter's MODELING SIMPLIFICATION note
    on the single-stop/blended-cost-basis approximation this uses).
    MAX_CONCURRENT=6, fractional-share sizing, FOMC blackout active,
    circuit breaker disabled (established daily-bar over-blocking
    finding), full 2022-present window, $capital starting capital.

    Reports whether/when the equity curve first closes at or above
    `target`, read off the actual simulated path (see _time_to_reach()) —
    against the requested "1.5 years" reference point, not as a target
    this function enforces or guarantees.
    """
    symbols = EXTREME_SPRINT_UNIVERSE
    print(f"Extreme Sprint Phase test ({len(symbols)}-symbol high-vol/leverage "
         f"universe): {START} -> {END} ...", flush=True)
    res, stats = run_risk_managed_backtest(
        symbols, START, END, capital, use_circuit_breaker=False,
        fractional_shares=True, max_concurrent=6,
        regime_filter={"sma_window": 50,
                      "aggressive": (0.0300, 0.0200),   # bull: 3.00% DIP / 2.00% BREAKOUT
                      "base": (0.0050, 0.0050)},         # bear: 0.50% DIP / 0.50% BREAKOUT
        pyramid_config={"trigger_atr_mult": 1.0, "add_fraction": 0.5})
    metrics = compute_metrics(res, capital)

    mom_trades = [t for t in res.trades if t.style == "MOM"]
    pyramided = sum(1 for t in mom_trades if t.scaled)
    time_desc, hit_date = _time_to_reach(res.equity, target)

    width = 84
    print("=" * width)
    print("EXTREME SPRINT PHASE — 10-SYMBOL HIGH-VOLATILITY/LEVERAGE UNIVERSE")
    print(f"${capital:,.0f} starting capital, {len(symbols)} symbols "
         f"({', '.join(symbols)}), MAX_CONCURRENT=6, fractional sizing, "
         f"{START} -> {END}")
    print("Bull: 3.00% DIP / 2.00% BREAKOUT   Bear: 0.50% DIP / 0.50% BREAKOUT")
    print("BREAKOUT pyramiding: +1.0x ATR trigger, +50% size, breakeven stop on tranche 1")
    print("=" * width)
    print(f"{'Total Return':<24}{metrics['total_return_pct']:>+10.2f}%")
    print(f"{'CAGR (Annualized)':<24}{metrics['ann_return_pct']:>+10.2f}%")
    print(f"{'Max Drawdown':<24}{metrics['max_dd_pct']:>10.2f}%")
    print(f"{'Sharpe Ratio':<24}{metrics['sharpe']:>10.3f}")
    print(f"{'Total Trades':<24}{metrics['n_trades']:>10}")
    print(f"{'  BREAKOUT trades':<24}{len(mom_trades):>10}  ({pyramided} pyramided)")
    print("-" * width)
    print(f"Time to grow ${capital:,.0f} -> ${target:,.0f}: {time_desc}")
    print("=" * width)
    return metrics


def run_margin_leverage_test(capital: float = 1_900.0, target: float = 4_000.0,
                             margin_multiplier: float = 1.5) -> dict:
    """
    Aggressive Margin/Leverage test: pure position-scaling, isolating
    buying-power leverage from the universe/pyramiding confounds of the
    prior Extreme Sprint test — back to the standard 30-symbol Mixed
    universe (which the prior test's diagnosis showed matters: signal
    scarcity, not risk-per-trade size, was the real bottleneck on a
    narrow 10-symbol book). Unconstrained regime risk (bull: 3.50% DIP /
    2.00% BREAKOUT; bear: 0.50% DIP / 0.25% BREAKOUT), MAX_CONCURRENT=8,
    and `margin_multiplier`=1.5 (buying power = 1.5x equity, i.e. $2,850
    on $1,900 — see that parameter's docstring on run_risk_managed_backtest
    for exactly what it does and does NOT model, notably: no financing
    cost on the borrowed balance).

    Whole-share sizing (not fractional — not requested this time), FOMC
    blackout active, circuit breaker disabled (established daily-bar
    over-blocking finding), full 2022-present window, $capital starting
    capital. Reports whether/when the equity curve first closes at or
    above `target`, off the actual simulated path (see _time_to_reach()).
    """
    symbols = bt.UNIVERSES["Mixed"]
    print(f"Margin/Leverage test ({margin_multiplier}x buying power): "
         f"{START} -> {END} ...", flush=True)
    res, stats = run_risk_managed_backtest(
        symbols, START, END, capital, use_circuit_breaker=False,
        fractional_shares=False, max_concurrent=8,
        regime_filter={"sma_window": 50,
                      "aggressive": (0.0350, 0.0200),   # bull: 3.50% DIP / 2.00% BREAKOUT
                      "base": (0.0050, 0.0025)},         # bear: 0.50% DIP / 0.25% BREAKOUT
        margin_multiplier=margin_multiplier)
    metrics = compute_metrics(res, capital)
    time_desc, _ = _time_to_reach(res.equity, target)

    width = 82
    print("=" * width)
    print("MARGIN/LEVERAGE TEST — PURE POSITION SCALING")
    print(f"${capital:,.0f} starting capital ({margin_multiplier}x buying power = "
         f"${capital * margin_multiplier:,.0f}), Mixed universe (30 symbols), "
         f"MAX_CONCURRENT=8, {START} -> {END}")
    print("Bull: 3.50% DIP / 2.00% BREAKOUT   Bear: 0.50% DIP / 0.25% BREAKOUT")
    print("=" * width)
    print(f"{'Total Return':<24}{metrics['total_return_pct']:>+10.2f}%")
    print(f"{'CAGR (Annualized)':<24}{metrics['ann_return_pct']:>+10.2f}%")
    print(f"{'Max Drawdown':<24}{metrics['max_dd_pct']:>10.2f}%")
    print(f"{'Sharpe Ratio':<24}{metrics['sharpe']:>10.3f}")
    print(f"{'Total Trades':<24}{metrics['n_trades']:>10}")
    print(f"{'Avg Open Positions':<24}{stats.avg_open_positions:>10.2f}  (of {stats.max_concurrent_used} cap)")
    print("-" * width)
    print(f"Time to grow ${capital:,.0f} -> ${target:,.0f}: {time_desc}")
    print("=" * width)
    return metrics


def run_sp500_universe_test(capital: float = 1_900.0, target: float = 4_000.0
                            ) -> dict:
    """
    Dynamic universe screener test: the current S&P 500 constituents
    (data.sp500_symbols(), ~503 tickers as of the run date — includes a
    few dual-class-share companies, hence >500) as the tradeable universe,
    under the Widened Spread regime filter (bull: 1.50% DIP / 1.00%
    BREAKOUT; bear: 0.50% DIP / 0.25% BREAKOUT — same as
    run_widened_regime_spread()), MAX_CONCURRENT=6, fractional-share
    sizing. FOMC blackout active, circuit breaker disabled (established
    daily-bar over-blocking finding), full 2022-present window, $capital
    starting capital.

    SURVIVORSHIP CAVEAT, disclosed rather than silently assumed: this is
    TODAY's S&P 500 membership run backward over 2022-present — any
    company added to the index after 2022 is included for its full listed
    history (fine), but any company REMOVED from the index since 2022 is
    absent for the whole window, even for the period it genuinely was a
    constituent. This is the same "hindsight-chosen universe" caveat this
    project's own prior research (the old fundamental-screener line of
    work) already flagged for a dynamic index-membership universe.

    Reports whether/when the equity curve first closes at or above
    `target`, off the actual simulated path (see _time_to_reach()).
    """
    symbols = data.sp500_symbols()
    print(f"S&P 500 universe test ({len(symbols)} nominal symbols — some may "
         f"lack sufficient 2022-present history and get skipped, see below): "
         f"{START} -> {END} ...", flush=True)
    res, stats = run_risk_managed_backtest(
        symbols, START, END, capital, use_circuit_breaker=False,
        fractional_shares=True, max_concurrent=6,
        regime_filter={"sma_window": 50,
                      "aggressive": (0.0150, 0.0100),   # bull: 1.50% DIP / 1.00% BREAKOUT
                      "base": (0.0050, 0.0025)})         # bear: 0.50% DIP / 0.25% BREAKOUT
    metrics = compute_metrics(res, capital)
    time_desc, _ = _time_to_reach(res.equity, target)

    width = 82
    print("=" * width)
    print("S&P 500 DYNAMIC UNIVERSE TEST — SOLVING SIGNAL SCARCITY HORIZONTALLY")
    print(f"${capital:,.0f} starting capital, {len(symbols)} nominal symbols "
         f"(S&P 500), MAX_CONCURRENT=6, fractional sizing, {START} -> {END}")
    print("Widened regime: bull 1.50% DIP/1.00% BRK | bear 0.50% DIP/0.25% BRK")
    print("=" * width)
    print(f"{'Total Return':<24}{metrics['total_return_pct']:>+10.2f}%")
    print(f"{'CAGR (Annualized)':<24}{metrics['ann_return_pct']:>+10.2f}%")
    print(f"{'Max Drawdown':<24}{metrics['max_dd_pct']:>10.2f}%")
    print(f"{'Sharpe Ratio':<24}{metrics['sharpe']:>10.3f}")
    print(f"{'Total Trades':<24}{metrics['n_trades']:>10}")
    print(f"{'Avg Open Positions':<24}{stats.avg_open_positions:>10.2f}  (of {stats.max_concurrent_used} cap)")
    print("-" * width)
    print(f"Time to grow ${capital:,.0f} -> ${target:,.0f}: {time_desc}")
    print("=" * width)
    return metrics


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
