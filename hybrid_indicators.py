"""
Data loading + indicator math for Step 1 of the Fixed ADX Hybrid Strategy —
built in COMPLETE ISOLATION from the live paper-trading engine. This module
(and hybrid_engine.py) never imports intraday.py or execution_engine.py, and
never will.

What this DOES reuse — shared, read-only library code, not the live engine:
  * backtest.MIXED_UNIVERSE                          — the 30-symbol universe
  * backtest.{ADX_RANGE_MAX, ADX_TREND_MIN, ADX_PERCENTILE_WINDOW,
    ADX_PERCENTILE_Q, ATR_PERIOD, VOLUME_MULT, VOLUME_SMA_WINDOW,
    BREAKOUT_LOOKBACK}                                — the Fixed ADX
    Hybrid's own constants from backtest.py's Phase 3.1 section, so this
    stays consistent with the already-validated backtest instead of
    drifting on a second, hand-typed copy of the same numbers.
  * strategy.{rsi, bollinger, adx, atr}               — shared Wilder-smoothed
    indicator primitives (same functions the backtest and the live engine
    both use).
  * config.{RSI_PERIOD, BB_PERIOD, BB_K, RSI_OVERSOLD} — the same oversold
    thresholds Phase 1 validated.
  * data.daily_bars / data.last_price                 — the existing
    yfinance data layer (disk-cached under .cache/; that cache is the only
    thing this module ever writes to disk).

DATA WINDOW — fetches 2 years, not 50 bars
    50 raw bars cannot support this module's own math. ADX(14) needs ~14
    bars of warmup before its first reading exists at all, the 80th-
    percentile threshold needs 50 of THOSE readings in a rolling window, and
    the DIP trend gate below needs a 200-session SMA plus one more day of
    lag on top of that. This fetches ~2 years (~504 trading days): well
    past all three warmups with a comfortable buffer, matching the backtest's
    own ~400-calendar-day seed window in spirit.

    DIP TREND GATE — reinstated to match backtest.py's validated
    daily_hybrid_frame() exactly, after a live/backtest divergence was found
    and quantified (see git history around the date this comment was added):
    this module originally computed DIP eligibility as bare
    `RSI<35 OR Close<=BB_low`, with no trend filter at all, because the data
    window was too short to support one. Measured against the identical
    2022-present backtest that otherwise validates this strategy, that bare
    trigger alone degrades CAGR from +17.87% to +10.96% and roughly DOUBLES
    max drawdown (9.84% -> 20.04%) — removing the trend gate lets the engine
    buy "oversold" readings inside long-term downtrends (catching falling
    knives), which the backtest's trend-filtered version never permits. The
    gate is now: `prev_close > prev_sma200 AND close > sma200` (both the
    prior day's AND today's close above the 200-day SMA — the same
    OVERSOLD_REQUIRE_ABOVE_SMA200=True behavior config.py already uses for
    the backtest), ANDed with the RSI/BB trigger below.

SIGNAL TIMING — last COMPLETED daily bar, never today's still-forming one
    Same non-repainting discipline the live engine's --screen fix uses:
    yfinance's "today" row updates live during the session, so reading it
    for indicator values would let this snapshot's classification change
    mid-day without a bar actually closing. Only the live PRICE column uses
    a genuinely current quote (data.last_price) — everything else (RSI, ADX,
    volume ratio, ATR) is read off the last bar that actually closed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import data
import strategy
from backtest import (
    MIXED_UNIVERSE,
    ADX_RANGE_MAX, ADX_TREND_MIN, ADX_PERCENTILE_WINDOW, ADX_PERCENTILE_Q,
    ATR_PERIOD, VOLUME_MULT, VOLUME_SMA_WINDOW, BREAKOUT_LOOKBACK,
)
from calendar_util import now_ny
from config import (
    RSI_PERIOD, BB_PERIOD, BB_K, RSI_OVERSOLD, STOP_PCT, TARGET_R,
    OVERSOLD_REQUIRE_ABOVE_SMA200,
)

DATA_PERIOD = "2y"                               # see module docstring — must
                                                  # cover the 200-day SMA trend gate
SMA200_WINDOW = 200
MIN_BARS_REQUIRED = max(ADX_PERCENTILE_WINDOW + 30, SMA200_WINDOW + 30)

# ---------------------------------------------------------------- Step 2/3: sizing
# Asymmetric risk: a DIP is a confirmed mean-reversion setup (oversold, prior
# edge validated in Phase 1); a BREAKOUT is taken the moment it prints, with
# no confirmation yet that the move continues, so it gets a smaller risk
# budget than DIP within the same regime.
#
# Stop placement aligns with backtest.py's validated Fixed Hybrid exactly:
#   DIP      — fixed STOP_PCT (2.5%, config.py — same constant Phase 1's
#              backtest uses, not a second hand-typed 0.025) off entry.
#              Also gets a TARGET_R (2.5R, config.py) profit target, since
#              a fixed-% stop needs a defined R-unit to size a target off —
#              BREAKOUT has no target, exiting purely via the trailing stop.
#   BREAKOUT — 2.0x ATR14, unchanged.
STOP_ATR_MULT = 2.0           # BREAKOUT initial stop = Entry - STOP_ATR_MULT x ATR14

# ---- Widened Spread Regime Filter (permanent production config) -----------
# Risk_pct is no longer a flat constant: it is selected live, per signal, by
# whether SPY's last COMPLETED daily close is above its own rolling 50-day
# SMA (current_spy_regime() below) — the same "Widened Spread" config
# validated across hypothetical_backtester.py's regime-filter research this
# session (best isolated backtest: ~+115% total return / ~9.8% max
# drawdown / Sharpe ~1.2 over 2022-present, $100k, though never re-run
# through the same live dry-run/fill-verification process the ORIGINAL
# 0.625%/0.25% baseline went through before that one was cleared for
# forward paper testing).
REGIME_SMA_WINDOW = 50
RISK_PCT_DIP_BULL = 0.0150         # 1.50% DIP when SPY > its 50-day SMA
RISK_PCT_BREAKOUT_BULL = 0.0100    # 1.00% BREAKOUT when SPY > its 50-day SMA
RISK_PCT_DIP_BEAR = 0.0050         # 0.50% DIP when SPY <= its 50-day SMA
RISK_PCT_BREAKOUT_BEAR = 0.0025    # 0.25% BREAKOUT when SPY <= its 50-day SMA


def current_spy_regime() -> str:
    """
    Live SPY 50-day SMA regime check — same non-repainting discipline as
    the rest of this module (snapshot()/current_atr()): reads only the
    last COMPLETED daily bar, never today's still-forming one. Returns
    "BULL" if that close is above its own trailing REGIME_SMA_WINDOW-day
    SMA, "BEAR" otherwise (a tie counts as BEAR, matching the backtest's
    `<=` bear-regime convention exactly). Defensively returns "BEAR" — the
    lower-risk regime — if SPY data can't be fetched or there isn't enough
    history to compute the SMA, rather than risking the aggressive bull
    sizing on missing/incomplete data.
    """
    try:
        daily = data.daily_bars("SPY", period="6mo")
    except Exception:
        return "BEAR"
    if daily.empty:
        return "BEAR"
    today = now_ny().date()
    hist = daily[daily.index.date < today]
    if len(hist) < REGIME_SMA_WINDOW:
        return "BEAR"
    close = hist["Close"].to_numpy(dtype=float)
    sma = close[-REGIME_SMA_WINDOW:].mean()
    return "BULL" if close[-1] > sma else "BEAR"


@dataclass
class SymbolSnapshot:
    symbol: str
    price: float               # live quote
    rsi: float                 # last completed bar
    bb_low: float
    adx: float
    adx_80th: float
    volume: float
    vol_sma20: float
    vol_ratio: float           # volume / vol_sma20
    atr: float
    prior_high: float
    dip_ok: bool
    breakout_ok: bool
    mode: str                  # "DIP" | "BREAKOUT" | "NONE"


def load_symbol_frame(symbol: str) -> pd.DataFrame | None:
    """
    Daily bars for `symbol` plus every indicator this module needs, or None
    if there is not enough history yet. Adds: RSI, BB_LOW, SMA200, ADX,
    ADX_80TH, ATR14, VOL_SMA20, VOL_RATIO, PRIOR_HIGH, DIP_OK, BREAKOUT_OK.
    """
    try:
        daily = data.daily_bars(symbol, period=DATA_PERIOD)
    except Exception:
        return None
    if daily.empty or len(daily) < MIN_BARS_REQUIRED:
        return None

    close = daily["Close"].to_numpy(dtype=float)
    high = daily["High"].to_numpy(dtype=float)
    low = daily["Low"].to_numpy(dtype=float)
    volume = daily["Volume"].to_numpy(dtype=float)

    rsi_v = strategy.rsi(close, RSI_PERIOD)
    _, bb_low, _ = strategy.bollinger(close, BB_PERIOD, BB_K)
    sma200 = pd.Series(close).rolling(SMA200_WINDOW).mean().to_numpy()
    adx_v = strategy.adx(high, low, close, ATR_PERIOD)
    atr_v = strategy.atr(high, low, close, ATR_PERIOD)

    # Dynamic momentum threshold: rolling ADX_PERCENTILE_WINDOW-session
    # ADX_PERCENTILE_Q percentile, shifted one day so it never includes the
    # value being tested against it, floored at ADX_TREND_MIN so a quiet
    # stretch can't drop the bar below the original fixed threshold — exactly
    # backtest.py's daily_hybrid_fixed_frame() logic.
    adx_80th = (pd.Series(adx_v).shift(1)
               .rolling(ADX_PERCENTILE_WINDOW).quantile(ADX_PERCENTILE_Q)
               .clip(lower=ADX_TREND_MIN).to_numpy())

    vol_sma20 = pd.Series(volume).rolling(VOLUME_SMA_WINDOW).mean().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_ratio = np.where(vol_sma20 > 0, volume / vol_sma20, np.nan)

    prior_high = pd.Series(high).shift(1).rolling(BREAKOUT_LOOKBACK).max().to_numpy()

    # Trend gate + RSI/BB trigger — matches backtest.py's daily_hybrid_frame()
    # exactly (same prev_close/prev_sma200 shift, same OVERSOLD_REQUIRE_
    # ABOVE_SMA200 branch, same Low-based BB touch) rather than a second,
    # independently-typed copy of this math. See module docstring for why.
    prev_close = np.empty_like(close)
    prev_close[0] = np.nan
    prev_close[1:] = close[:-1]
    prev_sma200 = np.empty_like(sma200)
    prev_sma200[0] = np.nan
    prev_sma200[1:] = sma200[:-1]
    trend_ok = np.where(np.isnan(prev_close) | np.isnan(prev_sma200), False,
                        prev_close > prev_sma200)
    if OVERSOLD_REQUIRE_ABOVE_SMA200:
        above_now = np.where(np.isnan(sma200), False, close > sma200)
    else:
        above_now = np.full(len(close), True)

    dip_rsi_ok = np.where(np.isnan(rsi_v), False, rsi_v < RSI_OVERSOLD)
    dip_bb_ok = np.where(np.isnan(bb_low), False, low <= bb_low)
    dip_ok = trend_ok & above_now & (dip_rsi_ok | dip_bb_ok)

    vol_ok = np.where(np.isnan(vol_ratio), False, vol_ratio > VOLUME_MULT)
    breakout_price_ok = np.where(np.isnan(prior_high), False, close > prior_high)
    breakout_ok = breakout_price_ok & vol_ok

    out = daily.copy()
    out["RSI"] = rsi_v
    out["BB_LOW"] = bb_low
    out["SMA200"] = sma200
    out["ADX"] = adx_v
    out["ADX_80TH"] = adx_80th
    out["ATR14"] = atr_v
    out["VOL_SMA20"] = vol_sma20
    out["VOL_RATIO"] = vol_ratio
    out["PRIOR_HIGH"] = prior_high
    out["DIP_OK"] = dip_ok
    out["BREAKOUT_OK"] = breakout_ok
    return out


def classify_mode(adx_v: float, adx_80th: float, dip_ok: bool,
                  breakout_ok: bool) -> str:
    """
    Fixed ADX Hybrid's regime routing, matching
    backtest.run_adx_hybrid_fixed_backtest() exactly: ranging
    (ADX < ADX_RANGE_MAX) is DIP-eligible; trending (ADX above this symbol's
    own dynamic 80th-percentile threshold) is BREAKOUT-eligible; the
    transition band between routes to neither.
    """
    if adx_v != adx_v:                    # NaN — ADX not warmed up
        return "NONE"
    if adx_v < ADX_RANGE_MAX and dip_ok:
        return "DIP"
    if adx_80th == adx_80th and adx_v > adx_80th and breakout_ok:
        return "BREAKOUT"
    return "NONE"


def snapshot(symbol: str) -> SymbolSnapshot | None:
    """Live snapshot for one symbol, using the last COMPLETED daily bar."""
    frame = load_symbol_frame(symbol)
    if frame is None:
        return None

    today = now_ny().date()
    hist = frame[frame.index.date < today]
    if hist.empty:
        return None
    last = hist.iloc[-1]

    rsi_v = float(last["RSI"])
    bb_low = float(last["BB_LOW"])
    adx_v = float(last["ADX"])
    adx_80th = float(last["ADX_80TH"])
    vol_ratio = float(last["VOL_RATIO"])
    atr_v = float(last["ATR14"])
    prior_high = float(last["PRIOR_HIGH"])
    dip_ok = bool(last["DIP_OK"])
    breakout_ok = bool(last["BREAKOUT_OK"])

    price = data.last_price(symbol)
    if price is None or price <= 0:
        price = float(last["Close"])       # fall back to last completed close

    mode = classify_mode(adx_v, adx_80th, dip_ok, breakout_ok)

    return SymbolSnapshot(
        symbol=symbol, price=price, rsi=rsi_v, bb_low=bb_low, adx=adx_v,
        adx_80th=adx_80th, volume=float(last["Volume"]),
        vol_sma20=float(last["VOL_SMA20"]), vol_ratio=vol_ratio, atr=atr_v,
        prior_high=prior_high, dip_ok=dip_ok, breakout_ok=breakout_ok,
        mode=mode)


def snapshot_universe(symbols: list[str] | None = None) -> list[SymbolSnapshot]:
    """Snapshot every symbol in `symbols` (default MIXED_UNIVERSE), silently
    skipping any with insufficient data."""
    symbols = symbols if symbols is not None else MIXED_UNIVERSE
    out: list[SymbolSnapshot] = []
    for sym in symbols:
        snap = snapshot(sym)
        if snap is not None:
            out.append(snap)
    return out


@dataclass
class HybridOrder:
    symbol: str
    trade_type: str             # "DIP" | "BREAKOUT"
    entry_price: float
    atr14: float | None         # None only if ATR happened to be undefined
    stop_price: float
    target_price: float | None  # DIP only; None for BREAKOUT
    risk_pct: float
    risk_dollars: float
    shares: float                # fractional — see size_order()'s docstring
    notional: float


# Fractional-share precision this module sizes to before the broker layer's
# own (venue-specific) rounding at submission time — see
# hybrid_engine.execute_signals()/buy_market(). 6dp matches Robinhood's
# documented fractional-order precision; Alpaca's buy_market() rounds to
# 4dp internally, so a 6dp value sized here is never LESS precise than
# either venue actually accepts, just possibly rounded slightly further at
# submission.
SHARE_PRECISION = 6


def size_order(snap: SymbolSnapshot, equity: float) -> HybridOrder | None:
    """
    Widened Spread Regime Filter sizing: risk_pct is selected live by
    current_spy_regime() — DIP risks RISK_PCT_DIP_BULL/BEAR off a fixed
    STOP_PCT stop with a TARGET_R profit target; BREAKOUT risks
    RISK_PCT_BREAKOUT_BULL/BEAR off a STOP_ATR_MULT x ATR14 stop with no
    fixed target (it exits via the trailing stop in --manage instead).

    Shares = risk_dollars / (entry - stop), sized to a PRECISE FRACTIONAL
    quantity (rounded only to SHARE_PRECISION, never floored to a whole
    share) — entries submit via Broker.buy_market() (see
    hybrid_engine.execute_signals()), which both Alpaca and Robinhood
    accept fractional quantities on; buy_limit() (whole-share only on both
    venues) is no longer used for entries. Returns None if the signal
    isn't DIP/BREAKOUT, ATR isn't defined (BREAKOUT only — DIP's stop
    doesn't need it), the stop doesn't land below entry, or the sized
    quantity rounds to <= 0.
    """
    if snap.mode not in ("DIP", "BREAKOUT"):
        return None

    entry = snap.price
    target: float | None = None
    regime = current_spy_regime()

    if snap.mode == "DIP":
        risk_pct = RISK_PCT_DIP_BULL if regime == "BULL" else RISK_PCT_DIP_BEAR
        stop = entry * (1.0 - STOP_PCT)
        risk_per_share = entry - stop
        if risk_per_share > 0:
            target = entry + risk_per_share * TARGET_R
    else:
        risk_pct = RISK_PCT_BREAKOUT_BULL if regime == "BULL" else RISK_PCT_BREAKOUT_BEAR
        if snap.atr != snap.atr or snap.atr <= 0:      # NaN/invalid guard
            return None
        stop = entry - STOP_ATR_MULT * snap.atr
        risk_per_share = entry - stop

    if risk_per_share <= 0:
        return None

    risk_dollars = equity * risk_pct
    shares = round(risk_dollars / risk_per_share, SHARE_PRECISION)
    if shares <= 0:
        return None

    return HybridOrder(
        symbol=snap.symbol, trade_type=snap.mode, entry_price=round(entry, 4),
        atr14=round(snap.atr, 4) if snap.atr == snap.atr else None,
        stop_price=round(stop, 4),
        target_price=round(target, 4) if target is not None else None,
        risk_pct=risk_pct, risk_dollars=round(risk_dollars, 2),
        shares=shares, notional=round(shares * entry, 2))


def current_atr(symbol: str) -> float | None:
    """
    Today's updated ATR14 for an already-open position, off the last
    COMPLETED daily bar (same non-repainting discipline as snapshot()) — used
    by hybrid_engine.py --manage to ratchet a BREAKOUT trailing stop. None if
    there isn't enough history or ATR isn't defined yet.
    """
    frame = load_symbol_frame(symbol)
    if frame is None:
        return None
    today = now_ny().date()
    hist = frame[frame.index.date < today]
    if hist.empty:
        return None
    atr_v = float(hist.iloc[-1]["ATR14"])
    return atr_v if atr_v == atr_v else None
