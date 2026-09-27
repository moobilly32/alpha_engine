"""
Signal logic — shared by the live engine and every backtester.

Two entry modes live here:

  PULLBACK (current)  Buy strength on a dip. Requires an established daily
                      uptrend, price that traded ABOVE an intraday support level
                      earlier in the session, a dip back to that level, and a
                      reclaim. You enter near support, so the stop sits just
                      under it and the reward:risk is set by structure rather
                      than by a round number.

  BREAKOUT (legacy)   Buy new highs: price clears the previous daily high, the
                      premarket high and the day's high. Retained because it is
                      the baseline the earlier work measured, and a new entry
                      model is only interesting relative to the old one.

WHY THE PULLBACK STOP IS STRUCTURAL. A breakout entry has no natural stop —
1% below entry is arbitrary, which is why sweeping it never helped. A pullback
entry has one: the low of the dip. If price returns through the level it just
defended, the premise is dead. That is a reason to be out, not a distance.

NON-REPAINTING. Every daily input is the last COMPLETED daily bar; intraday
levels exclude the current bar wherever including it would let the bar's own
extreme satisfy its own condition.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- containers
@dataclass
class DailyContext:
    """Daily-timeframe facts, constant for a whole session."""
    prev_high: float | None = None
    prev_low: float | None = None
    prev_close: float | None = None
    sma200: float | None = None
    sma50: float | None = None
    sma20: float | None = None
    atr: float | None = None          # ATR(14), in price units

    def ok(self) -> bool:
        return None not in (self.prev_high, self.prev_close, self.sma200, self.sma50)


@dataclass
class Levels:
    """Everything a signal needs, resolved for one symbol at one instant."""
    symbol: str
    # daily
    prev_high: float | None = None
    prev_close: float | None = None
    sma200: float | None = None
    sma50: float | None = None
    atr: float | None = None
    # breakout inputs
    pmh: float | None = None
    hod_prev: float | None = None
    # pullback inputs
    support: float | None = None        # the level being defended
    support_name: str = "vwap"
    was_above: bool = False             # traded above support earlier today
    dip_low: float | None = None        # lowest low of the dip
    dipped: bool = False                # touched support recently

    @property
    def trigger(self) -> float | None:
        """
        The reference price for the execution range.

        Breakout: the highest level that must be cleared. Pullback: the support
        level itself — "in range" then means near support rather than just past
        a high, which is what buying a discount actually means.
        """
        if self.support is not None:
            return self.support
        parts = [v for v in (self.prev_high, self.pmh, self.hod_prev) if v is not None]
        return max(parts) if parts else None

    @property
    def trend_ok(self) -> bool:
        return (self.prev_close is not None and self.sma200 is not None
                and self.prev_close > self.sma200)

    @property
    def strong_trend_ok(self) -> bool:
        """Uptrend on both the long and medium daily averages."""
        return (self.trend_ok and self.sma50 is not None
                and self.prev_close > self.sma50)


@dataclass
class Signal:
    fired: bool
    reason: str
    levels: Levels


# ---------------------------------------------------------------- daily
def daily_context_full(daily: pd.DataFrame, asof: dt.date) -> DailyContext:
    """
    Daily facts from bars that closed STRICTLY before `asof`.

    The strict `<` is the whole point: on any intraday evaluation today's
    partial daily bar is already in the frame, and including it would hand the
    strategy the answer.
    """
    ctx = DailyContext()
    if daily is None or daily.empty:
        return ctx

    hist = daily[daily.index.date < asof]
    if len(hist) < 200:
        return ctx

    prev = hist.iloc[-1]
    ctx.prev_high = float(prev["High"])
    ctx.prev_low = float(prev["Low"])
    ctx.prev_close = float(prev["Close"])
    ctx.sma200 = float(hist["Close"].tail(200).mean())
    ctx.sma50 = float(hist["Close"].tail(50).mean())
    ctx.sma20 = float(hist["Close"].tail(20).mean())

    # ATR(14), Wilder's true range. Used for volatility-scaled stops: a 2% stop
    # is loose on KO and tight on NVDA, and over a multi-day hold that
    # difference decides whether noise or thesis takes you out.
    tail = hist.tail(15)
    if len(tail) >= 15:
        h, l, c = tail["High"].to_numpy(), tail["Low"].to_numpy(), tail["Close"].to_numpy()
        tr = np.maximum(h[1:] - l[1:],
                        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        ctx.atr = float(np.mean(tr))
    return ctx


def daily_context(daily: pd.DataFrame, asof: dt.date):
    """Legacy 3-tuple form, kept so existing callers keep working."""
    c = daily_context_full(daily, asof)
    return c.prev_high, c.prev_close, c.sma200


# ---------------------------------------------------------------- intraday
def session_vwap(bars: pd.DataFrame) -> np.ndarray:
    """
    Session-anchored VWAP over the supplied bars, cumulative bar by bar.

    Anchored at the first bar passed in (the RTH open), which is what every
    chart means by "the VWAP" intraday. Typical price times volume, divided by
    cumulative volume.
    """
    if bars is None or bars.empty:
        return np.array([])
    tp = (bars["High"].to_numpy() + bars["Low"].to_numpy() + bars["Close"].to_numpy()) / 3.0
    vol = bars["Volume"].to_numpy(dtype=float)
    cv = np.cumsum(vol)
    cpv = np.cumsum(tp * vol)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(cv > 0, cpv / np.maximum(cv, 1e-9), tp)
    return out


def ema(values: np.ndarray, span: int) -> np.ndarray:
    """Standard EMA. Computed on the continuous series, not reset per session."""
    if len(values) == 0:
        return values
    return pd.Series(values).ewm(span=span, adjust=False).mean().to_numpy()


def session_levels(session_bars: pd.DataFrame, upto_idx: int):
    """(premarket_high, high_of_day_before the current bar) — breakout inputs."""
    if session_bars is None or session_bars.empty:
        return None, None

    idx = session_bars.index
    mins = idx.hour * 60 + idx.minute

    pm_mask = (mins >= 4 * 60) & (mins < 9 * 60 + 30)
    pmh = float(session_bars.loc[pm_mask, "High"].max()) if pm_mask.any() else None

    rth_mask = (mins >= 9 * 60 + 30) & (mins < 16 * 60)
    positional = np.arange(len(session_bars)) < upto_idx
    before = np.asarray(rth_mask) & positional
    hod = float(session_bars.loc[before, "High"].max()) if before.any() else None

    return pmh, (hod if hod == hod else None)


def compute_levels(symbol: str, daily: pd.DataFrame, session_bars: pd.DataFrame,
                   asof_date: dt.date, upto_idx: int) -> Levels:
    """Breakout-mode levels (legacy path, used by the live breakout engine)."""
    c = daily_context_full(daily, asof_date)
    pmh, hod = session_levels(session_bars, upto_idx)
    return Levels(symbol=symbol, prev_high=c.prev_high, prev_close=c.prev_close,
                  sma200=c.sma200, sma50=c.sma50, atr=c.atr, pmh=pmh, hod_prev=hod)


# ---------------------------------------------------------------- breakout
def evaluate_signal(levels: Levels, price: float) -> Signal:
    """Legacy breakout entry. Both legs, returning the blocking one."""
    lv = levels
    if lv.prev_high is None or lv.prev_close is None or lv.sma200 is None:
        return Signal(False, "insufficient daily history (need 200 bars)", lv)
    if lv.pmh is None:
        return Signal(False, "no premarket bars — extended-hours data missing", lv)
    if lv.hod_prev is None:
        return Signal(False, "no completed RTH bars yet today", lv)
    if not lv.trend_ok:
        return Signal(False, f"trend: prev close {lv.prev_close:.2f} <= SMA200 {lv.sma200:.2f}", lv)

    hi = max(lv.prev_high, lv.pmh, lv.hod_prev)
    if price <= hi:
        return Signal(False, f"price {price:.2f} has not cleared {hi:.2f}", lv)
    return Signal(True, f"breakout above {hi:.2f} (PDH/PMH/HOD cleared)", lv)


# ---------------------------------------------------------------- pullback
def evaluate_pullback(levels: Levels, price: float,
                      max_extension: float = 0.004,
                      require_strong_trend: bool = True) -> Signal:
    """
    Pullback entry: dip into support inside an uptrend, then reclaim.

    Four conditions, in the order a trader would check them:

      1. TREND      the daily uptrend is intact (close above SMA200, and SMA50
                    when require_strong_trend). Without this the "support" is
                    just a level on the way down.
      2. WAS ABOVE  price traded above support earlier in the session. This is
                    what makes it a pullback rather than a downtrend that has
                    never been above the line all day.
      3. DIPPED     price actually reached support recently. No dip, no discount.
      4. RECLAIM    the current bar is back above support, and not extended far
                    above it. Requiring the reclaim avoids catching a knife;
                    capping the extension keeps the entry near the level so the
                    structural stop stays tight.
    """
    lv = levels

    if lv.prev_close is None or lv.sma200 is None:
        return Signal(False, "insufficient daily history (need 200 bars)", lv)
    if lv.support is None:
        return Signal(False, "support level unavailable", lv)

    if require_strong_trend:
        if not lv.strong_trend_ok:
            ref = lv.sma50 if lv.sma50 is not None else float("nan")
            return Signal(False, f"trend: prev close {lv.prev_close:.2f} not above "
                                 f"SMA50 {ref:.2f} and SMA200", lv)
    elif not lv.trend_ok:
        return Signal(False, f"trend: prev close {lv.prev_close:.2f} <= SMA200 "
                             f"{lv.sma200:.2f}", lv)

    if not lv.was_above:
        return Signal(False, f"never traded above {lv.support_name} today "
                             f"— downtrend, not a pullback", lv)
    if not lv.dipped:
        return Signal(False, f"no dip to {lv.support_name} {lv.support:.2f} yet", lv)
    if price <= lv.support:
        return Signal(False, f"price {price:.2f} still below {lv.support_name} "
                             f"{lv.support:.2f} — no reclaim", lv)

    ext = price / lv.support - 1.0
    if ext > max_extension:
        return Signal(False, f"price {price:.2f} is {ext*100:.2f}% above "
                             f"{lv.support_name} — too extended to be a discount", lv)

    return Signal(True, f"pullback to {lv.support_name} {lv.support:.2f} reclaimed "
                        f"at {price:.2f} ({ext*100:+.2f}%)", lv)


def structural_stop(levels: Levels, entry: float, buffer_pct: float = 0.0015,
                    max_stop_pct: float = 0.06) -> float | None:
    """
    Stop just under the defended structure: the lower of the dip low and support.

    Capped at max_stop_pct so that a violently wide dip cannot create a position
    whose risk budget buys a rounding error, and floored implicitly by the
    buffer so it is never exactly at the level everyone else's stop sits on.
    """
    parts = [v for v in (levels.dip_low, levels.support) if v is not None]
    if not parts or entry <= 0:
        return None
    stop = min(parts) * (1.0 - buffer_pct)
    if stop >= entry:
        return None
    if (entry - stop) / entry > max_stop_pct:
        stop = entry * (1.0 - max_stop_pct)
    return stop


# ---------------------------------------------------------------- helpers
def split_sessions(bars: pd.DataFrame) -> dict[dt.date, pd.DataFrame]:
    """Group NY-localised intraday bars into per-session frames."""
    if bars is None or bars.empty:
        return {}
    return {d: g for d, g in bars.groupby(bars.index.date)}


# ---------------------------------------------------------------- oscillators
def rsi(values: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's RSI. Leading `period` entries are NaN (not yet defined)."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return np.full(len(v), np.nan)
    delta = np.diff(v, prepend=v[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ag = pd.Series(gain).ewm(alpha=1.0 / period, adjust=False).mean().to_numpy()
    al = pd.Series(loss).ewm(alpha=1.0 / period, adjust=False).mean().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(al > 1e-12, ag / np.maximum(al, 1e-12), np.inf)
    out = 100.0 - 100.0 / (1.0 + rs)
    out[:period] = np.nan
    return out


def bollinger(values: np.ndarray, period: int = 20, k: float = 2.0):
    """(middle, lower, upper) Bollinger bands, population sigma."""
    s = pd.Series(np.asarray(values, dtype=float))
    ma = s.rolling(period).mean()
    sd = s.rolling(period).std(ddof=0)
    return ma.to_numpy(), (ma - k * sd).to_numpy(), (ma + k * sd).to_numpy()


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    n = len(high)
    prev_close = np.empty(n)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]
    return np.maximum(high - low,
                      np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
       period: int = 14) -> np.ndarray:
    """
    Wilder's Average True Range — Wilder smoothing (ewm alpha=1/period,
    adjust=False), same idiom as rsi()/adx(). Leading `period` entries are
    NaN (smoothing not yet warmed up).
    """
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    if len(h) < 2:
        return np.full(len(h), np.nan)
    tr = _true_range(h, l, c)
    out = pd.Series(tr).ewm(alpha=1.0 / period, adjust=False).mean().to_numpy(copy=True)
    out[:period] = np.nan
    return out


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
       period: int = 14) -> np.ndarray:
    """
    Wilder's Average Directional Index — trend STRENGTH, not direction.
    Conventionally: below ~20 the market is range-bound (chop), above ~25 a
    trend (up or down) is established. Uses the same Wilder smoothing
    (ewm alpha=1/period, adjust=False) as rsi() above, for consistency.
    Leading `period` entries are NaN (smoothing not yet warmed up).
    """
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    n = len(h)
    if n < 2:
        return np.full(n, np.nan)

    up_move = np.diff(h, prepend=h[0])
    down_move = -np.diff(l, prepend=l[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0

    alpha = 1.0 / period
    tr = _true_range(h, l, c)
    atr_val = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean()
    smoothed_plus_dm = pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean()
    smoothed_minus_dm = pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean()

    atr_safe = atr_val.replace(0.0, np.nan)
    plus_di = 100.0 * smoothed_plus_dm / atr_safe
    minus_di = 100.0 * smoothed_minus_dm / atr_safe

    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx_val = dx.ewm(alpha=alpha, adjust=False).mean().to_numpy(copy=True)
    adx_val[:period] = np.nan
    return adx_val


def to_hourly(rth: pd.DataFrame) -> pd.DataFrame:
    """
    Resample RTH 5-minute bars to hourly, anchored at 09:30.

    `offset="30min"` matters: without it the bins start on the clock hour and
    the first bar of every session is a 30-minute stub that is not the hourly
    candle any chart would show.
    """
    if rth is None or rth.empty:
        return pd.DataFrame()
    agg = {"Open": "first", "High": "max", "Low": "min",
           "Close": "last", "Volume": "sum"}
    h = rth.resample("60min", offset="30min").agg(agg)
    return h.dropna(subset=["Close"])


def hourly_signals_at_5m(rth: pd.DataFrame, rsi_period: int = 14,
                         bb_period: int = 20, bb_k: float = 2.0):
    """
    Hourly RSI and lower Bollinger band, aligned onto the 5-minute index.

    NON-REPAINTING, and this is the whole difficulty. While a 5-minute bar is
    inside the current hourly bin, that hourly bar is INCOMPLETE — its close,
    and therefore its RSI and its bands, are still moving. Reading it would let
    the strategy see the hour's own outcome before the hour ends. So each hourly
    value is stamped as available only from bin_start + 60min, and every
    5-minute bar reads the last hourly bar that had actually CLOSED by then.
    """
    h = to_hourly(rth)
    if h.empty or len(h) < max(rsi_period, bb_period) + 1:
        n = len(rth)
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)

    closes = h["Close"].to_numpy(dtype=float)
    r = rsi(closes, rsi_period)
    _mid, low, _up = bollinger(closes, bb_period, bb_k)

    # available_from = bin start + one hour = the moment the bar is complete
    avail = (h.index + pd.Timedelta(minutes=60)).to_numpy()
    target = rth.index.to_numpy()

    # For each 5-minute bar, the last hourly bar that had CLOSED by then.
    # searchsorted with side="right" minus one gives exactly that; -1 marks
    # bars early in the window with no completed hourly bar yet.
    pos = np.searchsorted(avail, target, side="right") - 1
    valid = pos >= 0
    pos_safe = np.where(valid, pos, 0)

    def _pick(arr):
        out = np.where(valid, np.asarray(arr, dtype=float)[pos_safe], np.nan)
        return out

    return _pick(r), _pick(low), _pick(closes)


def evaluate_oversold(levels: Levels, price: float, bar_low: float,
                      rsi_val: float, bb_low: float,
                      rsi_threshold: float = 35.0,
                      require_above_sma200: bool = True) -> Signal:
    """
    Deep-oversold dip entry: buy weakness inside an intact long-term uptrend.

      TREND     previous daily close above the 200 SMA, and — the literal
                reading of "while remaining above its Daily 200 SMA" — the live
                price above it too. A stock that has just lost its 200 SMA is
                not oversold, it is in a downtrend.
      OVERSOLD  hourly RSI below the threshold, OR the bar touching the lower
                hourly Bollinger band. Either is sufficient, as specified.

    Note what this does NOT require: any sign that the fall has stopped. A
    pullback entry demands a reclaim of support before committing; this one
    buys while price is still falling, which is a materially different risk and
    shows up in the stop distance.
    """
    lv = levels
    if lv.prev_close is None or lv.sma200 is None:
        return Signal(False, "insufficient daily history (need 200 bars)", lv)
    if not lv.trend_ok:
        return Signal(False, f"prev close {lv.prev_close:.2f} <= SMA200 "
                             f"{lv.sma200:.2f}", lv)
    if require_above_sma200 and price <= lv.sma200:
        return Signal(False, f"price {price:.2f} has lost the 200 SMA "
                             f"{lv.sma200:.2f}", lv)

    rsi_ok = (rsi_val == rsi_val) and rsi_val < rsi_threshold
    bb_ok = (bb_low == bb_low) and bar_low <= bb_low
    if not (rsi_ok or bb_ok):
        shown = f"{rsi_val:.1f}" if rsi_val == rsi_val else "n/a"
        return Signal(False, f"not oversold (1h RSI {shown}, "
                             f"low {bar_low:.2f} above band)", lv)

    why = []
    if rsi_ok:
        why.append(f"1h RSI {rsi_val:.1f} < {rsi_threshold:.0f}")
    if bb_ok:
        why.append(f"touched lower BB {bb_low:.2f}")
    return Signal(True, "oversold: " + " and ".join(why), lv)


def hourly_ema_at_5m(rth: pd.DataFrame, span: int = 20) -> np.ndarray:
    """
    Hourly EMA aligned onto the 5-minute index, non-repainting.

    Same discipline as `hourly_signals_at_5m`: an EMA computed on a still-open
    hourly bar moves with every tick inside that hour, so trailing a stop on it
    would let the exit react to the hour's own close before the hour ends. Each
    hourly value becomes visible only at bin_start + 60min.
    """
    h = to_hourly(rth)
    if h.empty or len(h) < span:
        return np.full(len(rth), np.nan)

    e = ema(h["Close"].to_numpy(dtype=float), span)
    avail = (h.index + pd.Timedelta(minutes=60)).to_numpy()
    pos = np.searchsorted(avail, rth.index.to_numpy(), side="right") - 1
    valid = pos >= 0
    return np.where(valid, e[np.where(valid, pos, 0)], np.nan)


def trail_stop(kind: str, current_stop: float, high_water: float,
               ema_val: float, atr: float | None, atr_mult: float,
               floor: float, trail_pct: float = 0.0) -> float:
    """
    New trailing stop for the runner. Ratchets UP only, never below `floor`.

    `floor` is breakeven — the runner has already had its stop moved there when
    the first half was sold, and letting a trail drag it back below that would
    give away the locked-in outcome the scale was taken for.

      ema20h  : trail at the last completed 1-hour 20 EMA.
      atr     : chandelier — highest high since entry minus atr_mult x ATR.
      pct     : fixed percentage — highest high since entry x (1 - trail_pct).
    """
    proposed = current_stop
    if kind == "ema20h":
        if ema_val == ema_val:
            proposed = max(proposed, float(ema_val))
    elif kind == "atr":
        if atr and atr > 0:
            proposed = max(proposed, high_water - atr_mult * atr)
    elif kind == "pct":
        if trail_pct > 0:
            proposed = max(proposed, high_water * (1.0 - trail_pct))
    return max(float(floor), float(proposed), float(current_stop))
