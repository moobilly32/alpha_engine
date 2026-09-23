"""
59-day intraday backtest of the hybrid screener + Trend Join Long system.

WHAT THIS DOES AND DOES NOT MEASURE — read before trusting a number.

  * It DOES measure the technical layer: signal, execution-range gate, risk
    sizing, stop/target/EOD-flat exits, the 15-minute cadence, the concurrency
    cap, and slippage.

  * It does NOT measure the screener. yfinance serves only CURRENT fundamentals,
    so the watchlist is chosen with today's balance sheets and then run backwards
    over the last 59 days. Every name is one that survived to today looking
    healthy. That is look-ahead and survivorship bias, and it flatters results.
    The honest reading is: "this is how the technical system performed on a
    universe I would have liked in hindsight."

FIDELITY CHOICES, each of which makes the result WORSE than a naive backtest:

  * Entries are only evaluated on 15-minute cadence bars, because the live
    engine only looks at the market every 15 minutes. Checking all 78 daily
    5-minute bars would hand the strategy three times the opportunities and
    better prices than it can actually get.
  * Entries are also gated on the ideal execution range, so breakouts that had
    already run away are skipped rather than filled at a fantasy price.
  * When a bar's range contains BOTH the stop and the target, the stop is
    assumed to fill first. 5-minute OHLC cannot say which came first, and the
    pessimistic assumption is the only defensible one.
  * Fills pay SLIPPAGE_BPS on both sides.
  * MAX_CONCURRENT is enforced, so signals arriving while the book is full are
    counted and dropped, exactly as they would be live.

Usage:
    python3 backtest.py                         # uses data/watchlist.json
    python3 backtest.py --symbols AMD,NVDA,MU
    python3 backtest.py --days 59 --capital 10000
    python3 backtest.py --no-range-gate         # ablation: ignore exec range
    python3 backtest.py --regime-stress-test    # 6-way SPY-regime x universe
                                                 # test, Jan 2022-present (see
                                                 # the section below the 5-min
                                                 # engine for why this is a
                                                 # separate, daily-bar path)
    python3 backtest.py --ibs-variant-test      # Phase 1: RSI/BB vs. pure-IBS
                                                 # vs. combined oversold trigger,
                                                 # Small-Cap + Mixed, filter OFF
    python3 backtest.py --sector-rotation-test  # Phase 2: Variant A gated by
                                                 # SPDR sector momentum rank
                                                 # (Unfiltered/Top-2/Top-4)
    python3 backtest.py --adx-hybrid-test       # Phase 3: Mean-Reversion vs
                                                 # Momentum vs ADX-switched
                                                 # Hybrid
    python3 backtest.py --fixed-hybrid-test     # Benchmark vs Unfixed vs
                                                 # Fixed (volume/ATR/dynamic
                                                 # ADX) Hybrid, Mixed universe
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import data
import screener
import strategy
from calendar_util import NY, is_cadence_bar, flat_time, is_trading_day
from config import (
    BACKTEST_DAYS, BACKTEST_CAPITAL, BACKTEST_REQUIRE_EXEC_RANGE,
    SCAN_INTERVAL_MIN, WINDOW_START, WINDOW_END, MAX_CONCURRENT,
    STOP_PCT, TARGET_R, COMMISSION_PER_TRADE, DATA,
    MAX_POSITION_PCT, RSI_PERIOD, RSI_OVERSOLD, BB_PERIOD, BB_K,
    OVERSOLD_REQUIRE_ABOVE_SMA200, SCALE_ENABLED, SCALE_R, SCALE_FRACTION,
)
from execution import State, classify, size_position, apply_slippage


# ---------------------------------------------------------------- records
@dataclass
class Trade:
    symbol: str
    entry_ts: pd.Timestamp
    entry_px: float
    shares: float
    stop: float
    target: float
    exit_ts: pd.Timestamp | None = None
    exit_px: float | None = None
    reason: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0
    bars_held: int = 0

    @property
    def is_win(self) -> bool:
        return self.pnl > 0


@dataclass
class Position:
    trade: Trade
    bars: int = 0


@dataclass
class Result:
    trades: list[Trade] = field(default_factory=list)
    equity: pd.Series = field(default_factory=pd.Series)
    skipped_concurrency: int = 0
    skipped_extended: int = 0
    signals_seen: int = 0
    sessions: int = 0
    symbols: list[str] = field(default_factory=list)
    start: dt.date | None = None
    end: dt.date | None = None


# ---------------------------------------------------------------- stats
def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score interval for a win rate.

    Reported because a headline win rate without an interval invites exactly the
    mistake of reading an 8-trade sample as evidence. If the interval spans 50%,
    the strategy has not been shown to beat a coin.
    """
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def max_drawdown(eq: pd.Series) -> tuple[float, float]:
    """(dollar drawdown, fractional drawdown) at the worst peak-to-trough."""
    if eq.empty:
        return 0.0, 0.0
    peak = eq.cummax()
    dd = eq - peak
    i = dd.idxmin()
    return float(-dd.loc[i]), float(-(dd / peak).loc[i])


def sharpe(eq: pd.Series, periods: int = 252) -> float:
    """Annualised Sharpe on daily equity returns, zero risk-free."""
    if len(eq) < 3:
        return 0.0
    r = eq.pct_change().dropna()
    if r.empty or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * math.sqrt(periods))


def sortino(eq: pd.Series, periods: int = 252) -> float:
    if len(eq) < 3:
        return 0.0
    r = eq.pct_change().dropna()
    down = r[r < 0]
    if down.empty or down.std() == 0:
        return 0.0
    return float(r.mean() / down.std() * math.sqrt(periods))


# ---------------------------------------------------------------- data prep
def load_symbol(sym: str) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    try:
        d = data.daily_bars(sym, period="2y")
        i = data.intraday_5m(sym, period="60d")
        return sym, d, i
    except Exception:
        return sym, pd.DataFrame(), pd.DataFrame()


def prepare(symbols: list[str], days: int, verbose: bool = True):
    """
    Download everything, then precompute per-(symbol, session) context.

    The daily context and the premarket high are constant for a whole session,
    so computing them once per session instead of once per bar turns an O(bars)
    problem into an O(sessions) one — the difference between a 4-minute run and
    a 40-second one.
    """
    if verbose:
        print(f"Loading {len(symbols)} symbols ...", flush=True)

    loaded = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for sym, d, i in ex.map(load_symbol, symbols):
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

    ctx: dict[tuple[str, dt.date], dict] = {}
    for sym, (daily, intra) in loaded.items():
        for sdate, sess in strategy.split_sessions(intra).items():
            if sdate not in keep:
                continue
            ph, pc, sma = strategy.daily_context(daily, sdate)
            if ph is None:
                continue
            mins = sess.index.hour * 60 + sess.index.minute
            pm = sess[(mins >= 240) & (mins < 570)]
            pmh = float(pm["High"].max()) if not pm.empty else None
            rth = sess[(mins >= 570) & (mins < 960)]
            if rth.empty or pmh is None:
                continue
            ctx[(sym, sdate)] = {"prev_high": ph, "prev_close": pc, "sma200": sma,
                                 "pmh": pmh, "rth": rth}

    if verbose:
        print(f"  {len(loaded)} symbols · {len(all_dates)} sessions · "
              f"{len(ctx)} symbol-sessions with complete context", flush=True)
    return ctx, all_dates


# ---------------------------------------------------------------- engine
def run_backtest(symbols: list[str], days: int = BACKTEST_DAYS,
                 capital: float = BACKTEST_CAPITAL,
                 require_range: bool = BACKTEST_REQUIRE_EXEC_RANGE,
                 scores: dict[str, float] | None = None,
                 stop_pct: float = STOP_PCT, target_r: float = TARGET_R,
                 max_concurrent: int = MAX_CONCURRENT,
                 verbose: bool = True) -> Result:
    """
    Portfolio-level event loop, stepped one TIMESTAMP at a time.

    Stepping by timestamp rather than by (timestamp, symbol) matters. At a
    15-minute scan the live engine sees all 30 names at once and chooses among
    them. A per-symbol stream instead fills the concurrency cap in whatever
    order the symbols happen to sort — which, with a cap of 2 and ~4 signals a
    day, means the alphabet decides roughly half the trades taken. Here the
    candidates at each scan are RANKED (screener score, then least-extended)
    and the best ones get the slots.
    """
    ctx, dates = prepare(symbols, days, verbose)
    res = Result(symbols=sorted({s for s, _ in ctx.keys()}), sessions=len(dates))
    if not ctx:
        return res

    scores = scores or {}
    res.start, res.end = dates[0], dates[-1]
    cash = capital
    open_pos: dict[str, Position] = {}
    equity_points: list[tuple[dt.date, float]] = []

    win_start = WINDOW_START[0] * 60 + WINDOW_START[1]
    win_end = WINDOW_END[0] * 60 + WINDOW_END[1]

    for sdate in dates:
        todays = {sym: c for (sym, d), c in ctx.items() if d == sdate}
        if not todays:
            if equity_points:
                equity_points.append((sdate, equity_points[-1][1]))
            continue

        # bar lookup per symbol per timestamp
        frames = {sym: c["rth"] for sym, c in todays.items()}
        pos_of = {sym: {ts: i for i, ts in enumerate(f.index)}
                  for sym, f in frames.items()}
        timestamps = sorted({ts for f in frames.values() for ts in f.index})

        hod: dict[str, float] = {}
        entered_today: set[str] = set()
        last_px: dict[str, float] = {}
        flat_at = flat_time(sdate)
        flat_min = flat_at.hour * 60 + flat_at.minute

        for ts in timestamps:
            tmin = ts.hour * 60 + ts.minute
            live = {}
            for sym, f in frames.items():
                i = pos_of[sym].get(ts)
                if i is None:
                    continue
                b = f.iloc[i]
                live[sym] = (float(b["High"]), float(b["Low"]), float(b["Close"]))
                last_px[sym] = live[sym][2]

            # ---------- phase 1: exits. A resting stop or target fills before
            # ---------- the engine makes any new decision on the same bar.
            for sym, (h, l, cl) in live.items():
                pos = open_pos.get(sym)
                if not pos:
                    continue
                pos.bars += 1
                t = pos.trade
                exit_px = exit_reason = None

                if l <= t.stop:
                    # Pessimistic: if the bar also touched the target, assume
                    # the stop went first. 5-minute OHLC cannot order them.
                    exit_px, exit_reason = t.stop, "STOP"
                elif h >= t.target:
                    exit_px, exit_reason = t.target, "TARGET"
                elif tmin >= flat_min:
                    exit_px, exit_reason = cl, "EOD_FLAT"

                if exit_px is not None:
                    fill = apply_slippage(exit_px, "sell")
                    proceeds = fill * t.shares - COMMISSION_PER_TRADE
                    cash += proceeds
                    t.exit_ts, t.exit_px, t.reason = ts, fill, exit_reason
                    t.pnl = proceeds - (t.entry_px * t.shares)
                    t.pnl_pct = fill / t.entry_px - 1.0
                    t.bars_held = pos.bars
                    res.trades.append(t)
                    del open_pos[sym]

            # ---------- phase 2: entries, ranked
            if win_start <= tmin <= win_end and is_cadence_bar(ts, SCAN_INTERVAL_MIN):
                candidates = []
                for sym, (h, l, cl) in live.items():
                    if sym in open_pos or sym in entered_today:
                        continue
                    hod_prev = hod.get(sym)
                    if hod_prev is None:
                        continue
                    c = todays[sym]
                    lv = strategy.Levels(symbol=sym, prev_high=c["prev_high"],
                                         prev_close=c["prev_close"],
                                         sma200=c["sma200"], pmh=c["pmh"],
                                         hod_prev=hod_prev)
                    if not strategy.evaluate_signal(lv, cl).fired:
                        continue
                    res.signals_seen += 1
                    state, ext = classify(cl, lv.trigger)
                    if require_range and state is not State.EXECUTE:
                        res.skipped_extended += 1
                        continue
                    candidates.append((sym, cl, ext))

                # Best first: screener conviction, then the least-extended
                # entry (a fill nearer the trigger has a tighter effective risk).
                candidates.sort(key=lambda x: (-scores.get(x[0], 0.0), x[2]))

                for sym, cl, _ext in candidates:
                    if len(open_pos) >= max_concurrent:
                        res.skipped_concurrency += 1
                        continue
                    eq = cash + sum(p.trade.shares * last_px.get(s, p.trade.entry_px)
                                    for s, p in open_pos.items())
                    fill = apply_slippage(cl, "buy")
                    stop = fill * (1.0 - stop_pct)
                    shares, _note = size_position(eq, fill, stop)
                    cost = shares * fill + COMMISSION_PER_TRADE
                    if shares > 0 and cost <= cash:
                        cash -= cost
                        open_pos[sym] = Position(Trade(
                            symbol=sym, entry_ts=ts, entry_px=fill,
                            shares=shares, stop=stop,
                            target=fill + (fill - stop) * target_r))
                        entered_today.add(sym)

            # ---------- phase 3: HOD update, deliberately LAST so that
            # ---------- hod_prev above never includes the current bar
            for sym, (h, _l, _cl) in live.items():
                hod[sym] = max(hod.get(sym, h), h)

        # ---------- safety net: force-close anything still open at session end
        for sym, pos in list(open_pos.items()):
            t = pos.trade
            px = apply_slippage(last_px.get(sym, t.entry_px), "sell")
            proceeds = px * t.shares - COMMISSION_PER_TRADE
            cash += proceeds
            t.exit_ts = frames[sym].index[-1]
            t.exit_px, t.reason = px, "SESSION_END"
            t.pnl = proceeds - (t.entry_px * t.shares)
            t.pnl_pct = px / t.entry_px - 1.0
            t.bars_held = pos.bars
            res.trades.append(t)
            del open_pos[sym]

        equity_points.append((sdate, cash))

    res.equity = pd.Series(dict(equity_points)).sort_index()
    return res


# ---------------------------------------------------------------- reporting
def report(res: Result, capital: float, require_range: bool) -> str:
    t = res.trades
    n = len(t)
    wins = [x for x in t if x.is_win]
    losses = [x for x in t if not x.is_win]

    gross_win = sum(x.pnl for x in wins)
    gross_loss = -sum(x.pnl for x in losses)
    net = sum(x.pnl for x in t)
    final = capital + net

    wr = len(wins) / n if n else 0.0
    lo, hi = wilson_ci(len(wins), n)
    pf = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
    dd_d, dd_p = max_drawdown(res.equity) if not res.equity.empty else (0.0, 0.0)

    L = []
    A = L.append
    A("=" * 78)
    A("  59-DAY INTRADAY BACKTEST — Hybrid Screener + Trend Join Long")
    A("=" * 78)
    A(f"  Window          {res.start} → {res.end}  ({res.sessions} sessions)")
    A(f"  Universe        {len(res.symbols)} symbols")
    A(f"  Starting cap    ${capital:,.2f}")
    A(f"  Cadence         every {SCAN_INTERVAL_MIN} min, "
      f"{WINDOW_START[0]:02d}:{WINDOW_START[1]:02d}–{WINDOW_END[0]:02d}:{WINDOW_END[1]:02d} ET")
    A(f"  Exits           {STOP_PCT*100:.1f}% stop · {TARGET_R:.1f}R target · flat 15:55 ET")
    A(f"  Exec-range gate {'ON' if require_range else 'OFF (ablation)'}")
    A("")
    A("  " + "-" * 74)
    A(f"  {'METRIC':<28} {'VALUE':>20}   {'NOTE':<20}")
    A("  " + "-" * 74)
    A(f"  {'Total trades':<28} {n:>20}   {'':<20}")
    A(f"  {'Win rate':<28} {wr*100:>19.1f}%   95% CI {lo*100:.0f}–{hi*100:.0f}%")
    A(f"  {'Winners / losers':<28} {f'{len(wins)} / {len(losses)}':>20}   {'':<20}")
    A(f"  {'Net P&L ($)':<28} {net:>+20,.2f}   {'':<20}")
    A(f"  {'Net P&L (%)':<28} {net/capital*100:>+19.2f}%   on starting capital")
    A(f"  {'Final equity':<28} {final:>20,.2f}   {'':<20}")
    A(f"  {'Max drawdown ($)':<28} {dd_d:>20,.2f}   {'':<20}")
    A(f"  {'Max drawdown (%)':<28} {dd_p*100:>19.2f}%   {'':<20}")
    A(f"  {'Sharpe ratio':<28} {sharpe(res.equity):>20.3f}   annualised, daily")
    A(f"  {'Sortino ratio':<28} {sortino(res.equity):>20.3f}   {'':<20}")
    A(f"  {'Profit factor':<28} {pf:>20.3f}   {'':<20}")
    A("  " + "-" * 74)

    # Benchmark. A long-only strategy's P&L is close to meaningless without
    # knowing what the tape did underneath it.
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

    if n:
        aw = gross_win / len(wins) if wins else 0.0
        al = gross_loss / len(losses) if losses else 0.0
        A(f"  {'Average win':<28} {aw:>+20,.2f}")
        A(f"  {'Average loss':<28} {-al:>+20,.2f}")
        A(f"  {'Expectancy / trade':<28} {net/n:>+20,.2f}")
        A(f"  {'Avg bars held (5m)':<28} {np.mean([x.bars_held for x in t]):>20.1f}")
        A("  " + "-" * 74)

    A("")
    A("  FUNNEL")
    A(f"    signals fired                 {res.signals_seen}")
    A(f"    skipped — outside exec range  {res.skipped_extended}")
    A(f"    skipped — book full           {res.skipped_concurrency}")
    A(f"    trades taken                  {n}")

    if n:
        by_reason: dict[str, list[Trade]] = {}
        for x in t:
            by_reason.setdefault(x.reason, []).append(x)
        A("")
        A("  EXITS")
        for r, xs in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
            A(f"    {r:<14} {len(xs):>3}  ({len(xs)/n*100:>4.0f}%)  "
              f"net {sum(x.pnl for x in xs):>+10,.2f}")

        by_sym: dict[str, list[Trade]] = {}
        for x in t:
            by_sym.setdefault(x.symbol, []).append(x)
        A("")
        A("  BY SYMBOL (most active first)")
        A(f"    {'SYM':<7}{'N':>4}{'WINS':>6}{'NET $':>12}")
        for s, xs in sorted(by_sym.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:15]:
            A(f"    {s:<7}{len(xs):>4}{sum(1 for x in xs if x.is_win):>6}"
              f"{sum(x.pnl for x in xs):>+12,.2f}")

    A("")
    A("  " + "!" * 74)
    A("  BIAS DISCLOSURE")
    A("  The watchlist was selected using TODAY's fundamentals and then run")
    A("  backwards over this window. yfinance serves no point-in-time balance")
    A("  sheets, so every name is one that survived to today looking healthy.")
    A("  This measures the TECHNICAL layer on a hindsight-chosen universe — it")
    A("  is not evidence that the screener adds value.")
    if n < 30:
        A("")
        A(f"  SAMPLE SIZE: {n} trades is too few to conclude anything. The win-rate")
        A(f"  interval above ({lo*100:.0f}–{hi*100:.0f}%) is the honest summary.")
    A("  " + "!" * 74)
    return "\n".join(L)


# =============================================================================
# MULTI-YEAR SPY-REGIME x UNIVERSE STRESS TEST — daily bars
#
# WHY THIS IS A SEPARATE ENGINE FROM run_backtest() ABOVE
#   data.intraday_5m() is hard-capped by Yahoo at 60 calendar days of history
#   (see its docstring) — no argument gets 5-minute bars back to Jan 2022, and
#   even hourly bars only reach back ~2 years. Only DAILY bars
#   (data.daily_bars(sym, period="max")) go back that far. So this section
#   re-expresses the oversold-dip signal on DAILY closes: strategy.rsi() and
#   strategy.bollinger() are plain array functions, so the same math applies,
#   but this is a genuinely lower-fidelity approximation of the live hourly
#   signal, not the live-fidelity 5-minute engine above. Read its results as
#   "does the edge and the regime filter survive at daily resolution over
#   years," not as a replacement for the 59-day intraday backtest.
# =============================================================================

REGIME_START = dt.date(2022, 1, 1)

# Curated, not pulled from an index provider — "e.g." lists of liquid names in
# each bucket, picked for continuous trading history back to 2022 (nothing
# delisted/bankrupt in the window) rather than official index membership.
LARGE_CAP_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "META", "BRK-B", "LLY", "AVGO", "JPM",
    "V", "UNH", "XOM", "MA", "PG", "HD", "COST", "MRK", "ABBV", "CVX",
    "PEP", "KO", "ADBE", "WMT", "BAC", "CRM", "TMO", "MCD", "ACN", "LIN",
]

SMALL_CAP_UNIVERSE = [
    "UPST", "AFRM", "SOFI", "OPEN", "CVNA", "FUBO", "PLUG", "RIOT", "MARA", "CLSK",
    "CHPT", "BLNK", "FCEL", "SPCE", "RKT", "LAZR", "ASTS", "JOBY", "ACHR", "BYND",
    "SFIX", "PTON", "CPRI", "DNUT", "YETI", "FIVE", "CAKE", "RUN", "GPRO", "IRDM",
]

MIXED_UNIVERSE = LARGE_CAP_UNIVERSE[:15] + SMALL_CAP_UNIVERSE[:15]

# 100 high-volume S&P 100 / Nasdaq 100 constituents, spanning tech, comms,
# discretionary, staples, financials, healthcare, industrials, energy,
# utilities and materials — for testing whether widening the universe raises
# setup frequency without degrading risk-adjusted returns. Every symbol here
# was verified (not assumed) to clear $10M/day average dollar volume: the
# lowest of the 100, CL, ran ~$380M/day over its trailing 20 sessions as of
# the check date — roughly 38x the $10M floor — so this is a "curated from
# obviously liquid mega/large-caps" list, not one built against the $10M bar
# specifically (nothing here comes remotely close to it).
EXPANDED_UNIVERSE_100 = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "AVGO", "ORCL",
    "CRM", "ADBE", "CSCO", "ACN", "AMD", "INTC", "QCOM", "TXN", "IBM", "NOW",
    "INTU", "AMAT", "MU", "ADI", "LRCX", "KLAC", "SNPS", "CDNS", "PANW", "CRWD",
    "FTNT", "ANET",
    "NFLX", "CMCSA", "TMUS", "CHTR", "DIS",
    "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "BKNG", "CMG", "ORLY", "MAR",
    "ABNB", "TGT",
    "WMT", "PG", "KO", "PEP", "COST", "PM", "CL", "MO",
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP", "SPGI", "BLK",
    "C", "SCHW", "PYPL",
    "UNH", "LLY", "JNJ", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR",
    "AMGN", "GILD", "VRTX", "ISRG", "CVS", "MDT", "REGN",
    "CAT", "BA", "HON", "UPS", "RTX", "LMT", "GE", "DE", "UNP",
    "XOM", "CVX", "COP",
    "NEE", "LIN",
]

# 100 liquid US large-caps, HEAVILY WEIGHTED to Technology/Media/Telecom (62/100,
# vs. EXPANDED_UNIVERSE_100's deliberately balanced ~37/100) — built to test
# whether concentrating in high-beta TMT momentum names raises trade frequency
# without the diversification EXPANDED_UNIVERSE_100 relies on. Every symbol
# checked (not assumed) via data.daily_bars() for >=210 sessions of history
# over the 2022-present backtest window before being kept: this caught EA
# (down to a single trading day, 2026-08-04, on unusually heavy volume —
# consistent with a take-private buyout closing) and IPG (zero rows —
# consistent with the real Omnicom/Interpublic merger having closed) as
# delisted by the current date; both were swapped for RBLX and FOXA
# respectively after independently verifying those trade normally through
# today.
TMT_TECH = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "AVGO", "ORCL",
    "CRM", "ADBE", "CSCO", "ACN", "AMD", "INTC", "QCOM", "TXN", "IBM", "NOW",
    "INTU", "AMAT", "MU", "ADI", "LRCX", "KLAC", "SNPS", "CDNS", "PANW", "CRWD",
    "FTNT", "ANET", "DELL", "HPQ", "HPE", "WDC", "STX", "MRVL", "ENPH", "FSLR",
    "PLTR", "SNOW", "DDOG", "ZS", "NET", "MDB", "TEAM", "WDAY", "KEYS", "ROP",
]
TMT_MEDIA_TELECOM = [
    "NFLX", "DIS", "CMCSA", "TMUS", "VZ", "CHTR", "T", "WBD", "RBLX", "TTWO",
    "OMC", "FOXA",
]
TMT_DIVERSIFIED_FILL = [
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP",
    "UNH", "LLY", "JNJ", "ABBV", "MRK", "PFE", "TMO", "ABT",
    "HD", "MCD", "NKE", "SBUX", "WMT", "PG", "KO", "PEP",
    "CAT", "BA", "HON", "UPS", "RTX", "GE",
    "XOM", "CVX", "COP",
    "LIN", "NEE", "COST", "TGT", "LOW",
]
TMT_HEAVY_UNIVERSE_100 = TMT_TECH + TMT_MEDIA_TELECOM + TMT_DIVERSIFIED_FILL

UNIVERSES = {
    "Large-Cap": LARGE_CAP_UNIVERSE,
    "Small-Cap": SMALL_CAP_UNIVERSE,
    "Mixed": MIXED_UNIVERSE,
    "Expanded100": EXPANDED_UNIVERSE_100,
    "TMT100": TMT_HEAVY_UNIVERSE_100,
}


@dataclass
class DailyTrade:
    symbol: str
    entry_date: dt.date
    entry_px: float
    shares: float
    stop: float
    target: float
    exit_date: dt.date | None = None
    exit_px: float | None = None
    reason: str = ""
    pnl: float = 0.0

    @property
    def is_win(self) -> bool:
        return self.pnl > 0


@dataclass
class DailyResult:
    trades: list[DailyTrade] = field(default_factory=list)
    equity: pd.Series = field(default_factory=pd.Series)
    universe_name: str = ""
    regime_filter: bool = False
    variant: str = "A"
    sector_top_n: int | None = None
    start: dt.date | None = None
    end: dt.date | None = None


def spy_regime_ok(start: dt.date, end: dt.date) -> pd.Series:
    """
    Daily boolean series (index: date): True on sessions where SPY's close is
    at/above its 200-day SMA — new entries allowed; False below it — halt new
    entries (open positions still manage normally). Pulls SPY's FULL history
    so the 200-day average is real at `start`, not cold-started at the
    window's edge.
    """
    spy = data.daily_bars("SPY", period="max")
    sma200 = spy["Close"].rolling(200).mean()
    ok = spy["Close"] >= sma200
    ok.index = pd.Index([ts.date() for ts in spy.index])
    return ok[(ok.index >= start) & (ok.index <= end)]


def ibs(daily: pd.DataFrame) -> np.ndarray:
    """
    Internal Bar Strength: (Close - Low) / (High - Low) — where in the day's
    own range the close landed. Near 0 = closed at the low (weak close, sold
    off into the bell); near 1 = closed at the high. A flat bar (High == Low,
    e.g. a halt) has no defined range and is scored neutral 0.5 so it neither
    triggers nor blocks a signal by accident.
    """
    high = daily["High"].to_numpy(dtype=float)
    low = daily["Low"].to_numpy(dtype=float)
    close = daily["Close"].to_numpy(dtype=float)
    rng = high - low
    safe_rng = np.where(rng > 1e-9, rng, 1.0)
    return np.where(rng > 1e-9, (close - low) / safe_rng, 0.5)


def consecutive_down_days(close: np.ndarray) -> np.ndarray:
    """
    Count of consecutive lower closes ending at (and including) each index.
    close[i] < close[i-1] extends yesterday's streak by one; anything else
    (an up day, a flat day, or index 0) resets it to 0.
    """
    n = len(close)
    streak = np.zeros(n, dtype=int)
    for i in range(1, n):
        if close[i] < close[i - 1]:
            streak[i] = streak[i - 1] + 1
    return streak


# Phase 1 research pipeline: which oversold TRIGGER fires the entry. All three
# share the same background trend gate (prior close above its 200-SMA, still
# above it today) as the live strategy, so the comparison isolates the
# trigger's own effect rather than also toggling whether an uptrend filter
# exists at all.
IBS_DIP_THRESHOLD = 0.25          # Variant B
IBS_DOWN_STREAK = 2               # Variant B
IBS_COMBINED_THRESHOLD = 0.30     # Variant C

IBS_VARIANT_LABELS = {
    "A": "RSI<35 or BB touch",
    "B": "IBS<0.25 & 2 down days",
    "C": "RSI<35 & IBS<0.30",
}
IBS_VARIANT_DESCRIPTIONS = {
    "A": "Standard Oversold Dip: RSI(14) < 35 OR a touch of the lower daily "
        "Bollinger(20, 2.0) band",
    "B": f"Pure IBS Dip: Internal Bar Strength < {IBS_DIP_THRESHOLD} AND "
        f"{IBS_DOWN_STREAK} consecutive down days",
    "C": f"Combined Trigger: RSI(14) < {RSI_OVERSOLD:.0f} AND Internal Bar "
        f"Strength < {IBS_COMBINED_THRESHOLD}",
}


def daily_signal_frame(daily: pd.DataFrame, variant: str = "A") -> pd.DataFrame:
    """
    Adds RSI(14), lower Bollinger(20, 2.0), SMA200, IBS and a consecutive
    down-day count to `daily`, then a SIGNAL column per `variant`
    (IBS_VARIANT_DESCRIPTIONS). Variant "A" reproduces the original oversold
    signal exactly — same trend gate, same RSI/BB trigger — so it is the
    apples-to-apples baseline the other two are measured against.
    """
    close = daily["Close"].to_numpy(dtype=float)
    low = daily["Low"].to_numpy(dtype=float)
    rsi_v = strategy.rsi(close, RSI_PERIOD)
    _, bb_low, _ = strategy.bollinger(close, BB_PERIOD, BB_K)
    sma200 = daily["Close"].rolling(200).mean().to_numpy()
    ibs_v = ibs(daily)
    down_streak = consecutive_down_days(close)

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

    rsi_ok = np.where(np.isnan(rsi_v), False, rsi_v < RSI_OVERSOLD)
    bb_ok = np.where(np.isnan(bb_low), False, low <= bb_low)
    ibs_b_ok = ibs_v < IBS_DIP_THRESHOLD
    ibs_c_ok = ibs_v < IBS_COMBINED_THRESHOLD
    down_ok = down_streak >= IBS_DOWN_STREAK

    trigger = {
        "A": rsi_ok | bb_ok,
        "B": ibs_b_ok & down_ok,
        "C": rsi_ok & ibs_c_ok,
    }[variant]

    out = daily.copy()
    out["RSI"] = rsi_v
    out["BB_LOW"] = bb_low
    out["SMA200"] = sma200
    out["IBS"] = ibs_v
    out["DOWN_STREAK"] = down_streak
    out["SIGNAL"] = trend_ok & above_now & trigger
    return out


def daily_oversold_frame(daily: pd.DataFrame) -> pd.DataFrame:
    """Back-compat alias for daily_signal_frame(daily, variant="A")."""
    return daily_signal_frame(daily, variant="A")


def _load_daily_frames(symbols: list[str], start: dt.date, end: dt.date,
                       variant: str = "A") -> dict[str, pd.DataFrame]:
    """
    Parallel-fetch full daily history per symbol (data.daily_bars caches by
    (symbol, period), so calling this again for an overlapping universe, e.g.
    Mixed reusing Large-Cap/Small-Cap names, or a second variant/filter run
    over the same universe, is served from cache rather than re-downloaded).
    Trims to a seed window before `start` (long enough for RSI/SMA200 to warm
    up) THROUGH `end` — the upper trim matters: data.daily_bars(period="max")
    always includes bars up to today, so without it a window ending before
    today would force-close any still-open position at TODAY's price instead
    of the price at `end`, a real look-ahead bug caught in testing.
    """
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
            frames[sym] = daily_signal_frame(d, variant=variant)
    return frames


def run_daily_backtest(symbols: list[str], start: dt.date, end: dt.date,
                       capital: float = BACKTEST_CAPITAL,
                       regime_filter: bool = False,
                       universe_name: str = "",
                       variant: str = "A",
                       sector_top_n: int | None = None,
                       sector_ranks: dict[dt.date, dict[str, int]] | None = None
                       ) -> DailyResult:
    """
    Portfolio-level daily-bar event loop over [start, end].

    Signal-to-fill has a one-day lag by construction: a SIGNAL seen at day T's
    close is queued and filled at day T+1's OPEN (with slippage) — never at
    T's own close, which the order could not have seen in time. Exits (stop
    hit -> pessimistic if the target also traded that bar, matching
    run_backtest()'s convention, else target) are checked first each day on
    positions already open. New entries beyond MAX_CONCURRENT are dropped
    (ranked most-oversold, i.e. lowest RSI, first — even for variant B/C,
    where RSI is not itself the trigger, so entries still prioritize the
    deepest dip when several signal the same day). When regime_filter=True, a
    day where SPY closed below its 200-day SMA THAT day queues no new entries
    for the next open — open positions still manage normally. `variant`
    selects the oversold trigger definition (see IBS_VARIANT_DESCRIPTIONS).

    `sector_top_n` (Phase 2 — see sector_momentum_ranks()) additionally
    requires a candidate's SECTOR_MAP-mapped SPDR ETF to rank in the top N by
    90-trading-day ROC on the signal day, using `sector_ranks` (precomputed
    once by the caller and shared across runs — see run_sector_rotation_test).
    A day with no rank yet (sector history still warming up) or a symbol with
    no sector mapping blocks that entry rather than admitting it by default.
    """
    frames = _load_daily_frames(symbols, start, end, variant=variant)
    res = DailyResult(universe_name=universe_name, regime_filter=regime_filter,
                      variant=variant, sector_top_n=sector_top_n,
                      start=start, end=end)
    if not frames:
        return res

    regime = spy_regime_ok(start, end) if regime_filter else None
    all_dates = sorted({ts.date() for d in frames.values()
                        for ts in d.index if start <= ts.date() <= end})
    if not all_dates:
        return res

    cap_pct = min(MAX_POSITION_PCT, 1.0 / MAX_CONCURRENT)
    cash = capital
    open_pos: dict[str, DailyTrade] = {}
    pending_entries: list[str] = []
    equity_points: list[tuple[dt.date, float]] = []

    for sdate in all_dates:
        todays = {}
        for sym, d in frames.items():
            row = d[d.index.date == sdate]
            if not row.empty:
                todays[sym] = row.iloc[0]

        # ---- phase 1: exits on positions already open, using today's OHLC
        for sym, t in list(open_pos.items()):
            row = todays.get(sym)
            if row is None:
                continue
            h, l = float(row["High"]), float(row["Low"])
            exit_px = exit_reason = None
            if l <= t.stop:
                exit_px, exit_reason = t.stop, "STOP"
            elif h >= t.target:
                exit_px, exit_reason = t.target, "TARGET"
            if exit_px is not None:
                fill = apply_slippage(exit_px, "sell")
                proceeds = fill * t.shares - COMMISSION_PER_TRADE
                cash += proceeds
                t.exit_date, t.exit_px, t.reason = sdate, fill, exit_reason
                t.pnl = proceeds - (t.entry_px * t.shares)
                res.trades.append(t)
                del open_pos[sym]

        # ---- phase 2: fill yesterday's queued signals at TODAY's open
        for sym in pending_entries:
            if sym in open_pos or len(open_pos) >= MAX_CONCURRENT:
                continue
            row = todays.get(sym)
            if row is None:
                continue
            open_px = float(row["Open"])
            eq = cash + sum(
                p.shares * float(todays[s]["Close"]) if s in todays else p.shares * p.entry_px
                for s, p in open_pos.items())
            fill = apply_slippage(open_px, "buy")
            stop = fill * (1.0 - STOP_PCT)
            shares, _note = size_position(eq, fill, stop, max_position_pct=cap_pct)
            cost = shares * fill + COMMISSION_PER_TRADE
            if shares > 0 and cost <= cash:
                cash -= cost
                open_pos[sym] = DailyTrade(
                    symbol=sym, entry_date=sdate, entry_px=fill, shares=shares,
                    stop=stop, target=fill + (fill - stop) * TARGET_R)
        pending_entries = []

        # ---- phase 3: scan today's closes for new signals, queue for tomorrow
        allow_entries = True if regime is None else bool(regime.get(sdate, False))
        if allow_entries:
            candidates = []
            day_ranks = sector_ranks.get(sdate) if sector_ranks else None
            for sym, row in todays.items():
                if sym in open_pos or not bool(row["SIGNAL"]):
                    continue
                if sector_top_n is not None:
                    etf = SECTOR_MAP.get(sym)
                    rank = day_ranks.get(etf) if (day_ranks and etf) else None
                    if rank is None or rank > sector_top_n:
                        continue
                rsi_v = float(row["RSI"])
                candidates.append((sym, rsi_v if rsi_v == rsi_v else 999.0))
            candidates.sort(key=lambda x: x[1])
            pending_entries = [sym for sym, _ in candidates]

        mark = cash + sum(
            p.shares * float(todays[s]["Close"]) if s in todays else p.shares * p.entry_px
            for s, p in open_pos.items())
        equity_points.append((sdate, mark))

    # force-close anything still open at the end of the window
    for sym, t in list(open_pos.items()):
        last_close = float(frames[sym]["Close"].iloc[-1])
        fill = apply_slippage(last_close, "sell")
        proceeds = fill * t.shares - COMMISSION_PER_TRADE
        cash += proceeds
        t.exit_date, t.exit_px, t.reason = all_dates[-1], fill, "WINDOW_END"
        t.pnl = proceeds - (t.entry_px * t.shares)
        res.trades.append(t)
    if equity_points:
        equity_points[-1] = (equity_points[-1][0], cash)

    res.equity = pd.Series(dict(equity_points)).sort_index()
    return res


def run_six_way_stress_test(start: dt.date = REGIME_START,
                            end: dt.date | None = None,
                            capital: float = BACKTEST_CAPITAL,
                            verbose: bool = True) -> dict[str, DailyResult]:
    """Runs all 3 universes x {filter off, on} = 6 backtests over one window."""
    end = end or dt.date.today()
    results: dict[str, DailyResult] = {}
    for uni_name, symbols in UNIVERSES.items():
        for filt in (False, True):
            label = f"{uni_name} / Filter {'ON' if filt else 'OFF'}"
            if verbose:
                print(f"  running {label} ...", flush=True)
            results[label] = run_daily_backtest(
                symbols, start, end, capital=capital,
                regime_filter=filt, universe_name=uni_name)
    return results


def print_six_way_table(results: dict[str, DailyResult], capital: float,
                        start: dt.date, end: dt.date) -> None:
    rows = []
    for res in results.values():
        t = res.trades
        n = len(t)
        wins = [x for x in t if x.is_win]
        losses = [x for x in t if not x.is_win]
        gross_win = sum(x.pnl for x in wins)
        gross_loss = -sum(x.pnl for x in losses)
        net = sum(x.pnl for x in t)
        wr = len(wins) / n if n else 0.0
        pf = (gross_win / gross_loss if gross_loss > 0
              else (float("inf") if gross_win > 0 else 0.0))
        _, dd_p = max_drawdown(res.equity) if not res.equity.empty else (0.0, 0.0)
        rows.append((res.universe_name, "ON" if res.regime_filter else "OFF",
                    net / capital * 100.0, dd_p * 100.0, wr * 100.0, pf, n))

    width = 84
    print("=" * width)
    print(f"6-WAY SPY-REGIME x UNIVERSE STRESS TEST  ({start} -> {end}, daily bars)")
    print("=" * width)
    header = (f"{'Universe':<12}{'SPY Filter':<12}{'Total Ret':>11}"
             f"{'Max DD':>10}{'Win Rate':>10}{'Profit Fac':>12}{'Trades':>9}")
    print(header)
    print("-" * width)
    for uni, filt, ret, dd, wr, pf, n in rows:
        pf_s = f"{pf:.2f}" if pf != float("inf") else "inf"
        print(f"{uni:<12}{filt:<12}{ret:>+10.2f}%{dd:>9.2f}%{wr:>9.1f}%"
              f"{pf_s:>12}{n:>9}")
    print("=" * width)

    try:
        spy = data.daily_bars("SPY", period="max")
        w = spy[(spy.index.date >= start) & (spy.index.date <= end)]
        if len(w) > 1:
            r = float(w["Close"].iloc[-1] / w["Close"].iloc[0] - 1.0)
            print(f"SPY buy & hold, same window: {r*100:+.2f}%")
    except Exception:
        pass

    print()
    print("!" * width)
    print("METHODOLOGY NOTE: this test runs on DAILY bars, not the 5-minute")
    print("engine above — Yahoo caps intraday history at 60 days, which cannot")
    print("reach 2022. The signal is a daily-bar analogue of the live oversold")
    print("entry (RSI(14)/Bollinger(20,2) on daily closes, not hourly), so this")
    print("shows whether the EDGE and the REGIME FILTER survive at daily")
    print("resolution over years — it is not a live-fidelity backtest, and the")
    print("universes are curated examples, not an official index pull.")
    print("!" * width)


# =============================================================================
# PHASE 1 — IBS & CONSECUTIVE DOWN-DAY TRIGGER COMPARISON
#
# Same daily-bar engine as the regime stress test above, same trend gate, SPY
# filter held OFF throughout (per the request — this isolates the trigger's
# own effect; the regime filter's effect was already measured separately).
# Only the oversold TRIGGER changes between variants A/B/C — see
# IBS_VARIANT_DESCRIPTIONS above daily_signal_frame().
# =============================================================================

def run_ibs_variant_test(start: dt.date = REGIME_START,
                         end: dt.date | None = None,
                         capital: float = BACKTEST_CAPITAL,
                         universes: tuple[str, ...] = ("Small-Cap", "Mixed"),
                         verbose: bool = True) -> dict[str, DailyResult]:
    """Runs variants A/B/C on each of `universes`, SPY filter OFF throughout."""
    end = end or dt.date.today()
    results: dict[str, DailyResult] = {}
    for uni_name in universes:
        symbols = UNIVERSES[uni_name]
        for variant in ("A", "B", "C"):
            label = f"{uni_name} / Variant {variant}"
            if verbose:
                print(f"  running {label} ({IBS_VARIANT_LABELS[variant]}) ...",
                      flush=True)
            results[label] = run_daily_backtest(
                symbols, start, end, capital=capital, regime_filter=False,
                universe_name=uni_name, variant=variant)
    return results


def print_ibs_variant_table(results: dict[str, DailyResult], capital: float,
                            start: dt.date, end: dt.date) -> None:
    stats: dict[str, dict] = {}
    for label, res in results.items():
        t = res.trades
        n = len(t)
        wins = [x for x in t if x.is_win]
        losses = [x for x in t if not x.is_win]
        gross_win = sum(x.pnl for x in wins)
        gross_loss = -sum(x.pnl for x in losses)
        net = sum(x.pnl for x in t)
        wr = len(wins) / n if n else 0.0
        pf = (gross_win / gross_loss if gross_loss > 0
              else (float("inf") if gross_win > 0 else 0.0))
        _, dd_p = max_drawdown(res.equity) if not res.equity.empty else (0.0, 0.0)
        stats[label] = dict(universe=res.universe_name, variant=res.variant,
                            ret=net / capital * 100.0, dd=dd_p * 100.0,
                            wr=wr * 100.0, pf=pf, n=n)

    universes = sorted({s["universe"] for s in stats.values()})
    width = 96
    print("=" * width)
    print(f"PHASE 1 — IBS & CONSECUTIVE DOWN-DAY TRIGGERS  ({start} -> {end}, "
          f"daily bars, SPY filter OFF)")
    print("=" * width)
    header = (f"{'Universe':<11}{'Variant':<9}{'Trigger':<24}{'Total Ret':>10}"
             f"{'Δ vs A':>9}{'Max DD':>9}{'Win %':>8}{'PF':>7}{'Trades':>8}")
    print(header)
    print("-" * width)

    for uni in universes:
        baseline_ret = stats[f"{uni} / Variant A"]["ret"]
        for variant in ("A", "B", "C"):
            s = stats[f"{uni} / Variant {variant}"]
            delta = s["ret"] - baseline_ret
            pf_s = f"{s['pf']:.2f}" if s["pf"] != float("inf") else "inf"
            print(f"{uni:<11}{variant:<9}{IBS_VARIANT_LABELS[variant]:<24}"
                  f"{s['ret']:>+9.2f}%{delta:>+8.2f}%{s['dd']:>8.2f}%"
                  f"{s['wr']:>7.1f}%{pf_s:>7}{s['n']:>8}")
        print("-" * width)
    print("=" * width)

    print()
    print("Trigger definitions (all three share the SAME background trend gate")
    print("as the live strategy — prior close above its 200-SMA, still above it")
    print("today — so this isolates the trigger's own effect):")
    for v, desc in IBS_VARIANT_DESCRIPTIONS.items():
        print(f"  {v}: {desc}")
    print()
    print("Δ vs A is each universe's own Variant A run in THIS session, not a")
    print("hardcoded number — the prior regime-stress-test's SPY-filter-OFF")
    print("baselines (Small-Cap +73.24%, Mixed +56.20%) should reappear here as")
    print("Variant A, confirming no behavior change to the baseline signal.")


# =============================================================================
# PHASE 2 — SECTOR ROTATION (DUAL MOMENTUM)
#
# Carries forward Variant A (the winning trigger from Phase 1) unchanged. The
# new layer is a RELATIVE-STRENGTH gate on top of it: a dip-buy is admitted
# only if its parent SPDR sector ETF currently ranks near the top by
# 90-trading-day momentum — the "dual" in dual momentum being this absolute
# sector ranking combined with the existing absolute per-stock oversold
# trigger. SPY filter stays OFF throughout, per the request — this isolates
# the sector filter's own effect from the (already separately measured)
# market-regime filter.
# =============================================================================

SECTOR_ETFS = ["XLK", "XLE", "XLF", "XLI", "XLV", "XLP", "XLU", "XLY", "XLC",
              "XLB", "XLRE"]

SECTOR_ROC_LOOKBACK = 90    # TRADING days, not calendar days

# Curated GICS-sector mapping (2023 GICS revision: V/MA/UPST/AFRM/SOFI/RKT
# sit under Financials, not Technology) — not pulled from a data provider, to
# match how LARGE_CAP_UNIVERSE / SMALL_CAP_UNIVERSE above were hand-picked.
SECTOR_MAP = {
    # Large-cap
    "AAPL": "XLK", "MSFT": "XLK", "GOOGL": "XLC", "NVDA": "XLK", "AMZN": "XLY",
    "META": "XLC", "BRK-B": "XLF", "LLY": "XLV", "AVGO": "XLK", "JPM": "XLF",
    "V": "XLF", "UNH": "XLV", "XOM": "XLE", "MA": "XLF", "PG": "XLP",
    "HD": "XLY", "COST": "XLP", "MRK": "XLV", "ABBV": "XLV", "CVX": "XLE",
    "PEP": "XLP", "KO": "XLP", "ADBE": "XLK", "WMT": "XLP", "BAC": "XLF",
    "CRM": "XLK", "TMO": "XLV", "MCD": "XLY", "ACN": "XLK", "LIN": "XLB",
    # Small-cap
    "UPST": "XLF", "AFRM": "XLF", "SOFI": "XLF", "OPEN": "XLRE", "CVNA": "XLY",
    "FUBO": "XLC", "PLUG": "XLI", "RIOT": "XLK", "MARA": "XLK", "CLSK": "XLK",
    "CHPT": "XLI", "BLNK": "XLI", "FCEL": "XLU", "SPCE": "XLI", "RKT": "XLF",
    "LAZR": "XLK", "ASTS": "XLC", "JOBY": "XLI", "ACHR": "XLI", "BYND": "XLP",
    "SFIX": "XLY", "PTON": "XLY", "CPRI": "XLY", "DNUT": "XLY", "YETI": "XLY",
    "FIVE": "XLY", "CAKE": "XLY", "RUN": "XLU", "GPRO": "XLY", "IRDM": "XLC",
}

SECTOR_MODES = {"Unfiltered": None, "Top-2": 2, "Top-4": 4}


def sector_momentum_ranks(start: dt.date, end: dt.date
                          ) -> dict[dt.date, dict[str, int]]:
    """
    For every trading day in [start, end] that all 11 sector ETFs have enough
    history for, ranks them 1 (highest SECTOR_ROC_LOOKBACK-trading-day ROC —
    strongest momentum) through 11 (weakest). ROC = close / close.shift(N) - 1.
    Returns {date: {etf: rank}}; a date is simply absent until every ETF has
    SECTOR_ROC_LOOKBACK trading days behind it (XLC/XLRE, the newest, both
    predate 2022 by years, so this only affects the first ~4-5 months of the
    seed window, not the reported [start, end] typically).
    """
    seed_start = start - dt.timedelta(days=int(SECTOR_ROC_LOOKBACK * 2.2))

    def _load(etf: str):
        try:
            return etf, data.daily_bars(etf, period="max")
        except Exception:
            return etf, pd.DataFrame()

    closes: dict[str, dict[dt.date, float]] = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for etf, d in ex.map(_load, SECTOR_ETFS):
            if d.empty:
                continue
            d = d[(d.index.date >= seed_start) & (d.index.date <= end)]
            roc = d["Close"] / d["Close"].shift(SECTOR_ROC_LOOKBACK) - 1.0
            closes[etf] = {ts.date(): v for ts, v in zip(d.index, roc)
                          if pd.notna(v)}

    all_dates = sorted(set.union(*(set(v) for v in closes.values()))) if closes else []
    ranks: dict[dt.date, dict[str, int]] = {}
    for sdate in all_dates:
        if not (start <= sdate <= end):
            continue
        vals = {etf: closes[etf][sdate] for etf in SECTOR_ETFS
                if sdate in closes.get(etf, {})}
        if len(vals) < len(SECTOR_ETFS):
            continue    # not every sector has a defined ROC yet — skip the day
        ordered = sorted(vals.items(), key=lambda kv: -kv[1])
        ranks[sdate] = {etf: i + 1 for i, (etf, _) in enumerate(ordered)}
    return ranks


def run_sector_rotation_test(start: dt.date = REGIME_START,
                             end: dt.date | None = None,
                             capital: float = BACKTEST_CAPITAL,
                             universes: tuple[str, ...] = ("Small-Cap", "Mixed"),
                             verbose: bool = True) -> dict[str, DailyResult]:
    """Runs Unfiltered / Top-2 / Top-4 sector modes on each of `universes`."""
    end = end or dt.date.today()
    if verbose:
        print(f"  computing {SECTOR_ROC_LOOKBACK}-day ROC ranks for "
              f"{len(SECTOR_ETFS)} SPDR sector ETFs ...", flush=True)
    sector_ranks = sector_momentum_ranks(start, end)

    results: dict[str, DailyResult] = {}
    for uni_name in universes:
        symbols = UNIVERSES[uni_name]
        for mode_name, top_n in SECTOR_MODES.items():
            label = f"{uni_name} / {mode_name}"
            if verbose:
                print(f"  running {label} ...", flush=True)
            results[label] = run_daily_backtest(
                symbols, start, end, capital=capital, regime_filter=False,
                universe_name=uni_name, variant="A",
                sector_top_n=top_n, sector_ranks=sector_ranks)
    return results


def print_sector_rotation_table(results: dict[str, DailyResult], capital: float,
                                start: dt.date, end: dt.date) -> None:
    stats: dict[str, dict] = {}
    for label, res in results.items():
        t = res.trades
        n = len(t)
        wins = [x for x in t if x.is_win]
        losses = [x for x in t if not x.is_win]
        gross_win = sum(x.pnl for x in wins)
        gross_loss = -sum(x.pnl for x in losses)
        net = sum(x.pnl for x in t)
        wr = len(wins) / n if n else 0.0
        pf = (gross_win / gross_loss if gross_loss > 0
              else (float("inf") if gross_win > 0 else 0.0))
        _, dd_p = max_drawdown(res.equity) if not res.equity.empty else (0.0, 0.0)
        stats[label] = dict(universe=res.universe_name,
                            ret=net / capital * 100.0, dd=dd_p * 100.0,
                            wr=wr * 100.0, pf=pf, n=n)

    universes = sorted({s["universe"] for s in stats.values()})
    width = 92
    print("=" * width)
    print(f"PHASE 2 — SECTOR ROTATION (DUAL MOMENTUM), Variant A entry "
          f"({start} -> {end}, daily bars, SPY filter OFF)")
    print("=" * width)
    header = (f"{'Universe':<11}{'Sector Filter':<20}{'Total Ret':>10}"
             f"{'Max DD':>9}{'Win %':>8}{'PF':>7}{'Trades':>8}")
    print(header)
    print("-" * width)
    for uni in universes:
        for mode_name in SECTOR_MODES:
            s = stats[f"{uni} / {mode_name}"]
            pf_s = f"{s['pf']:.2f}" if s["pf"] != float("inf") else "inf"
            print(f"{uni:<11}{mode_name:<20}{s['ret']:>+9.2f}%{s['dd']:>8.2f}%"
                  f"{s['wr']:>7.1f}%{pf_s:>7}{s['n']:>8}")
        print("-" * width)
    print("=" * width)

    try:
        spy = data.daily_bars("SPY", period="max")
        w = spy[(spy.index.date >= start) & (spy.index.date <= end)]
        if len(w) > 1:
            r = float(w["Close"].iloc[-1] / w["Close"].iloc[0] - 1.0)
            print(f"SPY buy & hold, same window: {r*100:+.2f}%")
    except Exception:
        pass

    print()
    print("Sector Filter: 'Unfiltered' allows dip-buys in any sector; 'Top-2'/")
    print("'Top-4' additionally require the stock's SECTOR_MAP-mapped SPDR ETF")
    print(f"to rank in the top 2/4 of 11 by {SECTOR_ROC_LOOKBACK}-trading-day ROC")
    print("on the signal day (a day with no rank yet, or a symbol with no")
    print("sector mapping, blocks that entry). SECTOR_MAP is a curated GICS")
    print("mapping, not a data-provider pull — see its comment for specifics.")


# =============================================================================
# PHASE 3 — ADX REGIME-SWITCHING HYBRID STRATEGY
#
# Two trade STYLES with genuinely different lifecycles, hence a dedicated
# trade record and engine rather than reusing DailyTrade/run_daily_backtest:
#   MR  (mean-reversion, Variant A entry) — the SAME scale-then-target
#       mechanics execution_engine.py runs live (SCALE_FRACTION at SCALE_R,
#       remainder to TARGET_R, fixed STOP_PCT stop) — not the simplified
#       single fixed-stop/target model Phase 1/2's run_daily_backtest used.
#       Mode "Mean-Reversion"'s numbers will therefore NOT exactly match
#       Phase 1/2's "Unfiltered Variant A" baseline: the entry trigger is
#       identical, the exit management is now the fuller live mechanic.
#   MOM (momentum breakout) — buy a new BREAKOUT_LOOKBACK-day high close; no
#       fixed stop or target at all — the ONLY exit is close < 10-EMA. Sized
#       off (entry price - EMA10-at-entry) as the initial risk distance, the
#       nearest thing to a "stop" this style has, through the same
#       size_position() the rest of the engine uses.
# ADX is computed PER SYMBOL (there is no single market-wide ADX here), and in
# the Hybrid mode decides PER SYMBOL PER DAY which style, if either, may open
# a new position on that name — one symbol can be ranging while another
# trends on the same day.
# =============================================================================

ADX_PERIOD = 14
ADX_RANGE_MAX = 20.0        # ADX below this: ranging -> mean-reversion regime
ADX_TREND_MIN = 25.0        # ADX above this: trending -> momentum regime
                             # between the two: transition band -> no new entry
BREAKOUT_LOOKBACK = 20      # trading days; prior high, today excluded
MOMENTUM_EMA_SPAN = 10

HYBRID_MODES = {
    "Mean-Reversion": "MR_ONLY",
    "Momentum": "MOM_ONLY",
    "ADX Hybrid": "HYBRID",
}


@dataclass
class HybridTrade:
    symbol: str
    style: str                          # "MR" or "MOM"
    entry_date: dt.date
    entry_px: float
    shares_total: float
    shares_open: float
    stop: float
    r_unit: float = 0.0                 # MR: initial (entry - stop) distance
    target: float | None = None         # MR only
    scaled: bool = False                # MR only
    high_water: float = 0.0             # MOM (fixed variant): trailing-stop anchor
    realized_pnl: float = 0.0
    exit_date: dt.date | None = None
    reason: str = ""

    @property
    def is_win(self) -> bool:
        return self.realized_pnl > 0


@dataclass
class HybridResult:
    trades: list[HybridTrade] = field(default_factory=list)
    equity: pd.Series = field(default_factory=pd.Series)
    universe_name: str = ""
    mode: str = ""
    fixed: bool = False
    start: dt.date | None = None
    end: dt.date | None = None


def daily_hybrid_frame(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Indicators for Phase 3: the Variant A mean-reversion SIGNAL_MR (identical
    trend gate + RSI/BB trigger to Phase 1/2's daily_signal_frame variant
    "A"), ADX(14) for the regime switch, the prior BREAKOUT_LOOKBACK-day high
    for the momentum SIGNAL_MOM (today's close breaking above it), and the
    MOMENTUM_EMA_SPAN EMA for the momentum trade's exit / initial-risk level.
    """
    close = daily["Close"].to_numpy(dtype=float)
    high = daily["High"].to_numpy(dtype=float)
    low = daily["Low"].to_numpy(dtype=float)

    rsi_v = strategy.rsi(close, RSI_PERIOD)
    _, bb_low, _ = strategy.bollinger(close, BB_PERIOD, BB_K)
    sma200 = daily["Close"].rolling(200).mean().to_numpy()
    adx_v = strategy.adx(high, low, close, ADX_PERIOD)
    ema10 = strategy.ema(close, MOMENTUM_EMA_SPAN)
    prior_high = pd.Series(high).shift(1).rolling(BREAKOUT_LOOKBACK).max().to_numpy()

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
    rsi_ok = np.where(np.isnan(rsi_v), False, rsi_v < RSI_OVERSOLD)
    bb_ok = np.where(np.isnan(bb_low), False, low <= bb_low)
    signal_mr = trend_ok & above_now & (rsi_ok | bb_ok)
    signal_mom = np.where(np.isnan(prior_high), False, close > prior_high)

    out = daily.copy()
    out["RSI"] = rsi_v
    out["BB_LOW"] = bb_low
    out["SMA200"] = sma200
    out["ADX"] = adx_v
    out["EMA10"] = ema10
    out["PRIOR20_HIGH"] = prior_high
    out["SIGNAL_MR"] = signal_mr
    out["SIGNAL_MOM"] = signal_mom
    return out


def _load_hybrid_frames(symbols: list[str], start: dt.date,
                        end: dt.date) -> dict[str, pd.DataFrame]:
    """Same fetch/seed/trim pattern as _load_daily_frames, hybrid indicators."""
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
            frames[sym] = daily_hybrid_frame(d)
    return frames


def run_adx_hybrid_backtest(symbols: list[str], start: dt.date, end: dt.date,
                            mode: str, capital: float = BACKTEST_CAPITAL,
                            universe_name: str = "") -> HybridResult:
    """
    Portfolio-level daily-bar event loop with two entry/exit styles.

    `mode` is one of HYBRID_MODES' values:
      "MR_ONLY"  — every candidate is a mean-reversion (Variant A) entry,
                   regardless of ADX.
      "MOM_ONLY" — every candidate is a momentum-breakout entry, regardless
                   of ADX.
      "HYBRID"   — per symbol per day: ADX < ADX_RANGE_MAX routes it to MR,
                   ADX > ADX_TREND_MIN routes it to MOM, the band between
                   takes no new entry for that symbol that day.

    Signal-to-fill keeps the same one-day lag as run_daily_backtest() (queued
    at T's close, filled at T+1's open) for both entries and the momentum
    exit (close < EMA10 can only be known at T's close, so it cannot fill at
    that same close). The MR stop/scale/target levels are known in advance,
    so those exits fire same-day off T's own High/Low, exactly as
    run_daily_backtest() does.

    MR exit precedence when more than one level could have traded the same
    bar (pessimistic, matching run_backtest()/run_daily_backtest()): stop
    (via Low) is checked first and closes the whole remaining position; only
    if the stop held do the scale level and then the target (both via High)
    get checked, in that order — the same order execution_engine.py's live
    state machine uses.
    """
    frames = _load_hybrid_frames(symbols, start, end)
    res = HybridResult(universe_name=universe_name, mode=mode, start=start, end=end)
    if not frames:
        return res

    all_dates = sorted({ts.date() for d in frames.values()
                        for ts in d.index if start <= ts.date() <= end})
    if not all_dates:
        return res

    cap_pct = min(MAX_POSITION_PCT, 1.0 / MAX_CONCURRENT)
    cash = capital
    open_pos: dict[str, HybridTrade] = {}
    pending_entries: list[tuple[str, str]] = []     # (symbol, style)
    pending_mom_exits: list[str] = []
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

        # ---- phase 1: MR exits — stop/scale/target on today's OHLC
        for sym, t in list(open_pos.items()):
            if t.style != "MR":
                continue
            row = todays.get(sym)
            if row is None:
                continue
            h, l = float(row["High"]), float(row["Low"])

            if l <= t.stop:
                fill = apply_slippage(t.stop, "sell")
                proceeds = fill * t.shares_open - COMMISSION_PER_TRADE
                cash += proceeds
                t.realized_pnl += proceeds - t.entry_px * t.shares_open
                t.shares_open = 0.0
                t.exit_date = sdate
                t.reason = "BE_STOP" if t.scaled else "STOP"
                res.trades.append(t)
                del open_pos[sym]
                continue

            if SCALE_ENABLED and not t.scaled and h >= t.entry_px + t.r_unit * SCALE_R:
                scale_price = t.entry_px + t.r_unit * SCALE_R
                qty = round(t.shares_total * SCALE_FRACTION, 6)
                fill = apply_slippage(scale_price, "sell")
                proceeds = fill * qty - COMMISSION_PER_TRADE
                cash += proceeds
                t.realized_pnl += proceeds - t.entry_px * qty
                t.shares_open = round(t.shares_open - qty, 6)
                t.scaled = True
                t.stop = t.entry_px    # breakeven on the runner

            if t.shares_open > 0 and h >= t.target:
                fill = apply_slippage(t.target, "sell")
                proceeds = fill * t.shares_open - COMMISSION_PER_TRADE
                cash += proceeds
                t.realized_pnl += proceeds - t.entry_px * t.shares_open
                t.shares_open = 0.0
                t.exit_date = sdate
                t.reason = "TARGET"
                res.trades.append(t)
                del open_pos[sym]

        # ---- phase 1b: fill yesterday's queued momentum exits at TODAY's open
        for sym in pending_mom_exits:
            t = open_pos.get(sym)
            if t is None or t.style != "MOM":
                continue
            row = todays.get(sym)
            if row is None:
                continue
            fill = apply_slippage(float(row["Open"]), "sell")
            proceeds = fill * t.shares_open - COMMISSION_PER_TRADE
            cash += proceeds
            t.realized_pnl += proceeds - t.entry_px * t.shares_open
            t.shares_open = 0.0
            t.exit_date, t.reason = sdate, "EMA_EXIT"
            res.trades.append(t)
            del open_pos[sym]
        pending_mom_exits = []

        # ---- phase 2: fill yesterday's queued entries at TODAY's open
        for sym, style in pending_entries:
            if sym in open_pos or len(open_pos) >= MAX_CONCURRENT:
                continue
            row = todays.get(sym)
            if row is None:
                continue
            open_px = float(row["Open"])
            fill = apply_slippage(open_px, "buy")
            stop = fill * (1.0 - STOP_PCT) if style == "MR" else float(row["EMA10"])
            if fill <= stop:
                continue
            eq = _mark(todays)
            shares, _note = size_position(eq, fill, stop, max_position_pct=cap_pct)
            cost = shares * fill + COMMISSION_PER_TRADE
            if shares <= 0 or cost > cash:
                continue
            cash -= cost
            r_unit = fill - stop
            target = fill + r_unit * TARGET_R if style == "MR" else None
            open_pos[sym] = HybridTrade(
                symbol=sym, style=style, entry_date=sdate, entry_px=fill,
                shares_total=shares, shares_open=shares, stop=stop,
                r_unit=r_unit, target=target)
        pending_entries = []

        # ---- phase 3: scan today's closes -> queue tomorrow's exits/entries
        for sym, t in open_pos.items():
            if t.style != "MOM":
                continue
            row = todays.get(sym)
            if row is not None and float(row["Close"]) < float(row["EMA10"]):
                pending_mom_exits.append(sym)

        for sym, row in todays.items():
            if sym in open_pos:
                continue
            style = None
            if mode == "MR_ONLY":
                style = "MR" if bool(row["SIGNAL_MR"]) else None
            elif mode == "MOM_ONLY":
                style = "MOM" if bool(row["SIGNAL_MOM"]) else None
            else:  # HYBRID: per-symbol ADX regime
                adx_v = float(row["ADX"])
                if adx_v == adx_v:    # not NaN — ADX warmed up
                    if adx_v < ADX_RANGE_MAX and bool(row["SIGNAL_MR"]):
                        style = "MR"
                    elif adx_v > ADX_TREND_MIN and bool(row["SIGNAL_MOM"]):
                        style = "MOM"
            if style:
                pending_entries.append((sym, style))

        # Rank: MR candidates (most oversold first) ahead of MOM candidates
        # (strongest trend first) — an arbitrary but disclosed tie-break for
        # the rare day both styles compete for the same MAX_CONCURRENT slots.
        mr_c = [(s, st) for s, st in pending_entries if st == "MR"]
        mom_c = [(s, st) for s, st in pending_entries if st == "MOM"]
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
    return res


def run_adx_hybrid_test(start: dt.date = REGIME_START,
                        end: dt.date | None = None,
                        capital: float = BACKTEST_CAPITAL,
                        universes: tuple[str, ...] = ("Small-Cap", "Mixed"),
                        verbose: bool = True) -> dict[str, HybridResult]:
    """Runs Mean-Reversion / Momentum / ADX Hybrid on each of `universes`."""
    end = end or dt.date.today()
    results: dict[str, HybridResult] = {}
    for uni_name in universes:
        symbols = UNIVERSES[uni_name]
        for mode_name, mode in HYBRID_MODES.items():
            label = f"{uni_name} / {mode_name}"
            if verbose:
                print(f"  running {label} ...", flush=True)
            results[label] = run_adx_hybrid_backtest(
                symbols, start, end, mode=mode, capital=capital,
                universe_name=uni_name)
    return results


def print_adx_hybrid_table(results: dict[str, HybridResult], capital: float,
                           start: dt.date, end: dt.date) -> None:
    stats: dict[str, dict] = {}
    for label, res in results.items():
        t = res.trades
        n = len(t)
        wins = [x for x in t if x.is_win]
        losses = [x for x in t if not x.is_win]
        gross_win = sum(x.realized_pnl for x in wins)
        gross_loss = -sum(x.realized_pnl for x in losses)
        net = sum(x.realized_pnl for x in t)
        wr = len(wins) / n if n else 0.0
        pf = (gross_win / gross_loss if gross_loss > 0
              else (float("inf") if gross_win > 0 else 0.0))
        _, dd_p = max_drawdown(res.equity) if not res.equity.empty else (0.0, 0.0)
        stats[label] = dict(universe=res.universe_name,
                            ret=net / capital * 100.0, dd=dd_p * 100.0,
                            wr=wr * 100.0, pf=pf, n=n)

    universes = sorted({s["universe"] for s in stats.values()})
    width = 92
    print("=" * width)
    print(f"PHASE 3 — ADX REGIME-SWITCHING HYBRID  ({start} -> {end}, daily "
          f"bars, SPY filter OFF)")
    print("=" * width)
    header = (f"{'Universe':<11}{'Strategy':<18}{'Total Ret':>10}{'Max DD':>9}"
             f"{'Win %':>8}{'PF':>7}{'Trades':>8}")
    print(header)
    print("-" * width)
    for uni in universes:
        for mode_name in HYBRID_MODES:
            s = stats[f"{uni} / {mode_name}"]
            pf_s = f"{s['pf']:.2f}" if s["pf"] != float("inf") else "inf"
            print(f"{uni:<11}{mode_name:<18}{s['ret']:>+9.2f}%{s['dd']:>8.2f}%"
                  f"{s['wr']:>7.1f}%{pf_s:>7}{s['n']:>8}")
        print("-" * width)
    print("=" * width)

    try:
        spy = data.daily_bars("SPY", period="max")
        w = spy[(spy.index.date >= start) & (spy.index.date <= end)]
        if len(w) > 1:
            r = float(w["Close"].iloc[-1] / w["Close"].iloc[0] - 1.0)
            print(f"SPY buy & hold, same window: {r*100:+.2f}%")
    except Exception:
        pass

    print()
    print(f"Mean-Reversion: Variant A entry (RSI<35 or BB touch); "
          f"{SCALE_FRACTION:.0%} scaled at +{SCALE_R:.1f}R, stop to breakeven,")
    print(f"remainder to +{TARGET_R:.1f}R; {STOP_PCT:.1%} initial stop — the full")
    print("live scale/target mechanic, NOT Phase 1/2's simplified fixed-stop/")
    print("target model, so these numbers will not exactly match that baseline.")
    print(f"Momentum: buy a new {BREAKOUT_LOOKBACK}-day-high close; exit only when "
          f"close < {MOMENTUM_EMA_SPAN}-EMA (no fixed stop/target —")
    print("sized off entry price to EMA10-at-entry as the initial risk distance).")
    print(f"ADX Hybrid: PER SYMBOL PER DAY — ADX<{ADX_RANGE_MAX:.0f} routes to "
          f"Mean-Reversion, ADX>{ADX_TREND_MIN:.0f} routes to Momentum,")
    print(f"{ADX_RANGE_MAX:.0f}-{ADX_TREND_MIN:.0f} takes no new entry for that symbol that day.")


# =============================================================================
# PHASE 3.1 — FIXED ADX HYBRID (volume confirmation, ATR trailing stop,
# dynamic ADX threshold)
#
# Three changes, all confined to the MOMENTUM side — mean-reversion (Variant
# A) is untouched:
#   1. VOLUME GATE — a breakout must also print Volume > VOLUME_MULT x its own
#      VOLUME_SMA_WINDOW-day average volume. A price breakout nobody actually
#      traded is the textbook false-breakout signature.
#   2. ATR TRAILING STOP — replaces the 10-EMA close-based exit with a real
#      trailing STOP order: 2.0x ATR(14) under the highest High since entry,
#      ratcheted up (never down) once per day, triggered intraday off that
#      day's Low exactly like the mean-reversion stop — same-day fill, no
#      1-day lag, unlike the old EMA exit (which could only be evaluated
#      after the close it depended on).
#   3. DYNAMIC ADX THRESHOLD — the fixed ADX>25 momentum gate becomes ADX >
#      that SYMBOL's own rolling 80th percentile of ADX over the prior 50
#      sessions (shifted one day so the threshold never includes the value
#      being tested against it), floored at ADX_TREND_MIN so a quiet stretch
#      can't drop the bar below the original fixed threshold.
#
# A SEPARATE, ALSO-FIXED BUG: the original run_adx_hybrid_backtest() sized a
# fresh momentum entry off the FILL day's own EMA10 — but a T+1-open fill
# happens before T+1's close (and therefore its EMA10) exists, a small
# look-ahead leak. This engine sizes off the SIGNAL day's (T's) own High/ATR
# instead, known in full before the T+1 fill. Test A ("Unfixed ADX Hybrid")
# in run_fixed_hybrid_comparison() deliberately keeps calling the ORIGINAL
# run_adx_hybrid_backtest() unchanged so the "before" side of the comparison
# is genuinely the original code, leak included — see the printed disclosure.
# =============================================================================

VOLUME_MULT = 1.5
VOLUME_SMA_WINDOW = 20
ATR_PERIOD = 14
ATR_TRAIL_MULT = 2.0
ADX_PERCENTILE_WINDOW = 50
ADX_PERCENTILE_Q = 0.80


def daily_hybrid_fixed_frame(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Extends daily_hybrid_frame() with the three "Fixed ADX Hybrid" changes.
    SIGNAL_MR and the ADX<ADX_RANGE_MAX ranging gate are UNCHANGED — none of
    the three fixes touch the mean-reversion side. SIGNAL_MOM gains the
    volume gate; ATR14 and ADX_DYN_THRESH are added for the engine to use.
    """
    out = daily_hybrid_frame(daily)
    volume = out["Volume"].to_numpy(dtype=float)
    high = out["High"].to_numpy(dtype=float)
    low = out["Low"].to_numpy(dtype=float)
    close = out["Close"].to_numpy(dtype=float)

    vol_sma20 = pd.Series(volume).rolling(VOLUME_SMA_WINDOW).mean().to_numpy()
    vol_ok = np.where(np.isnan(vol_sma20), False, volume > VOLUME_MULT * vol_sma20)

    atr14 = strategy.atr(high, low, close, ATR_PERIOD)

    dyn_thresh_raw = (pd.Series(out["ADX"].to_numpy()).shift(1)
                      .rolling(ADX_PERCENTILE_WINDOW).quantile(ADX_PERCENTILE_Q))
    dyn_thresh = dyn_thresh_raw.clip(lower=ADX_TREND_MIN).to_numpy()

    out["ATR14"] = atr14
    out["ADX_DYN_THRESH"] = dyn_thresh
    out["SIGNAL_MOM"] = out["SIGNAL_MOM"].to_numpy() & vol_ok
    return out


def _load_hybrid_fixed_frames(symbols: list[str], start: dt.date,
                              end: dt.date) -> dict[str, pd.DataFrame]:
    """Same fetch/seed/trim pattern as _load_hybrid_frames, fixed indicators."""
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
            frames[sym] = daily_hybrid_fixed_frame(d)
    return frames


def run_adx_hybrid_fixed_backtest(symbols: list[str], start: dt.date, end: dt.date,
                                  mode: str, capital: float = BACKTEST_CAPITAL,
                                  universe_name: str = "",
                                  risk_pct_mr: float | None = None,
                                  risk_pct_mom: float | None = None,
                                  max_concurrent: int | None = None
                                  ) -> HybridResult:
    """
    Same portfolio loop shape as run_adx_hybrid_backtest(), with the three
    fixes applied to the momentum side (see the section note above). Momentum
    exits are now same-day (Low-triggered stop, like mean-reversion) instead
    of next-day-open (Close-vs-EMA10) — there is no more pending_mom_exits
    queue. Momentum entries keep the one-day lag (a breakout is a close-based
    signal), but size off the SIGNAL day's own High/ATR rather than the fill
    day's, closing the look-ahead noted above.

    `risk_pct_mr`/`risk_pct_mom`: OPTIONAL asymmetric risk-based sizing
    (shares = equity * risk_pct / (entry - stop), no position-pct cap) —
    the same scheme hybrid_indicators.size_order() uses live, for a backtest
    that validates that exact tool. Leaving both None (the default) keeps
    this function's original behavior byte-for-byte: size_position() capped
    at min(MAX_POSITION_PCT, 1/MAX_CONCURRENT) of equity, uniform across
    MR/MOM — so existing callers (e.g. run_fixed_hybrid_comparison()) are
    unaffected by this parameter's addition.

    `max_concurrent`: OPTIONAL override of the book-size cap (defaults to
    config.MAX_CONCURRENT when None) — added for optimize_capacity.py's
    universe-size x slot-count sweep. Only changes how many positions may be
    open at once (the `len(open_pos) >= max_concurrent` gate); when the
    asymmetric risk sizing above is ALSO active, this cap has no effect on
    per-trade position size (that path never uses cap_pct at all) — the two
    parameters are cleanly separable, which is the point of the sweep.
    """
    frames = _load_hybrid_fixed_frames(symbols, start, end)
    res = HybridResult(universe_name=universe_name, mode=mode, fixed=True,
                       start=start, end=end)
    if not frames:
        return res

    all_dates = sorted({ts.date() for d in frames.values()
                        for ts in d.index if start <= ts.date() <= end})
    if not all_dates:
        return res

    max_concurrent = MAX_CONCURRENT if max_concurrent is None else max_concurrent
    cap_pct = min(MAX_POSITION_PCT, 1.0 / max_concurrent)
    cash = capital
    open_pos: dict[str, HybridTrade] = {}
    # (symbol, style, basis) — basis carries the SIGNAL day's High/ATR for a
    # queued MOM entry; None for MR (whose stop is a pure function of its own
    # fill price, so it needs no snapshot).
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

        # ---- phase 1: same-day exits for BOTH styles off today's OHLC
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
                if atr_today == atr_today:    # not NaN
                    t.high_water = max(t.high_water, h)
                    new_stop = t.high_water - ATR_TRAIL_MULT * atr_today
                    if new_stop > t.stop:
                        t.stop = new_stop

        # ---- phase 2: fill yesterday's queued entries at TODAY's open
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
                stop = high_water - ATR_TRAIL_MULT * basis["atr"]

            if fill <= stop:
                continue
            eq = _mark(todays)
            if risk_pct_mr is not None and risk_pct_mom is not None:
                risk_pct = risk_pct_mr if style == "MR" else risk_pct_mom
                shares = math.floor(eq * risk_pct / (fill - stop))
            else:
                shares, _note = size_position(eq, fill, stop, max_position_pct=cap_pct)
            cost = shares * fill + COMMISSION_PER_TRADE
            if shares <= 0 or cost > cash:
                continue
            cash -= cost
            r_unit = fill - stop
            target = fill + r_unit * TARGET_R if style == "MR" else None
            open_pos[sym] = HybridTrade(
                symbol=sym, style=style, entry_date=sdate, entry_px=fill,
                shares_total=shares, shares_open=shares, stop=stop,
                r_unit=r_unit, target=target, high_water=high_water)
        pending_entries = []

        # ---- phase 3: scan today's closes -> queue tomorrow's entries
        for sym, row in todays.items():
            if sym in open_pos:
                continue
            style = None
            basis = None
            if mode == "MR_ONLY":
                style = "MR" if bool(row["SIGNAL_MR"]) else None
            elif mode == "MOM_ONLY":
                if bool(row["SIGNAL_MOM"]):
                    style = "MOM"
            else:  # HYBRID: per-symbol ADX regime, dynamic momentum threshold
                adx_v = float(row["ADX"])
                if adx_v == adx_v:
                    if adx_v < ADX_RANGE_MAX and bool(row["SIGNAL_MR"]):
                        style = "MR"
                    else:
                        dyn = float(row["ADX_DYN_THRESH"])
                        if dyn == dyn and adx_v > dyn and bool(row["SIGNAL_MOM"]):
                            style = "MOM"
            if style == "MOM":
                atr_v = float(row["ATR14"])
                if atr_v != atr_v:
                    style = None    # ATR not warmed up yet — skip this signal
                else:
                    basis = {"high": float(row["High"]), "atr": atr_v}
            if style:
                pending_entries.append((sym, style, basis))

        # Rank: MR (most oversold first) ahead of MOM (strongest trend first)
        # — same disclosed tie-break as the original hybrid engine.
        mr_c = [x for x in pending_entries if x[1] == "MR"]
        mom_c = [x for x in pending_entries if x[1] == "MOM"]
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
    return res


def run_fixed_hybrid_comparison(start: dt.date = REGIME_START,
                                end: dt.date | None = None,
                                capital: float = BACKTEST_CAPITAL,
                                universe_name: str = "Mixed",
                                verbose: bool = True) -> dict[str, object]:
    """Benchmark (Phase 1 Variant A) vs Test A (original Hybrid) vs Test B
    (Fixed Hybrid), all on `universe_name`, SPY filter OFF."""
    end = end or dt.date.today()
    symbols = UNIVERSES[universe_name]
    results: dict[str, object] = {}

    if verbose:
        print("  running Benchmark: Phase 1 Dip-Buying (Variant A) ...", flush=True)
    results["Benchmark: Phase 1 Dip-Buying"] = run_daily_backtest(
        symbols, start, end, capital=capital, regime_filter=False,
        universe_name=universe_name, variant="A")

    if verbose:
        print("  running Test A: Unfixed ADX Hybrid (Original) ...", flush=True)
    results["Test A: Unfixed ADX Hybrid"] = run_adx_hybrid_backtest(
        symbols, start, end, mode="HYBRID", capital=capital,
        universe_name=universe_name)

    if verbose:
        print("  running Test B: Fixed ADX Hybrid (Volume + ATR + Dynamic ADX) ...",
              flush=True)
    results["Test B: Fixed ADX Hybrid"] = run_adx_hybrid_fixed_backtest(
        symbols, start, end, mode="HYBRID", capital=capital,
        universe_name=universe_name)

    return results


def _generic_metrics(res, capital: float) -> dict:
    """Works for both DailyResult (t.pnl) and HybridResult (t.realized_pnl)."""
    t = res.trades
    n = len(t)
    pnl_of = (lambda x: x.pnl) if isinstance(res, DailyResult) else (lambda x: x.realized_pnl)
    wins = [x for x in t if pnl_of(x) > 0]
    losses = [x for x in t if pnl_of(x) <= 0]
    gross_win = sum(pnl_of(x) for x in wins)
    gross_loss = -sum(pnl_of(x) for x in losses)
    net = sum(pnl_of(x) for x in t)
    wr = len(wins) / n if n else 0.0
    pf = (gross_win / gross_loss if gross_loss > 0
          else (float("inf") if gross_win > 0 else 0.0))
    _, dd_p = max_drawdown(res.equity) if not res.equity.empty else (0.0, 0.0)
    return dict(ret=net / capital * 100.0, dd=dd_p * 100.0, wr=wr * 100.0,
               pf=pf, n=n)


def print_fixed_hybrid_table(results: dict[str, object], capital: float,
                             start: dt.date, end: dt.date) -> None:
    width = 92
    print("=" * width)
    print(f"FIXED ADX HYBRID COMPARISON  ({start} -> {end}, daily bars, "
          f"Mixed universe, SPY filter OFF)")
    print("=" * width)
    header = (f"{'Scenario':<34}{'Total Ret':>10}{'Max DD':>9}{'Win %':>8}"
             f"{'PF':>7}{'Trades':>8}")
    print(header)
    print("-" * width)
    for label, res in results.items():
        s = _generic_metrics(res, capital)
        pf_s = f"{s['pf']:.2f}" if s["pf"] != float("inf") else "inf"
        print(f"{label:<34}{s['ret']:>+9.2f}%{s['dd']:>8.2f}%{s['wr']:>7.1f}%"
              f"{pf_s:>7}{s['n']:>8}")
    print("=" * width)

    try:
        spy = data.daily_bars("SPY", period="max")
        w = spy[(spy.index.date >= start) & (spy.index.date <= end)]
        if len(w) > 1:
            r = float(w["Close"].iloc[-1] / w["Close"].iloc[0] - 1.0)
            print(f"SPY buy & hold, same window: {r*100:+.2f}%")
    except Exception:
        pass

    print()
    print("Fixes applied to Test B's MOMENTUM side only (mean-reversion is")
    print(f"identical to Test A): Volume > {VOLUME_MULT}x {VOLUME_SMA_WINDOW}-day avg volume;")
    print(f"a real {ATR_TRAIL_MULT}x ATR(14) trailing STOP (same-day, Low-triggered,")
    print("ratchets up only) replacing the close-vs-10-EMA exit; and a DYNAMIC")
    print(f"momentum threshold — ADX > that symbol's own rolling {ADX_PERCENTILE_WINDOW}-session "
          f"{ADX_PERCENTILE_Q:.0%}ile")
    print(f"(floored at {ADX_TREND_MIN:.0f}) instead of the fixed ADX>{ADX_TREND_MIN:.0f}.")
    print("Test B also sizes a momentum entry off the SIGNAL day's own")
    print("High/ATR, closing a small look-ahead in Test A (which sized off the")
    print("FILL day's EMA10 — not yet fully known at that day's open). Test A")
    print("is left unchanged from Phase 3 so the 'before' side of this")
    print("comparison is genuinely the original code, leak included.")


def save_trades(res: Result, path) -> None:
    if not res.trades:
        return
    rows = [{"symbol": t.symbol,
             "entry_ts": t.entry_ts.isoformat(), "entry_px": round(t.entry_px, 4),
             "exit_ts": t.exit_ts.isoformat() if t.exit_ts else None,
             "exit_px": round(t.exit_px, 4) if t.exit_px else None,
             "shares": t.shares, "stop": round(t.stop, 4), "target": round(t.target, 4),
             "reason": t.reason, "pnl": round(t.pnl, 2),
             "pnl_pct": round(t.pnl_pct * 100, 3), "bars_held": t.bars_held}
            for t in res.trades]
    pd.DataFrame(rows).to_csv(path, index=False)


# ---------------------------------------------------------------- cli
def main() -> int:
    ap = argparse.ArgumentParser(description="59-day intraday backtest")
    ap.add_argument("--symbols", type=str, default=None,
                    help="comma-separated override for the watchlist")
    ap.add_argument("--days", type=int, default=BACKTEST_DAYS)
    ap.add_argument("--capital", type=float, default=BACKTEST_CAPITAL)
    ap.add_argument("--no-range-gate", action="store_true",
                    help="ablation: take every signal regardless of extension")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--regime-stress-test", action="store_true",
                    help="6-way SPY-regime x universe test on daily bars "
                         "(Jan 2022-present by default) and exit")
    ap.add_argument("--regime-start", type=str,
                    default=REGIME_START.isoformat(),
                    help="start date for --regime-stress-test / "
                         "--ibs-variant-test / --sector-rotation-test / "
                         "--adx-hybrid-test / --fixed-hybrid-test (YYYY-MM-DD)")
    ap.add_argument("--ibs-variant-test", action="store_true",
                    help="Phase 1: compare oversold triggers A (baseline) / "
                         "B (pure IBS dip) / C (RSI+IBS combined) on daily "
                         "bars, SPY filter OFF, and exit")
    ap.add_argument("--sector-rotation-test", action="store_true",
                    help="Phase 2: Variant A entry gated by SPDR sector "
                         "momentum rank (Unfiltered / Top-2 / Top-4), daily "
                         "bars, SPY filter OFF, and exit")
    ap.add_argument("--adx-hybrid-test", action="store_true",
                    help="Phase 3: Mean-Reversion vs Momentum vs ADX-switched "
                         "Hybrid, daily bars, SPY filter OFF, and exit")
    ap.add_argument("--fixed-hybrid-test", action="store_true",
                    help="Benchmark (Phase 1) vs Unfixed vs Fixed (volume + "
                         "ATR trailing stop + dynamic ADX) Hybrid, Mixed "
                         "universe, SPY filter OFF, and exit")
    args = ap.parse_args()

    if args.regime_stress_test:
        start = dt.date.fromisoformat(args.regime_start)
        end = dt.date.today()
        print(f"Loading daily bars for {len(set(LARGE_CAP_UNIVERSE) | set(SMALL_CAP_UNIVERSE))} "
              f"symbols + SPY ({start} -> {end}) ...", flush=True)
        results = run_six_way_stress_test(start=start, end=end, capital=args.capital)
        print_six_way_table(results, args.capital, start, end)
        return 0

    if args.ibs_variant_test:
        start = dt.date.fromisoformat(args.regime_start)
        end = dt.date.today()
        print(f"Loading daily bars for Small-Cap + Mixed universes "
              f"({start} -> {end}) ...", flush=True)
        results = run_ibs_variant_test(start=start, end=end, capital=args.capital)
        print_ibs_variant_table(results, args.capital, start, end)
        return 0

    if args.sector_rotation_test:
        start = dt.date.fromisoformat(args.regime_start)
        end = dt.date.today()
        print(f"Loading daily bars for Small-Cap + Mixed universes + "
              f"{len(SECTOR_ETFS)} sector ETFs ({start} -> {end}) ...", flush=True)
        results = run_sector_rotation_test(start=start, end=end, capital=args.capital)
        print_sector_rotation_table(results, args.capital, start, end)
        return 0

    if args.adx_hybrid_test:
        start = dt.date.fromisoformat(args.regime_start)
        end = dt.date.today()
        print(f"Loading daily bars for Small-Cap + Mixed universes "
              f"({start} -> {end}) ...", flush=True)
        results = run_adx_hybrid_test(start=start, end=end, capital=args.capital)
        print_adx_hybrid_table(results, args.capital, start, end)
        return 0

    if args.fixed_hybrid_test:
        start = dt.date.fromisoformat(args.regime_start)
        end = dt.date.today()
        print(f"Loading daily bars for Mixed universe ({start} -> {end}) ...",
              flush=True)
        results = run_fixed_hybrid_comparison(start=start, end=end,
                                              capital=args.capital)
        print_fixed_hybrid_table(results, args.capital, start, end)
        return 0

    scores: dict[str, float] = {}
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        wl = screener.load_watchlist()
        if not wl:
            print("No watchlist found. Run:  python3 screener.py")
            return 1
        symbols = [s["symbol"] for s in wl["watchlist"]]
        scores = {s["symbol"]: s.get("score", 0.0) for s in wl["watchlist"]}

    require_range = not args.no_range_gate
    res = run_backtest(symbols, days=args.days, capital=args.capital,
                       require_range=require_range, scores=scores,
                       verbose=not args.quiet)

    out = report(res, args.capital, require_range)
    print("\n" + out)

    tag = "norange" if args.no_range_gate else "base"
    (DATA / f"backtest_{tag}.txt").write_text(out)
    save_trades(res, DATA / f"backtest_trades_{tag}.csv")
    if not res.equity.empty:
        res.equity.to_csv(DATA / f"backtest_equity_{tag}.csv", header=["equity"])
    print(f"\nwrote {DATA}/backtest_{tag}.txt and trade/equity CSVs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
