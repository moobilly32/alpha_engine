"""
Monte Carlo projection for the oversold-dip strategy, scaled to a $1,500
account trading the locked 4-concurrent-slot book.

SOURCE METRICS — "oversold / 2.5% fixed stop / scale at 1.0R" backtest
(README.md "High-win-rate modifications", scale_test.py): 85 trades over 59
sessions, 55.3% win rate, profit factor 1.35, net +$747.91 (+7.48%) on a
$10,000 book, average hold 4.4 days. This script does not re-run that
backtest — it takes the win rate and per-outcome R-multiples as given
(BT_* below) and simulates them forward on a different account size, using
the SAME position-sizing formula the live engine applies
(execution.py:size_position, called from intraday.py's oversold branch with
max_position_pct=min(MAX_POSITION_PCT, 1/MAX_CONCURRENT)) rather than
inventing a separate risk number here.

WHY MONTE CARLO, NOT A SMOOTH EXPECTED-VALUE LINE
    An earlier version of this script applied the backtest's *average*
    outcome (+0.17R) to every single trade. That is mathematically
    guaranteed to go up every active day, forever — no losing trade can ever
    occur, because every trade IS the average. That is not what the strategy
    actually does; it wins 55.3% of the time and loses the other 44.7%, and
    real accounts experience the variance in between, including losing
    streaks and drawdowns.

    Each of the 1,000 simulated paths below instead randomizes two
    independent things per trade:
      1. WHEN it happens — a day's trade count is Poisson(BT_TRADES /
         BT_SESSIONS), bursty by construction: a broad sell-off can trip
         several watchlist names the same session, calm days trip none.
      2. WHAT it does — a coin flip weighted at BT_WIN_RATE (55.3%): a win
         scales 50% out at +1.0R and lets the other 50% run to the +2.5R
         target, netting BT_AVG_WIN_R (+1.12R, the backtest's realized
         average, not the naive 1.75R blend, since real wins don't all run
         the full distance); a loss rides the full fixed 2.5% stop, -1.0R,
         with no partial exit. (Sanity check: 0.553*1.12 - 0.447*1.00 = 0.17,
         matching the backtest's measured expectancy.)

REAL-WORLD FRICTION — the backtest is clean-fill; live fills are not
    Two costs a clean backtest never pays, layered onto every trade above:
      * SLIPPAGE — bid-ask spread plus market-order slippage on a Robinhood
        fill, taken as SLIPPAGE_PCT (0.15%) of position notional, on every
        trade regardless of outcome. Converted to R-multiple terms via the
        same relationship effective_risk_pct() uses (position size is
        risk_dollars / STOP_PCT, so a cost that is a % of position notional
        is SLIPPAGE_PCT / STOP_PCT in R): SLIPPAGE_R = 0.15% / 2.5% = 0.06R,
        subtracted from every trade's R-multiple, win or lose.
      * GAP-DOWN TAIL RISK — a fixed 2.5% stop assumes the exit fills AT the
        stop. It does not on an overnight gap or a fast-moving name: with
        probability GAP_DOWN_PROB (5%), a losing trade instead realizes a
        GAP_DOWN_PCT (5.0%) loss — GAP_DOWN_R (-2.0R) instead of the assumed
        -1.0R. This does not touch win outcomes; only losses can gap through
        their own stop.

Real trade-by-trade variance is large — see README: "Do not treat +7.48% as
an expectation." This script's job is to show that variance, not hide it.

Usage:
    python3 project_pnl.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotext as plt

from config import MAX_CONCURRENT, MAX_POSITION_PCT, RISK_PCT, STOP_PCT

# ---------------------------------------------------------------- baseline metrics
BT_TRADES = 85
BT_SESSIONS = 59
BT_WIN_RATE = 0.553
BT_PROFIT_FACTOR = 1.35
BT_AVG_WIN_R = 1.12             # net win, after the 50%@1R / 50%@2.5R scale-out
BT_AVG_LOSS_R = 1.00            # full fixed 2.5% stop, no partial exit

# ---------------------------------------------------------------- real-world friction
SLIPPAGE_PCT = 0.0015           # bid-ask spread + Robinhood market-order slippage
GAP_DOWN_PROB = 0.05            # of LOSING trades, fraction that gap through the stop
GAP_DOWN_PCT = 0.05             # gap-down loss size (vs. the 2.5% fixed stop)
GAP_DOWN_R = GAP_DOWN_PCT / STOP_PCT   # -5.0% / -2.5% stop = -2.0R

START_CAPITAL = 1_500.0
PROJECTION_DAYS = 60
N_SIMULATIONS = 1_000
MASTER_SEED = 7                 # each run gets MASTER_SEED + run_index


@dataclass
class DaySummary:
    day: int
    n_trades: int
    equity_start: float
    equity_end: float

    @property
    def pnl(self) -> float:
        return self.equity_end - self.equity_start


@dataclass
class RunResult:
    equity_curve: list[float]        # index 0 = start, before day 1
    days_log: list[DaySummary]
    total_trades: int
    final_equity: float
    net_pnl: float
    pct_return: float
    max_drawdown_pct: float          # peak-to-trough, positive number


def effective_risk_pct() -> float:
    """
    Dollar fraction of equity actually put at risk per trade under the LIVE
    sizing rule — not the nominal RISK_PCT. intraday.py caps every oversold
    entry's position at min(MAX_POSITION_PCT, 1/MAX_CONCURRENT) of equity so
    that MAX_CONCURRENT slots filled at once can never demand more than 100%
    of the book; with a fixed % stop that position cap translates into a
    risk cap:

        by_risk (shares) = (equity * RISK_PCT)  / (price * STOP_PCT)
        by_cap  (shares) = (equity * cap_pct)   / price
        effective risk %  = min(RISK_PCT, cap_pct * STOP_PCT)
    """
    cap_pct = min(MAX_POSITION_PCT, 1.0 / MAX_CONCURRENT)
    return min(RISK_PCT, cap_pct * STOP_PCT)


def slippage_r() -> float:
    """
    SLIPPAGE_PCT is a cost against position notional (spread + market-order
    slippage), but every trade in this simulation is denominated in R. Since
    position size is risk_dollars / STOP_PCT (see effective_risk_pct()), a
    notional-based cost converts to R the same way: SLIPPAGE_R = SLIPPAGE_PCT
    / STOP_PCT. Charged on every trade, independent of win/loss.
    """
    return SLIPPAGE_PCT / STOP_PCT


def simulate_run(days: int, seed: int) -> RunResult:
    """
    One Monte Carlo path. Trade arrivals are Poisson-clustered by day. Each
    trade's outcome draws THREE independent random things, not the
    backtest's smooth average (see the module docstring for why that
    distinction is the entire point of this simulation):
      1. win or lose, weighted at BT_WIN_RATE;
      2. for a loss only, whether it gaps through the stop instead of
         filling at it (GAP_DOWN_PROB chance of -GAP_DOWN_R instead of
         -BT_AVG_LOSS_R);
      3. slippage_r() shaved off every trade's R-multiple regardless of 1-2.
    Drawdown is tracked trade-by-trade (not just at day boundaries) so a
    same-day losing cluster is not hidden behind a single end-of-day print.
    """
    trades_per_day = BT_TRADES / BT_SESSIONS
    risk_pct = effective_risk_pct()
    slip_r = slippage_r()
    rng = np.random.default_rng(seed)

    equity = START_CAPITAL
    peak = equity
    max_dd = 0.0
    equity_curve = [equity]
    days_log: list[DaySummary] = []
    total_trades = 0

    for day in range(1, days + 1):
        n_trades = int(rng.poisson(trades_per_day))
        equity_start = equity
        for _ in range(n_trades):
            risk_dollars = equity * risk_pct
            is_win = rng.random() < BT_WIN_RATE
            if is_win:
                r_multiple = BT_AVG_WIN_R
            else:
                gapped = rng.random() < GAP_DOWN_PROB
                r_multiple = -GAP_DOWN_R if gapped else -BT_AVG_LOSS_R
            r_multiple -= slip_r
            equity += r_multiple * risk_dollars
            total_trades += 1
            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / peak if peak > 0 else 0.0)
        days_log.append(DaySummary(day, n_trades, equity_start, equity))
        equity_curve.append(equity)

    net_pnl = equity - START_CAPITAL
    return RunResult(
        equity_curve=equity_curve, days_log=days_log, total_trades=total_trades,
        final_equity=equity, net_pnl=net_pnl,
        pct_return=net_pnl / START_CAPITAL * 100.0,
        max_drawdown_pct=max_dd * 100.0,
    )


def run_monte_carlo(n_sims: int = N_SIMULATIONS) -> list[RunResult]:
    return [simulate_run(PROJECTION_DAYS, MASTER_SEED + i) for i in range(n_sims)]


def print_summary(runs: list[RunResult]) -> dict:
    net_pnls = np.array([r.net_pnl for r in runs])
    pct_returns = np.array([r.pct_return for r in runs])
    drawdowns = np.array([r.max_drawdown_pct for r in runs])

    stats = {
        "median_pnl": float(np.median(net_pnls)),
        "median_pct": float(np.median(pct_returns)),
        "p95_pnl": float(np.percentile(net_pnls, 95)),
        "p95_pct": float(np.percentile(pct_returns, 95)),
        "p5_pnl": float(np.percentile(net_pnls, 5)),
        "p5_pct": float(np.percentile(pct_returns, 5)),
        "median_dd": float(np.median(drawdowns)),
        "worst_dd": float(np.max(drawdowns)),
    }

    rows = [
        ("Simulations", f"{len(runs):,} runs x {PROJECTION_DAYS} trading days"),
        ("Backtest sample", f"{BT_TRADES} trades / {BT_SESSIONS} sessions "
                           f"(PF {BT_PROFIT_FACTOR})"),
        ("Per-trade model", f"{BT_WIN_RATE:.1%} win @ +{BT_AVG_WIN_R:.2f}R  /  "
                           f"{1 - BT_WIN_RATE:.1%} loss @ -{BT_AVG_LOSS_R:.2f}R"),
        ("Slippage drag", f"-{SLIPPAGE_PCT:.2%} notional/trade "
                          f"(-{slippage_r():.3f}R), every trade"),
        ("Gap-down tail risk", f"{GAP_DOWN_PROB:.0%} of losses gap to "
                               f"-{GAP_DOWN_PCT:.1%} (-{GAP_DOWN_R:.1f}R) "
                               f"instead of the -2.5% stop"),
        ("Trade frequency", f"{BT_TRADES / BT_SESSIONS:.3f} trades/day (Poisson)"),
        ("Effective risk / trade", f"{effective_risk_pct():.3%} of equity "
                                   f"(capped by {MAX_CONCURRENT} concurrent slots)"),
        ("Starting capital", f"${START_CAPITAL:,.2f}"),
        ("Median net PnL", f"${stats['median_pnl']:,.2f}  "
                          f"({stats['median_pct']:+.2f}%)"),
        ("95th pct — best case", f"${stats['p95_pnl']:,.2f}  "
                                f"({stats['p95_pct']:+.2f}%)"),
        ("5th pct — worst case", f"${stats['p5_pnl']:,.2f}  "
                                f"({stats['p5_pct']:+.2f}%)"),
        ("Max expected drawdown", f"{stats['median_dd']:.2f}% median "
                                 f"(worst of {len(runs)} runs: {stats['worst_dd']:.2f}%)"),
    ]

    label_w = max(len(label) for label, _ in rows) + 2
    width = label_w + 40
    print("=" * width)
    print(f"MONTE CARLO PROJECTION — Oversold Dip w/ Real-World Friction "
          f"({len(runs):,} runs, {PROJECTION_DAYS}d)")
    print("=" * width)
    for label, value in rows:
        print(f"{label:<{label_w}}{value}")
    print("=" * width)
    print("NOTE: every run randomizes WHEN trades fire, WHETHER each wins or")
    print("loses at the backtest's actual 55.3% hit rate, and WHETHER a loss")
    print("gaps through its stop — on top of that, slippage is deducted from")
    print("every single trade. Losing streaks, red weeks, real drawdowns and")
    print("this friction are all now in scope, not just upside variance.")

    return stats


def pick_representative(runs: list[RunResult], stats: dict) -> RunResult:
    """
    The run whose final net P&L sits closest to the 1,000-run median — a
    single path that illustrates a typical outcome, not the luckiest or the
    unluckiest one.
    """
    diffs = [abs(r.net_pnl - stats["median_pnl"]) for r in runs]
    return runs[int(np.argmin(diffs))]


def plot_run(run: RunResult) -> None:
    """
    Two panels, one figure: the equity curve on top (its dips ARE the
    drawdowns) and each day's P&L as a red/green bar underneath, so a losing
    day reads as a losing day rather than only a flatter slope up top.

    plotext 6.x uses an object API on `plt.figure` — figure.signal()/.bar()
    build a data series but only reach the canvas once fig.draw(series) is
    called on it explicitly; bar color comes from the `marker` argument via
    plotext.colorize(), not from theme/color-cycling.
    """
    days = list(range(len(run.equity_curve)))
    day_idx = [d.day for d in run.days_log]
    daily_pnl = [d.pnl for d in run.days_log]
    gains = [p if p > 0 else 0 for p in daily_pnl]
    losses = [p if p < 0 else 0 for p in daily_pnl]

    fig = plt.figure
    fig.clear()
    fig.subplots(2, 1)
    top = fig.subplot(1, 1)
    bottom = fig.subplot(2, 1)

    top.theme("dark")
    sig = top.signal(days, run.equity_curve, marker="braille").lines()
    top.draw(sig)
    # plotext blanks a subplot title outright rather than wrapping/truncating
    # it when it doesn't fit the panel width, so keep this short.
    top.title(f"Equity Curve — ${START_CAPITAL:,.0f} start, {PROJECTION_DAYS}d "
              f"(median-outcome path)")
    top.label("Equity ($)", axis="y")

    bottom.theme("dark")
    g = bottom.bar(day_idx, gains, marker=plt.colorize("█", pixel="green"))
    l = bottom.bar(day_idx, losses, marker=plt.colorize("█", pixel="red"))
    bottom.draw(g)
    bottom.draw(l)
    bottom.title("Daily P&L (green = up day, red = down day)")
    bottom.label("Trading day", axis="x")
    bottom.label("Day P&L ($)", axis="y")

    fig.plot_size(100, 40)
    fig.show()


def print_run_detail(run: RunResult) -> None:
    """Numbers for the specific path just plotted, so chart and text agree."""
    print()
    print(f"Representative path: {run.total_trades} trades -> "
          f"${run.final_equity:,.2f} ({run.net_pnl:+,.2f}, "
          f"{run.pct_return:+.2f}%), max drawdown {run.max_drawdown_pct:.2f}%")


def main() -> int:
    runs = run_monte_carlo()
    stats = print_summary(runs)
    rep = pick_representative(runs, stats)
    print()
    plot_run(rep)
    print_run_detail(rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
