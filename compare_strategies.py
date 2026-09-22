"""
compare_strategies.py — side-by-side backtest comparison: the Phase 1
Baseline (Strategy A) vs the Fixed ADX Hybrid with asymmetric risk sizing
(Strategy B), same 30-symbol Mixed Universe, same historical window,
$100,000 starting capital each.

Reuses backtest.py's validated engines directly rather than reimplementing
any signal/exit logic:
    Strategy A — backtest.run_daily_backtest(..., variant="A",
                 regime_filter=False): RSI<35/lower-BB-touch entry, fixed
                 2.5% stop, 2.5R target.
    Strategy B — backtest.run_adx_hybrid_fixed_backtest(..., mode="HYBRID",
                 risk_pct_mr=RISK_PCT_MR, risk_pct_mom=RISK_PCT_MOM):
                 dynamic 50-session ADX-percentile regime routing, 1.5x
                 volume-SMA gating on breakouts, 2.0x ATR trailing stop —
                 PLUS the asymmetric 0.625%/0.25% risk-based sizing
                 hybrid_indicators.py's live tool uses. That sizing scheme
                 was added to run_adx_hybrid_fixed_backtest() as a new
                 OPT-IN pair of parameters (risk_pct_mr/risk_pct_mom,
                 default None) specifically to run this comparison — passing
                 neither preserves that function's original behavior
                 (size_position() capped at equity's position-pct, uniform
                 across styles) exactly, so this file's existing callers
                 (e.g. backtest.py --fixed-hybrid-test) are unaffected.

OUT-OF-SAMPLE EFFICIENCY uses the SAME methodology validate_stability.py
already established: two independent, freshly-capitalized runs over
IS (2022-01-01 -> 2024-06-01) and OOS (2024-06-02 -> present), each strategy's
own annualized Sharpe at a 2% risk-free rate, OOS Sharpe / IS Sharpe x 100%.
This is a SEPARATE Sharpe basis from the headline "Sharpe Ratio" column
(which uses backtest.py's existing 0%-risk-free convention, matching every
other report in this pipeline) — labeled explicitly in the footnote so the
two aren't confused.

CALMAR RATIO here is exactly Total Return% / Max Drawdown% (not the
textbook CAGR/MaxDD definition) — this is the literal formula specified for
this comparison; flagged since it differs from the standard usage.

Usage:
    python3 compare_strategies.py                  # Strategy A vs B, Mixed universe
    python3 compare_strategies.py --universe-scaling  # Strategy B only, 30 vs 100 symbols
"""

from __future__ import annotations

import datetime as dt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import backtest as bt
import data
from config import BASE

START = dt.date(2022, 1, 1)
END = dt.date.today()
IS_END = dt.date(2024, 6, 1)
OOS_START = dt.date(2024, 6, 2)

STARTING_CAPITAL = 100_000.0
UNIVERSE_NAME = "Mixed"

RISK_PCT_MR = 0.00625     # DIP / mean-reversion — matches hybrid_indicators.py
RISK_PCT_MOM = 0.0025     # BREAKOUT / momentum  — matches hybrid_indicators.py

RISK_FREE_ANNUAL = 0.02   # OOS-efficiency Sharpe basis only (see module docstring)
TRADING_DAYS = 252

OUT_PNG = BASE / "strategy_comparison.png"


# ---------------------------------------------------------------- OOS efficiency
def _daily_rf() -> float:
    return (1.0 + RISK_FREE_ANNUAL) ** (1.0 / TRADING_DAYS) - 1.0


def _annualized_sharpe_2pct(equity: pd.Series) -> float:
    if len(equity) < 3:
        return 0.0
    r = equity.pct_change().dropna()
    if r.empty or r.std() == 0:
        return 0.0
    excess = r - _daily_rf()
    return float(excess.mean() / r.std() * np.sqrt(TRADING_DAYS))


def oos_efficiency(is_equity: pd.Series, oos_equity: pd.Series) -> float | None:
    """OOS Sharpe / IS Sharpe x 100%, both at 2% risk-free. None if IS Sharpe
    <= 0 (a ratio against a non-positive base isn't a meaningful efficiency)."""
    is_sharpe = _annualized_sharpe_2pct(is_equity)
    if is_sharpe <= 0:
        return None
    oos_sharpe = _annualized_sharpe_2pct(oos_equity)
    return oos_sharpe / is_sharpe * 100.0


# ---------------------------------------------------------------- metrics
def _pnl_of(trade) -> float:
    return trade.pnl if hasattr(trade, "pnl") else trade.realized_pnl


def avg_active_positions_per_day(res) -> float:
    """
    Mean number of positions simultaneously open on an average trading day
    in the backtest window — book utilization, not just entry rate. Directly
    answers "does a wider universe actually keep more slots filled more
    often," which is the whole premise being tested by expanding to 100
    symbols. O(days x trades); a few hundred thousand ops even at 100
    symbols over 4+ years, so no need to optimize further.
    """
    days = list(res.equity.index)
    if not days:
        return 0.0
    trades = [t for t in res.trades if t.entry_date and t.exit_date]
    counts = [sum(1 for t in trades if t.entry_date <= day <= t.exit_date)
             for day in days]
    return float(np.mean(counts)) if counts else 0.0


def compute_metrics(res, capital: float, oos_eff: float | None = None) -> dict:
    trades = res.trades
    n = len(trades)
    wins = [t for t in trades if _pnl_of(t) > 0]
    losses = [t for t in trades if _pnl_of(t) <= 0]
    gross_win = sum(_pnl_of(t) for t in wins)
    gross_loss = -sum(_pnl_of(t) for t in losses)
    net = sum(_pnl_of(t) for t in trades)
    win_rate = len(wins) / n if n else 0.0
    profit_factor = (gross_win / gross_loss if gross_loss > 0
                    else (float("inf") if gross_win > 0 else 0.0))

    eq = res.equity
    _, dd_frac = bt.max_drawdown(eq) if not eq.empty else (0.0, 0.0)
    max_dd_pct = dd_frac * 100.0

    total_return_pct = net / capital * 100.0
    n_days = (res.end - res.start).days if res.start and res.end else 0
    years = n_days / 365.25 if n_days > 0 else 0.0
    final_equity = capital + net
    ann_return_pct = (((final_equity / capital) ** (1.0 / years) - 1.0) * 100.0
                      if years > 0 and final_equity > 0 else 0.0)

    calmar = (total_return_pct / max_dd_pct if max_dd_pct > 0
             else (float("inf") if total_return_pct > 0 else 0.0))

    hold_days = [(t.exit_date - t.entry_date).days for t in trades
                if t.entry_date and t.exit_date]
    avg_hold = float(np.mean(hold_days)) if hold_days else 0.0

    return dict(
        total_return_pct=total_return_pct, ann_return_pct=ann_return_pct,
        max_dd_pct=max_dd_pct, calmar=calmar,
        sharpe=bt.sharpe(eq), sortino=bt.sortino(eq),
        win_rate_pct=win_rate * 100.0, profit_factor=profit_factor,
        n_trades=n, avg_hold_days=avg_hold, oos_efficiency=oos_eff,
        active_trades_per_day=avg_active_positions_per_day(res),
    )


# ---------------------------------------------------------------- backtest runs
def run_all():
    symbols = bt.UNIVERSES[UNIVERSE_NAME]

    print(f"Strategy A (Phase 1 Baseline): full window {START} -> {END} ...",
          flush=True)
    a_full = bt.run_daily_backtest(
        symbols, START, END, capital=STARTING_CAPITAL, regime_filter=False,
        universe_name=UNIVERSE_NAME, variant="A")
    print("Strategy A: IS/OOS sub-windows ...", flush=True)
    a_is = bt.run_daily_backtest(
        symbols, START, IS_END, capital=STARTING_CAPITAL, regime_filter=False,
        universe_name=UNIVERSE_NAME, variant="A")
    a_oos = bt.run_daily_backtest(
        symbols, OOS_START, END, capital=STARTING_CAPITAL, regime_filter=False,
        universe_name=UNIVERSE_NAME, variant="A")

    print(f"Strategy B (Fixed ADX Hybrid): full window {START} -> {END} ...",
          flush=True)
    b_full = bt.run_adx_hybrid_fixed_backtest(
        symbols, START, END, mode="HYBRID", capital=STARTING_CAPITAL,
        universe_name=UNIVERSE_NAME, risk_pct_mr=RISK_PCT_MR,
        risk_pct_mom=RISK_PCT_MOM)
    print("Strategy B: IS/OOS sub-windows ...", flush=True)
    b_is = bt.run_adx_hybrid_fixed_backtest(
        symbols, START, IS_END, mode="HYBRID", capital=STARTING_CAPITAL,
        universe_name=UNIVERSE_NAME, risk_pct_mr=RISK_PCT_MR,
        risk_pct_mom=RISK_PCT_MOM)
    b_oos = bt.run_adx_hybrid_fixed_backtest(
        symbols, OOS_START, END, mode="HYBRID", capital=STARTING_CAPITAL,
        universe_name=UNIVERSE_NAME, risk_pct_mr=RISK_PCT_MR,
        risk_pct_mom=RISK_PCT_MOM)

    a_metrics = compute_metrics(a_full, STARTING_CAPITAL,
                                oos_efficiency(a_is.equity, a_oos.equity))
    b_metrics = compute_metrics(b_full, STARTING_CAPITAL,
                                oos_efficiency(b_is.equity, b_oos.equity))
    return a_full, b_full, a_metrics, b_metrics


# ---------------------------------------------------------------- reporting
def _fmt_ratio(x: float) -> str:
    return "inf" if x == float("inf") else f"{x:.2f}"


def _fmt_oos(x: float | None) -> str:
    return f"{x:.1f}%" if x is not None else "n/a (IS Sharpe<=0)"


def print_table(a_metrics: dict, b_metrics: dict) -> None:
    rows = [
        ("Total Return (%)", f"{a_metrics['total_return_pct']:+.2f}%",
         f"{b_metrics['total_return_pct']:+.2f}%"),
        ("Annualized Return (%)", f"{a_metrics['ann_return_pct']:+.2f}%",
         f"{b_metrics['ann_return_pct']:+.2f}%"),
        ("Max Drawdown (%)", f"{a_metrics['max_dd_pct']:.2f}%",
         f"{b_metrics['max_dd_pct']:.2f}%"),
        ("Calmar Ratio (Ret/MaxDD)", _fmt_ratio(a_metrics["calmar"]),
         _fmt_ratio(b_metrics["calmar"])),
        ("Sharpe Ratio", f"{a_metrics['sharpe']:.3f}", f"{b_metrics['sharpe']:.3f}"),
        ("Sortino Ratio", f"{a_metrics['sortino']:.3f}", f"{b_metrics['sortino']:.3f}"),
        ("Win Rate (%)", f"{a_metrics['win_rate_pct']:.1f}%",
         f"{b_metrics['win_rate_pct']:.1f}%"),
        ("Profit Factor", _fmt_ratio(a_metrics["profit_factor"]),
         _fmt_ratio(b_metrics["profit_factor"])),
        ("Total Trades", f"{a_metrics['n_trades']}", f"{b_metrics['n_trades']}"),
        ("Avg Hold Duration (days)", f"{a_metrics['avg_hold_days']:.1f}",
         f"{b_metrics['avg_hold_days']:.1f}"),
        ("OOS Efficiency (%)", _fmt_oos(a_metrics["oos_efficiency"]),
         _fmt_oos(b_metrics["oos_efficiency"])),
    ]

    width = 78
    print("=" * width)
    print("STRATEGY COMPARISON — Phase 1 Baseline vs Fixed ADX Hybrid")
    print(f"Mixed Universe, {START} -> {END}, ${STARTING_CAPITAL:,.0f} "
          f"starting capital each")
    print("=" * width)
    print(f"{'Metric':<28}{'A: Phase 1 Baseline':>24}{'B: Fixed ADX Hybrid':>24}")
    print("-" * width)
    for label, a_v, b_v in rows:
        print(f"{label:<28}{a_v:>24}{b_v:>24}")
    print("=" * width)

    try:
        spy = data.daily_bars("SPY", period="max")
        w = spy[(spy.index.date >= START) & (spy.index.date <= END)]
        if len(w) > 1:
            r = float(w["Close"].iloc[-1] / w["Close"].iloc[0] - 1.0)
            print(f"SPY buy & hold, same window: {r*100:+.2f}%")
    except Exception:
        pass

    print()
    print("Sharpe/Sortino above use backtest.py's standard 0%-risk-free")
    print("convention (matching every other report in this pipeline). OOS")
    print("Efficiency uses a SEPARATE 2%-risk-free annualized Sharpe, per")
    print("validate_stability.py's own methodology — the two are not the")
    print("same number and shouldn't be compared to each other directly.")
    print("Calmar here is literal Total Return% / Max Drawdown%, not the")
    print("textbook CAGR/MaxDD definition.")


def print_verdict(a_metrics: dict, b_metrics: dict) -> None:
    sharpe_winner = "A" if a_metrics["sharpe"] > b_metrics["sharpe"] else "B"
    calmar_winner = "A" if a_metrics["calmar"] > b_metrics["calmar"] else "B"
    names = {"A": "Phase 1 Baseline", "B": "Fixed ADX Hybrid"}

    print()
    print("=" * 78)
    print("QUANTITATIVE DECISION VERDICT")
    print("=" * 78)
    print(f"Sharpe Ratio -> Strategy {sharpe_winner} ({names[sharpe_winner]})  "
          f"[A={a_metrics['sharpe']:.3f}  B={b_metrics['sharpe']:.3f}]")
    print(f"Calmar Ratio -> Strategy {calmar_winner} ({names[calmar_winner]})  "
          f"[A={_fmt_ratio(a_metrics['calmar'])}  "
          f"B={_fmt_ratio(b_metrics['calmar'])}]")
    print()

    if sharpe_winner == calmar_winner:
        print(f"VERDICT: Strategy {sharpe_winner} ({names[sharpe_winner]}) is "
              f"superior on BOTH risk-adjusted measures over this window.")
    else:
        print(f"VERDICT: MIXED — Sharpe favors {names[sharpe_winner]}, Calmar "
              f"favors {names[calmar_winner]}. Sharpe rewards smooth, "
              f"consistent returns overall; Calmar specifically penalizes "
              f"deep peak-to-trough drawdowns. Which matters more depends on "
              f"whether day-to-day volatility or a large drawdown is the "
              f"bigger practical concern for how this would actually be traded.")
    print("=" * 78)


# ---------------------------------------------------------------- visualization
def _drawdown_series(eq: pd.Series) -> pd.Series:
    if eq.empty:
        return eq
    peak = eq.cummax()
    return (eq - peak) / peak * 100.0


def _spy_equity(start: dt.date, end: dt.date, capital: float) -> pd.Series:
    try:
        spy = data.daily_bars("SPY", period="max")
    except Exception:
        return pd.Series(dtype=float)
    w = spy[(spy.index.date >= start) & (spy.index.date <= end)]
    if w.empty:
        return pd.Series(dtype=float)
    norm = capital * (w["Close"] / w["Close"].iloc[0])
    return pd.Series(norm.to_numpy(), index=pd.to_datetime([ts.date() for ts in w.index]))


def make_plot(a_res, b_res, out_path) -> None:
    a_eq = pd.Series(a_res.equity.to_numpy(),
                     index=pd.to_datetime(list(a_res.equity.index)))
    b_eq = pd.Series(b_res.equity.to_numpy(),
                     index=pd.to_datetime(list(b_res.equity.index)))
    spy_eq = _spy_equity(START, END, STARTING_CAPITAL)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(a_eq.index, a_eq.values, label="A: Phase 1 Baseline",
             color="#2ecc71", linewidth=1.4)
    ax1.plot(b_eq.index, b_eq.values, label="B: Fixed ADX Hybrid",
             color="#3498db", linewidth=1.4)
    if not spy_eq.empty:
        ax1.plot(spy_eq.index, spy_eq.values, label="SPY Buy & Hold",
                 color="#95a5a6", linewidth=1.2, linestyle="--")
    ax1.set_title("Cumulative Equity Curves")
    ax1.set_ylabel("Equity ($)")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    dd_a = _drawdown_series(a_eq)
    dd_b = _drawdown_series(b_eq)
    ax2.fill_between(dd_a.index, dd_a.values, 0, color="#2ecc71", alpha=0.35,
                     label="A: Phase 1 Baseline")
    ax2.fill_between(dd_b.index, dd_b.values, 0, color="#3498db", alpha=0.35,
                     label="B: Fixed ADX Hybrid")
    ax2.set_title("Drawdown Profile")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3)

    fig.suptitle(f"Strategy Comparison — Mixed Universe, {START} to {END}",
                fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- universe scaling
def run_universe_scaling_comparison():
    """
    Strategy B (Fixed ADX Hybrid, asymmetric risk sizing) ONLY — run once on
    the 30-symbol Mixed universe and once on the 100-symbol Expanded100
    universe, same window, same starting capital each. No OOS split here
    (not requested for this comparison) — just the full-window metrics.
    """
    print(f"Strategy B on Mixed (30 symbols): {START} -> {END} ...", flush=True)
    mixed = bt.run_adx_hybrid_fixed_backtest(
        bt.UNIVERSES["Mixed"], START, END, mode="HYBRID",
        capital=STARTING_CAPITAL, universe_name="Mixed",
        risk_pct_mr=RISK_PCT_MR, risk_pct_mom=RISK_PCT_MOM)

    print(f"Strategy B on Expanded100 (100 symbols): {START} -> {END} ...",
          flush=True)
    expanded = bt.run_adx_hybrid_fixed_backtest(
        bt.EXPANDED_UNIVERSE_100, START, END, mode="HYBRID",
        capital=STARTING_CAPITAL, universe_name="Expanded100",
        risk_pct_mr=RISK_PCT_MR, risk_pct_mom=RISK_PCT_MOM)

    mixed_metrics = compute_metrics(mixed, STARTING_CAPITAL)
    expanded_metrics = compute_metrics(expanded, STARTING_CAPITAL)
    return mixed, expanded, mixed_metrics, expanded_metrics


def print_universe_table(m30: dict, m100: dict) -> None:
    rows = [
        ("Total Return (%)", f"{m30['total_return_pct']:+.2f}%",
         f"{m100['total_return_pct']:+.2f}%"),
        ("Annualized Return (%)", f"{m30['ann_return_pct']:+.2f}%",
         f"{m100['ann_return_pct']:+.2f}%"),
        ("Max Drawdown (%)", f"{m30['max_dd_pct']:.2f}%", f"{m100['max_dd_pct']:.2f}%"),
        ("Sharpe Ratio", f"{m30['sharpe']:.3f}", f"{m100['sharpe']:.3f}"),
        ("Calmar Ratio (Ret/MaxDD)", _fmt_ratio(m30["calmar"]),
         _fmt_ratio(m100["calmar"])),
        ("Win Rate (%)", f"{m30['win_rate_pct']:.1f}%", f"{m100['win_rate_pct']:.1f}%"),
        ("Total Trade Count", f"{m30['n_trades']}", f"{m100['n_trades']}"),
        ("Avg Trade Duration (days)", f"{m30['avg_hold_days']:.1f}",
         f"{m100['avg_hold_days']:.1f}"),
        ("Active Trades / Day", f"{m30['active_trades_per_day']:.2f}",
         f"{m100['active_trades_per_day']:.2f}"),
    ]

    width = 74
    print("=" * width)
    print("UNIVERSE SCALING — Strategy B (Fixed ADX Hybrid), 30 vs 100 symbols")
    print(f"{START} -> {END}, ${STARTING_CAPITAL:,.0f} starting capital each")
    print("=" * width)
    print(f"{'Metric':<28}{'30-Symbol (Mixed)':>22}{'100-Symbol (Expanded)':>24}")
    print("-" * width)
    for label, a_v, b_v in rows:
        print(f"{label:<28}{a_v:>22}{b_v:>24}")
    print("=" * width)


def print_universe_verdict(m30: dict, m100: dict) -> None:
    sharpe_winner = "30" if m30["sharpe"] > m100["sharpe"] else "100"
    calmar_winner = "30" if m30["calmar"] > m100["calmar"] else "100"
    freq_ratio = (m100["active_trades_per_day"] / m30["active_trades_per_day"]
                 if m30["active_trades_per_day"] > 0 else float("inf"))

    print()
    print("=" * 74)
    print("QUANTITATIVE ASSESSMENT — does expanding to 100 symbols help?")
    print("=" * 74)
    print(f"Setup frequency: {m100['active_trades_per_day']:.2f} active "
          f"trades/day at 100 symbols vs {m30['active_trades_per_day']:.2f} "
          f"at 30 ({freq_ratio:.2f}x); trade count {m100['n_trades']} vs "
          f"{m30['n_trades']}.")
    print(f"Sharpe Ratio -> {sharpe_winner}-symbol universe  "
          f"[30={m30['sharpe']:.3f}  100={m100['sharpe']:.3f}]")
    print(f"Calmar Ratio -> {calmar_winner}-symbol universe  "
          f"[30={_fmt_ratio(m30['calmar'])}  100={_fmt_ratio(m100['calmar'])}]")
    print()

    if sharpe_winner == calmar_winner == "100":
        print("VERDICT: Expanding to 100 symbols IMPROVES risk-adjusted "
              "performance on both Sharpe and Calmar, in addition to raising "
              "setup frequency — a clean win for widening the universe.")
    elif sharpe_winner == calmar_winner == "30":
        print("VERDICT: Expanding to 100 symbols DEGRADES risk-adjusted "
              "performance on both Sharpe and Calmar despite the higher "
              "setup frequency — more opportunities did not translate into "
              "a better-quality book; the extra 70 names are diluting "
              "average trade quality rather than adding good ones.")
    else:
        print(f"VERDICT: MIXED — Sharpe favors the {sharpe_winner}-symbol "
              f"universe, Calmar favors the {calmar_winner}-symbol universe. "
              f"Higher setup frequency alone ({freq_ratio:.2f}x) is not "
              f"sufficient evidence expansion helps — it bought more trades, "
              f"not unambiguously better risk-adjusted returns.")
    print("=" * 74)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Backtest comparison: Phase 1 Baseline vs Fixed ADX Hybrid")
    ap.add_argument("--universe-scaling", action="store_true",
                    help="Strategy B only, 30-symbol Mixed vs 100-symbol "
                         "Expanded100 universe, instead of the default "
                         "Strategy A vs B comparison")
    args = ap.parse_args()

    if args.universe_scaling:
        m30_res, m100_res, m30, m100 = run_universe_scaling_comparison()
        print_universe_table(m30, m100)
        print_universe_verdict(m30, m100)
        return 0

    a_res, b_res, a_metrics, b_metrics = run_all()
    print_table(a_metrics, b_metrics)
    print_verdict(a_metrics, b_metrics)
    make_plot(a_res, b_res, OUT_PNG)
    print(f"\nSaved: {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
