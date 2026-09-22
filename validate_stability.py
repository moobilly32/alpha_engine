"""
Phase 4 — Walk-Forward Validation & Sharpe Stability Analysis for Test B, the
FIXED ADX Hybrid strategy: Mixed universe, mean-reversion side unchanged
(Variant A), momentum side gated by volume confirmation (>1.5x 20-day avg
volume), exited via a 2.0x ATR(14) trailing stop instead of the 10-EMA, and
routed by a dynamic (rolling 80th-percentile) ADX threshold instead of the
fixed ADX>25 — backtest.py's run_adx_hybrid_fixed_backtest(mode="HYBRID").

Prior runs of this same report: the original (unfixed) ADX Hybrid FAILED
(OOS SSR 1.13, OOS Efficiency 35.8%); the plain Phase 1 Variant A baseline
PASSED convincingly (OOS SSR 10.38, OOS Efficiency 210.6%). This run checks
whether Test B's full-window outperformance of both (+89.49% vs +87.07% vs
+56.20%, and the best drawdown/win-rate/profit-factor of the three) reflects
a genuinely more stable edge, or whether the fixes mainly improved the
in-sample-heavy full-window number without fixing the underlying instability
the original Hybrid showed.

METHODOLOGY
    Two INDEPENDENT backtests — each starting fresh from BACKTEST_CAPITAL, not
    one continuous run cut in two — over disjoint windows:
        IN-SAMPLE (IS):      2022-01-01 -> 2024-06-01
        OUT-OF-SAMPLE (OOS): 2024-06-02 -> 2026-09-14
    Independent runs isolate "did the return profile hold up in a later era"
    from "OOS position sizing was inflated/deflated by whatever equity level
    the IS run happened to end at" — a single continuous run split in two
    would conflate the two questions.

    IMPORTANT: there is no parameter-FITTING step here. Variant A's
    thresholds (RSI<35, BB(20,2), STOP_PCT, TARGET_R, ...) are the same fixed
    constants from backtest.py/config.py in both windows — nothing is tuned
    on IS data and re-run on OOS. This script is therefore a
    STABILITY/CONSISTENCY check ("does this one fixed strategy behave
    similarly in a later, unseen era"), not a classic fit-then-validate
    walk-forward optimization test. Framing it as the latter would overstate
    what a script with no free parameters can actually demonstrate.

STATISTICS
    Sharpe uses a 2% annual risk-free rate throughout — both the rolling
    30-trading-day windows and each period's single annualized Sharpe —
    converted to a daily rate via (1.02)**(1/252) - 1.

    Sharpe Stability Ratio (SSR) = mean(rolling 30-day Sharpe) /
    standard_error(rolling 30-day Sharpe). A large SSR means the rolling
    Sharpe estimate has been both positive AND consistent period-to-period,
    not just positive on average once. Computed for both IS and OOS for
    context, but the pass/fail VERDICT uses OOS's SSR only — that is the
    series actually being validated.

    OOS Efficiency Ratio = OOS annualized Sharpe / IS annualized Sharpe x
    100% — how much of the in-sample risk-adjusted return survived into an
    era the fixed strategy was not shaped by.

    VERDICT: PASS requires BOTH OOS SSR > 1.96 AND OOS Efficiency > 70%.

Usage:
    python3 validate_stability.py
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

import backtest as bt
from config import BACKTEST_CAPITAL

IS_START = dt.date(2022, 1, 1)
IS_END = dt.date(2024, 6, 1)
OOS_START = dt.date(2024, 6, 2)
OOS_END = dt.date(2026, 9, 14)

RISK_FREE_ANNUAL = 0.02
ROLLING_WINDOW = 30           # trading days
TRADING_DAYS = 252

SSR_PASS_THRESHOLD = 1.96
OOS_EFFICIENCY_PASS_THRESHOLD = 70.0   # percent

UNIVERSE_NAME = "Mixed"
MODE = "HYBRID"            # ADX-routed per symbol per day — see backtest.py


def daily_rf() -> float:
    return (1.0 + RISK_FREE_ANNUAL) ** (1.0 / TRADING_DAYS) - 1.0


def annualized_sharpe(equity: pd.Series) -> float:
    """Annualized Sharpe on daily equity returns, RISK_FREE_ANNUAL risk-free."""
    if len(equity) < 3:
        return 0.0
    r = equity.pct_change().dropna()
    if r.empty or r.std() == 0:
        return 0.0
    excess = r - daily_rf()
    return float(excess.mean() / r.std() * np.sqrt(TRADING_DAYS))


def rolling_sharpe(equity: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    """
    Trailing `window`-trading-day annualized Sharpe, one value per day once
    `window` daily returns are available (dropped before that, not padded
    with NaN, so callers get a clean series to take mean/std of directly).

    A window with zero return variance (a stretch with no open positions, so
    every day's return is exactly 0) divides mean/std by zero — pandas gives
    +-inf, not NaN, when the numerator is nonzero, and an infinity left in
    the series poisons every downstream mean/std (mean becomes +-inf, std
    becomes NaN). Treated as undefined and dropped here, matching how
    backtest.py's own sharpe()/annualized_sharpe() above treat zero std.
    """
    r = equity.pct_change().dropna()
    if len(r) <= window:
        return pd.Series(dtype=float)
    excess = r - daily_rf()
    mean_roll = excess.rolling(window).mean()
    std_roll = r.rolling(window).std()
    sharpe = (mean_roll / std_roll * np.sqrt(TRADING_DAYS))
    return sharpe.replace([np.inf, -np.inf], np.nan).dropna()


def sharpe_stability_ratio(roll: pd.Series) -> float:
    """
    SSR = mean(rolling Sharpe) / standard_error(rolling Sharpe), i.e. a
    t-statistic-style measure of whether the rolling Sharpe has been
    reliably positive rather than merely positive on average with wide
    swings. SSR > 1.96 is the classic two-tailed 95%-confidence threshold
    for "reliably different from zero."
    """
    if len(roll) < 2:
        return 0.0
    se = roll.std(ddof=1) / np.sqrt(len(roll))
    if se == 0:
        return float("inf") if roll.mean() > 0 else 0.0
    return float(roll.mean() / se)


def period_metrics(res: "bt.HybridResult", capital: float) -> dict:
    eq = res.equity
    net = sum(t.realized_pnl for t in res.trades)
    _, dd_p = bt.max_drawdown(eq) if not eq.empty else (0.0, 0.0)
    roll = rolling_sharpe(eq)
    return dict(
        equity=eq,
        return_pct=net / capital * 100.0,
        max_dd_pct=dd_p * 100.0,
        sharpe=annualized_sharpe(eq),
        rolling_sharpe=roll,
        ssr=sharpe_stability_ratio(roll),
        n_trades=len(res.trades),
    )


def run() -> tuple[dict, dict]:
    symbols = bt.UNIVERSES[UNIVERSE_NAME]

    print(f"Running IN-SAMPLE      {IS_START} -> {IS_END}   "
          f"({UNIVERSE_NAME}, Fixed ADX Hybrid) ...", flush=True)
    is_res = bt.run_adx_hybrid_fixed_backtest(
        symbols, IS_START, IS_END, mode=MODE,
        capital=BACKTEST_CAPITAL, universe_name=UNIVERSE_NAME)

    print(f"Running OUT-OF-SAMPLE  {OOS_START} -> {OOS_END}   "
          f"({UNIVERSE_NAME}, Fixed ADX Hybrid) ...", flush=True)
    oos_res = bt.run_adx_hybrid_fixed_backtest(
        symbols, OOS_START, OOS_END, mode=MODE,
        capital=BACKTEST_CAPITAL, universe_name=UNIVERSE_NAME)

    return (period_metrics(is_res, BACKTEST_CAPITAL),
            period_metrics(oos_res, BACKTEST_CAPITAL))


def print_report(is_m: dict, oos_m: dict) -> None:
    is_sharpe, oos_sharpe = is_m["sharpe"], oos_m["sharpe"]
    eff_defined = is_sharpe > 0
    oos_efficiency = (oos_sharpe / is_sharpe * 100.0) if eff_defined else float("nan")

    ssr_ok = oos_m["ssr"] > SSR_PASS_THRESHOLD
    eff_ok = eff_defined and oos_efficiency > OOS_EFFICIENCY_PASS_THRESHOLD
    verdict = "PASS" if (ssr_ok and eff_ok) else "FAIL"

    width = 78

    def block(title: str, m: dict) -> None:
        roll = m["rolling_sharpe"]
        print(title)
        print(f"  Return                     {m['return_pct']:>+9.2f}%")
        print(f"  Max Drawdown               {m['max_dd_pct']:>9.2f}%")
        print(f"  Sharpe (annualized, 2% rf) {m['sharpe']:>9.3f}")
        if len(roll):
            print(f"  Rolling {ROLLING_WINDOW}d Sharpe          "
                  f"mean={roll.mean():>6.3f}  std={roll.std():>6.3f}  "
                  f"n={len(roll)}")
        else:
            print(f"  Rolling {ROLLING_WINDOW}d Sharpe          "
                  f"insufficient history")
        print(f"  Sharpe Stability Ratio     {m['ssr']:>9.2f}")
        print(f"  Trades                     {m['n_trades']:>9}")

    print("=" * width)
    print("PHASE 4 — WALK-FORWARD VALIDATION & SHARPE STABILITY")
    print(f"Mixed Universe / Test B: Fixed ADX Hybrid   "
          f"IS {IS_START} -> {IS_END}   OOS {OOS_START} -> {OOS_END}")
    print("=" * width)
    block("IN-SAMPLE (IS)", is_m)
    print("-" * width)
    block("OUT-OF-SAMPLE (OOS)", oos_m)
    print("-" * width)
    if eff_defined:
        print(f"OOS Efficiency Ratio (OOS Sharpe / IS Sharpe x 100%): "
              f"{oos_efficiency:>+7.1f}%")
    else:
        print("OOS Efficiency Ratio: undefined (IS Sharpe <= 0 — a ratio "
              "against a non-positive base is not a meaningful 'efficiency')")
    print("=" * width)
    print(f"VALIDATION VERDICT: {verdict}")
    print(f"  OOS SSR        {oos_m['ssr']:>7.2f}  "
          f"{'>' if ssr_ok else '<='} {SSR_PASS_THRESHOLD}   "
          f"[{'OK' if ssr_ok else 'FAIL'}]")
    if eff_defined:
        print(f"  OOS Efficiency {oos_efficiency:>7.1f}%  "
              f"{'>' if eff_ok else '<='} {OOS_EFFICIENCY_PASS_THRESHOLD:.0f}%   "
              f"[{'OK' if eff_ok else 'FAIL'}]")
    else:
        print(f"  OOS Efficiency        n/a  [FAIL — undefined counts as fail]")
    print("=" * width)
    print("NOTE: no parameters were fit on IS data — Test B's thresholds")
    print("(volume 1.5x, ATR 2.0x, ADX percentile window/floor) are the same")
    print("fixed constants from backtest.py in both windows. This is a")
    print("stability/consistency check across two independent eras, not a")
    print("classic fit-then-validate walk-forward optimization test. IS and")
    print("OOS are separately-capitalized runs, not one equity curve cut in two.")


def main() -> int:
    is_m, oos_m = run()
    print_report(is_m, oos_m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
