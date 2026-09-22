"""
optimize_capacity.py — parameter sweep: Universe Size x MAX_CONCURRENT (book
capacity), searching for the highest risk-adjusted configuration of Strategy
B (the Fixed ADX Hybrid). Temporary/exploratory by design, per the request —
reuses backtest.py's run_adx_hybrid_fixed_backtest() (extended with a new
`max_concurrent` parameter specifically for this sweep — default behavior
for every existing caller is unchanged, verified by regression test before
this file was written) and compare_strategies.py's compute_metrics(), rather
than reimplementing either.

CONFIGS TESTED (2022-01-01 -> present, $100,000 starting capital each):
  1. Baseline:          30 symbols (Mixed, the original curated universe),
                        MAX_CONCURRENT=4, full risk_pct.
  2. Optimal Universe:  50 symbols — the top 50 of EXPANDED_UNIVERSE_100 by
                        REAL trailing-60-session average dollar volume (that
                        list is grouped by sector, not liquidity-ordered, so
                        this re-ranks it rather than just slicing [:50]),
                        MAX_CONCURRENT=4, full risk_pct.
  3. Optimal Universe:  75 symbols, same top-N-by-liquidity method,
                        MAX_CONCURRENT=4, full risk_pct.
  4. Capacity Scaling: 100 symbols (EXPANDED_UNIVERSE_100), MAX_CONCURRENT=8,
                        risk_pct x0.5 — 8 slots x 0.5 risk = 4 slots x 1.0
                        risk, the same worst-case aggregate exposure as
                        Config 1's 4 full-risk slots.
  5. Capacity Scaling: 100 symbols, MAX_CONCURRENT=10, risk_pct x0.4 —
                        10 x 0.4 = 4 x 1.0, same logic as Config 4.

Usage:
    python3 optimize_capacity.py
"""

from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import backtest as bt
import data
from compare_strategies import compute_metrics

START = dt.date(2022, 1, 1)
END = dt.date.today()
STARTING_CAPITAL = 100_000.0

BASE_RISK_PCT_MR = 0.00625     # matches hybrid_indicators.py / compare_strategies.py
BASE_RISK_PCT_MOM = 0.0025

LIQUIDITY_WINDOW = 60           # trading days, for ranking EXPANDED_UNIVERSE_100


def _fmt_ratio(x: float) -> str:
    return "inf" if x == float("inf") else f"{x:.2f}"


def rank_by_liquidity(symbols: list[str]) -> list[str]:
    """
    Real trailing LIQUIDITY_WINDOW-session average DOLLAR volume ranking,
    most liquid first. EXPANDED_UNIVERSE_100 is grouped by sector in
    backtest.py, not liquidity-ordered, so Configs 2/3's "top N most liquid"
    requires this re-ranking rather than a plain [:50]/[:75] slice.
    """
    def _avg_dollar_vol(sym: str) -> tuple[str, float]:
        try:
            d = data.daily_bars(sym, period="6mo")
            if d.empty:
                return sym, 0.0
            v = (d["Close"] * d["Volume"]).tail(LIQUIDITY_WINDOW).mean()
            return sym, float(v) if v == v else 0.0
        except Exception:
            return sym, 0.0

    ranked: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        for sym, v in ex.map(_avg_dollar_vol, symbols):
            ranked[sym] = v
    return sorted(symbols, key=lambda s: -ranked[s])


def build_configs() -> list[dict]:
    print(f"Ranking {len(bt.EXPANDED_UNIVERSE_100)} symbols by trailing "
          f"{LIQUIDITY_WINDOW}-session $ volume ...", flush=True)
    ranked_100 = rank_by_liquidity(bt.EXPANDED_UNIVERSE_100)
    top_50, top_75 = ranked_100[:50], ranked_100[:75]

    return [
        dict(name="1: Baseline (30 sym, MC=4)", symbols=bt.UNIVERSES["Mixed"],
            max_concurrent=4, risk_pct_mr=BASE_RISK_PCT_MR,
            risk_pct_mom=BASE_RISK_PCT_MOM),
        dict(name="2: 50-Symbol, top liquidity (MC=4)", symbols=top_50,
            max_concurrent=4, risk_pct_mr=BASE_RISK_PCT_MR,
            risk_pct_mom=BASE_RISK_PCT_MOM),
        dict(name="3: 75-Symbol, top liquidity (MC=4)", symbols=top_75,
            max_concurrent=4, risk_pct_mr=BASE_RISK_PCT_MR,
            risk_pct_mom=BASE_RISK_PCT_MOM),
        dict(name="4: 100-Symbol (MC=8, 0.5x risk)", symbols=bt.EXPANDED_UNIVERSE_100,
            max_concurrent=8, risk_pct_mr=BASE_RISK_PCT_MR * 0.5,
            risk_pct_mom=BASE_RISK_PCT_MOM * 0.5),
        dict(name="5: 100-Symbol (MC=10, 0.4x risk)", symbols=bt.EXPANDED_UNIVERSE_100,
            max_concurrent=10, risk_pct_mr=BASE_RISK_PCT_MR * 0.4,
            risk_pct_mom=BASE_RISK_PCT_MOM * 0.4),
    ]


def run_sweep() -> list[dict]:
    results = []
    for cfg in build_configs():
        print(f"Running {cfg['name']} ({len(cfg['symbols'])} symbols, "
              f"MAX_CONCURRENT={cfg['max_concurrent']}) ...", flush=True)
        res = bt.run_adx_hybrid_fixed_backtest(
            cfg["symbols"], START, END, mode="HYBRID", capital=STARTING_CAPITAL,
            universe_name=cfg["name"], risk_pct_mr=cfg["risk_pct_mr"],
            risk_pct_mom=cfg["risk_pct_mom"], max_concurrent=cfg["max_concurrent"])
        metrics = compute_metrics(res, STARTING_CAPITAL)
        results.append(dict(name=cfg["name"], n_symbols=len(cfg["symbols"]),
                            max_concurrent=cfg["max_concurrent"], metrics=metrics))
    return results


def print_table(results: list[dict]) -> None:
    width = 92
    print("=" * width)
    print("PARAMETER SWEEP — Universe Size x MAX_CONCURRENT "
          "(Strategy B: Fixed ADX Hybrid)")
    print(f"{START} -> {END}, ${STARTING_CAPITAL:,.0f} starting capital each")
    print("=" * width)
    header = (f"{'Config':<36}{'Total Ret':>10}{'Max DD':>9}{'Sharpe':>8}"
             f"{'Calmar':>8}{'Win %':>7}{'Trades':>8}")
    print(header)
    print("-" * width)
    for r in results:
        m = r["metrics"]
        print(f"{r['name']:<36}{m['total_return_pct']:>+9.2f}%{m['max_dd_pct']:>8.2f}%"
              f"{m['sharpe']:>8.3f}{_fmt_ratio(m['calmar']):>8}"
              f"{m['win_rate_pct']:>6.1f}%{m['n_trades']:>8}")
    print("=" * width)


def print_verdict(results: list[dict]) -> None:
    width = 92
    print()
    print("=" * width)
    print("QUANTITATIVE VERDICT — best risk-adjusted configuration")
    print("=" * width)
    print("Selection rule: Sharpe > 1.0 first; among those, lowest Max Drawdown.")
    print()

    qualifying = [r for r in results if r["metrics"]["sharpe"] > 1.0]
    if qualifying:
        best = min(qualifying, key=lambda r: r["metrics"]["max_dd_pct"])
        print(f"Configs clearing Sharpe > 1.0: "
              f"{', '.join(r['name'] for r in qualifying)}")
        print()
        print(f"RECOMMENDATION: {best['name']}")
        m = best["metrics"]
        print(f"  Sharpe {m['sharpe']:.3f} (>1.0)  |  Max Drawdown "
              f"{m['max_dd_pct']:.2f}% (lowest among qualifying)  |  "
              f"Calmar {_fmt_ratio(m['calmar'])}  |  Total Return "
              f"{m['total_return_pct']:+.2f}%  |  {m['n_trades']} trades")
    else:
        best = max(results, key=lambda r: r["metrics"]["sharpe"])
        m = best["metrics"]
        print("No configuration cleared Sharpe > 1.0.")
        print()
        print(f"BEST AVAILABLE (by Sharpe): {best['name']}")
        print(f"  Sharpe {m['sharpe']:.3f}  |  Max Drawdown {m['max_dd_pct']:.2f}%  |  "
              f"Calmar {_fmt_ratio(m['calmar'])}  |  Total Return "
              f"{m['total_return_pct']:+.2f}%  |  {m['n_trades']} trades")
    print("=" * width)


def main() -> int:
    results = run_sweep()
    print_table(results)
    print_verdict(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
