"""
Swing parameter sweep — profit target against stop definition.

The brief asked for a target sweep at 2.0R / 2.5R / 3.0R / 3.5R. A target is
meaningless without the stop it is a multiple of, so the grid varies both: 3.0R
off a 1.5% stop is a 4.5% move, off a 3% stop it is 9%, and those are different
strategies wearing the same label.

Data is prepared ONCE and reused across every configuration — the download and
the VWAP/EMA/daily-context precomputation dominate the runtime, and they do not
depend on the stop or the target.

READ THE RESULT AS IN-SAMPLE. Sweeping ~64 configurations over one 59-day window
and reporting the best five is a ranking of how well each fits THIS window. The
honest questions are whether the profitable cells form a contiguous region
(consistent with a real effect) or scatter (noise), and whether the best cell
beats holding SPY. Both are answered below the table.

Usage:
    python3 swing_sweep.py
    python3 swing_sweep.py --supports vwap --targets 2.0,2.5,3.0,3.5
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import data
import screener
import swing_backtest as sw
from config import BACKTEST_DAYS, BACKTEST_CAPITAL, MAX_CONCURRENT, DATA


def main() -> int:
    ap = argparse.ArgumentParser(description="Swing stop/target sweep")
    ap.add_argument("--symbols", type=str, default=None)
    ap.add_argument("--days", type=int, default=BACKTEST_DAYS)
    ap.add_argument("--capital", type=float, default=BACKTEST_CAPITAL)
    ap.add_argument("--targets", type=str, default="2.0,2.5,3.0,3.5")
    ap.add_argument("--supports", type=str, default="vwap,ema20")
    ap.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    ap.add_argument("--top", type=int, default=5)
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

    targets = [float(x) for x in args.targets.split(",")]
    supports = [s.strip() for s in args.supports.split(",")]

    # (label, method, pct, atr_mult)
    stops = [
        ("structure", "structure", 0.02, 1.5),
        ("1.5% fixed", "pct", 0.015, 0.0),
        ("2.0% fixed", "pct", 0.020, 0.0),
        ("2.5% fixed", "pct", 0.025, 0.0),
        ("3.0% fixed", "pct", 0.030, 0.0),
        ("1.0x ATR", "atr", 0.02, 1.0),
        ("1.5x ATR", "atr", 0.02, 1.5),
        ("2.0x ATR", "atr", 0.02, 2.0),
    ]

    print(f"Preparing data once for {len(symbols)} symbols ...", flush=True)
    prepped, dates = sw.prepare(symbols, args.days, "vwap", verbose=True)
    if not prepped:
        print("no usable data")
        return 1

    total = len(supports) * len(stops) * len(targets)
    print(f"\nSweeping {len(supports)} supports x {len(stops)} stops x "
          f"{len(targets)} targets = {total} configurations ...\n", flush=True)

    rows = []
    done = 0
    for sup in supports:
        for label, method, spct, amult in stops:
            for tr in targets:
                res = sw.run(symbols, days=args.days, capital=args.capital,
                             scores=scores, entry_mode="pullback",
                             support_kind=sup, stop_method=method,
                             stop_pct=spct, atr_mult=amult, target_r=tr,
                             max_concurrent=args.max_concurrent,
                             eod_flat=False, prepped=prepped, dates=dates,
                             verbose=False)
                s = sw.summarise(res, args.capital)
                rows.append({"support": sup, "stop": label, "target_r": tr, **s})
                done += 1
                if done % 8 == 0:
                    print(f"  {done}/{total}", flush=True)

    df = pd.DataFrame(rows).sort_values("net", ascending=False).reset_index(drop=True)

    # benchmark
    bench = {}
    for b in ("SPY", "QQQ"):
        try:
            d = data.daily_bars(b, period="6mo")
            w = d[(d.index.date >= dates[0]) & (d.index.date <= dates[-1])]
            bench[b] = float(w["Close"].iloc[-1] / w["Close"].iloc[0] - 1.0)
        except Exception:
            pass

    L, A = [], None
    A = L.append
    A("=" * 96)
    A("  SWING PARAMETER SWEEP — top configurations by net profit")
    A("=" * 96)
    A(f"  Window {dates[0]} → {dates[-1]} ({len(dates)} sessions) · "
      f"{len(prepped)} symbols · ${args.capital:,.0f} · max {args.max_concurrent} concurrent")
    A(f"  Entry: pullback to support, held until stop or target — no EOD flatten")
    A("")
    A(f"  {'#':>2}  {'SUPPORT':<8}{'STOP':<12}{'TGT R':>7}{'TRADES':>8}{'WIN%':>7}"
      f"{'95% CI':>11}{'NET $':>11}{'NET %':>8}{'MAXDD $':>10}{'MAXDD%':>8}"
      f"{'SHARPE':>8}{'PF':>7}{'HOLD':>6}")
    A("  " + "-" * 92)
    for i, r in df.head(args.top).iterrows():
        A(f"  {i+1:>2}  {r['support']:<8}{r['stop']:<12}{r['target_r']:>7.1f}"
          f"{int(r['trades']):>8}{r['win_rate']*100:>6.1f}%"
          f"{f'{r[chr(99)+chr(105)+chr(95)+chr(108)+chr(111)]*100:.0f}-{r[chr(99)+chr(105)+chr(95)+chr(104)+chr(105)]*100:.0f}%':>11}"
          f"{r['net']:>+11,.2f}{r['net_pct']*100:>+7.2f}%"
          f"{r['dd_d']:>10,.2f}{r['dd_p']*100:>7.2f}%{r['sharpe']:>8.2f}"
          f"{r['pf']:>7.2f}{r['avg_hold']:>6.1f}")
    A("  " + "-" * 92)

    A("")
    A("  BY TARGET R  (median across every stop and support)")
    A(f"  {'TGT R':>7}{'MEDIAN NET':>13}{'MEDIAN WIN%':>13}{'MEDIAN PF':>11}"
      f"{'MEDIAN SHARPE':>15}{'PROFITABLE':>12}")
    A("  " + "-" * 72)
    for tr in sorted(targets):
        g = df[df["target_r"] == tr]
        A(f"  {tr:>7.1f}{g['net'].median():>+13,.2f}{g['win_rate'].median()*100:>12.1f}%"
          f"{g['pf'].median():>11.2f}{g['sharpe'].median():>15.2f}"
          f"{f'{int((g[chr(110)+chr(101)+chr(116)]>0).sum())}/{len(g)}':>12}")

    A("")
    A("  BY STOP  (median across every target and support)")
    A(f"  {'STOP':<14}{'MEDIAN NET':>13}{'MEDIAN WIN%':>13}{'MEDIAN PF':>11}{'PROFITABLE':>12}")
    A("  " + "-" * 66)
    for label, *_ in stops:
        g = df[df["stop"] == label]
        A(f"  {label:<14}{g['net'].median():>+13,.2f}{g['win_rate'].median()*100:>12.1f}%"
          f"{g['pf'].median():>11.2f}"
          f"{f'{int((g[chr(110)+chr(101)+chr(116)]>0).sum())}/{len(g)}':>12}")

    npos = int((df["net"] > 0).sum())
    A("")
    A("  " + "!" * 92)
    A(f"  {npos} of {len(df)} configurations profitable "
      f"({npos/len(df)*100:.0f}%).")
    for b, v in bench.items():
        A(f"  {b} buy & hold over the same window: {v*100:+.2f}% "
          f"(${args.capital*v:+,.2f} on this capital)")
    best = df.iloc[0]
    if bench.get("SPY") is not None:
        verdict = ("beats" if best["net_pct"] > bench["SPY"] else "does NOT beat")
        A(f"  Best configuration {verdict} SPY buy & hold.")
    A("")
    A("  THIS TABLE IS IN-SAMPLE. The window chose these parameters. A single")
    A("  profitable cell in a noisy grid is noise; a contiguous profitable")
    A("  region is weaker evidence than it looks but at least consistent with a")
    A("  real effect. Nothing here is out-of-sample, and the watchlist was still")
    A("  selected with today's fundamentals, so survivorship applies on top.")
    A("  " + "!" * 92)

    out = "\n".join(L)
    print("\n" + out)
    (DATA / "swing_sweep.txt").write_text(out)
    df.to_csv(DATA / "swing_sweep.csv", index=False)
    print(f"\nwrote {DATA}/swing_sweep.txt and swing_sweep.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
