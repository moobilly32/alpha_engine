"""
Test the two high-win-rate modifications against their own baselines.

  1. PARTIAL PROFIT SCALING — sell SCALE_FRACTION at SCALE_R, move the stop on
     the remainder to breakeven, let the runner go to TARGET_R.
  2. DEEP OVERSOLD DIP ENTRY — 1-hour RSI below threshold OR a touch of the
     lower hourly Bollinger band, while price holds above the daily 200 SMA.

WHY EVERY ROW IS PAIRED WITH A NO-SCALING BASELINE. Scaling raises win rate
almost by construction: a trade that reaches the scale level banks a profit, so
even when the runner is stopped at breakeven the trade closes positive and
counts as a win. It also caps every winner at a blend of SCALE_R and TARGET_R
instead of TARGET_R. Win rate and net P&L therefore move in OPPOSITE directions,
and "win rate went up" is not evidence the change helped. The columns that
answer that are NET P&L, PROFIT FACTOR and EXPECTANCY IN R.

Expectancy in R is the fairest single number: it is P&L per unit of risk
actually taken, so it does not flatter a strategy for taking smaller bets.

Usage:
    python3 scale_test.py
    python3 scale_test.py --scale-rs 1.0,1.25,1.5 --target-r 2.5
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
    ap = argparse.ArgumentParser(description="Scaling + oversold entry test")
    ap.add_argument("--symbols", type=str, default=None)
    ap.add_argument("--days", type=int, default=BACKTEST_DAYS)
    ap.add_argument("--capital", type=float, default=BACKTEST_CAPITAL)
    ap.add_argument("--target-r", type=float, default=2.5)
    ap.add_argument("--scale-rs", type=str, default="1.0,1.25,1.5")
    ap.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    ap.add_argument("--no-earnings", action="store_true",
                    help="disable earnings blocks (to measure their effect)")
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

    scale_rs = [float(x) for x in args.scale_rs.split(",")]

    print(f"Preparing {len(symbols)} symbols ...", flush=True)
    prepped, dates = sw.prepare(symbols, args.days, "vwap", verbose=True)
    if not prepped:
        print("no usable data")
        return 1

    earn_block = 0 if args.no_earnings else 5
    earn_exit = not args.no_earnings

    stops = [("structure", "structure", 1.5), ("2.0x ATR", "atr", 2.0)]
    entries = ["pullback", "oversold"]

    rows = []
    for entry in entries:
        for stop_label, stop_method, amult in stops:
            variants = [("no scaling", False, 0.0)]
            variants += [(f"scale {r:.2f}R", True, r) for r in scale_rs]
            for vlabel, sc_on, sc_r in variants:
                res = sw.run(symbols, days=args.days, capital=args.capital,
                             scores=scores, entry_mode=entry,
                             support_kind="vwap", stop_method=stop_method,
                             atr_mult=amult, target_r=args.target_r,
                             max_concurrent=args.max_concurrent, eod_flat=False,
                             scale_enabled=sc_on, scale_r=sc_r,
                             earnings_block_days=earn_block,
                             earnings_exit=earn_exit,
                             prepped=prepped, dates=dates, verbose=False)
                s = sw.summarise(res, args.capital)
                rows.append({"entry": entry, "stop": stop_label,
                             "variant": vlabel, "scale_r": sc_r, **s})

    df = pd.DataFrame(rows)

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
    A("=" * 104)
    A("  HIGH-WIN-RATE MODIFICATIONS — 59-DAY SWING BACKTEST")
    A("=" * 104)
    A(f"  Window {dates[0]} → {dates[-1]} ({len(dates)} sessions) · "
      f"{len(prepped)} symbols · ${args.capital:,.0f} · max {args.max_concurrent} concurrent")
    A(f"  Runner target {args.target_r:.1f}R · scale sells 50% then stop → breakeven")
    A(f"  Earnings: {'BLOCKED (no entry within 5d, exit before the print)' if earn_exit else 'IGNORED'}"
      f" · overnight gaps modelled")
    A("")
    A(f"  {'ENTRY':<10}{'STOP':<11}{'VARIANT':<13}{'TRADES':>7}{'WIN%':>7}"
      f"{'95% CI':>10}{'NET $':>11}{'NET %':>8}{'MAXDD $':>10}{'MAXDD%':>8}"
      f"{'SHARPE':>8}{'PF':>7}{'E[R]':>7}")
    A("  " + "-" * 100)

    for entry in entries:
        for stop_label, _m, _a in stops:
            g = df[(df["entry"] == entry) & (df["stop"] == stop_label)]
            for _, r in g.iterrows():
                A(f"  {entry:<10}{stop_label:<11}{r['variant']:<13}"
                  f"{int(r['trades']):>7}{r['win_rate']*100:>6.1f}%"
                  f"{f'{r[chr(99)+chr(105)+chr(95)+chr(108)+chr(111)]*100:.0f}-{r[chr(99)+chr(105)+chr(95)+chr(104)+chr(105)]*100:.0f}%':>10}"
                  f"{r['net']:>+11,.2f}{r['net_pct']*100:>+7.2f}%"
                  f"{r['dd_d']:>10,.2f}{r['dd_p']*100:>7.2f}%"
                  f"{r['sharpe']:>8.2f}{r['pf']:>7.2f}{r['expectancy_r']:>7.2f}")
            A("  " + "-" * 100)

    # Did scaling actually help?
    A("")
    A("  DID SCALING HELP?  (each scaled variant vs. its own no-scaling baseline)")
    A(f"  {'ENTRY':<10}{'STOP':<11}{'VARIANT':<13}{'Δ WIN%':>9}{'Δ NET $':>11}"
      f"{'Δ PF':>8}{'Δ E[R]':>9}{'VERDICT':>22}")
    A("  " + "-" * 92)
    for entry in entries:
        for stop_label, _m, _a in stops:
            g = df[(df["entry"] == entry) & (df["stop"] == stop_label)]
            base = g[g["variant"] == "no scaling"]
            if base.empty:
                continue
            b = base.iloc[0]
            for _, r in g[g["variant"] != "no scaling"].iterrows():
                dwin = (r["win_rate"] - b["win_rate"]) * 100
                dnet = r["net"] - b["net"]
                dpf = r["pf"] - b["pf"]
                der = r["expectancy_r"] - b["expectancy_r"]
                if dnet > 0 and dwin > 0:
                    v = "better on both"
                elif dnet > 0:
                    v = "better P&L"
                elif dwin > 0:
                    v = "win% up, P&L DOWN"
                else:
                    v = "worse on both"
                A(f"  {entry:<10}{stop_label:<11}{r['variant']:<13}"
                  f"{dwin:>+8.1f}%{dnet:>+11,.2f}{dpf:>+8.2f}{der:>+9.2f}{v:>22}")
            A("  " + "-" * 92)

    hit50 = df[df["win_rate"] >= 0.50]
    prof = df[df["net"] > 0]
    A("")
    A("  " + "!" * 100)
    A(f"  Configurations reaching the 50%+ win-rate goal: {len(hit50)}/{len(df)}")
    if len(hit50):
        best_wr = hit50.sort_values("net", ascending=False).iloc[0]
        A(f"    best P&L among them: {best_wr['entry']}/{best_wr['stop']}/"
          f"{best_wr['variant']} — win {best_wr['win_rate']*100:.1f}%, "
          f"net {best_wr['net']:+,.2f} ({best_wr['net_pct']*100:+.2f}%), "
          f"PF {best_wr['pf']:.2f}")
    A(f"  Profitable configurations: {len(prof)}/{len(df)}")
    if len(prof):
        best = prof.sort_values("net", ascending=False).iloc[0]
        A(f"    best P&L overall: {best['entry']}/{best['stop']}/{best['variant']} — "
          f"net {best['net']:+,.2f} ({best['net_pct']*100:+.2f}%), "
          f"win {best['win_rate']*100:.1f}%, PF {best['pf']:.2f}, "
          f"E[R] {best['expectancy_r']:.2f}, {int(best['trades'])} trades")
    for b, v in bench.items():
        A(f"  {b} buy & hold, same window: {v*100:+.2f}%")
    A("")
    A("  Win rate here is NOT comparable across scaled and unscaled rows: scaling")
    A("  converts would-be breakeven-or-worse trades into small winners, which")
    A("  lifts win rate while capping the winners that pay for the losers. Judge")
    A("  by NET, PF and E[R]. All of this is in-sample on one 59-day window.")
    A("  " + "!" * 100)

    out = "\n".join(L)
    print("\n" + out)
    (DATA / "scale_test.txt").write_text(out)
    df.to_csv(DATA / "scale_test.csv", index=False)
    print(f"\nwrote {DATA}/scale_test.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
