"""
Risk sizing x runner exit, on a $1,500 account.

THE ARITHMETIC THAT GOVERNS THIS WHOLE TEST. Position size is set by

    position $ = equity x risk% / stop%

so with the winning config's 2.5% stop, 2.5% risk per trade means 100% of
equity in ONE name. Requesting both "2.5% risk" and "3-4 concurrent positions"
on a cash account is asking for 250-400% of capital. The position cap
(1/max_concurrent) silently wins that argument, and REALIZED risk per trade
lands far below what was requested:

    concurrent  position cap   realized risk @2.5% stop
        3           33%              0.83%
        4           25%              0.62%

So this table reports REQUESTED risk and REALIZED risk side by side. Where they
diverge, the concurrency requirement — not the risk setting — is what actually
sized the book.

ON THE ANNUALISED COLUMN. 59 sessions is 23% of a trading year, so annualising
raises the period return to the power 252/59 = 4.27 and multiplies the noise
with it. A bootstrap 90% interval is printed beside every point estimate for
exactly that reason; if the interval spans zero, the point estimate is
decoration.

Usage:
    python3 risk_test.py
    python3 risk_test.py --capital 1500 --risks 1.0,2.0,2.5
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import screener
import swing_backtest as sw
from config import DATA, BACKTEST_DAYS

TRADING_DAYS = 252


def annualise(period_return: float, sessions: int) -> float:
    """Compound a period return up to a year. Fragile by nature — see module doc."""
    if sessions <= 0:
        return 0.0
    base = 1.0 + period_return
    if base <= 0:
        return -1.0
    return base ** (TRADING_DAYS / sessions) - 1.0


def bootstrap_annual(trades, capital: float, sessions: int,
                     iters: int = 4000, seed: int = 5):
    """90% interval for the annualised return, resampling trades with replacement."""
    if not trades:
        return 0.0, 0.0
    pnl = np.array([t.pnl for t in trades], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pnl), size=(iters, len(pnl)))
    tot = pnl[idx].sum(axis=1)
    ann = np.array([annualise(x / capital, sessions) for x in tot])
    return float(np.percentile(ann, 5)), float(np.percentile(ann, 95))


def realized_risk(trades, capital: float) -> float:
    """Mean dollar risk actually taken per trade, as a fraction of capital."""
    if not trades:
        return 0.0
    return float(np.mean([t.r_unit * t.shares for t in trades])) / capital


def main() -> int:
    ap = argparse.ArgumentParser(description="Risk sizing and runner-exit test")
    ap.add_argument("--symbols", type=str, default=None)
    ap.add_argument("--days", type=int, default=BACKTEST_DAYS)
    ap.add_argument("--capital", type=float, default=1500.0)
    ap.add_argument("--risks", type=str, default="1.0,2.0,2.5")
    ap.add_argument("--concurrent", type=str, default="3,4")
    ap.add_argument("--iters", type=int, default=4000)
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

    risks = [float(x) / 100.0 for x in args.risks.split(",")]
    concurrents = [int(x) for x in args.concurrent.split(",")]

    print(f"Preparing {len(symbols)} symbols ...", flush=True)
    prepped, dates = sw.prepare(symbols, args.days, "vwap", verbose=True)
    if not prepped:
        print("no usable data")
        return 1
    n_sessions = len(dates)

    exits = [("fixed 2.5R", "fixed"), ("1h 20EMA trail", "ema20h"),
             ("1.5x ATR trail", "atr")]

    rows = []
    for mc in concurrents:
        for rp in risks:
            for elabel, ekind in exits:
                res = sw.run(symbols, days=args.days, capital=args.capital,
                             scores=scores, entry_mode="oversold",
                             support_kind="vwap", stop_method="pct",
                             stop_pct=0.025, target_r=2.5,
                             max_concurrent=mc, eod_flat=False,
                             scale_enabled=True, scale_r=1.0,
                             runner_exit=ekind, trail_atr_mult=1.5,
                             risk_pct=rp, earnings_block_days=5,
                             earnings_exit=True, prepped=prepped, dates=dates,
                             verbose=False)
                s = sw.summarise(res, args.capital)
                lo, hi = bootstrap_annual(res.trades, args.capital,
                                          n_sessions, args.iters)
                rows.append({
                    "concurrent": mc, "risk_req": rp,
                    "risk_real": realized_risk(res.trades, args.capital),
                    "exit": elabel, **s,
                    "annual": annualise(s["net_pct"], n_sessions),
                    "ann_lo": lo, "ann_hi": hi,
                })

    df = pd.DataFrame(rows)

    L, A = [], None
    A = L.append
    A("=" * 118)
    A("  RISK SIZING x RUNNER EXIT — 59-DAY SWING BACKTEST, $%s ACCOUNT"
      % f"{args.capital:,.0f}")
    A("=" * 118)
    A(f"  Window {dates[0]} → {dates[-1]} ({n_sessions} sessions) · "
      f"{len(prepped)} symbols · oversold entry · 2.5% stop · sell 50% at 1.0R → breakeven")
    A("")
    A(f"  {'CONC':<5}{'RISK REQ':>9}{'RISK REAL':>10}{'EXIT':<17}{'TRADES':>7}"
      f"{'WIN%':>7}{'NET $':>10}{'NET %':>8}{'MAXDD%':>8}{'SHARPE':>8}"
      f"{'PF':>6}{'ANNUAL':>9}{'90% INTERVAL':>22}")
    A("  " + "-" * 114)
    for mc in concurrents:
        for rp in risks:
            g = df[(df["concurrent"] == mc) & (df["risk_req"] == rp)]
            for _, r in g.iterrows():
                capped = "*" if r["risk_real"] < r["risk_req"] * 0.9 else " "
                A(f"  {mc:<5}{r['risk_req']*100:>8.1f}%"
                  f"{r['risk_real']*100:>9.2f}%{capped}{r['exit']:<17}"
                  f"{int(r['trades']):>7}{r['win_rate']*100:>6.1f}%"
                  f"{r['net']:>+10.2f}{r['net_pct']*100:>+7.2f}%"
                  f"{r['dd_p']*100:>7.2f}%{r['sharpe']:>8.2f}{r['pf']:>6.2f}"
                  f"{r['annual']*100:>+8.1f}%"
                  f"{f'[{r[chr(97)+chr(110)+chr(110)+chr(95)+chr(108)+chr(111)]*100:+.0f}%, {r[chr(97)+chr(110)+chr(110)+chr(95)+chr(104)+chr(105)]*100:+.0f}%]':>22}")
            A("  " + "-" * 114)

    A("")
    A("  * = REALIZED risk fell more than 10% short of the request, because the")
    A("      position cap (1/concurrent) bound before the risk budget did.")

    # Does raising risk actually raise return?
    A("")
    A("  DOES RAISING RISK RAISE RETURN?  (fixed 2.5R exit)")
    A(f"  {'CONC':<6}{'RISK REQ':>10}{'RISK REAL':>11}{'NET %':>9}{'MAXDD%':>9}"
      f"{'RETURN/DD':>11}")
    A("  " + "-" * 58)
    for mc in concurrents:
        g = df[(df["concurrent"] == mc) & (df["exit"] == "fixed 2.5R")]
        for _, r in g.iterrows():
            rd = r["net_pct"] / r["dd_p"] if r["dd_p"] > 0 else 0.0
            A(f"  {mc:<6}{r['risk_req']*100:>9.1f}%{r['risk_real']*100:>10.2f}%"
              f"{r['net_pct']*100:>+8.2f}%{r['dd_p']*100:>8.2f}%{rd:>11.2f}")
        A("  " + "-" * 58)

    A("")
    A("  RUNNER EXIT — does uncapped upside beat a fixed target?")
    A(f"  {'EXIT':<17}{'MEDIAN NET%':>13}{'MEDIAN WIN%':>13}{'MEDIAN PF':>11}"
      f"{'PROFITABLE':>12}")
    A("  " + "-" * 68)
    for elabel, _k in exits:
        g = df[df["exit"] == elabel]
        A(f"  {elabel:<17}{g['net_pct'].median()*100:>+12.2f}%"
          f"{g['win_rate'].median()*100:>12.1f}%{g['pf'].median():>11.2f}"
          f"{f'{int((g[chr(110)+chr(101)+chr(116)]>0).sum())}/{len(g)}':>12}")

    best = df.sort_values("net", ascending=False).iloc[0]
    A("")
    A("  " + "!" * 114)
    A(f"  Best: {int(best['concurrent'])} concurrent · {best['risk_req']*100:.1f}% requested "
      f"({best['risk_real']*100:.2f}% realized) · {best['exit']}")
    A(f"        {int(best['trades'])} trades · win {best['win_rate']*100:.1f}% · "
      f"net {best['net']:+,.2f} ({best['net_pct']*100:+.2f}%) · maxDD "
      f"{best['dd_p']*100:.2f}% · annualised {best['annual']*100:+.1f}% "
      f"[{best['ann_lo']*100:+.0f}%, {best['ann_hi']*100:+.0f}%]")
    A("")
    A("  The 90% intervals are the honest content of the annualised column.")
    A("  Compounding a 59-day result to a year raises it to the power 4.27 and")
    A("  the uncertainty with it. Every row here is also in-sample: the entry,")
    A("  the stop and the scale level were all chosen on this same window.")
    A("  " + "!" * 114)

    out = "\n".join(L)
    print("\n" + out)
    (DATA / "risk_test.txt").write_text(out)
    df.to_csv(DATA / "risk_test.csv", index=False)
    print(f"\nwrote {DATA}/risk_test.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
