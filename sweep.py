"""
Stop / target parameter sweep.

Purpose: settle whether the base configuration lost money because the EXITS
were wrong, or because the signal has nothing to give.

The edge test reports a mean maximum favourable excursion of roughly 0.9% after
a signal. A 1.0% stop with a 2.0R (i.e. 2.0%) target asks the move to travel
more than twice as far as the average signal ever travels — so the target is
almost never reached and the strategy is left exiting flat at 15:55. That is a
concrete, checkable hypothesis, and this grid checks it.

READ THE OUTPUT WITH SUSPICION. Sweeping 20 combinations over one 59-day window
and keeping the best is how overfitting happens. A grid where only one cell is
positive is noise. A grid where a whole neighbourhood is positive is at least
consistent with a real effect — and still needs out-of-sample confirmation
before it means anything.

Usage:
    python3 sweep.py
    python3 sweep.py --stops 0.3,0.5,0.75,1.0 --targets 0.75,1.0,1.5,2.0
"""

from __future__ import annotations

import argparse
import sys

import screener
from backtest import run_backtest, max_drawdown, sharpe, wilson_ci
from config import BACKTEST_DAYS, BACKTEST_CAPITAL, DATA


def main() -> int:
    ap = argparse.ArgumentParser(description="Stop/target sweep")
    ap.add_argument("--symbols", type=str, default=None)
    ap.add_argument("--days", type=int, default=BACKTEST_DAYS)
    ap.add_argument("--capital", type=float, default=BACKTEST_CAPITAL)
    ap.add_argument("--stops", type=str, default="0.3,0.5,0.75,1.0,1.5")
    ap.add_argument("--targets", type=str, default="0.5,1.0,1.5,2.0")
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

    stops = [float(x) / 100.0 for x in args.stops.split(",")]
    targets = [float(x) for x in args.targets.split(",")]

    rows = []
    print(f"Sweeping {len(stops)}x{len(targets)} = {len(stops)*len(targets)} "
          f"combinations ...\n", flush=True)

    first = True
    for sp in stops:
        for tr in targets:
            res = run_backtest(symbols, days=args.days, capital=args.capital,
                               scores=scores, stop_pct=sp, target_r=tr,
                               verbose=first)
            first = False
            t = res.trades
            n = len(t)
            wins = sum(1 for x in t if x.is_win)
            net = sum(x.pnl for x in t)
            gw = sum(x.pnl for x in t if x.pnl > 0)
            gl = -sum(x.pnl for x in t if x.pnl <= 0)
            pf = gw / gl if gl > 0 else float("inf") if gw > 0 else 0.0
            dd_d, dd_p = max_drawdown(res.equity) if not res.equity.empty else (0, 0)
            lo, hi = wilson_ci(wins, n)
            rows.append({
                "stop": sp, "target_r": tr, "target_pct": sp * tr * 100,
                "n": n, "wins": wins, "wr": wins / n if n else 0.0,
                "lo": lo, "hi": hi, "net": net, "pf": pf,
                "sharpe": sharpe(res.equity), "dd": dd_p,
            })

    rows.sort(key=lambda r: -r["net"])

    L = []
    A = L.append
    A("=" * 86)
    A("  STOP / TARGET SWEEP")
    A("=" * 86)
    A(f"  {'STOP':>6}{'TGT R':>7}{'TGT %':>7}{'N':>6}{'WIN%':>7}"
      f"{'95% CI':>12}{'NET $':>11}{'PF':>7}{'SHARPE':>8}{'MAXDD':>8}")
    A("  " + "-" * 82)
    for r in rows:
        A(f"  {r['stop']*100:>5.2f}%{r['target_r']:>7.1f}{r['target_pct']:>6.2f}%"
          f"{r['n']:>6}{r['wr']*100:>6.1f}%"
          f"{f'{r['lo']*100:.0f}-{r['hi']*100:.0f}%':>12}"
          f"{r['net']:>+11,.2f}{r['pf']:>7.2f}{r['sharpe']:>8.2f}"
          f"{r['dd']*100:>7.2f}%")
    A("  " + "-" * 82)

    pos = [r for r in rows if r["net"] > 0]
    A("")
    A(f"  {len(pos)} of {len(rows)} combinations profitable.")
    if pos:
        best = pos[0]
        A(f"  Best: {best['stop']*100:.2f}% stop / {best['target_r']:.1f}R "
          f"→ {best['net']:+,.2f} on {best['n']} trades, PF {best['pf']:.2f}")
        A("")
        if len(pos) <= 2:
            A("  CAUTION: only a couple of cells are positive. That is the shape of")
            A("  noise, not of an edge. Do not adopt this configuration.")
        else:
            A("  A contiguous profitable region is weak evidence of a real effect —")
            A("  but this window chose the parameters, so the numbers above are")
            A("  in-sample. Confirm out-of-sample before trusting any of it.")
    else:
        A("  No combination is profitable. The exits are not the problem.")

    out = "\n".join(L)
    print("\n" + out)
    (DATA / "sweep.txt").write_text(out)
    print(f"\nwrote {DATA}/sweep.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
