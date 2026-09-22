"""
Does the Trend Join Long breakout predict anything?

The backtest answers "did this configuration make money", which confounds the
signal with the exits, the sizing and the concurrency cap. A losing backtest
could mean the signal is worthless OR that a 1% stop is simply too tight for
these names. Those two diagnoses lead to opposite decisions — abandon it, or
retune it — so they have to be separated.

METHOD

  * Signal sample:   forward returns from every bar where the breakout fires.
  * Baseline sample: forward returns from EVERY OTHER cadence bar in the same
                     symbol-sessions and the same 10:00-14:00 window.

The baseline is the control that matters. Comparing breakout returns against
zero would only prove that these stocks drifted up over the window; comparing
against arbitrary bars in the same names, days and hours asks the real
question — given that you were going to be in this window anyway, does waiting
for a breakout beat picking a moment at random?

Significance is bootstrapped (no normality assumption on 5-minute returns,
which are visibly fat-tailed) and reported against the round-trip friction
hurdle, because an edge smaller than the cost of trading it is not an edge.

Usage:
    python3 edge_test.py
    python3 edge_test.py --symbols AMD,NVDA,MU --iters 20000
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import screener
import strategy
from backtest import prepare
from calendar_util import is_cadence_bar
from config import (
    BACKTEST_DAYS, SCAN_INTERVAL_MIN, WINDOW_START, WINDOW_END, SLIPPAGE_BPS,
    DATA,
)

HORIZONS = [3, 6, 12]          # 5-minute bars: 15 / 30 / 60 minutes
FRICTION = 2 * SLIPPAGE_BPS / 10_000.0   # round-trip slippage


def collect(symbols: list[str], days: int, verbose: bool = True):
    ctx, dates = prepare(symbols, days, verbose)
    win_start = WINDOW_START[0] * 60 + WINDOW_START[1]
    win_end = WINDOW_END[0] * 60 + WINDOW_END[1]

    sig_rows: list[dict] = []
    base_rows: list[dict] = []

    for (sym, sdate), c in ctx.items():
        rth = c["rth"]
        closes = rth["Close"].to_numpy(dtype=float)
        highs = rth["High"].to_numpy(dtype=float)
        idx = rth.index
        n = len(closes)
        eod = closes[-1]

        hod = None
        fired = False

        for i in range(n):
            ts = idx[i]
            tmin = ts.hour * 60 + ts.minute
            in_win = win_start <= tmin <= win_end and is_cadence_bar(ts, SCAN_INTERVAL_MIN)

            if in_win and hod is not None:
                cl = closes[i]
                lv = strategy.Levels(symbol=sym, prev_high=c["prev_high"],
                                     prev_close=c["prev_close"], sma200=c["sma200"],
                                     pmh=c["pmh"], hod_prev=hod)
                hit = strategy.evaluate_signal(lv, cl).fired

                row = {"symbol": sym, "date": sdate, "ts": ts, "px": cl,
                       "trend_ok": lv.trend_ok}
                for k in HORIZONS:
                    j = min(i + k, n - 1)
                    row[f"f{k}"] = closes[j] / cl - 1.0
                row["eod"] = eod / cl - 1.0
                row["mfe"] = (highs[i:].max() / cl - 1.0) if i < n else 0.0

                # One signal per symbol per day, matching the live rule.
                if hit and not fired:
                    fired = True
                    sig_rows.append(row)
                elif not hit:
                    base_rows.append(row)

            hod = highs[i] if hod is None else max(hod, highs[i])

    return pd.DataFrame(sig_rows), pd.DataFrame(base_rows)


def bootstrap_p(sig: np.ndarray, base: np.ndarray, iters: int = 20000,
                seed: int = 7) -> float:
    """
    Two-sided bootstrap p-value for (mean(sig) - mean(base)) under the null
    that both come from the same pool.
    """
    if len(sig) == 0 or len(base) == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    obs = sig.mean() - base.mean()
    pool = np.concatenate([sig, base])
    ns = len(sig)
    hits = 0
    for _ in range(iters):
        rng.shuffle(pool)
        if abs(pool[:ns].mean() - pool[ns:].mean()) >= abs(obs):
            hits += 1
    return (hits + 1) / (iters + 1)


def report(sig: pd.DataFrame, base: pd.DataFrame, iters: int) -> str:
    L = []
    A = L.append
    A("=" * 78)
    A("  SIGNAL EDGE TEST — breakout bars vs. all other cadence bars")
    A("=" * 78)
    A(f"  Signal observations    {len(sig):,}")
    A(f"  Baseline observations  {len(base):,}")
    A(f"  Friction hurdle        {FRICTION*100:.3f}%  (round-trip slippage)")
    A("")
    A(f"  {'HORIZON':<12}{'SIGNAL':>11}{'BASELINE':>11}{'EDGE':>11}"
      f"{'p-value':>10}   {'VERDICT':<24}")
    A("  " + "-" * 74)

    labels = {3: "+15 min", 6: "+30 min", 12: "+60 min", "eod": "to close"}
    for key in HORIZONS + ["eod"]:
        col = f"f{key}" if key != "eod" else "eod"
        if col not in sig.columns or sig.empty:
            continue
        s = sig[col].dropna().to_numpy()
        b = base[col].dropna().to_numpy()
        edge = s.mean() - b.mean()
        p = bootstrap_p(s, b, iters)

        if p >= 0.05:
            verdict = "not significant"
        elif edge <= 0:
            verdict = "SIGNIFICANT, WRONG WAY"
        elif edge < FRICTION:
            verdict = "real but < costs"
        else:
            verdict = "tradeable"

        A(f"  {labels[key]:<12}{s.mean()*100:>10.4f}%{b.mean()*100:>10.4f}%"
          f"{edge*100:>10.4f}%{p:>10.4f}   {verdict:<24}")

    A("  " + "-" * 74)

    if not sig.empty:
        A("")
        A("  SIGNAL DISTRIBUTION")
        A(f"    max favourable excursion (mean)  {sig['mfe'].mean()*100:>7.3f}%")
        A(f"    median forward 30-min return     {sig['f6'].median()*100:>7.3f}%")
        A(f"    share of signals green at +30m   "
          f"{(sig['f6'] > 0).mean()*100:>7.1f}%")
        A(f"    share of signals green at close  {(sig['eod'] > 0).mean()*100:>7.1f}%")
        A(f"    signals per session (mean)       "
          f"{len(sig)/max(sig['date'].nunique(), 1):>7.2f}")

    A("")
    A("  HOW TO READ THIS")
    A("  'not significant'  — the breakout bar is indistinguishable from any")
    A("                       other bar in the same window. No exit tuning")
    A("                       rescues a signal with no forward information.")
    A("  'real but < costs' — there IS an effect, but it is smaller than the")
    A("                       friction of capturing it.")
    A("  'tradeable'        — edge survives costs; exits are worth optimising.")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="Signal edge test")
    ap.add_argument("--symbols", type=str, default=None)
    ap.add_argument("--days", type=int, default=BACKTEST_DAYS)
    ap.add_argument("--iters", type=int, default=20000)
    args = ap.parse_args()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        wl = screener.load_watchlist()
        if not wl:
            print("No watchlist. Run: python3 screener.py")
            return 1
        symbols = [s["symbol"] for s in wl["watchlist"]]

    sig, base = collect(symbols, args.days)
    if sig.empty:
        print("No signals fired — nothing to test.")
        return 1

    out = report(sig, base, args.iters)
    print("\n" + out)
    (DATA / "edge_test.txt").write_text(out)
    sig.to_csv(DATA / "edge_signals.csv", index=False)
    print(f"\nwrote {DATA}/edge_test.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
