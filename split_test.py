"""
Does the fundamental score do any work?

THE QUESTION. The backtest showed the technical signal has a real but
sub-friction edge. That result says nothing about the screener: the watchlist
was chosen by fundamental score, but every name in it cleared the same gate, so
the score's contribution was never isolated. This script isolates it.

THE CONFOUND, and why a naive comparison would be worthless. The top scorers are
NVDA / MU / AVGO / VLO / EOG — high-beta semiconductors and refiners. The bottom
scorers are MCD / COST / LOW / ISRG — low-beta defensives. Comparing their raw
post-signal returns would mostly measure beta and sector drift over one 59-day
window, and would "prove" the score works on any window where tech outperformed.

THE FIX. Each cohort is differenced against ITS OWN baseline: the forward returns
from every other cadence bar in the same names, same days, same hours. A cohort's
"edge" is therefore

    edge = mean(return | breakout fired) - mean(return | same names, no breakout)

which cancels whatever those names were doing anyway. The test statistic is the
difference-in-differences:

    DiD = edge(top decile) - edge(bottom decile)

If the fundamental score adds information, high-scoring names should convert a
breakout into forward return better than low-scoring names do. DiD > 0 is that
claim; DiD ~ 0 says the score is decoration.

A THIRD COHORT tests the GATE rather than the SCORE: names that FAILED the
quality gate entirely. The score only ranks within survivors, so the gate could
be doing all the work, or none.

PRIMARY ENDPOINT is +30 minutes, fixed in advance because that is where the base
edge test found the signal strongest. The other horizons are secondary and carry
a Bonferroni note — testing four horizons and reporting the best one is how a
null result gets dressed up as a finding.

Usage:
    python3 split_test.py
    python3 split_test.py --n 15 --iters 20000
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import screener
from backtest import run_backtest, max_drawdown, sharpe, wilson_ci
from edge_test import collect, FRICTION
from config import BACKTEST_DAYS, BACKTEST_CAPITAL, DATA

HORIZON_COLS = {"+15 min": "f3", "+30 min": "f6", "+60 min": "f12", "to close": "eod"}
PRIMARY = "+30 min"


# ---------------------------------------------------------------- volatility
def realized_vol(symbol: str) -> float | None:
    """Standard deviation of the name's own 5-minute RTH returns over the window."""
    import data
    d = data.intraday_5m(symbol, period="60d")
    if d.empty:
        return None
    m = d.index.hour * 60 + d.index.minute
    r = d[(m >= 570) & (m < 960)]["Close"].pct_change().dropna()
    return float(r.std()) if len(r) > 100 else None


_VOL: dict[str, float] = {}


def vol_of(symbol: str) -> float | None:
    if symbol not in _VOL:
        v = realized_vol(symbol)
        if v:
            _VOL[symbol] = v
    return _VOL.get(symbol)


def add_vol_normalized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Express every forward return in units of the symbol's own 5-minute vol.

    THE REASON THIS PANEL EXISTS. The fundamental score correlates about +0.60
    with realized volatility, and the top decile is roughly 1.6x more volatile
    than the bottom. Percentage returns therefore scale with the score
    mechanically, whether or not the score carries any information. Dividing by
    each name's own volatility removes that scaling, so what remains is "how far
    did this move go, relative to how far this stock normally moves" — which is
    the question the raw panel cannot answer.
    """
    if df.empty:
        return df
    out = df.copy()
    v = out["symbol"].map(lambda s: vol_of(s))
    for col in HORIZON_COLS.values():
        if col in out.columns:
            out[col + "_z"] = out[col] / v
    return out.dropna(subset=[c + "_z" for c in HORIZON_COLS.values()
                              if c + "_z" in out.columns])


# ---------------------------------------------------------------- statistics
def boot_mean_diff(a: np.ndarray, b: np.ndarray, iters: int, rng) -> np.ndarray:
    """Bootstrap distribution of mean(a) - mean(b)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros(iters)
    ia = rng.integers(0, len(a), size=(iters, len(a)))
    ib = rng.integers(0, len(b), size=(iters, len(b)))
    return a[ia].mean(axis=1) - b[ib].mean(axis=1)


def boot_did(sig_a, base_a, sig_b, base_b, iters=20000, seed=11):
    """
    Bootstrap the difference-in-differences.

    Resamples within each of the four samples independently, so the resulting
    distribution carries the uncertainty of all four means — which is the whole
    point, since a DiD built from two noisy edges is noisier than either.
    """
    rng = np.random.default_rng(seed)
    d_a = boot_mean_diff(sig_a, base_a, iters, rng)
    d_b = boot_mean_diff(sig_b, base_b, iters, rng)
    did = d_a - d_b
    obs = ((sig_a.mean() - base_a.mean()) - (sig_b.mean() - base_b.mean())
           if len(sig_a) and len(sig_b) else 0.0)
    # Two-sided p from how much of the bootstrap mass sits on the other side of 0
    p = 2.0 * min((did <= 0).mean(), (did >= 0).mean())
    return obs, min(p, 1.0), np.percentile(did, [2.5, 97.5])


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman rho and a permutation p-value, without scipy."""
    n = len(x)
    if n < 4:
        return 0.0, 1.0
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(3)
    null = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(5000)])
    return rho, float((np.abs(null) >= abs(rho)).mean())


# ---------------------------------------------------------------- cohorts
def build_cohorts(n: int, verbose: bool = True, include_financials: bool = False):
    """
    Cohorts for the split test.

    Financials are excluded from the SCORE cohorts by default. The gate now has
    a carve-out that admits banks (their cash-flow and leverage metrics are not
    comparable to an industrial's), but the SCORE still leans on FCF yield and a
    DCF, neither of which is defined for them — so banks pile up at the bottom
    of the ranking for a structural reason rather than a quality one. Leaving
    them in makes "low score" mean "is a bank", and the test then measures
    sector membership instead of fundamental quality.
    """
    from fundamentals import FINANCIAL_SECTORS

    snaps, stats = screener.screen(verbose=verbose)
    pool = [s for s in snaps if s.passed]
    if not include_financials:
        pool = [s for s in pool if s.sector not in FINANCIAL_SECTORS]
    passed = sorted(pool, key=lambda s: -s.score)
    if len(passed) < 2 * n:
        raise SystemExit(f"only {len(passed)} names passed the gate; need {2*n}")

    top = passed[:n]
    bot = passed[-n:]

    # Gate failures, largest first — big liquid names so the comparison is not
    # confounded by illiquidity on top of everything else.
    fails = sorted([s for s in snaps if not s.passed and s.market_cap and s.price],
                   key=lambda s: -(s.market_cap or 0))[:n]

    return {"TOP": top, "BOTTOM": bot, "FAILED_GATE": fails}, passed, stats


def describe(name: str, cohort) -> str:
    scores = [s.score for s in cohort]
    secs: dict[str, int] = {}
    for s in cohort:
        secs[s.sector] = secs.get(s.sector, 0) + 1
    top_secs = ", ".join(f"{k.split()[0]} {v}" for k, v in
                         sorted(secs.items(), key=lambda x: -x[1])[:3])
    return (f"  {name:<12} {', '.join(s.symbol for s in cohort)}\n"
            f"  {'':<12} score {min(scores):.1f}–{max(scores):.1f} · {top_secs}")


# ---------------------------------------------------------------- run
def main() -> int:
    ap = argparse.ArgumentParser(description="Fundamental score split test")
    ap.add_argument("--n", type=int, default=10, help="cohort size")
    ap.add_argument("--days", type=int, default=BACKTEST_DAYS)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--capital", type=float, default=BACKTEST_CAPITAL)
    ap.add_argument("--include-financials", action="store_true",
                    help="keep banks in the score cohorts (see build_cohorts)")
    args = ap.parse_args()

    cohorts, passed, stats = build_cohorts(
        args.n, include_financials=args.include_financials)

    L, A = [], None
    A = L.append
    A("=" * 84)
    A("  FUNDAMENTAL SCORE SPLIT TEST")
    A("=" * 84)
    A(f"  Pool {stats['pool']} · cleared gate {stats['passed']} · cohort size {args.n}")
    A(f"  Financials in score cohorts: "
      f"{'yes' if args.include_financials else 'NO (score formula undefined for banks)'}")
    A("")
    for k, c in cohorts.items():
        A(describe(k, c))
    A("")
    A("  Note the sector skew: this is why every comparison below is differenced")
    A("  against each cohort's OWN non-signal bars, not against zero.")

    # ---- collect signal & baseline samples per cohort
    samples = {}
    for name, cohort in cohorts.items():
        syms = [s.symbol for s in cohort]
        sig, base = collect(syms, args.days, verbose=False)
        samples[name] = (sig, base)

    # ---- per-cohort edge table
    A("")
    A("=" * 84)
    A("  PER-COHORT EDGE  (signal bars minus that cohort's own other bars)")
    A("=" * 84)
    A(f"  {'COHORT':<13}{'SIGNALS':>8}{'HORIZON':>11}{'SIGNAL':>10}"
      f"{'BASELINE':>10}{'EDGE':>10}{'vs FRICTION':>14}")
    A("  " + "-" * 80)
    for name, (sig, base) in samples.items():
        if sig.empty:
            A(f"  {name:<13}{0:>8}   no signals fired")
            continue
        for h, col in HORIZON_COLS.items():
            s = sig[col].dropna().to_numpy()
            b = base[col].dropna().to_numpy()
            edge = s.mean() - b.mean()
            mark = "clears" if edge > FRICTION else "below"
            A(f"  {name if h == '+15 min' else '':<13}"
              f"{len(s) if h == '+15 min' else '':>8}{h:>11}"
              f"{s.mean()*100:>9.4f}%{b.mean()*100:>9.4f}%{edge*100:>9.4f}%"
              f"{mark:>14}")
        A("  " + "-" * 80)

    # ---- difference-in-differences
    A("")
    A("=" * 84)
    A("  DIFFERENCE-IN-DIFFERENCES")
    A("=" * 84)

    def did_block(a_name: str, b_name: str, title: str, norm: bool = False):
        A(f"\n  {title}")
        unit = "sd" if norm else "%"
        A(f"  {'HORIZON':<12}{'EDGE(A)':>11}{'EDGE(B)':>11}{'DiD':>11}"
          f"{'95% CI':>22}{'p':>9}")
        A("  " + "-" * 80)
        sa, ba = samples[a_name]
        sb, bb = samples[b_name]
        if norm:
            sa, ba, sb, bb = (add_vol_normalized(x) for x in (sa, ba, sb, bb))
        if sa.empty or sb.empty:
            A("    insufficient signals")
            return
        for h, base_col in HORIZON_COLS.items():
            col = base_col + "_z" if norm else base_col
            if col not in sa.columns or col not in sb.columns:
                continue
            obs, p, ci = boot_did(sa[col].dropna().to_numpy(),
                                  ba[col].dropna().to_numpy(),
                                  sb[col].dropna().to_numpy(),
                                  bb[col].dropna().to_numpy(),
                                  iters=args.iters)
            ea = sa[col].mean() - ba[col].mean()
            eb = sb[col].mean() - bb[col].mean()
            k = 1.0 if norm else 100.0
            star = " *" if p < 0.05 else ""
            tag = "  <-- PRIMARY" if h == PRIMARY else ""
            A(f"  {h:<12}{ea*k:>10.4f}{unit}{eb*k:>10.4f}{unit}{obs*k:>10.4f}{unit}"
              f"{f'[{ci[0]*k:+.3f}, {ci[1]*k:+.3f}]':>22}{p:>9.4f}{star}{tag}")

    did_block("TOP", "BOTTOM", "A = top-scoring decile   B = bottom-scoring decile   "
                               "(tests the SCORE)")
    did_block("TOP", "FAILED_GATE", "A = top-scoring decile   B = gate failures        "
                                    "(tests the GATE)")

    # ---- volatility-normalised repeat of the headline test
    A("")
    A("=" * 84)
    A("  VOLATILITY-NORMALISED  —  returns in units of each name's own 5-min sd")
    A("=" * 84)
    A("  The score correlates ~+0.60 with realized volatility and the top decile")
    A("  is ~1.6x more volatile than the bottom, so percentage returns scale with")
    A("  the score mechanically. If the effect above is real rather than a beta")
    A("  artifact, it must survive here.")
    did_block("TOP", "BOTTOM", "A = top decile   B = bottom decile   (SCORE, vol-normalised)",
              norm=True)
    did_block("TOP", "FAILED_GATE", "A = top decile   B = gate failures   (GATE, vol-normalised)",
              norm=True)

    # ---- volatility profile, so the confound is on the page
    A("")
    A("  COHORT VOLATILITY PROFILE")
    A(f"  {'COHORT':<14}{'MEAN SCORE':>12}{'MEAN 5m SD':>13}{'RATIO vs BOTTOM':>18}")
    vols = {}
    for name, cohort in cohorts.items():
        vs = [vol_of(s.symbol) for s in cohort]
        vs = [v for v in vs if v]
        vols[name] = float(np.mean(vs)) if vs else float("nan")
    base_v = vols.get("BOTTOM", float("nan"))
    for name, cohort in cohorts.items():
        ms = float(np.mean([s.score for s in cohort]))
        A(f"  {name:<14}{ms:>12.1f}{vols[name]*100:>12.4f}%"
          f"{vols[name]/base_v:>17.2f}x")

    # ---- robustness
    A("")
    A("=" * 84)
    A("  ROBUSTNESS  (primary horizon, volatility-normalised)")
    A("=" * 84)
    sa, ba = (add_vol_normalized(x) for x in samples["TOP"])
    sb, bb = (add_vol_normalized(x) for x in samples["BOTTOM"])
    zc = HORIZON_COLS[PRIMARY] + "_z"

    if not sa.empty and not sb.empty:
        # Time split. This is the look-ahead probe, not just a stability check.
        # The score is built from TODAY's fundamentals, so if it is merely
        # recording what already happened, the effect should be CONCENTRATED in
        # the recent half of the window — the part closest to the snapshot. An
        # effect that is flat or stronger in the older half is evidence against
        # that mechanism (it does not rule out survivorship in cohort membership).
        dates = sorted(set(sa["date"]) | set(sb["date"]))
        mid = dates[len(dates) // 2]
        A("")
        A("  TIME SPLIT — look-ahead probe")
        A(f"  {'PERIOD':<24}{'n(A)':>7}{'n(B)':>7}{'EDGE(A)':>11}"
          f"{'EDGE(B)':>11}{'DiD':>10}{'p':>9}")
        A("  " + "-" * 80)
        for label, mask in (("first half (older)", lambda d: d["date"] < mid),
                            ("second half (recent)", lambda d: d["date"] >= mid)):
            a, ab = sa[mask(sa)], ba[mask(ba)]
            b, bf = sb[mask(sb)], bb[mask(bb)]
            if len(a) < 5 or len(b) < 5:
                A(f"  {label:<24} too few observations")
                continue
            obs, p, _ = boot_did(a[zc].dropna().to_numpy(), ab[zc].dropna().to_numpy(),
                                 b[zc].dropna().to_numpy(), bf[zc].dropna().to_numpy(),
                                 iters=args.iters)
            A(f"  {label:<24}{len(a):>7}{len(b):>7}"
              f"{a[zc].mean()-ab[zc].mean():>10.3f}sd"
              f"{b[zc].mean()-bf[zc].mean():>10.3f}sd{obs:>9.3f}sd{p:>9.4f}")

        # Jackknife: is the result carried by one lucky name?
        rows = []
        for sym in [s.symbol for s in cohorts["TOP"] + cohorts["BOTTOM"]]:
            a, ab = sa[sa.symbol != sym], ba[ba.symbol != sym]
            b, bf = sb[sb.symbol != sym], bb[bb.symbol != sym]
            if len(a) < 5 or len(b) < 5:
                continue
            obs, p, _ = boot_did(a[zc].dropna().to_numpy(), ab[zc].dropna().to_numpy(),
                                 b[zc].dropna().to_numpy(), bf[zc].dropna().to_numpy(),
                                 iters=4000)
            rows.append((sym, obs, p))
        if rows:
            jk = pd.DataFrame(rows, columns=["dropped", "did", "p"])
            full, _, _ = boot_did(sa[zc].dropna().to_numpy(), ba[zc].dropna().to_numpy(),
                                  sb[zc].dropna().to_numpy(), bb[zc].dropna().to_numpy(),
                                  iters=args.iters)
            A("")
            A("  LEAVE-ONE-NAME-OUT")
            A(f"    full sample DiD      {full:>7.3f} sd")
            A(f"    jackknife DiD range  {jk['did'].min():>7.3f} .. {jk['did'].max():.3f} sd")
            A(f"    significant (p<0.05) {int((jk['p'] < 0.05).sum())}/{len(jk)} drops")
            worst = jk.reindex(jk["did"].sub(full).abs().sort_values(ascending=False).index)
            A("    most influential:")
            for _, r in worst.head(3).iterrows():
                A(f"      without {r['dropped']:<6} DiD {r['did']:>6.3f} sd  p={r['p']:.4f}")

    # ---- per-name rank correlation
    A("")
    A("=" * 84)
    A("  RANK CORRELATION — does a higher score mean a bigger edge?")
    A("=" * 84)
    rows = []
    for name in ("TOP", "BOTTOM"):
        sig, base = samples[name]
        if sig.empty:
            continue
        bmean = base.groupby("symbol")["f6"].mean()
        for sym, g in sig.groupby("symbol"):
            sc = next((s.score for s in cohorts[name] if s.symbol == sym), None)
            if sc is None or sym not in bmean.index or len(g) < 2:
                continue
            rows.append({"symbol": sym, "score": sc, "n": len(g),
                         "edge30": g["f6"].mean() - bmean[sym]})
    df = pd.DataFrame(rows)
    if len(df) >= 6:
        rho, p = spearman(df["score"].to_numpy(), df["edge30"].to_numpy())
        A(f"  names with >=2 signals: {len(df)}")
        A(f"  Spearman rho(score, +30min edge) = {rho:+.3f}   permutation p = {p:.4f}")
        A("")
        A(f"  {'SYM':<7}{'SCORE':>8}{'SIGNALS':>9}{'EDGE +30m':>12}")
        for _, r in df.sort_values("score", ascending=False).iterrows():
            A(f"  {r['symbol']:<7}{r['score']:>8.1f}{int(r['n']):>9}"
              f"{r['edge30']*100:>11.4f}%")
    else:
        A("  too few names with repeated signals to correlate")

    # ---- portfolio backtests
    A("")
    A("=" * 84)
    A("  PORTFOLIO BACKTEST PER COHORT  (same rules, same window)")
    A("=" * 84)
    A(f"  {'COHORT':<14}{'TRADES':>8}{'WIN%':>8}{'95% CI':>12}{'NET $':>11}"
      f"{'NET %':>9}{'PF':>7}{'SHARPE':>8}{'MAXDD':>8}")
    A("  " + "-" * 80)
    for name, cohort in cohorts.items():
        syms = [s.symbol for s in cohort]
        sc = {s.symbol: s.score for s in cohort}
        r = run_backtest(syms, days=args.days, capital=args.capital,
                         scores=sc, verbose=False)
        t = r.trades
        n = len(t)
        w = sum(1 for x in t if x.is_win)
        net = sum(x.pnl for x in t)
        gw = sum(x.pnl for x in t if x.pnl > 0)
        gl = -sum(x.pnl for x in t if x.pnl <= 0)
        pf = gw / gl if gl > 0 else 0.0
        _, ddp = max_drawdown(r.equity) if not r.equity.empty else (0, 0)
        lo, hi = wilson_ci(w, n)
        A(f"  {name:<14}{n:>8}{(w/n*100 if n else 0):>7.1f}%"
          f"{f'{lo*100:.0f}-{hi*100:.0f}%':>12}{net:>+11,.2f}"
          f"{net/args.capital*100:>+8.2f}%{pf:>7.2f}{sharpe(r.equity):>8.2f}"
          f"{ddp*100:>7.2f}%")

    A("")
    A("  " + "!" * 78)
    A("  HOW TO READ THIS")
    A("  The DiD row at the PRIMARY horizon is the answer. Positive and")
    A("  significant = the score carries information. Straddling zero = it does")
    A("  not, on this evidence.")
    A("  Four horizons are tested, so a Bonferroni-corrected threshold is")
    A("  p < 0.0125. Secondary horizons clearing 0.05 but not 0.0125 are not")
    A("  findings.")
    A("  A null here is WEAK evidence of no effect, not proof of none: cohorts")
    A("  of this size over 59 sessions can only detect a fairly large effect.")
    A("  " + "!" * 78)

    out = "\n".join(L)
    print("\n" + out)
    (DATA / "split_test.txt").write_text(out)
    if len(df):
        df.to_csv(DATA / "split_test_per_name.csv", index=False)
    print(f"\nwrote {DATA}/split_test.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
