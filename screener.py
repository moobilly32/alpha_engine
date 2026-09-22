"""
Daily pre-market screener.

Scores the candidate pool on fundamentals, applies sector and industry
isolation, and writes data/watchlist.json for the intraday engine to consume.

Sector isolation is enforced twice:
  * MAX_PER_INDUSTRY (default 1) — the literal reading of "no two companies
    with identical business models". One semiconductor-equipment name, one
    integrated oil, one drug manufacturer.
  * MAX_PER_SECTOR — stops a hot sector eating the whole list even when the
    industries differ.

Usage:
    python3 screener.py                # full run, writes the watchlist
    python3 screener.py --limit 60     # faster smoke test
    python3 screener.py --dry-run      # print, don't write
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import data
import fundamentals
import universe
from calendar_util import now_ny
from config import (
    WATCHLIST_FILE, WATCHLIST_SIZE, MAX_PER_SECTOR, MAX_PER_INDUSTRY,
    MACRO_SYMBOLS, VIX_PROXIES, VIX_RISK_OFF, REALIZED_VOL_WINDOW,
    REALIZED_VOL_RISK_OFF, MACRO_SIZE_HAIRCUT, LOGS,
)


# ---------------------------------------------------------------- macro
def macro_regime() -> dict:
    """
    Cheap, honest regime read: index trend vs. its own 50-day SMA, SPY realized
    volatility, and the 10-year yield.

    This does not predict anything. It exists to halve position size when the
    tape is hostile, because a long breakout strategy in a risk-off tape is the
    same strategy with a worse hit rate.
    """
    import numpy as np

    out: dict = {"asof": now_ny().isoformat(timespec="seconds")}
    for key, sym in MACRO_SYMBOLS.items():
        try:
            d = data.daily_bars(sym, period="6mo")
            if d.empty:
                continue
            last = float(d["Close"].iloc[-1])
            sma50 = float(d["Close"].tail(50).mean())
            chg = float(d["Close"].pct_change().iloc[-1] * 100)
            out[key] = {"last": round(last, 2), "sma50": round(sma50, 2),
                        "above_sma50": last > sma50, "chg_pct": round(chg, 2)}
        except Exception:
            continue

    # Realized vol replaces ^VIX, which serves no data to this host.
    rvol = None
    try:
        spy = data.daily_bars("SPY", period="6mo")
        if not spy.empty:
            rets = spy["Close"].pct_change().dropna().tail(REALIZED_VOL_WINDOW)
            if len(rets) >= 10:
                rvol = float(rets.std() * np.sqrt(252))
                out["realized_vol"] = round(rvol, 4)
    except Exception:
        pass

    # VIX is a bonus, not a dependency.
    vix = None
    for proxy in VIX_PROXIES:
        try:
            d = data.daily_bars(proxy, period="1mo")
            if not d.empty and proxy.startswith("^"):
                vix = float(d["Close"].iloc[-1])
                out["vix"] = {"symbol": proxy, "last": round(vix, 2)}
                break
        except Exception:
            continue

    spy_ok = out.get("spy", {}).get("above_sma50", True)
    qqq_ok = out.get("qqq", {}).get("above_sma50", True)

    risk_off = (not spy_ok and not qqq_ok)
    if rvol is not None and rvol >= REALIZED_VOL_RISK_OFF:
        risk_off = True
    if vix is not None and vix >= VIX_RISK_OFF:
        risk_off = True

    out["risk_off"] = bool(risk_off)
    out["size_multiplier"] = MACRO_SIZE_HAIRCUT if risk_off else 1.0

    bits = ["SPY above 50d" if spy_ok else "SPY below 50d",
            "QQQ above 50d" if qqq_ok else "QQQ below 50d"]
    if rvol is not None:
        bits.append(f"SPY realized vol {rvol*100:.1f}%")
    if vix is not None:
        bits.append(f"VIX {vix:.1f}")
    if "tnx" in out:
        # ^TNX now quotes the yield directly (4.81 = 4.81%), not x10 as it
        # historically did. Verified against the live feed.
        bits.append(f"10y {out['tnx']['last']:.2f}%")
    out["summary"] = " · ".join(bits) + (" — RISK OFF, sizing halved" if risk_off else "")
    return out


# ---------------------------------------------------------------- screen
def screen(limit: int | None = None, workers: int = 8,
           verbose: bool = True) -> tuple[list[fundamentals.Snapshot], dict]:
    pool = universe.candidates(limit=limit)
    if verbose:
        print(f"Evaluating {len(pool)} candidates ...", flush=True)

    t0 = time.time()
    snaps: list[fundamentals.Snapshot] = []

    def _one(sym):
        try:
            return fundamentals.evaluate(sym)
        except Exception as e:                      # one bad ticker must not
            s = fundamentals.Snapshot(symbol=sym)   # abort a 140-name screen
            s.fails.append(f"error: {type(e).__name__}")
            return s

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, s in enumerate(ex.map(_one, pool), 1):
            snaps.append(s)
            if verbose and i % 25 == 0:
                print(f"  {i}/{len(pool)}  ({time.time()-t0:.0f}s)", flush=True)

    passed = [s for s in snaps if s.passed]
    fundamentals.add_comps(passed)
    for s in passed:                      # comps land after the base score
        if s.comp_discount is not None:
            s.score += min(max(-s.comp_discount, -0.4), 0.6) * 12.0
            s.score = round(s.score, 2)

    passed.sort(key=lambda s: s.score, reverse=True)

    if verbose:
        print(f"  {len(passed)}/{len(snaps)} cleared the quality gate "
              f"({time.time()-t0:.0f}s)", flush=True)

    return snaps, {"pool": len(pool), "passed": len(passed)}


def diversify(passed: list[fundamentals.Snapshot]) -> list[fundamentals.Snapshot]:
    """Greedy pick down the score ranking, respecting sector/industry caps."""
    chosen: list[fundamentals.Snapshot] = []
    per_sector: dict[str, int] = {}
    per_industry: dict[str, int] = {}

    for s in passed:
        if len(chosen) >= WATCHLIST_SIZE:
            break
        if per_sector.get(s.sector, 0) >= MAX_PER_SECTOR:
            continue
        if per_industry.get(s.industry, 0) >= MAX_PER_INDUSTRY:
            continue
        chosen.append(s)
        per_sector[s.sector] = per_sector.get(s.sector, 0) + 1
        per_industry[s.industry] = per_industry.get(s.industry, 0) + 1

    # If the industry cap starved the list, relax it (never the sector cap) so
    # the engine gets a full watchlist rather than 11 names.
    if len(chosen) < WATCHLIST_SIZE:
        have = {s.symbol for s in chosen}
        for s in passed:
            if len(chosen) >= WATCHLIST_SIZE:
                break
            if s.symbol in have:
                continue
            if per_sector.get(s.sector, 0) >= MAX_PER_SECTOR:
                continue
            chosen.append(s)
            per_sector[s.sector] = per_sector.get(s.sector, 0) + 1

    return chosen


def build_watchlist(limit: int | None = None, write: bool = True,
                    verbose: bool = True) -> dict:
    macro = macro_regime()
    snaps, stats = screen(limit=limit, verbose=verbose)
    passed = [s for s in snaps if s.passed]
    chosen = diversify(passed)

    payload = {
        "generated_at": now_ny().isoformat(timespec="seconds"),
        "session_date": now_ny().date().isoformat(),
        "macro": macro,
        "stats": {**stats, "selected": len(chosen)},
        "watchlist": [s.to_dict() for s in chosen],
    }

    if write:
        WATCHLIST_FILE.write_text(json.dumps(payload, indent=2, default=str))
        if verbose:
            print(f"\nwrote {WATCHLIST_FILE}")

    return payload


def load_watchlist() -> dict | None:
    if not WATCHLIST_FILE.exists():
        return None
    try:
        return json.loads(WATCHLIST_FILE.read_text())
    except Exception:
        return None


# ---------------------------------------------------------------- cli
def main() -> int:
    ap = argparse.ArgumentParser(description="Daily pre-market fundamental screener")
    ap.add_argument("--limit", type=int, default=None,
                    help="only evaluate the first N candidates (smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="print, do not write")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    payload = build_watchlist(limit=args.limit, write=not args.dry_run,
                              verbose=not args.quiet)

    print(f"\nMacro: {payload['macro']['summary']}")
    print(f"\n{'#':>3}  {'SYM':<6} {'SECTOR':<24} {'INDUSTRY':<34} "
          f"{'PRICE':>9} {'FWD PE':>7} {'SCORE':>7}")
    print("-" * 100)
    for i, s in enumerate(payload["watchlist"], 1):
        print(f"{i:>3}  {s['symbol']:<6} {str(s['sector'])[:23]:<24} "
              f"{str(s['industry'])[:33]:<34} "
              f"{(s['price'] or 0):>9.2f} {(s['forward_pe'] or 0):>7.1f} "
              f"{s['score']:>7.2f}")

    sectors: dict[str, int] = {}
    for s in payload["watchlist"]:
        sectors[s["sector"]] = sectors.get(s["sector"], 0) + 1
    print("\nSector spread: " + ", ".join(f"{k} {v}" for k, v in
                                          sorted(sectors.items(), key=lambda x: -x[1])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
