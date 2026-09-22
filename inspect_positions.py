"""
inspect_positions.py — read-only diagnostic snapshot of live positions,
combining Alpaca's broker-confirmed reality (list_positions()) with
hybrid_engine.py's own risk-management formulas, not just whatever is
currently stored in data/hybrid_positions.json.

ISOLATION: reuses hybrid_engine.make_broker() (the SAME broker factory the
live engine uses) and hybrid_indicators.STOP_ATR_MULT/current_atr, but
makes ZERO order calls and writes NO state files — read-only and
side-effect-free, in the same spirit as hybrid_engine.py --screen/--status.

WHY RECOMPUTE THE STOP RATHER THAN JUST READING THE STORED VALUE:
  - DIP: stop/target are fixed at entry (STOP_PCT/TARGET_R off the entry
    price) — the stored value in local state is already final, so this
    just echoes it.
  - BREAKOUT: the stop is a LIVE TRAILING stop
    (highest_high - STOP_ATR_MULT x TODAY's ATR14) that only ratchets
    UP, on every --manage cycle. This script recomputes it live — using
    today's ATR14 and max(stored highest_high, current live price) — so
    the printed number matches what --manage would compute RIGHT NOW,
    not whatever was last written to disk (which could be stale if a
    --manage cycle hasn't run since the price made a new high).

Position membership itself is broker-authoritative (list_positions()),
never local state alone — see hybrid_engine.reconcile_local_state()'s
docstring for why that distinction matters on this account (a real
phantom-position incident, JPM, 2026-09-17).

Usage:
    python3 inspect_positions.py
"""

from __future__ import annotations

import json

from hybrid_engine import HYBRID_POSITIONS_FILE, make_broker
from hybrid_indicators import STOP_ATR_MULT, current_atr


def _load_local_meta() -> dict[str, dict]:
    """Local trade_type/stop/target/highest_high bookkeeping ONLY — used to
    annotate a broker-confirmed position, never to decide which symbols
    are open (that's list_positions()'s job, in main())."""
    if not HYBRID_POSITIONS_FILE.exists():
        return {}
    try:
        state = json.loads(HYBRID_POSITIONS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return {s: p for s, p in state.items() if p.get("status") == "OPEN"}


def _row_for(p, meta: dict) -> dict:
    """
    Combines one broker-confirmed BrokerPosition (`p`) with local
    trade_type/stop/target bookkeeping (`meta` — {} if this symbol isn't
    tracked locally), recomputing the BREAKOUT trailing stop LIVE per
    hybrid_engine.py's own formula rather than trusting a possibly-stale
    stored value.
    """
    trade_type = meta.get("trade_type", "UNTRACKED")
    current_price = p.current_price

    if trade_type == "BREAKOUT":
        stored_stop = meta.get("current_stop")
        highest_high = max(float(meta.get("highest_high", p.avg_entry_price)),
                           current_price)
        atr_now = current_atr(p.symbol)
        if atr_now is not None:
            candidate = highest_high - STOP_ATR_MULT * atr_now
            live_stop = max(candidate,
                            float(stored_stop) if stored_stop is not None else candidate)
        else:
            live_stop = float(stored_stop) if stored_stop is not None else None
        target = None      # BREAKOUT has no fixed target — exits via trailing stop only
        target_label = "TRAIL"
    elif trade_type == "DIP":
        live_stop = (float(meta["current_stop"])
                     if meta.get("current_stop") is not None else None)
        target = (float(meta["target_price"])
                 if meta.get("target_price") is not None else None)
        target_label = "TARGET"
    else:
        # Broker holds this symbol but this engine has no local trade_type/
        # stop/target metadata for it (opened outside this engine, or a
        # local-state gap) — nothing to compute; flagged in main()'s footer.
        live_stop = None
        target = None
        target_label = "n/a"

    stop_dist_pct = ((current_price - live_stop) / current_price * 100.0
                     if live_stop is not None and current_price else None)
    target_dist_pct = ((target - current_price) / current_price * 100.0
                       if target is not None and current_price else None)

    cost_basis = p.avg_entry_price * p.qty
    pnl_pct = (p.unrealized_pnl / cost_basis * 100.0) if cost_basis else None

    return dict(
        symbol=p.symbol, qty=p.qty, avg_entry=p.avg_entry_price,
        current=current_price, trade_type=trade_type,
        stop=live_stop, stop_dist_pct=stop_dist_pct,
        target=target, target_label=target_label, target_dist_pct=target_dist_pct,
        pnl=p.unrealized_pnl, pnl_pct=pnl_pct,
    )


def print_table(rows: list[dict]) -> None:
    width = 130
    print("=" * width)
    print("LIVE POSITION RISK SNAPSHOT — broker-confirmed positions x hybrid_engine.py "
         "risk formulas")
    print("=" * width)
    header = (f"{'SYMBOL':<8}{'QTY':>10}{'ENTRY':>10}{'CURRENT':>10}{'TYPE':>10}"
             f"{'STOP $':>10}{'STOP %':>9}{'TGT/TRAIL':>12}{'TGT %':>9}"
             f"{'PNL $':>12}{'PNL %':>9}")
    print(header)
    print("-" * width)
    total_pnl = 0.0
    for r in rows:
        stop_txt = f"{r['stop']:.2f}" if r["stop"] is not None else "n/a"
        stop_pct_txt = (f"{r['stop_dist_pct']:+.2f}%"
                        if r["stop_dist_pct"] is not None else "n/a")
        if r["trade_type"] == "BREAKOUT":
            target_txt, target_pct_txt = "TRAIL", "n/a"
        elif r["target"] is not None:
            target_txt = f"{r['target']:.2f}"
            target_pct_txt = f"{r['target_dist_pct']:+.2f}%"
        else:
            target_txt, target_pct_txt = "n/a", "n/a"
        pnl_pct_txt = f"{r['pnl_pct']:+.2f}%" if r["pnl_pct"] is not None else "n/a"

        total_pnl += r["pnl"]
        # .6f, not .0f: qty can be fractional now (buy_market() entries) —
        # truncating it here would misreport a real fractional holding.
        print(f"{r['symbol']:<8}{r['qty']:>10.6f}{r['avg_entry']:>10.2f}"
              f"{r['current']:>10.2f}{r['trade_type']:>10}"
              f"{stop_txt:>10}{stop_pct_txt:>9}{target_txt:>12}{target_pct_txt:>9}"
              f"{r['pnl']:>+12.2f}{pnl_pct_txt:>9}")
    print("-" * width)
    print(f"{'TOTAL':<8}{'':>10}{'':>10}{'':>10}{'':>10}{'':>10}{'':>9}{'':>12}{'':>9}"
         f"{total_pnl:>+12.2f}")
    print("=" * width)


def main() -> int:
    try:
        b = make_broker()
        equity = b.get_equity()
    except Exception as e:                                          # noqa: BLE001
        print(f"inspect_positions: could not connect to broker "
             f"({type(e).__name__}: {e}) — aborting")
        return 1

    live = b.list_positions()
    if not live:
        print("No open positions at the broker right now.")
        return 0

    meta_by_symbol = _load_local_meta()
    rows = [_row_for(p, meta_by_symbol.get(p.symbol, {}))
           for p in sorted(live, key=lambda p: p.symbol)]

    print(f"Equity: ${equity:,.2f}  |  {len(rows)} open position(s)\n")
    print_table(rows)

    live_symbols = {p.symbol for p in live}
    ghosts = sorted(s for s in meta_by_symbol if s not in live_symbols)
    if ghosts:
        print()
        print(f"NOTE: {len(ghosts)} symbol(s) tracked locally but NOT held at the "
             f"broker (stale local state): {', '.join(ghosts)}")

    untracked = sorted(p.symbol for p in live if p.symbol not in meta_by_symbol)
    if untracked:
        print()
        print(f"NOTE: {len(untracked)} broker position(s) have no local trade_type/"
             f"stop/target metadata in {HYBRID_POSITIONS_FILE.name} (opened outside "
             f"this engine?): {', '.join(untracked)} — stop/target shown as n/a.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
