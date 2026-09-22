"""
dashboard.py — lightweight local Streamlit monitor for the live Alpha Engine
(the Fixed ADX Hybrid, running via hybrid_engine.py under launchd).

READ-ONLY: this file never calls buy_limit/sell_market/--execute/--manage,
and never writes data/hybrid_positions.json or data/open_positions.json. The
only "live" call it makes is a read-only get_price() per open position, to
show current price / unrealized PnL — via hybrid_engine.make_broker(), the
SAME isolated broker factory the live engine itself uses (so credentials/
BROKER_MODE selection come from the existing .env/config.py setup, not a
new copy of that logic here).

NOTE ON LOGS: the request asked for `logs/engine.log`, which does not exist
in this project. There is no single unified engine log — each launchd job
writes its own via its wrapper script. This dashboard shows the three real,
currently-active ones instead: execution_manager.log (hybrid_engine.py
--manage, 60s), intraday.log (--execute, 15min), screener.log (--screen,
daily).

Run:
    cd ~/alpha_engine
    streamlit run dashboard.py
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

from config import DATA, LOGS

st.set_page_config(page_title="Alpha Engine Dashboard", layout="wide")

HYBRID_POSITIONS_FILE = DATA / "hybrid_positions.json"

LAUNCHD_JOBS = [
    "com.billy.alpha-screener",
    "com.billy.alpha-intraday",
    "com.billy.alpha-manager",
]

LOG_FILES = {
    "Manager  (hybrid_engine.py --manage, 60s)": LOGS / "execution_manager.log",
    "Intraday (hybrid_engine.py --execute, 15min)": LOGS / "intraday.log",
    "Screener (hybrid_engine.py --screen, daily)": LOGS / "screener.log",
}


# ---------------------------------------------------------------- system health
def get_launchd_status() -> pd.DataFrame:
    """
    Parses `launchctl list` for the three alpha-* jobs. The PID column being
    "-" just means the job is idle, waiting for its next scheduled fire —
    normal for StartInterval/StartCalendarInterval jobs, not a failure.
    Health is judged by (a) the job being loaded at all and (b) its last
    exit code being 0.
    """
    try:
        out = subprocess.run(["launchctl", "list"], capture_output=True,
                             text=True, timeout=5).stdout
    except Exception as e:                                       # noqa: BLE001
        return pd.DataFrame([{"Job": "launchctl error", "Status": str(e)}])

    by_label = {ln.split()[-1]: ln.split() for ln in out.splitlines() if ln.strip()}

    rows = []
    for job in LAUNCHD_JOBS:
        parts = by_label.get(job)
        if parts is None or len(parts) < 3:
            rows.append(dict(Job=job, Loaded="No", PID="-", **{"Last Exit": "-"},
                             Status="⛔ NOT LOADED"))
            continue
        pid, exit_code = parts[0], parts[1]
        running_now = pid != "-"
        clean = exit_code in ("0", "-")
        if not clean:
            status = f"🔴 FAILING (exit {exit_code})"
        elif running_now:
            status = "🟢 Running now"
        else:
            status = "🟢 Idle (healthy)"
        rows.append(dict(Job=job, Loaded="Yes", PID=pid,
                         **{"Last Exit": exit_code}, Status=status))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- positions
@st.cache_resource(show_spinner=False)
def get_broker():
    """Same isolated broker factory hybrid_engine.py itself uses — never
    execution_engine.py's. Cached as a resource so the dashboard doesn't
    re-authenticate on every rerun; only actually connects once there is at
    least one open position to price."""
    from hybrid_engine import make_broker
    return make_broker()


@st.cache_data(ttl=10, show_spinner=False)
def get_live_price(symbol: str) -> float | None:
    try:
        return get_broker().get_price(symbol)
    except Exception:                                             # noqa: BLE001
        return None


@st.cache_data(ttl=10, show_spinner=False)
def get_broker_positions() -> list:
    """
    Ground truth for "what's actually open": broker.list_positions()
    directly, not data/hybrid_positions.json. Found live on this account:
    a JPM entry limit order that expired unfilled stayed marked OPEN in
    local state indefinitely (nothing ever checked the fill) — see
    hybrid_engine.reconcile_local_state()'s docstring for the full incident
    and the fix applied there. This dashboard must never let local JSON
    alone decide what counts as a position for the same reason.
    """
    try:
        return get_broker().list_positions()
    except Exception:                                             # noqa: BLE001
        return []


def load_hybrid_positions_meta() -> dict[str, dict]:
    """Local bookkeeping ONLY (trade_type/stop/target) — used to annotate a
    broker-confirmed position, never to decide which symbols are open."""
    if not HYBRID_POSITIONS_FILE.exists():
        return {}
    try:
        state = json.loads(HYBRID_POSITIONS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return {s: p for s, p in state.items() if p.get("status") == "OPEN"}


def build_positions_df() -> tuple[pd.DataFrame, list[float], list[str]]:
    """Returns (df, pnl_values, ghost_symbols). `ghost_symbols` are symbols
    marked OPEN in local state but not actually held at the broker right
    now — surfaced as an explicit warning below, never silently counted as
    a live position."""
    live = get_broker_positions()
    meta = load_hybrid_positions_meta()

    rows = []
    pnl_values: list[float] = []
    for p in live:
        m = meta.get(p.symbol, {})
        pnl_values.append(p.unrealized_pnl)
        stop = m.get("current_stop")
        stop_dist_pct = ((p.current_price - float(stop)) / p.current_price * 100.0
                         if stop is not None and p.current_price else None)
        rows.append(dict(
            Symbol=p.symbol, Type=m.get("trade_type", "n/a"),
            Entry=round(p.avg_entry_price, 2), Shares=p.qty,
            Current=round(p.current_price, 2),
            UnrealizedPnL=round(p.unrealized_pnl, 2),
            StopDistance=f"{stop_dist_pct:.2f}%" if stop_dist_pct is not None else "n/a",
        ))

    live_symbols = {p.symbol for p in live}
    ghosts = sorted(s for s in meta if s not in live_symbols)
    return pd.DataFrame(rows), pnl_values, ghosts


# ---------------------------------------------------------------- logs
def tail_log(path: Path, n: int = 15) -> str:
    if not path.exists():
        return f"(no such file: {path})"
    try:
        lines = path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:]) if lines else "(empty)"
    except OSError as e:                                          # noqa: BLE001
        return f"(error reading {path}: {e})"


# ---------------------------------------------------------------- layout
st.title("Alpha Engine — Live Dashboard")
st.caption(f"Read-only monitor. Last loaded: {dt.datetime.now():%Y-%m-%d %H:%M:%S}")

if st.button("🔄 Refresh now"):
    st.cache_data.clear()
    st.rerun()

st.header("System Health — launchd jobs")
st.dataframe(get_launchd_status(), width='stretch', hide_index=True)

st.header("Active Positions — live broker (list_positions())")
st.caption("Position count/qty come from the broker directly, not "
          "data/hybrid_positions.json — see build_positions_df()'s "
          "docstring for why (a real local/broker drift incident on this "
          "account). Local state only supplies Type/StopDistance.")
df, pnl_values, ghosts = build_positions_df()
if df.empty:
    st.info("No active positions at the broker right now.")
else:
    st.dataframe(df, width='stretch', hide_index=True)
    if pnl_values:
        st.metric("Total Unrealized PnL", f"${sum(pnl_values):,.2f}")

if ghosts:
    st.warning(f"{len(ghosts)} symbol(s) marked OPEN in "
              f"data/hybrid_positions.json but NOT held at the broker — "
              f"local state is stale: {', '.join(ghosts)}. "
              f"Run `hybrid_engine.py --manage` to reconcile.")

st.header("Recent Logs")
st.caption("`logs/engine.log` doesn't exist in this project — showing the "
          "three real per-job logs instead.")
for label, path in LOG_FILES.items():
    with st.expander(label, expanded=(path.name == "execution_manager.log")):
        st.code(tail_log(path, 15), language="text")
