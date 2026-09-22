# Alpha Engine v1.1

A hybrid quantitative execution engine combining asymmetric ATR-based risk sizing,
automated FOMC macro-calendar synchronization, and fractional-share order execution
for micro-capital accounts. Runs the **Fixed ADX Hybrid strategy** — a regime-routed
DIP (mean-reversion) / BREAKOUT (momentum) system — against Alpaca or Robinhood via
a broker-agnostic interface, fully automated through macOS `launchd`.

---

## Production Status

**CLEARED FOR FORWARD PAPER TESTING — $1,900 Account Baseline.**

- Broker: Alpaca paper (`BROKER_MODE=alpaca_paper`), account `PA3YCYLCPWC9`.
- Live equity as of the account swap: **$1,900.00**, 0 open positions (fresh account).
- This status reflects a validated backtest and a live-verified execution pipeline
  (see *Backtested Performance* and *Testing Performed* below) — it is **not** a
  live track record yet. The $1,900 figure appears in two different places in this
  document with two different meanings: the account's *actual current balance*
  (real, today), and the backtest's *starting capital assumption* (simulated, over
  2022–present). Forward paper results will replace the backtest table over time as
  real trades accumulate.

---

## Backtested Performance (2022–Present, $1,900 Starting Capital)

Full 4-year daily-bar simulation (`hypothetical_backtester.py`), production risk
config, fractional-share sizing, 30-symbol Mixed universe, `MAX_CONCURRENT=4`.

| Metric | Value |
|---|---|
| Total Return | **+65.24%** |
| Final Account Value | **$3,139.60** |
| Win Rate | 52.4% |
| Max Drawdown | 8.55% |
| Sharpe Ratio | 1.156 |
| Executed Trades | 357 |
| Skipped Trades (0-share) | **0**, via fractional market entries |

For comparison, the same window/config with the OLD whole-share `floor()` sizing
skipped **91 signals** entirely (couldn't afford 1 whole share at the target risk)
and returned +58.64% / $3,014.14 final value — fractional sizing is what makes a
$1,900 account viable for this strategy at all. See *Known Limitations* for the one
execution-mechanics caveat this comparison doesn't capture (backtest fills are
frictionless; live market orders carry real slippage).

This is a **backtest**, not a live result — see *Production Status* above.

---

## Key Engineering Modules

### Asymmetric Risk Parameters
`hybrid_indicators.py` sizes every entry as `shares = risk_dollars / (entry − stop)`,
with the risk target split by signal type:

- **DIP** (mean-reversion): **0.625%** of equity, off a fixed 2.5% stop, with a
  2.5R profit target.
- **BREAKOUT** (momentum): **0.25%** of equity, off a 2.0× ATR(14) stop that
  trails up (never down) — no fixed target; exits purely via the trailing stop.

Regime routing: ADX(14) < 20 → DIP-eligible (ranging); ADX above the symbol's own
rolling 50-session 80th-percentile threshold (floored at 25) → BREAKOUT-eligible
(trending). The asymmetry is deliberate — a DIP is a confirmed oversold setup with
a validated historical edge; a BREAKOUT is taken the moment it prints, with no
confirmation yet that the move continues, so it gets a much smaller risk budget.

### Fractional Execution Engine
Precision floating-point order sizing (`round(shares, 6)`) with **no whole-share
floor** — the old `math.floor()` constraint is gone from the entry path. Because
neither Alpaca nor Robinhood accepts a fractional quantity on a LIMIT order (only
on MARKET/notional orders), entries submit via a new `Broker.buy_market()` method
rather than `buy_limit()`. Stops and targets are always recomputed off the
**actual fill price** after submission (confirmed via `Broker.get_order_status()`),
never the pre-trade signal snapshot — this matters more now than it did under
limit orders, since a market order carries no resting price protecting the entry
from slippage.

### Macro Safety Filters
Two independent gates block **new entries only** — an already-open position always
keeps managing off its own stop/target/trailing-stop regardless of either gate:

- **FOMC macro blackout**: blocks new entries from **1:45 PM to 3:30 PM ET** on a
  scheduled Fed announcement day. The date list (`FOMC_DATES`) is kept in sync with
  the official Federal Reserve calendar automatically (see *Automation
  Infrastructure* below) rather than hand-maintained.
- **SPY circuit breaker**: blocks new entries for 60 minutes after any exact
  rolling **10-minute SPY decline greater than 0.75%**, measured on native 5-minute
  bars, grouped per trading day so the rolling return never trips on an ordinary
  overnight gap.

### Automation Infrastructure
Four macOS `launchd` background jobs, all invoking `hybrid_engine.py` directly
(no long-running daemon — each fire is a fresh `python3` process, so code changes
take effect on the very next scheduled run with no reload needed):

| Job | Schedule | Command |
|---|---|---|
| `com.billy.alpha-screener` | 8:30 AM ET, Mon–Fri (**pre-market**) | `--screen` (read-only DIP/BREAKOUT snapshot) |
| `com.billy.alpha-intraday` | 10:00 AM–2:00 PM ET, every 15 min, Mon–Fri (85 fires/week) | `--execute` (submits real entries) |
| `com.billy.alpha-manager` | every 60 seconds | `--manage` (checks stops/targets, exits, reconciles state) |
| `com.billy.alpha-fomc-sync` | 1st of each month, 6:00 AM | `update_fomc_calendar.py` (syncs `FOMC_DATES` against the live Fed calendar) |

All scheduling gates (weekday, US holiday, time window, once-per-cycle dedup) live
in each job's bash wrapper script, not the plist — `StartCalendarInterval` re-fires
a run that was missed while the Mac was asleep, so the wrapper must independently
decide whether a late-firing run is still valid.

---

## Architecture

### Broker abstraction
`broker_interface.py` defines an abstract `Broker` with nine methods
(`get_price`, `buy_limit`, `buy_market`, `sell_market`, `place_stop_loss`,
`cancel_order`, `list_positions`, `get_equity`, `get_order_status`), implemented
identically for `alpaca_client.AlpacaBroker` and `robinhood_client.RobinhoodBroker`.
The strategy/state-machine layer never branches on which broker is live.

### The live engine (`hybrid_engine.py`)
Built in complete isolation from an earlier, separate execution engine
(`execution_engine.py`/`intraday.py`) — different broker connection, different
state files (`data/hybrid_positions.json` / `data/hybrid_closed_positions.jsonl`,
never `open_positions.json`). Commands:

```bash
python3 hybrid_engine.py --screen      # read-only DIP/BREAKOUT scan, no state written
python3 hybrid_engine.py --dry-run     # size + print orders for active signals, no broker order calls
python3 hybrid_engine.py --execute     # submit real fractional market orders
python3 hybrid_engine.py --manage      # check stops/targets, reconcile state, exit if breached
python3 hybrid_engine.py --status      # broker-authoritative position/PnL snapshot
```

Every entry goes through fill verification before ever being recorded as an open
position: a submitted order is polled briefly for a confirmed fill; anything still
working is tracked as `PENDING` (not `OPEN`) and resolved by `--manage` on a later
cycle. Position membership for `--status` and the dashboard is always read from
`broker.list_positions()` directly, never from local state alone.

### Diagnostic / research tools
- `dashboard.py` — Streamlit live monitor (launchd job status, positions, logs).
- `inspect_positions.py` — read-only terminal snapshot combining broker-confirmed
  positions with live-recomputed stop/target risk levels.
- `hypothetical_backtester.py` — isolated backtesting harness (never imports or
  modifies `hybrid_engine.py`); both the multi-year daily-bar engine and a
  60-day/15-minute intraday engine with multi-timeframe macro filters.
- `update_fomc_calendar.py` — fetches and parses the official Fed calendar,
  diffs it against `hybrid_engine.py`'s `FOMC_DATES`, and rewrites that file's
  source (with a timestamped backup and a post-write syntax check) if it changed.

---

## Setup

```
.env                    # BROKER_MODE, ALPACA_API_KEY/SECRET_KEY/BASE_URL,
                        # or RH_EMAIL/PASSWORD/MFA_SECRET/ACCOUNT_NUMBER
.env.template           # documented blank template — copy, never commit .env
```

`ALPACA_BASE_URL` must be the bare host (`https://paper-api.alpaca.markets`) — the
SDK appends `/v2` itself; including it in the URL double-appends and 404s.

```bash
python3 hybrid_engine.py --status     # verify broker connection + current book
```

---

## Testing Performed

- Unit tests (mocked orders) covering every fill-verification branch: confirmed
  fill → `OPEN`, still-working → `PENDING`, confirmed dead order → archived,
  broker lookup failure → left untouched for retry.
- Live end-to-end tests against the real paper account: `--dry-run`, `--status`,
  `--manage`, and a full `--execute --dry-run` signal-processing pass, all
  producing genuinely fractional share sizing (e.g. 0.091521 shares on a $1,900
  account) with zero errors.
- A reconciliation bug was found and fixed live on this account: an entry limit
  order that expired unfilled had been marked `OPEN` in local state indefinitely
  (a phantom position). `--manage` now reconciles local state against
  `broker.list_positions()` every cycle before doing anything else.

---

## Known Limitations

- **Backtest vs. live**: the performance table above is a historical simulation.
  It assumes frictionless fills at the strategy's computed price; live market
  orders (required for fractional sizing) will fill at whatever price the market
  prints at submission, which can differ meaningfully from the backtest's assumed
  fill, especially in a fast-moving name.
- **FOMC calendar currency**: dates beyond ~1 year out are marked "tentative" by
  the Fed itself and can move. `alpha-fomc-sync` re-syncs monthly, but a moved
  date between syncs is a real (if narrow) gap.
- **Robinhood's `get_order_status()`** field mapping (`state`,
  `cumulative_quantity`, `average_price`) is implemented per Robinhood's
  documented schema but has never been exercised against a real Robinhood order —
  this account trades exclusively `alpaca_paper`. Verify before it ever gates a
  real-money entry under `BROKER_MODE=robinhood_live`.
- **`execution_engine.py`/`intraday.py`** are a separate, earlier engine kept in
  the repo but not part of the live automation (superseded by `hybrid_engine.py`).
  It still enters via whole-share `buy_limit()`, unchanged by any of the work
  described here.
- **SPY circuit breaker is intraday-only**: the daily-bar backtest disables it
  deliberately (its only daily-resolution proxy, an Open→Low check, was found to
  over-block and distort results) — the backtested performance table above
  reflects FOMC blackout only, not the circuit breaker, whereas the live engine
  runs both.
