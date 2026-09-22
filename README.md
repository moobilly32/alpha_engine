# Alpha Engine

Fundamental screener + intraday Trend Join Long execution engine + Discord alerts,
with a 59-day backtest and a signal edge test.

**Status: NOT CLEARED FOR LIVE TRADING.** The backtest is negative and the signal's
measured edge is smaller than its trading costs. See *Results* below. The code is
complete and validated; the strategy is not.

---

## Results (2026-06-15 → 2026-09-08, 59 sessions, 29 names)

| Metric | Value |
|---|---|
| Total trades | 120 |
| Win rate | 41.7% (95% CI 33–51%) |
| Net P&L | **−$210.46 (−2.10%)** on $10,000 |
| Max drawdown | $268.63 (2.70%) |
| Sharpe | −1.155 |
| Profit factor | 0.860 |
| SPY / QQQ same window | +1.47% / −3.45% |

**The exits are not the problem.** A 5×4 sweep of stop (0.30–1.50%) against target
(0.5–2.0R) produced **0 of 20 profitable** combinations, profit factor 0.30–0.87.

**The signal is the problem.** `edge_test.py` compares forward returns from breakout
bars against every other cadence bar in the same names, days and hours:

| Horizon | Signal | Baseline | Edge | p | Verdict |
|---|---|---|---|---|---|
| +15 min | +0.0546% | −0.0051% | +0.0597% | 0.007 | real but < costs |
| +30 min | +0.0558% | −0.0105% | +0.0664% | 0.028 | real but < costs |
| +60 min | +0.0493% | −0.0187% | +0.0680% | 0.096 | not significant |
| to close | −0.0160% | −0.0769% | +0.0609% | 0.411 | not significant |

The breakout carries genuine short-horizon information — about **+0.06%**, decaying
to noise within an hour. Round-trip slippage at 5 bps a side is **0.10%**. The edge
is real and it is smaller than the cost of harvesting it.

Mean maximum favourable excursion after a signal is **0.89%**, which is why a 2.0%
target was hit on only 10% of trades: the target sat beyond where these moves
typically travel.

---

## High-win-rate modifications (`scale_test.py`)

Partial profit scaling (sell 50% at 1R → stop to breakeven → 2.5R runner) and a
deep-oversold dip entry (1h RSI < 35 OR a touch of the lower hourly Bollinger
band, above the daily 200 SMA).

**Best config — oversold / 2.5% fixed stop / scale at 1.0R:**
85 trades · **55.3% win** (CI 45–65%) · **+$747.91 (+7.48%)** · maxDD $417 (3.84%)
· **Sharpe 2.59** · PF 1.35 · E[R] 0.17 · avg hold 4.4 days. SPY +1.67%.

**Do not treat +7.48% as an expectation.** Testing the oversold entry against
random timing at stop widths that were NOT selected gives a mixed picture:

| Stop | Real net | n | Real win% | Random win% | p |
|---|---|---|---|---|---|
| 2.5% *(selected)* | +7.48% | 85 | 55.3% | 44.8% | 0.000 |
| 3.0% | −4.89% | 61 | 47.5% | 44.7% | **0.520** |
| 4.0% | +4.10% | 40 | 55.0% | 42.5% | 0.030 |
| 1.0× ATR | −1.83% | 45 | 51.1% | 44.8% | 0.210 |
| 1.5× ATR | +6.46% | 27 | 59.3% | 43.3% | 0.000 |

Net P&L is **non-monotonic in stop width** — great at 2.5%, exactly random at
3.0%, good at 4.0%. That is the signature of noise, not edge. Mean real net
across the five is **+2.26%** vs random ≈ −4.4%. Expect ~+2%, not +7.5%.

**What IS consistent is the win rate**: oversold beat random on hit rate at all
five stop widths (+3 to +16pp, mean ~+8pp).

### Scaling raises win rate mechanically — check P&L, not win rate

Random entries carrying the same scale-at-1R structure hit **44.8%**, so a 50%+
win rate is not by itself evidence of anything. On the largest sample scaling
cost money:

| Entry / stop | Δ Win% | Δ Net $ | Verdict |
|---|---|---|---|
| pullback / structure | +16.8% | **−568** | win% up, P&L down |
| oversold / structure | +22.5% | +601 | better on both |
| oversold / 2.0× ATR | +0.0% | −238 | worse on both |

Scaling earns its place only with the oversold entry and a wide stop. Judge by
net P&L, profit factor and expectancy-in-R.

**Oversold needs a wide stop** (2.5–4%) because you are buying while price still
falls; with a tight structural stop it was the worst config tested (−12.46%,
PF 0.57). **Earnings blocks are variance reduction**, cutting max drawdown
(6.10→5.10%, 8.95→7.74%) at a small cost in return.

---

## Swing results (`swing_backtest.py`, `swing_sweep.py`)

Multi-day holds, no EOD flatten, pullback entry, gap-aware exits.

**Defensible config — structural stop / 3.0R / VWAP:** 148 trades, **+1.13%**,
win 27.7%, PF 1.04, Sharpe 0.42, max DD 5.32%, avg hold 2.1 days.

**Sweep top cell — 2.0× ATR / 2.0R / VWAP:** +10.08%, Sharpe 2.88 — but on
**11 trades** with 28-day holds and a 35–85% win-rate CI. Every profitable cell
sits in the widest-stop column; only 21 of 64 configs are profitable. Do not
trade the top cell.

By target R (median across all stops and supports): 2.0R −$198 · 2.5R −$228 ·
**3.0R −$111 (best)** · 3.5R −$267. Widening past 3.0R lowers hit rate faster
than it raises payoff. The stop matters far more than the target.

### Does the pullback entry beat random timing?

Control arm (`--entry random`): identical trend filter, exits, sizing and
concurrency — only the entry *timing* becomes a coin flip.

| Config | Real | Trades | Random mean | Percentile | p |
|---|---|---|---|---|---|
| 2.0× ATR / 2.0R (sweep winner) | +10.08% | 11 | +1.61% | 98th | 0.020 |
| **structure / 3.0R (a priori)** | **+1.13%** | **148** | **−3.29%** | **87th** | **0.133** |

The first row is **circular** — that cell was selected as best of 64, so it beats
random by construction. The second is the honest test: a config chosen for
sample size, not performance. It lands at the 87th percentile, **not
significant**.

The supporting detail is the hit rate: 27.7% real vs 21.0% random at matched
trade counts (148 vs 144). Breakeven at 3.0R is 25%, which is exactly why random
loses 3.3% and the strategy makes 1.1%.

**Verdict: the pullback entry is probably better than random timing, and clearly
better than the breakout it replaced — but it is not established at p<0.05 on
the non-circular test.**

### Overnight gap risk — the cost of the new architecture

14% of closed trades exit through a gap, and gapped stops fill **1.20% worse than
the stop price**. Without modelling this, every gap-through is silently repaired
to the stop and losses are understated. **Earnings are not yet handled for
multi-day holds** — the screener blacks out 3 days, but a position opened before
that window can run into a report, and that is the fattest tail here.

---

## Does the fundamental filter work? (`split_test.py`)

Cohorts are differenced against **their own** non-signal bars, then re-expressed
in units of each name's realized volatility — the score correlates **+0.60 with
5-minute realized vol** and the top decile is ~1.6× more volatile, so a raw
percentage comparison would measure beta, not quality.

| Test | DiD @ +30 min | p | Jackknife |
|---|---|---|---|
| Top decile vs bottom decile (**the score**) | +0.860 sd | 0.017 | 19/20 sig |
| Top decile vs gate failures (**the gate**) | +0.772 sd | 0.019 | 18/20 sig |

Significant at 0.05, **not** at a Bonferroni-corrected 0.0125 across four
horizons. Spearman(score, edge) = +0.32, p=0.27 — a top-vs-bottom effect, not a
smooth gradient, so don't read the score as a cardinal ranking.

| Cohort | Trades | Win% | Net | PF |
|---|---|---|---|---|
| TOP | 63 | 42.9% | −0.38% | 0.96 |
| BOTTOM | 37 | 24.3% | −4.04% | 0.35 |
| FAILED_GATE | 55 | 30.9% | −6.14% | 0.40 |

Ordering is monotone — the filter carries information — but nothing is
profitable. Best cohort is break-even at PF 0.96.

### The first run of this test gave the opposite answer, because of a bug

The gate read `if s.fcf is None or s.fcf <= 0`, treating **missing data as
failing**. 62 of 122 rejections were "FCF not positive" and every large one had
`fcf = None`, not a negative number — WMT ($14.9B), XOM ($23.6B) and V ($21.6B)
were all rejected for data *absence*. yfinance compounds this by returning
truncated `.info` payloads (~90 keys vs ~185) with sector, FCF and currentRatio
missing.

Fixed by: falling back to the cash-flow statement for FCF; not penalising
missing values; requiring `sector` before accepting an `.info` payload; relaxing
the current-ratio gate 1.00 → 0.60 (sub-1.0 is normal for retail and
restaurants — it was rejecting 43 names); and carving out financials, whose
FCF, D/E and current ratio are not comparable to an industrial's (14 of 17 were
being rejected on that category error).

**Gate pass rate went 73 → 138 of 195**, and the full watchlist backtest
improved from −2.10% to −0.88% (PF 0.86 → 0.93). Before the fix the test read
"the score works, the gate is useless"; after it, the reverse. Neither extreme
was real.

**Look-ahead is still unresolved.** The time-split probe — if the score merely
records what already happened, the effect should concentrate in the recent half
— flipped direction between the buggy and fixed runs, so it settles nothing.
Cohorts are still formed from today's fundamentals. Only point-in-time data or a
forward walk resolves it.

---

## Layout

```
config.py          every tunable, read by all entry points
calendar_util.py   NY clock, holidays, half-days, cadence
data.py            yfinance access, caching, shape normalisation
fundamentals.py    quality gate, DCF, comps, headline sentiment
universe.py        candidate pool (S&P 500 scrape + built-in seed)
screener.py        daily pre-market screen  → data/watchlist.json
strategy.py        THE SIGNAL — pullback + legacy breakout, shared everywhere
execution.py       execution range, risk sizing, stops/targets
notify.py          Discord webhook
intraday.py        15-minute engine
swing_backtest.py  MULTI-DAY engine — continuous holds, gap-aware exits
swing_sweep.py     stop x target x support grid
backtest.py        legacy day-trading backtest (kept as a reference)
edge_test.py       does the signal predict anything?
split_test.py      does the fundamental filter add anything?
sweep.py           legacy day-trading stop/target grid
```

`strategy.py` is imported by `intraday.py`, `swing_backtest.py` and the tests on
purpose. `swing_backtest.py --entry breakout --eod-flat` reproduces the old
day-trading engine **exactly** (verified across four stop/concurrency
combinations), which is the regression check that the swing rewrite did not
quietly change the old answer.

`strategy.py` is imported by both `intraday.py` and `backtest.py` on purpose. If the
logic ever forks into two implementations the backtest stops describing the system
you actually run.

---

## Usage

```bash
python3 screener.py                      # build today's watchlist
python3 screener.py --limit 40 --dry-run # fast smoke test

python3 intraday.py --dry-run            # evaluate, send nothing
python3 intraday.py --asof "2026-09-08 11:30" --dry-run   # replay a past scan
python3 intraday.py --force              # ignore the clock gate

python3 backtest.py                      # 59-day backtest
python3 backtest.py --no-range-gate      # ablation
python3 edge_test.py                     # signal edge vs. baseline
python3 sweep.py                         # stop/target grid
```

---

## Discord

The webhook URL is a bearer secret — anyone holding it can post to your channel — so
it lives in the Keychain, never in a file, a log or a repo.

1. Discord → Server Settings → Integrations → Webhooks → New Webhook → Copy URL.
2. In **your** terminal (not in chat):

```bash
security add-generic-password -a "$USER" -s discord-alpha-webhook -w 'PASTE_URL_HERE' -U
```

3. Verify: `python3 notify.py --test`

If no webhook is configured, alerts fall back to a local macOS notification, so a
missing secret degrades the channel rather than losing the alert.

Each alert carries: what the company does · P/E, forward P/E, FCF, health · live
price vs ideal range · execute-or-wait · share and dollar size · target, stop and
the 15:55 flatten.

**Notification policy — quiet by default.** 17 scans × 30 names is 510 evaluations a
day. A name is announced when it *enters* a state, not while it remains in one:
`EXECUTE` once per symbol per day, `APPROACHING` once and only if it has not already
fired `EXECUTE`, `EXTENDED` never on its own. Verified on a full replay of
2026-09-08: 17 scans produced 1 summary + 3 alerts and 13 silent runs.

---

## Scheduling

Written but **deliberately not loaded** — loading them starts posting to Discord, and
the strategy is not cleared.

```bash
launchctl load ~/Library/LaunchAgents/com.billy.alpha-screener.plist   # 08:30 Mon-Fri
launchctl load ~/Library/LaunchAgents/com.billy.alpha-intraday.plist   # 85 triggers
launchctl list | grep alpha                                           # verify
launchctl unload ~/Library/LaunchAgents/com.billy.alpha-intraday.plist # stop
```

`StartCalendarInterval`, not cron: cron silently drops runs missed while the Mac
sleeps; launchd re-fires them on wake. Because it fires on wake regardless of the
clock, **all** gating lives in the wrapper scripts — weekend, US holiday, time
window, once-per-NY-date.

---

## Things that are true about this machine

Each verified, not assumed. Do not "fix" them back.

- **`^VIX` returns no data here** — empty via both `yf.download` and `Ticker.history`.
  `^VIX9D` resolves but serves one row. The regime gate therefore uses **SPY realized
  volatility**, which is always computable; VIX is read opportunistically.
- **`^TNX` now quotes the yield directly** (4.81 = 4.81%), not ×10 as it historically
  did.
- **`debtToEquity` from yfinance is a PERCENT** — AMD's 6.361 means 0.064x, not 6.4x.
  Always divide by 100. Getting this backwards rejects every healthy company or
  accepts every levered one.
- **Yahoo caps 5-minute history at 60 calendar days.** That cap is the reason the
  backtest window is 59 days, and it cannot be extended with this data source.
- **Intermittent HTTP 401 "Invalid Crumb"** on `.info` under concurrency. `data.info`
  retries three times; without it roughly one name in eight silently drops, and a
  dropped name is indistinguishable from one that failed the gate.
- **The Anaconda interpreter at `/opt/anaconda3/bin/python3` holds yfinance.** launchd
  gets a minimal PATH and cannot find it, so both wrappers and both plists pin it.
- **`flock` does not exist on macOS** — the intraday lock is a PID file.

---

## Known limitations

- **The screener cannot be backtested.** yfinance serves no point-in-time
  fundamentals, so the watchlist is chosen with today's balance sheets and run
  backwards. Every name survived to today looking healthy. The backtest measures the
  *technical* layer on a hindsight-chosen universe; it is not evidence the screener
  adds value. Fixing this needs a point-in-time fundamentals source.
- **The DCF is a relative ranker, not a fair value.** Single-stage, 9% WACC, growth
  capped at 25%. Median output is about −60% versus spot across the watchlist — it
  cannot justify a quality compounder's multiple. It ranks EOG/BLK/PSX cheap and
  MSFT/NVDA expensive, which is the correct *ordering*; the absolute level is not
  meaningful and is labelled "rel." in alerts.
- **Headline sentiment is keyword-based**, deliberately: an unattended job that
  depends on an authenticated LLM CLI turns a news blip into a silent failure. It
  degrades to neutral instead.
- **The execution-range gate is nearly inert in backtest** (1 skip in 220 signals).
  The signal fires on the first cadence bar clearing the trigger, so price is almost
  always barely above it. The gate matters live, where a scan can land well after a
  breakout began.
- **`MAX_CONCURRENT = 2` is the binding constraint**, dropping 99 of 220 signals.
  Candidates at each scan are ranked by screener score then least-extended, so the
  cap keeps the best available names rather than the alphabetically first — this was
  worth $124 of the result.
- **Sub-bar exit ordering is unknowable** from 5-minute OHLC. When a bar contains both
  the stop and the target, the stop is assumed to fill first.
