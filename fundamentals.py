"""
Fundamental quality gate, DCF, comps and headline sentiment.

Scope note, stated plainly because it changes how much the backtest is worth:
every number here is a CURRENT snapshot. yfinance does not serve point-in-time
fundamentals, so there is no way to ask "what did this balance sheet look like
on 12 June". Any backtest whose universe was chosen with this module inherits
look-ahead bias — the names are ones that still looked healthy today. See
backtest.py, which says so again in its own output.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field, asdict

import data
from config import (
    MIN_MARKET_CAP, MIN_AVG_DOLLAR_VOL, MIN_PRICE, MAX_PRICE,
    MAX_DEBT_TO_EQUITY, MIN_CURRENT_RATIO, MAX_FORWARD_PE, MIN_FORWARD_PE,
    REQUIRE_POSITIVE_FCF, MIN_REVENUE_GROWTH, MIN_PROFIT_MARGIN,
    EARNINGS_BLACKOUT_DAYS, DCF_WACC, DCF_TERMINAL_GROWTH, DCF_HORIZON_YEARS,
    DCF_FADE, DCF_MAX_INITIAL_GROWTH,
)


# ---------------------------------------------------------------- helpers
def _f(v) -> float | None:
    """Coerce to a finite float, or None. yfinance is liberal with junk."""
    try:
        if v is None:
            return None
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


# Banks and insurers do not have meaningful free cash flow, current ratios or
# industrial-style leverage: JPM's operating cash flow prints at -$148B because
# it swings with the loan book and trading inventory. Applying an industrial
# gate to them rejects the entire sector for a category error (14 of 17
# financials were rejected before this carve-out existed).
FINANCIAL_SECTORS = {"Financial Services", "Financials"}


def fcf_from_statement(symbol: str) -> float | None:
    """
    Free cash flow from the cash-flow statement, used when `.info` omits it.

    `.info["freeCashflow"]` is missing for a lot of very large companies, and
    treating that absence as "FCF is not positive" rejected WMT ($14.9B),
    XOM ($23.6B) and V ($21.6B) — all of which report it perfectly well one
    call away. Prefer the explicit Free Cash Flow row, else operating cash
    flow minus capex.
    """
    import data as _d
    try:
        cf = _d.financials(symbol).get("cashflow")
        if cf is None or cf.empty:
            return None
        col = cf.columns[0]

        def _row(*names):
            for want in names:
                for idx in cf.index:
                    if str(idx).strip().lower() == want:
                        v = _f(cf.loc[idx, col])
                        if v is not None:
                            return v
            return None

        direct = _row("free cash flow")
        if direct is not None:
            return direct
        ocf = _row("operating cash flow", "total cash from operating activities")
        capex = _row("capital expenditure", "capital expenditures")
        if ocf is not None:
            return ocf + (capex or 0.0)   # capex is reported negative
        return None
    except Exception:
        return None


def debt_to_equity(raw) -> float | None:
    """
    yfinance reports debtToEquity as a PERCENT (AMD 6.361 means 0.064x, not 6.4x).
    Getting this wrong rejects every healthy company or accepts every levered
    one, depending on which way you guess. Always divide.
    """
    v = _f(raw)
    return None if v is None else v / 100.0


# ---------------------------------------------------------------- snapshot
@dataclass
class Snapshot:
    symbol: str
    name: str = ""
    sector: str = "Unknown"
    industry: str = "Unknown"
    summary: str = ""

    price: float | None = None
    market_cap: float | None = None
    avg_dollar_vol: float | None = None

    trailing_pe: float | None = None
    forward_pe: float | None = None
    peg: float | None = None
    fcf: float | None = None
    fcf_yield: float | None = None
    debt_equity: float | None = None
    current_ratio: float | None = None
    roe: float | None = None
    revenue_growth: float | None = None
    profit_margin: float | None = None
    ev_ebitda: float | None = None

    dcf_value: float | None = None
    dcf_upside: float | None = None
    comp_discount: float | None = None      # vs. industry median forward P/E

    next_earnings: dt.date | None = None
    days_to_earnings: int | None = None
    news_score: float = 0.0
    news_titles: list[str] = field(default_factory=list)

    passed: bool = False
    score: float = 0.0
    fails: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def health_summary(self) -> str:
        bits = []
        if self.fcf_yield is not None:
            bits.append(f"FCF yield {self.fcf_yield*100:.1f}%")
        if self.debt_equity is not None:
            bits.append(f"D/E {self.debt_equity:.2f}x")
        if self.current_ratio is not None:
            bits.append(f"current ratio {self.current_ratio:.2f}")
        if self.roe is not None:
            bits.append(f"ROE {self.roe*100:.0f}%")
        return " · ".join(bits) if bits else "limited data"

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.next_earnings:
            d["next_earnings"] = self.next_earnings.isoformat()
        return d


# ---------------------------------------------------------------- DCF
def simple_dcf(fcf: float | None, shares: float | None, net_debt: float | None,
               growth: float | None) -> float | None:
    """
    A deliberately simple, fully-disclosed FCF DCF.

    Free cash flow is grown at a fading rate for DCF_HORIZON_YEARS, capped at
    DCF_MAX_INITIAL_GROWTH so that one euphoric revenue-growth print cannot
    manufacture a 400% upside. Terminal value uses Gordon growth. This is a
    sanity check on price, not a research-grade model — it exists to stop the
    engine buying something at 40x forward earnings with no cash generation.
    """
    fcf = _f(fcf)
    shares = _f(shares)
    if not fcf or not shares or fcf <= 0 or shares <= 0:
        return None
    if DCF_WACC <= DCF_TERMINAL_GROWTH:
        return None

    g = min(max(_f(growth) or 0.05, 0.0), DCF_MAX_INITIAL_GROWTH)
    pv, cash = 0.0, fcf
    for yr in range(1, DCF_HORIZON_YEARS + 1):
        cash *= (1.0 + g)
        pv += cash / ((1.0 + DCF_WACC) ** yr)
        g = max(g * DCF_FADE, DCF_TERMINAL_GROWTH)

    tv = cash * (1.0 + DCF_TERMINAL_GROWTH) / (DCF_WACC - DCF_TERMINAL_GROWTH)
    pv += tv / ((1.0 + DCF_WACC) ** DCF_HORIZON_YEARS)

    equity = pv - (_f(net_debt) or 0.0)
    return equity / shares if equity > 0 else None


# ---------------------------------------------------------------- sentiment
_POS = {"beat", "beats", "raises", "raised", "upgrade", "upgraded", "surge",
        "surges", "record", "wins", "win", "approval", "approved", "expands",
        "expansion", "partnership", "buyback", "outperform", "strong", "tops",
        "jumps", "rally", "breakthrough", "awarded", "contract", "guidance"}
_NEG = {"miss", "misses", "cuts", "cut", "downgrade", "downgraded", "plunge",
        "plunges", "lawsuit", "probe", "investigation", "recall", "warns",
        "warning", "layoff", "layoffs", "halt", "halted", "fraud", "delay",
        "delayed", "slumps", "falls", "weak", "bankruptcy", "sec", "subpoena"}


def score_headlines(titles: list[str]) -> float:
    """
    Keyword sentiment in [-1, 1].

    Deterministic on purpose. An LLM call would read these better, but this runs
    inside an unattended launchd job every 15 minutes — a dependency on an
    authenticated CLI turns a news blip into a silent failure. Keyword scoring
    degrades to 0.0 (neutral) instead.
    """
    if not titles:
        return 0.0
    pos = neg = 0
    for t in titles:
        words = {w.strip(".,:;!?'\"()").lower() for w in t.split()}
        pos += len(words & _POS)
        neg += len(words & _NEG)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


# ---------------------------------------------------------------- evaluation
def evaluate(symbol: str) -> Snapshot:
    """Pull everything for one ticker and apply the quality gate."""
    s = Snapshot(symbol=symbol)
    inf = data.info(symbol)
    if not inf:
        s.fails.append("no fundamental data")
        return s

    s.name = inf.get("shortName") or inf.get("longName") or symbol
    s.sector = inf.get("sector") or "Unknown"
    s.industry = inf.get("industry") or "Unknown"
    s.summary = (inf.get("longBusinessSummary") or "").strip()

    s.price = _f(inf.get("currentPrice")) or _f(inf.get("regularMarketPrice"))
    s.market_cap = _f(inf.get("marketCap"))
    s.trailing_pe = _f(inf.get("trailingPE"))
    s.forward_pe = _f(inf.get("forwardPE"))
    s.peg = _f(inf.get("trailingPegRatio"))
    s.fcf = _f(inf.get("freeCashflow"))
    s.debt_equity = debt_to_equity(inf.get("debtToEquity"))
    s.current_ratio = _f(inf.get("currentRatio"))
    s.roe = _f(inf.get("returnOnEquity"))
    s.revenue_growth = _f(inf.get("revenueGrowth"))
    s.profit_margin = _f(inf.get("profitMargins"))

    if s.fcf is None:
        s.fcf = fcf_from_statement(symbol)
        if s.fcf is not None:
            s.notes.append("FCF from cash-flow statement")

    ebitda = _f(inf.get("ebitda"))
    cash = _f(inf.get("totalCash")) or 0.0
    debt = _f(inf.get("totalDebt")) or 0.0
    net_debt = debt - cash
    if s.market_cap and ebitda and ebitda > 0:
        s.ev_ebitda = (s.market_cap + net_debt) / ebitda
    if s.market_cap and s.fcf:
        s.fcf_yield = s.fcf / s.market_cap

    # liquidity from real bars, not the info blob's stale average
    try:
        d = data.daily_bars(symbol, period="3mo")
        if not d.empty:
            s.avg_dollar_vol = float((d["Close"] * d["Volume"]).tail(20).mean())
            if s.price is None:
                s.price = float(d["Close"].iloc[-1])
    except Exception:
        pass

    # valuation
    s.dcf_value = simple_dcf(s.fcf, _f(inf.get("sharesOutstanding")),
                             net_debt, s.revenue_growth)
    if s.dcf_value and s.price:
        s.dcf_upside = s.dcf_value / s.price - 1.0

    # earnings blackout
    s.next_earnings = data.next_earnings(symbol)
    if s.next_earnings:
        s.days_to_earnings = (s.next_earnings - dt.date.today()).days

    # news
    hl = data.headlines(symbol)
    s.news_titles = [h["title"] for h in hl]
    s.news_score = score_headlines(s.news_titles)

    _apply_gate(s)
    return s


def _apply_gate(s: Snapshot) -> None:
    """Hard filters first, then a composite score over the survivors."""
    f = s.fails

    if s.price is None:
        f.append("no price")
    elif not (MIN_PRICE <= s.price <= MAX_PRICE):
        f.append(f"price {s.price:.2f} outside {MIN_PRICE:.0f}-{MAX_PRICE:.0f}")

    if s.market_cap is None or s.market_cap < MIN_MARKET_CAP:
        f.append(f"market cap < ${MIN_MARKET_CAP/1e9:.0f}B")

    if s.avg_dollar_vol is None or s.avg_dollar_vol < MIN_AVG_DOLLAR_VOL:
        f.append(f"avg $ volume < ${MIN_AVG_DOLLAR_VOL/1e6:.0f}M")

    is_fin = s.sector in FINANCIAL_SECTORS

    # MISSING IS NOT FAILING. Conflating "unknown" with "bad" turned this gate
    # into a data-completeness filter: 62 of 122 rejections were "free cash flow
    # not positive", and every large one of those had fcf = None rather than a
    # negative number. A gate that rejects on absent data selects for names
    # whose Yahoo payload happened to be complete, which is not a quality signal.
    if REQUIRE_POSITIVE_FCF and not is_fin:
        if s.fcf is None:
            s.notes.append("FCF unavailable — not penalised")
        elif s.fcf <= 0:
            f.append(f"free cash flow negative (${s.fcf/1e9:.2f}B)")

    if s.debt_equity is not None and not is_fin and s.debt_equity > MAX_DEBT_TO_EQUITY:
        f.append(f"D/E {s.debt_equity:.2f}x > {MAX_DEBT_TO_EQUITY:.2f}x")

    # A current ratio below 1.0 is normal for retail, restaurants and
    # subscription businesses that collect before they pay. Requiring >1.0
    # universally rejected 43 names on a metric that is business-model
    # dependent, so this is now a scoring input rather than a hard gate — kept
    # as a gate only where it signals genuine distress.
    if s.current_ratio is not None and not is_fin and s.current_ratio < 0.60:
        f.append(f"current ratio {s.current_ratio:.2f} < 0.60 (liquidity risk)")

    # Financials get their own test: leverage and cash-flow screens are
    # meaningless for them, so judge on returns and profitability instead.
    if is_fin:
        if s.roe is not None and s.roe <= 0:
            f.append(f"ROE {s.roe*100:.1f}% not positive")
        if s.profit_margin is not None and s.profit_margin <= 0:
            f.append(f"profit margin {s.profit_margin*100:.1f}% not positive")

    if s.forward_pe is None:
        f.append("no forward P/E")
    elif not (MIN_FORWARD_PE < s.forward_pe <= MAX_FORWARD_PE):
        f.append(f"forward P/E {s.forward_pe:.1f} outside "
                 f"{MIN_FORWARD_PE:.0f}-{MAX_FORWARD_PE:.0f}")

    if s.revenue_growth is not None and s.revenue_growth < MIN_REVENUE_GROWTH:
        f.append(f"revenue growth {s.revenue_growth*100:.1f}% negative")

    if s.profit_margin is not None and s.profit_margin < MIN_PROFIT_MARGIN:
        f.append(f"profit margin {s.profit_margin*100:.1f}% negative")

    if s.days_to_earnings is not None and 0 <= s.days_to_earnings <= EARNINGS_BLACKOUT_DAYS:
        f.append(f"earnings in {s.days_to_earnings}d")

    s.passed = not f
    if not s.passed:
        return

    # ---- composite score, each component bounded so none can dominate
    sc = 0.0
    if s.trailing_pe and s.forward_pe and s.forward_pe > 0:
        # forward below trailing = the market expects earnings to grow
        ratio = s.trailing_pe / s.forward_pe
        sc += min(max(ratio - 1.0, -0.5), 1.0) * 20.0
        s.notes.append(f"fwd P/E {s.forward_pe:.1f} vs trailing {s.trailing_pe:.1f}")
    if s.fcf_yield:
        sc += min(s.fcf_yield, 0.12) * 150.0
    if s.revenue_growth:
        sc += min(s.revenue_growth, 0.50) * 30.0
    if s.roe:
        sc += min(max(s.roe, 0.0), 0.60) * 20.0
    if s.profit_margin:
        sc += min(max(s.profit_margin, 0.0), 0.40) * 20.0
    if s.debt_equity is not None:
        sc += max(0.0, (MAX_DEBT_TO_EQUITY - s.debt_equity)) * 4.0
    if s.dcf_upside is not None:
        sc += min(max(s.dcf_upside, -0.5), 1.0) * 15.0
        s.notes.append(f"DCF ${s.dcf_value:.0f} ({s.dcf_upside*100:+.0f}%)")
    sc += s.news_score * 5.0

    s.score = round(sc, 2)


def add_comps(snaps: list[Snapshot]) -> None:
    """
    Peer comps: each name's forward P/E against its INDUSTRY median, falling
    back to the sector median when an industry has too few members to have a
    meaningful median.
    """
    def _median(vals):
        v = sorted(x for x in vals if x and x > 0)
        if not v:
            return None
        n = len(v)
        return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2

    by_ind: dict[str, list[float]] = {}
    by_sec: dict[str, list[float]] = {}
    for s in snaps:
        if s.forward_pe:
            by_ind.setdefault(s.industry, []).append(s.forward_pe)
            by_sec.setdefault(s.sector, []).append(s.forward_pe)

    for s in snaps:
        if not s.forward_pe:
            continue
        peers = by_ind.get(s.industry, [])
        med = _median(peers) if len(peers) >= 3 else _median(by_sec.get(s.sector, []))
        if med:
            s.comp_discount = s.forward_pe / med - 1.0
            s.notes.append(f"{s.comp_discount*100:+.0f}% vs peer median P/E {med:.1f}")
