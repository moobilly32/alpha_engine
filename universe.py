"""
Candidate pool the screener draws from.

The screener is "dynamic" in that it re-scores and re-ranks every morning, but
it still needs a pool to score. Two sources, in order:

1. S&P 500 constituents scraped live from Wikipedia.
2. This built-in seed pool, used when the scrape fails.

The seed pool is deliberately wider than the S&P in a few places (mid-cap
infrastructure, uranium, grid) because the brief asks for Infrastructure as a
first-class sector and the index under-represents it.
"""

from __future__ import annotations

import data

SEED = [
    # Technology — semis, software, hardware, networking
    "AAPL", "MSFT", "NVDA", "AVGO", "AMD", "MU", "INTC", "QCOM", "TXN", "ADI",
    "LRCX", "AMAT", "KLAC", "MRVL", "NXPI", "ON", "SNPS", "CDNS", "ANET", "CSCO",
    "ORCL", "CRM", "ADBE", "NOW", "INTU", "PANW", "FTNT", "CRWD", "ZS", "DDOG",
    "SNOW", "MDB", "NET", "WDAY", "TEAM", "HPQ", "DELL", "STX", "WDC", "SMCI",

    # Communication services / internet
    "GOOGL", "META", "NFLX", "DIS", "TMUS", "TTD", "RBLX", "SPOT", "EA", "TTWO",

    # Consumer
    "AMZN", "TSLA", "HD", "LOW", "MCD", "SBUX", "NKE", "TJX", "COST", "WMT",
    "TGT", "DG", "ORLY", "AZO", "CMG", "YUM", "PG", "KO", "PEP", "MDLZ",

    # Healthcare / pharma / biotech / devices
    "LLY", "JNJ", "MRK", "PFE", "ABBV", "AMGN", "GILD", "BMY", "VRTX", "REGN",
    "BIIB", "MRNA", "ZTS", "TMO", "DHR", "ABT", "SYK", "BSX", "ISRG", "MDT",
    "UNH", "ELV", "CI", "HCA", "MCK",

    # Energy — integrated, E&P, services, midstream, refining
    "XOM", "CVX", "COP", "EOG", "PXD", "DVN", "FANG", "OXY", "HES", "MRO",
    "SLB", "HAL", "BKR", "PSX", "VLO", "MPC", "KMI", "WMB", "OKE", "LNG",

    # Infrastructure — engineering, construction, aggregates, rail, power, grid
    "CAT", "DE", "HON", "GE", "ETN", "EMR", "PH", "ROK", "PWR", "J",
    "ACM", "MTZ", "FLR", "VMC", "MLM", "NUE", "STLD", "URI", "FAST", "GWW",
    "UNP", "CSX", "NSC", "ODFL", "CP", "NEE", "DUK", "SO", "AEP", "SRE",
    "VST", "CEG", "PCG", "EXC", "XEL",

    # Financials
    "JPM", "BAC", "GS", "MS", "WFC", "C", "SCHW", "BLK", "SPGI", "ICE",
    "CME", "AXP", "V", "MA", "PYPL", "COF", "USB", "PNC", "TFC", "MET",

    # Materials / industrials gases / chemicals
    "LIN", "APD", "SHW", "ECL", "DOW", "DD", "PPG", "ALB", "FCX", "NEM",

    # Aerospace & defence
    "BA", "LMT", "RTX", "NOC", "GD", "LHX", "TDG", "HWM", "AXON",

    # Real assets / REITs with infrastructure exposure
    "AMT", "CCI", "EQIX", "DLR", "PLD", "SPG",
]


def candidates(use_index: bool = True, limit: int | None = None) -> list[str]:
    """
    Merged, de-duplicated candidate pool.

    Order matters: seed names come first so that if `limit` truncates the pool,
    the hand-curated sector coverage survives and only the index tail is cut.
    """
    pool: list[str] = []
    seen: set[str] = set()

    for sym in SEED:
        if sym not in seen:
            seen.add(sym)
            pool.append(sym)

    if use_index:
        for sym in data.sp500_symbols():
            if sym and sym not in seen:
                seen.add(sym)
                pool.append(sym)

    return pool[:limit] if limit else pool
