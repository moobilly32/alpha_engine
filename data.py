"""
Market data access layer.

Everything that touches yfinance goes through here, for three reasons:

1. Caching. The screener asks ~140 tickers for fundamentals; the backtest asks
   30 tickers for 60 days of 5-minute bars. Without a cache you re-download on
   every iteration and Yahoo starts rate-limiting.
2. Shape normalisation. yfinance returns a MultiIndex column frame for a single
   ticker under some versions and a flat one under others; `.news` changed
   structure between releases. Normalise once, here.
3. Failure containment. A single ticker that 404s must not abort a 140-name
   screen.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import pickle
import time
import warnings
from typing import Any

import logging

import pandas as pd
import requests
import yfinance as yf

from config import CACHE, TZ
from calendar_util import NY

warnings.filterwarnings("ignore", category=FutureWarning)

# Yahoo intermittently answers a fundamentals request with a 401 "Invalid Crumb"
# and yfinance logs it at ERROR before transparently retrying. On a 140-name
# screen that is pages of noise for a condition we already handle. Silence the
# library's own logger; genuine failures still surface as empty results, which
# the callers check.
for _name in ("yfinance", "yfinance.data", "yfinance.utils", "peewee"):
    logging.getLogger(_name).setLevel(logging.CRITICAL)

_MEM: dict[str, Any] = {}


# ---------------------------------------------------------------- cache
def _cache_path(kind: str, key: str):
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
    return CACHE / f"{kind}__{safe}.pkl"


def _cached(kind: str, key: str, ttl_sec: float, producer):
    """Two-tier cache: process memory, then disk with a TTL."""
    mk = f"{kind}:{key}"
    if mk in _MEM:
        return _MEM[mk]

    p = _cache_path(kind, key)
    if p.exists() and (time.time() - p.stat().st_mtime) < ttl_sec:
        try:
            with open(p, "rb") as fh:
                val = pickle.load(fh)
            _MEM[mk] = val
            return val
        except Exception:
            pass  # corrupt cache entry — fall through and refetch

    val = producer()
    try:
        with open(p, "wb") as fh:
            pickle.dump(val, fh)
    except Exception:
        pass
    _MEM[mk] = val
    return val


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


# ---------------------------------------------------------------- bars
def daily_bars(symbol: str, period: str = "2y", ttl: float = 6 * 3600) -> pd.DataFrame:
    """Daily OHLCV, NY-localised, most recent bar last."""

    def _go():
        df = yf.download(symbol, period=period, interval="1d", auto_adjust=False,
                         progress=False, threads=False)
        df = _flatten(df)
        if df.empty:
            return df
        if df.index.tz is None:
            df.index = df.index.tz_localize(NY)
        else:
            df.index = df.index.tz_convert(NY)
        return df.dropna(subset=["Close"])

    return _cached("daily", f"{symbol}_{period}", ttl, _go)


def intraday_5m(symbol: str, period: str = "60d", ttl: float = 900) -> pd.DataFrame:
    """
    5-minute bars WITH pre/post market, NY-localised.

    Premarket bars are not optional here: without the 04:00-09:30 window the
    premarket high is undefined and the intraday leg of the signal can never
    fire. The strategy would silently take zero trades rather than error, which
    is the worst possible failure mode.

    Yahoo caps 5-minute history at 60 calendar days. That cap is the reason the
    backtest window is 59 days.
    """

    def _go():
        df = yf.download(symbol, period=period, interval="5m", prepost=True,
                         auto_adjust=False, progress=False, threads=False)
        df = _flatten(df)
        if df.empty:
            return df
        df.index = df.index.tz_convert(NY)
        return df.dropna(subset=["Close"])

    return _cached("i5m", f"{symbol}_{period}", ttl, _go)


def last_price(symbol: str) -> float | None:
    """Most recent traded price, extended hours included."""
    try:
        df = intraday_5m(symbol, period="5d", ttl=60)
        if not df.empty:
            return float(df["Close"].iloc[-1])
    except Exception:
        pass
    try:
        fi = yf.Ticker(symbol).fast_info
        v = fi.get("lastPrice") or fi.get("last_price")
        return float(v) if v else None
    except Exception:
        return None


# ---------------------------------------------------------------- fundamentals
def info(symbol: str, ttl: float = 12 * 3600, attempts: int = 3) -> dict:
    """
    The `.info` blob, with a bounded retry.

    Yahoo's crumb/cookie handshake fails often enough under concurrency that a
    single attempt drops roughly one name in eight — and a dropped name looks
    identical to a name that failed the quality gate, which would quietly bias
    the watchlist toward whatever happened to answer first.
    """

    def _go():
        best: dict = {}
        for i in range(attempts):
            try:
                d = dict(yf.Ticker(symbol).info or {})
                # A TRUNCATED payload is the dangerous case, not an empty one.
                # Yahoo sometimes answers with ~90 keys instead of ~185: price
                # and marketCap are present, but sector, freeCashflow and
                # currentRatio are missing. A gate that treats missing as bad
                # then rejects WMT, V and XOM for "no free cash flow". Requiring
                # `sector` is a cheap completeness check; keep the richest
                # payload seen in case every attempt comes back thin.
                if len(d) > len(best):
                    best = d
                if d.get("sector") and d.get("marketCap"):
                    return d
            except Exception:
                pass
            time.sleep(0.4 * (i + 1))
        return best

    return _cached("info", symbol, ttl, _go)


def financials(symbol: str, ttl: float = 24 * 3600) -> dict:
    """Balance-sheet / income-statement / cash-flow frames, best effort."""

    def _go():
        out = {}
        tk = yf.Ticker(symbol)
        for name, attr in (("balance", "balance_sheet"),
                           ("income", "income_stmt"),
                           ("cashflow", "cashflow")):
            try:
                df = getattr(tk, attr)
                out[name] = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
            except Exception:
                out[name] = pd.DataFrame()
        return out

    return _cached("fin", symbol, ttl, _go)


def next_earnings(symbol: str, ttl: float = 12 * 3600) -> dt.date | None:
    """Next earnings date, or None if unknown."""

    def _go():
        try:
            cal = yf.Ticker(symbol).calendar
        except Exception:
            return None
        vals = []
        if isinstance(cal, dict):
            vals = cal.get("Earnings Date") or []
            if not isinstance(vals, (list, tuple)):
                vals = [vals]
        elif isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
            vals = list(cal.loc["Earnings Date"].values)
        out = []
        for v in vals:
            try:
                out.append(pd.Timestamp(v).date())
            except Exception:
                continue
        return min(out) if out else None

    return _cached("earn", symbol, ttl, _go)


def headlines(symbol: str, limit: int = 8, ttl: float = 3 * 3600) -> list[dict]:
    """
    Recent headlines, normalised to {title, publisher, link, ts}.

    yfinance has moved this payload around between releases — older versions put
    fields at the top level, newer ones nest them under 'content'. Handle both
    rather than pinning a version.
    """

    def _go():
        try:
            raw = yf.Ticker(symbol).news or []
        except Exception:
            return []
        out = []
        for item in raw[:limit]:
            c = item.get("content", item) if isinstance(item, dict) else {}
            title = c.get("title") or item.get("title")
            if not title:
                continue
            prov = c.get("provider")
            publisher = (prov.get("displayName") if isinstance(prov, dict) else None) \
                or item.get("publisher") or "—"
            link = item.get("link") or ""
            if not link:
                cu = c.get("canonicalUrl")
                if isinstance(cu, dict):
                    link = cu.get("url", "")
            ts = c.get("pubDate") or item.get("providerPublishTime")
            out.append({"title": str(title), "publisher": str(publisher),
                        "link": link, "ts": ts})
        return out

    return _cached("news", symbol, ttl, _go)


# ---------------------------------------------------------------- universe
def sp500_symbols(ttl: float = 7 * 24 * 3600) -> list[str]:
    """
    S&P 500 constituents from Wikipedia. Returns [] on any failure — the caller
    falls back to the built-in seed pool, so a Wikipedia layout change degrades
    the universe rather than breaking the run.

    Fetches the page with `requests` (a real browser User-Agent) and hands
    `pd.read_html` the HTML text rather than the URL — `read_html` sends no
    User-Agent of its own, which Wikipedia's edge now answers with a 403
    (confirmed: this silently returned [] on every call until fixed).
    """

    def _go():
        try:
            resp = requests.get(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X "
                                       "10_15_7) AppleWebKit/537.36"},
                timeout=15)
            resp.raise_for_status()
            tables = pd.read_html(io.StringIO(resp.text))
            for t in tables:
                if "Symbol" in t.columns:
                    return [str(s).replace(".", "-").strip()
                            for s in t["Symbol"].tolist()]
        except Exception:
            pass
        return []

    return _cached("universe", "sp500", ttl, _go)


def earnings_dates(symbol: str, ttl: float = 24 * 3600) -> list:
    """
    HISTORICAL and future earnings dates.

    `next_earnings()` above serves only the upcoming report, which is all the
    live screener needs. A backtest that wants to avoid holding through a print
    needs the dates that fall INSIDE the test window, and those are historical.
    Returns [] on failure so a missing calendar degrades to "no earnings known"
    rather than aborting the run.
    """

    def _go():
        try:
            ed = yf.Ticker(symbol).get_earnings_dates(limit=32)
            if ed is None or ed.empty:
                return []
            return sorted({pd.Timestamp(x).date() for x in pd.DatetimeIndex(ed.index)})
        except Exception:
            return []

    return _cached("erndates", symbol, ttl, _go)
