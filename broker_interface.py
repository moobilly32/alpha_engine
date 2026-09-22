"""
Broker abstraction — the only surface the execution engine is allowed to touch.

WHY THIS EXISTS
    The strategy, the state file and the position-management logic must not know
    or care whether an order is going to Alpaca paper or a live Robinhood
    account. Everything broker-specific — SDK objects, auth, quirks of each
    order API, error taxonomies — is confined to a concrete `Broker` subclass.
    `execution_engine.py` calls five methods and nothing else.

THE CONTRACT
    get_price(ticker)                      -> float           (last trade / mark)
    buy_limit(ticker, shares, price)       -> str  order_id
    sell_market(ticker, shares)            -> str  order_id
    place_stop_loss(ticker, shares, stop)  -> str  order_id   (resting SELL stop)
    cancel_order(order_id)                 -> bool

    Order ids are opaque strings. A method that cannot complete raises one of
    the exceptions below; it never returns a sentinel the caller has to
    remember to check.

    list_positions()                       -> list[BrokerPosition]

    A sixth method, added for read-only introspection (e.g.
    `execution_engine.py --broker-positions`). It queries the broker directly
    for whatever it actually holds. The trading state machine still infers
    STOP/scale/target fills on an OPEN position from price alone (a
    resting stop/limit fills when price reaches it — the same "swing
    backtest" model this project uses elsewhere); this method's main
    consumer there is a human looking at the book, not the automation.

    get_equity()                           -> float

    A seventh method: the account's real, current equity (cash + long market
    value) — NOT buying_power, which includes margin leverage and would
    over-risk every position by the account's leverage multiple if used for
    sizing. Unlike list_positions(), this ONE IS live-path-critical:
    intraday.py calls it every run to size new entries off the broker's
    actual balance instead of the static ACCOUNT_SIZE fallback in config.py —
    that fallback exists only for when this call fails.

    get_order_status(order_id)             -> OrderStatusInfo

    An eighth method, live-path-critical on the ENTRY side:
    execution_engine.py's PENDING_ENTRY -> OPEN transition used to infer a
    fill from price alone ("price crossed the limit, so it must have
    filled" — the LIMITATION this module's docstring used to flag). That
    let a limit order that never actually filled get treated as a real
    open position, exactly the phantom-position bug found and fixed live
    on the hybrid_engine.py side of this project. This method queries the
    broker directly instead. `status` is normalized across venues into a
    small vocabulary so execution_engine.py never has to branch on which
    broker is live: "new" (still working, no fill yet), "partially_filled",
    "filled", "canceled" (canceled/expired/done-for-day/stopped/
    suspended — dead, no more fills coming), "rejected" (the venue refused
    it outright), or "unknown" (a raw status this mapping doesn't
    recognize — callers must treat this as "still working," never as a
    confirmed fill or a confirmed dead order).

    buy_market(ticker, shares)             -> str  order_id

    A ninth method, live-path-critical for hybrid_engine.py's fractional
    share position sizing: buy_limit() is whole-share only on BOTH venues
    (see _whole_shares() below) because neither accepts a fractional
    quantity on a LIMIT order — only on a MARKET/notional order. This
    method is that market-order entry path. execution_engine.py does NOT
    use this — it still enters via buy_limit(), unchanged.

ERROR TAXONOMY  (all subclass BrokerError)
    AuthError        credentials/session bad — do not retry, the engine stops.
    OrderRejected    the venue refused this specific order — do not retry.
    TransientError   network timeout, connection reset, 5xx — safe to retry.
    RateLimitError   429 / "throttled" — a TransientError with a longer backoff.

    `_resilient()` retries TransientError/RateLimitError with exponential
    backoff + jitter and re-raises everything else immediately. Concrete
    brokers only implement `_classify()` to map their SDK's exceptions onto
    this taxonomy; the retry loop is shared.
"""

from __future__ import annotations

import abc
import random
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

import requests

from calendar_util import now_ny
from config import LOGS

_T = TypeVar("_T")

_LOG_FILE = LOGS / "execution_engine.log"


@dataclass
class BrokerPosition:
    """
    One position exactly as the broker reports it right now — not the local
    open_positions.json book, which can drift from reality (see the
    LIMITATION note on execution_engine.py). For introspection only.
    """
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float


@dataclass
class OrderStatusInfo:
    """
    Broker-agnostic snapshot of one order's live status, returned by
    Broker.get_order_status(). See that method's docstring for the
    normalized `status` vocabulary.
    """
    order_id: str
    status: str                     # "new" | "partially_filled" | "filled" |
                                    # "canceled" | "rejected" | "unknown"
    filled_qty: float
    filled_avg_price: float | None


def log(msg: str) -> None:
    """Timestamped line to stdout and logs/execution_engine.log. Never raises."""
    line = f"{now_ny():%Y-%m-%d %H:%M:%S %Z}  {msg}"
    print(line, flush=True)
    try:
        with open(_LOG_FILE, "a") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------- exceptions
class BrokerError(Exception):
    """Base for every broker failure the engine might see."""


class AuthError(BrokerError):
    """Credentials or session are invalid. Not retryable."""


class OrderRejected(BrokerError):
    """The venue refused this specific order (bad symbol, size, buying power)."""


class TransientError(BrokerError):
    """Network/5xx/timeout. Retrying the same call is reasonable."""


class RateLimitError(TransientError):
    """HTTP 429 or an API 'request throttled' response. Back off harder."""


# ---------------------------------------------------------------- base class
class Broker(abc.ABC):
    """
    Abstract broker. Subclasses implement the five order methods plus
    `_classify`. Everything else here is shared plumbing.
    """

    name: str = "broker"
    is_paper: bool = False

    # -- abstract surface -------------------------------------------------
    @abc.abstractmethod
    def get_price(self, ticker: str) -> float:
        """Most recent trade price for `ticker`. Raises on no quote."""

    @abc.abstractmethod
    def buy_limit(self, ticker: str, shares: float, price: float) -> str:
        """Submit a BUY limit order, DAY. Whole shares only — both venues
        reject fractional quantities on limit orders (see _whole_shares()
        below). Returns the broker order id."""

    @abc.abstractmethod
    def buy_market(self, ticker: str, shares: float) -> str:
        """
        Submit a BUY market order, DAY. Fractional shares ALLOWED (both
        venues support fractional quantities on market/notional orders,
        not on limit/stop orders — this is the entry path fractional-share
        position sizing requires; see hybrid_engine.py's execute_signals()).
        Returns the broker order id.
        """

    @abc.abstractmethod
    def sell_market(self, ticker: str, shares: float) -> str:
        """Submit a SELL market order. Fractional shares allowed. Returns
        the broker order id."""

    @abc.abstractmethod
    def place_stop_loss(self, ticker: str, shares: float, stop_price: float) -> str:
        """Submit a resting SELL stop order, GTC. Returns the broker order id."""

    @abc.abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a working order. Returns True if the order is no longer working
        (cancelled now, or already filled/cancelled). Raises only on a
        transient failure that is worth retrying.
        """

    @abc.abstractmethod
    def list_positions(self) -> list[BrokerPosition]:
        """
        Every position currently held at the broker, queried live. Read-only
        introspection — the trading state machine never calls this.
        """

    @abc.abstractmethod
    def get_equity(self) -> float:
        """
        Real, current account equity (cash + long market value) — NOT
        buying_power/margin. This IS on the live entry path: intraday.py
        sizes new positions off this value every run.
        """

    @abc.abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatusInfo:
        """
        Live status of a previously-submitted order, queried directly from
        the broker — never inferred from price. This IS on the live entry
        path: execution_engine.py's _handle_pending() calls this to confirm
        a REAL fill before ever marking a position OPEN. See the module
        docstring above (the get_order_status section) for the normalized
        `status` vocabulary and the incident this method exists to prevent.
        """

    # -- error mapping (override in subclasses, then call super) ---------
    def _classify(self, exc: Exception) -> BrokerError:
        """
        Map an arbitrary SDK/HTTP exception onto the BrokerError taxonomy.
        Subclasses handle their own SDK types first, then delegate here for the
        `requests`-level cases both APIs share.
        """
        if isinstance(exc, BrokerError):
            return exc
        if isinstance(exc, (requests.exceptions.Timeout,
                            requests.exceptions.ConnectionError,
                            requests.exceptions.ChunkedEncodingError)):
            return TransientError(f"network: {type(exc).__name__}")
        if isinstance(exc, requests.exceptions.HTTPError):
            code = getattr(getattr(exc, "response", None), "status_code", None)
            if code == 429:
                return RateLimitError("HTTP 429 rate limited")
            if code in (401, 403):
                return AuthError(f"HTTP {code}")
            if code is not None and 500 <= code < 600:
                return TransientError(f"HTTP {code}")
            return BrokerError(f"HTTP {code}")
        if isinstance(exc, requests.exceptions.RequestException):
            return TransientError(f"request: {type(exc).__name__}")
        return BrokerError(f"{type(exc).__name__}: {exc}")

    # -- shared retry loop ---------------------------------------------------
    def _resilient(self, what: str, fn: Callable[[], _T], *,
                   tries: int = 4, base_delay: float = 1.0,
                   max_delay: float = 30.0) -> _T:
        """
        Call `fn`, retrying only TransientError/RateLimitError with exponential
        backoff and jitter. AuthError and OrderRejected propagate on the first
        hit — retrying them just burns time or double-submits.
        """
        last: BrokerError | None = None
        for attempt in range(1, tries + 1):
            try:
                return fn()
            except Exception as raw:                       # noqa: BLE001
                exc = self._classify(raw)
                last = exc
                if isinstance(exc, (AuthError, OrderRejected)):
                    raise exc
                if not isinstance(exc, TransientError):
                    raise exc
                if attempt == tries:
                    break
                # Rate limits get a fatter backoff than plain network blips.
                factor = 4.0 if isinstance(exc, RateLimitError) else 1.0
                delay = min(max_delay,
                            base_delay * factor * (2 ** (attempt - 1)))
                delay += random.uniform(0.0, base_delay)
                log(f"{self.name}: {what} failed "
                    f"({type(exc).__name__}: {exc}); "
                    f"retry {attempt}/{tries - 1} in {delay:.1f}s")
                time.sleep(delay)
        assert last is not None
        log(f"{self.name}: {what} gave up after {tries} attempts — {last}")
        raise last

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _whole_shares(shares: float, ctx: str) -> int:
        """
        Both venues reject fractional quantities on limit and stop orders.
        Round down, and refuse rather than silently send a 0-share order.
        """
        q = int(shares)
        if q <= 0:
            raise OrderRejected(
                f"{ctx}: {shares} shares rounds to 0 — venue requires whole "
                f"shares for limit/stop orders")
        if q != shares:
            log(f"{ctx}: {shares} -> {q} shares (venue is whole-share only here)")
        return q

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        tag = "paper" if self.is_paper else "LIVE"
        return f"<{type(self).__name__} {self.name} {tag}>"
