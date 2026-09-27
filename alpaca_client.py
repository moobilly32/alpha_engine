"""
Alpaca implementation of the Broker interface — used for paper trading.

Auth comes from the environment (loaded from .env by execution_engine, but this
module also loads it so it can be exercised standalone):

    ALPACA_API_KEY      key id
    ALPACA_SECRET_KEY   secret
    ALPACA_BASE_URL     https://paper-api.alpaca.markets  (paper)
                        https://api.alpaca.markets        (live)

`alpaca-py` takes a `paper` boolean rather than a URL. We keep ALPACA_BASE_URL
as the source of truth — it is what everyone recognises — and derive `paper`
from it, while also passing it through as `url_override` so a non-standard
endpoint (a mock, a proxy) still works.

QUIRKS HANDLED HERE
  * Fractional shares are allowed on MARKET/DAY orders only. Limit and stop
    orders are whole-share; `buy_limit` / `place_stop_loss` round down and
    refuse a sub-1-share order rather than let the SDK 422.
  * Prices are rounded to the penny (sub-penny limit/stop prices are rejected
    for anything >= $1).
  * `APIError` carries an HTTP status; it is mapped onto the shared taxonomy so
    429s back off and auth failures stop the engine.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from broker_interface import (
    AuthError, Broker, BrokerError, BrokerPosition, OrderRejected,
    OrderStatusInfo, RateLimitError, TransientError, log,
)
from config import BASE

load_dotenv(BASE / ".env")

_PAPER_HOSTS = ("paper-api.alpaca.markets",)


class AlpacaBroker(Broker):
    name = "alpaca"

    def __init__(self, *, api_key: str | None = None, secret_key: str | None = None,
                 base_url: str | None = None, paper: bool | None = None) -> None:
        key = api_key or os.getenv("ALPACA_API_KEY")
        secret = secret_key or os.getenv("ALPACA_SECRET_KEY")
        url = base_url or os.getenv("ALPACA_BASE_URL") \
            or "https://paper-api.alpaca.markets"
        if not key or not secret:
            raise AuthError("ALPACA_API_KEY / ALPACA_SECRET_KEY are not set")

        self.base_url = url.rstrip("/")
        self.is_paper = paper if paper is not None else \
            any(h in self.base_url for h in _PAPER_HOSTS)

        # Import inside __init__ so a Robinhood-only deployment need not have
        # alpaca-py installed to import this package.
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient
        except ImportError as e:  # pragma: no cover
            raise BrokerError(
                "alpaca-py is not installed — `pip install alpaca-py`") from e

        try:
            self._trading = TradingClient(
                key, secret, paper=self.is_paper, url_override=self.base_url)
            self._data = StockHistoricalDataClient(key, secret)
            acct = self._resilient("authenticate",
                                   lambda: self._trading.get_account())
        except BrokerError:
            raise
        except Exception as raw:  # noqa: BLE001
            raise self._classify(raw) from raw

        log(f"alpaca: connected {'paper' if self.is_paper else 'LIVE'} "
            f"acct {getattr(acct, 'account_number', '?')} "
            f"status={getattr(acct, 'status', '?')} "
            f"buying_power=${float(getattr(acct, 'buying_power', 0) or 0):,.2f}")

    # -- error mapping --------------------------------------------------------
    def _classify(self, exc: Exception) -> BrokerError:
        try:
            from alpaca.common.exceptions import APIError
        except ImportError:  # pragma: no cover
            APIError = ()  # type: ignore

        if APIError and isinstance(exc, APIError):
            code = getattr(exc, "status_code", None)
            msg = str(exc)
            if code == 429 or "too many requests" in msg.lower():
                return RateLimitError(f"alpaca 429: {msg[:200]}")
            if code in (401, 403):
                return AuthError(f"alpaca {code}: {msg[:200]}")
            if code is not None and 500 <= code < 600:
                return TransientError(f"alpaca {code}: {msg[:200]}")
            if code in (400, 422, 403):
                return OrderRejected(f"alpaca {code}: {msg[:200]}")
            return BrokerError(f"alpaca APIError {code}: {msg[:200]}")
        return super()._classify(exc)

    # -- prices ------------------------------------------------------------
    def get_price(self, ticker: str) -> float:
        from alpaca.data.requests import (
            StockLatestQuoteRequest, StockLatestTradeRequest,
        )
        sym = ticker.upper()

        def _do() -> float:
            try:
                tr = self._data.get_stock_latest_trade(
                    StockLatestTradeRequest(symbol_or_symbols=sym))
                px = float(tr[sym].price)
                if px > 0:
                    return round(px, 4)
            except Exception as raw:  # noqa: BLE001
                exc = self._classify(raw)
                if isinstance(exc, (AuthError, RateLimitError)):
                    raise exc
                log(f"alpaca: latest_trade({sym}) fell back to quote ({exc})")
            # Fall back to the quote midpoint if the trade feed is empty.
            q = self._data.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols=sym))[sym]
            bid, ask = float(q.bid_price or 0), float(q.ask_price or 0)
            mid = (bid + ask) / 2 if bid and ask else (bid or ask)
            if mid and mid > 0:
                return round(mid, 4)
            raise TransientError(f"alpaca: no quote for {sym}")

        return self._resilient(f"get_price({sym})", _do)

    # -- orders ----------------------------------------------------------
    def buy_limit(self, ticker: str, shares: float, price: float) -> str:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest
        sym = ticker.upper()
        qty = self._whole_shares(shares, f"alpaca buy_limit {sym}")
        limit = round(float(price), 2)

        def _do() -> str:
            req = LimitOrderRequest(
                symbol=sym, qty=qty, side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY, limit_price=limit)
            o = self._trading.submit_order(order_data=req)
            log(f"alpaca: BUY LIMIT {qty} {sym} @ {limit} -> {o.id} ({o.status})")
            return str(o.id)

        return self._resilient(f"buy_limit({sym})", _do)

    def buy_market(self, ticker: str, shares: float) -> str:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        sym = ticker.upper()
        # Market orders MAY be fractional on Alpaca; same 4dp convention as
        # sell_market() below, for symmetry.
        qty = round(float(shares), 4)
        if qty <= 0:
            raise OrderRejected(f"alpaca buy_market {sym}: qty {shares} <= 0")

        def _do() -> str:
            req = MarketOrderRequest(
                symbol=sym, qty=qty, side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY)
            o = self._trading.submit_order(order_data=req)
            log(f"alpaca: BUY MKT {qty} {sym} -> {o.id} ({o.status})")
            return str(o.id)

        return self._resilient(f"buy_market({sym})", _do)

    def sell_market(self, ticker: str, shares: float) -> str:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        sym = ticker.upper()
        # Market orders MAY be fractional on Alpaca; keep up to 4dp.
        qty = round(float(shares), 4)
        if qty <= 0:
            raise OrderRejected(f"alpaca sell_market {sym}: qty {shares} <= 0")

        def _do() -> str:
            req = MarketOrderRequest(
                symbol=sym, qty=qty, side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY)
            o = self._trading.submit_order(order_data=req)
            log(f"alpaca: SELL MKT {qty} {sym} -> {o.id} ({o.status})")
            return str(o.id)

        return self._resilient(f"sell_market({sym})", _do)

    def place_stop_loss(self, ticker: str, shares: float, stop_price: float) -> str:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import StopOrderRequest
        sym = ticker.upper()
        qty = self._whole_shares(shares, f"alpaca stop {sym}")
        stop = round(float(stop_price), 2)

        def _do() -> str:
            req = StopOrderRequest(
                symbol=sym, qty=qty, side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC, stop_price=stop)
            o = self._trading.submit_order(order_data=req)
            log(f"alpaca: SELL STOP {qty} {sym} @ {stop} -> {o.id} ({o.status})")
            return str(o.id)

        return self._resilient(f"place_stop_loss({sym})", _do)

    def cancel_order(self, order_id: str) -> bool:
        def _do() -> bool:
            try:
                self._trading.cancel_order_by_id(order_id)
                log(f"alpaca: cancelled {order_id}")
                return True
            except Exception as raw:  # noqa: BLE001
                exc = self._classify(raw)
                # 404 (unknown) or 422 (already filled/cancelled) both mean the
                # order is not working any more, which is what the caller wants.
                if isinstance(exc, (OrderRejected, BrokerError)) and \
                        not isinstance(exc, TransientError):
                    log(f"alpaca: cancel {order_id} — already inactive ({exc})")
                    return True
                raise exc

        return self._resilient(f"cancel_order({order_id})", _do)

    # -- introspection ---------------------------------------------------
    def list_positions(self) -> list[BrokerPosition]:
        def _do() -> list[BrokerPosition]:
            positions = self._trading.get_all_positions()
            out: list[BrokerPosition] = []
            for p in positions:
                qty = float(p.qty)
                avg_entry = float(p.avg_entry_price)
                current = (float(p.current_price)
                          if p.current_price is not None else avg_entry)
                pnl = (float(p.unrealized_pl)
                      if p.unrealized_pl is not None
                      else (current - avg_entry) * qty)
                out.append(BrokerPosition(
                    symbol=p.symbol, qty=qty,
                    avg_entry_price=round(avg_entry, 4),
                    current_price=round(current, 4),
                    unrealized_pnl=round(pnl, 2)))
            return out

        return self._resilient("list_positions", _do)

    def get_equity(self) -> float:
        def _do() -> float:
            acct = self._trading.get_account()
            eq = float(acct.equity)
            if eq <= 0:
                raise TransientError(f"alpaca: non-positive equity ({eq})")
            return eq

        return self._resilient("get_equity", _do)

    def get_buying_power(self) -> float:
        def _do() -> float:
            acct = self._trading.get_account()
            return max(0.0, float(acct.buying_power))

        return self._resilient("get_buying_power", _do)

    def get_order_status(self, order_id: str) -> OrderStatusInfo:
        from alpaca.trading.enums import OrderStatus as _AlpacaOrderStatus

        _DEAD = {
            _AlpacaOrderStatus.CANCELED, _AlpacaOrderStatus.EXPIRED,
            _AlpacaOrderStatus.DONE_FOR_DAY, _AlpacaOrderStatus.STOPPED,
            _AlpacaOrderStatus.SUSPENDED,
        }

        def _do() -> OrderStatusInfo:
            o = self._trading.get_order_by_id(order_id)
            filled_qty = float(o.filled_qty or 0)
            filled_avg = (float(o.filled_avg_price)
                         if o.filled_avg_price is not None else None)

            if o.status == _AlpacaOrderStatus.FILLED:
                status = "filled"
            elif o.status == _AlpacaOrderStatus.PARTIALLY_FILLED:
                status = "partially_filled"
            elif o.status == _AlpacaOrderStatus.REJECTED:
                status = "rejected"
            elif o.status in _DEAD:
                status = "canceled"
            elif filled_qty > 0:
                # Defensive: any status this mapping doesn't special-case
                # but that already has real shares filled is at least a
                # partial fill, never "new".
                status = "partially_filled"
            else:
                status = "new"

            return OrderStatusInfo(order_id=order_id, status=status,
                                   filled_qty=filled_qty,
                                   filled_avg_price=filled_avg)

        return self._resilient(f"get_order_status({order_id})", _do)


if __name__ == "__main__":  # pragma: no cover
    b = AlpacaBroker()
    print("SPY:", b.get_price("SPY"))
