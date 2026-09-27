"""
Robinhood implementation of the Broker interface — used for live trading.

Auth from the environment (see .env.template):

    RH_EMAIL        account email
    RH_PASSWORD     account password
    RH_MFA_SECRET   TOTP seed shown when you enable an authenticator app on the
                    Robinhood account. REQUIRED for unattended login — Robinhood
                    mandates MFA and robin_stocks will otherwise block on an
                    interactive input() prompt. Needs `pyotp` installed.
    RH_MFA_CODE     alternative: a single current 6-digit code (expires fast;
                    only useful for a one-off manual run).
    RH_ACCOUNT_NUMBER
                    optional. Robinhood account number (visible in the app under
                    Account -> Settings, or via rh.load_account_profile()) to
                    trade when the login has more than one brokerage account
                    (e.g. individual + IRA). Every order call is pinned to this
                    account; leave blank to let robin_stocks fall back to
                    whichever account it treats as default, which is only safe
                    when the login has exactly one.

SECURITY NOTE
    A live brokerage password in a plaintext .env is the weakest secret in this
    repo — everything else (the Discord webhook) lives in the macOS Keychain.
    Lock the file down (`chmod 600 .env`), keep it out of git (.gitignore), and
    prefer a dedicated Keychain entry if you can. robin_stocks caches a session
    token under ~/.tokens/ after the first login, so the password is read at
    most once per token lifetime.

QUIRKS HANDLED HERE
  * robin_stocks returns dicts, not exceptions: an error looks like
    {'detail': 'Request was throttled...'} or {'non_field_errors': [...]}.
    Those are inspected and turned into RateLimitError / OrderRejected.
  * A throttled response is detected by substring and backed off.
  * Fractional shares are not allowed on limit/stop orders — round down, refuse
    a 0-share order.
  * get_latest_price returns a list of strings, any of which may be None.
"""

from __future__ import annotations

import os
import threading

from dotenv import load_dotenv

from broker_interface import (
    AuthError, Broker, BrokerError, BrokerPosition, OrderRejected,
    OrderStatusInfo, RateLimitError, TransientError, log,
)
from config import BASE

load_dotenv(BASE / ".env")

_THROTTLE_MARKERS = ("throttl", "too many requests", "rate limit")


class RobinhoodBroker(Broker):
    name = "robinhood"
    is_paper = False

    def __init__(self, *, email: str | None = None, password: str | None = None,
                 mfa_secret: str | None = None,
                 account_number: str | None = None) -> None:
        self._email = email or os.getenv("RH_EMAIL")
        self._password = password or os.getenv("RH_PASSWORD")
        self._mfa_secret = mfa_secret or os.getenv("RH_MFA_SECRET")
        self._mfa_code = os.getenv("RH_MFA_CODE")
        self._account_number = (account_number
                                 or os.getenv("RH_ACCOUNT_NUMBER") or None)
        if not self._email or not self._password:
            raise AuthError("RH_EMAIL / RH_PASSWORD are not set")

        try:
            import robin_stocks.robinhood as rh
        except ImportError as e:  # pragma: no cover
            raise BrokerError(
                "robin-stocks is not installed — `pip install robin-stocks`") from e
        self._rh = rh
        self._lock = threading.Lock()      # robin_stocks holds global session state
        self._connect()

    # -- auth ------------------------------------------------------------
    def _totp_now(self) -> str | None:
        if self._mfa_code:
            return self._mfa_code
        if not self._mfa_secret:
            return None
        try:
            import pyotp
        except ImportError:  # pragma: no cover
            log("robinhood: RH_MFA_SECRET set but pyotp missing — "
                "`pip install pyotp`")
            return None
        return pyotp.TOTP(self._mfa_secret.replace(" ", "")).now()

    def _connect(self) -> None:
        mfa = self._totp_now()
        if not mfa and not self._mfa_code:
            log("robinhood: no RH_MFA_SECRET/RH_MFA_CODE — login will block on "
                "an interactive prompt if the session cache is cold")

        def _do() -> dict:
            out = self._rh.login(
                username=self._email, password=self._password,
                mfa_code=mfa, store_session=True, expiresIn=86400)
            if not isinstance(out, dict) or not out.get("access_token"):
                raise AuthError(f"robinhood login rejected: "
                                f"{str(out)[:200] if out else 'no response'}")
            return out

        try:
            self._resilient("authenticate", _do)
        except BrokerError:
            raise
        except Exception as raw:  # noqa: BLE001
            raise self._classify(raw) from raw
        log("robinhood: session established (cached under ~/.tokens/)"
            + (f" — pinned to account {self._account_number}"
               if self._account_number else
               " — no RH_ACCOUNT_NUMBER set, using robin_stocks default account"))

    # -- error mapping --------------------------------------------------------
    def _classify(self, exc: Exception) -> BrokerError:
        return super()._classify(exc)

    def _check_payload(self, ctx: str, payload) -> dict:
        """
        Turn a robin_stocks response dict into either a clean dict or the right
        BrokerError. robin_stocks reports failure in-band, not by raising.
        """
        if payload is None:
            raise TransientError(f"{ctx}: empty response from robin_stocks")
        if not isinstance(payload, dict):
            return {"raw": payload}
        blob = str(payload).lower()
        if any(m in blob for m in _THROTTLE_MARKERS):
            raise RateLimitError(f"{ctx}: {str(payload)[:200]}")
        if payload.get("detail") and not payload.get("id"):
            raise OrderRejected(f"{ctx}: {payload['detail']}")
        for k in ("non_field_errors", "account", "quantity", "price"):
            if k in payload and isinstance(payload[k], list) and payload[k]:
                raise OrderRejected(f"{ctx}: {k}={payload[k]}")
        return payload

    # -- prices ------------------------------------------------------------
    def get_price(self, ticker: str) -> float:
        sym = ticker.upper()

        def _do() -> float:
            with self._lock:
                res = self._rh.stocks.get_latest_price(
                    sym, includeExtendedHours=True)
            if not res or res[0] in (None, "None", ""):
                raise TransientError(f"robinhood: no price for {sym}")
            px = float(res[0])
            if px <= 0:
                raise TransientError(f"robinhood: bad price {px} for {sym}")
            return round(px, 4)

        return self._resilient(f"get_price({sym})", _do)

    # -- orders ----------------------------------------------------------
    def buy_limit(self, ticker: str, shares: float, price: float) -> str:
        sym = ticker.upper()
        qty = self._whole_shares(shares, f"robinhood buy_limit {sym}")
        limit = round(float(price), 2)

        def _do() -> str:
            with self._lock:
                res = self._rh.orders.order_buy_limit(
                    symbol=sym, quantity=qty, limitPrice=limit,
                    account_number=self._account_number, timeInForce="gtc")
            payload = self._check_payload(f"buy_limit {sym}", res)
            oid = payload.get("id")
            if not oid:
                raise OrderRejected(f"buy_limit {sym}: no order id in {payload}")
            log(f"robinhood: BUY LIMIT {qty} {sym} @ {limit} -> {oid}")
            return str(oid)

        return self._resilient(f"buy_limit({sym})", _do)

    def buy_market(self, ticker: str, shares: float) -> str:
        """
        Fractional-capable BUY market order. Robinhood does not accept a
        fractional quantity on order_buy_limit/order_buy_stop_loss (whole
        shares only, same as Alpaca) — order_buy_fractional_by_quantity is
        its dedicated fractional MARKET order call, documented by
        robin_stocks as supporting up to 6 decimal places.
        """
        sym = ticker.upper()
        qty = round(float(shares), 6)
        if qty <= 0:
            raise OrderRejected(f"robinhood buy_market {sym}: qty {shares} <= 0")

        def _do() -> str:
            with self._lock:
                res = self._rh.orders.order_buy_fractional_by_quantity(
                    symbol=sym, quantity=qty,
                    account_number=self._account_number, timeInForce="gfd")
            payload = self._check_payload(f"buy_market {sym}", res)
            oid = payload.get("id")
            if not oid:
                raise OrderRejected(f"buy_market {sym}: no order id in {payload}")
            log(f"robinhood: BUY MKT (fractional) {qty} {sym} -> {oid}")
            return str(oid)

        return self._resilient(f"buy_market({sym})", _do)

    def sell_market(self, ticker: str, shares: float) -> str:
        sym = ticker.upper()
        qty = round(float(shares), 6)
        if qty <= 0:
            raise OrderRejected(f"robinhood sell_market {sym}: qty {shares} <= 0")

        def _do() -> str:
            with self._lock:
                res = self._rh.orders.order_sell_market(
                    symbol=sym, quantity=qty,
                    account_number=self._account_number, timeInForce="gfd")
            payload = self._check_payload(f"sell_market {sym}", res)
            oid = payload.get("id")
            if not oid:
                raise OrderRejected(f"sell_market {sym}: no order id in {payload}")
            log(f"robinhood: SELL MKT {qty} {sym} -> {oid}")
            return str(oid)

        return self._resilient(f"sell_market({sym})", _do)

    def place_stop_loss(self, ticker: str, shares: float, stop_price: float) -> str:
        sym = ticker.upper()
        qty = self._whole_shares(shares, f"robinhood stop {sym}")
        stop = round(float(stop_price), 2)

        def _do() -> str:
            with self._lock:
                res = self._rh.orders.order_sell_stop_loss(
                    symbol=sym, quantity=qty, stopPrice=stop,
                    account_number=self._account_number, timeInForce="gtc")
            payload = self._check_payload(f"stop_loss {sym}", res)
            oid = payload.get("id")
            if not oid:
                raise OrderRejected(f"stop_loss {sym}: no order id in {payload}")
            log(f"robinhood: SELL STOP {qty} {sym} @ {stop} -> {oid}")
            return str(oid)

        return self._resilient(f"place_stop_loss({sym})", _do)

    def cancel_order(self, order_id: str) -> bool:
        def _do() -> bool:
            with self._lock:
                res = self._rh.orders.cancel_stock_order(order_id)
            blob = str(res).lower()
            if any(m in blob for m in _THROTTLE_MARKERS):
                raise RateLimitError(f"cancel {order_id}: {str(res)[:200]}")
            # robin_stocks returns "" / {} on success and a dict with 'detail'
            # like "Not found" when the order is already inactive — both fine.
            log(f"robinhood: cancel {order_id} -> {str(res)[:120] or 'ok'}")
            return True

        return self._resilient(f"cancel_order({order_id})", _do)

    # -- introspection ---------------------------------------------------
    def list_positions(self) -> list[BrokerPosition]:
        def _do() -> list[BrokerPosition]:
            with self._lock:
                raw = self._rh.account.get_open_stock_positions(
                    account_number=self._account_number)
            blob = str(raw).lower()
            if any(m in blob for m in _THROTTLE_MARKERS):
                raise RateLimitError(f"list_positions: {str(raw)[:200]}")

            out: list[BrokerPosition] = []
            for item in raw or []:
                if not item:
                    continue
                qty = float(item.get("quantity") or 0)
                if qty <= 0:
                    continue
                avg_entry = float(item.get("average_buy_price") or 0)
                with self._lock:
                    sym = self._rh.stocks.get_symbol_by_url(item["instrument"])
                    px = self._rh.stocks.get_latest_price(
                        sym, includeExtendedHours=True)
                current = (float(px[0])
                          if px and px[0] not in (None, "None", "")
                          else avg_entry)
                out.append(BrokerPosition(
                    symbol=sym, qty=qty,
                    avg_entry_price=round(avg_entry, 4),
                    current_price=round(current, 4),
                    unrealized_pnl=round((current - avg_entry) * qty, 2)))
            return out

        return self._resilient("list_positions", _do)

    def get_equity(self) -> float:
        def _do() -> float:
            with self._lock:
                payload = self._rh.profiles.load_portfolio_profile(
                    account_number=self._account_number)
            payload = self._check_payload("get_equity", payload)
            equity = payload.get("equity")
            if equity in (None, "None", ""):
                raise TransientError(f"robinhood: no equity in portfolio profile")
            eq = float(equity)
            extended = payload.get("extended_hours_equity")
            if extended not in (None, "None", ""):
                eq = max(eq, float(extended))
            if eq <= 0:
                raise TransientError(f"robinhood: non-positive equity ({eq})")
            return eq

        return self._resilient("get_equity", _do)

    def get_buying_power(self) -> float:
        """
        `cash_available_for_withdrawal` from load_account_profile() — the
        conservative, literal "settled cash you can actually spend right
        now" figure, NOT `buying_power` from the same payload (which
        includes margin/instant-deposit extension and was observed live to
        still overstate what a real order could execute — a $1,152 order
        was rejected by Robinhood even though `buying_power` reported
        $1,828 available at the time). Defaults to 0.0 (never negative,
        never a crash) if the field is missing/unparseable, since "assume
        nothing is available" is the safe failure mode for an affordability
        check — the caller skips or scales the order down, it does not
        guess a number and submit anyway.
        """
        def _do() -> float:
            with self._lock:
                payload = self._rh.profiles.load_account_profile(
                    account_number=self._account_number)
            payload = self._check_payload("get_buying_power", payload)
            cash = payload.get("cash_available_for_withdrawal")
            if cash in (None, "None", ""):
                return 0.0
            return max(0.0, float(cash))

        return self._resilient("get_buying_power", _do)

    def get_order_status(self, order_id: str) -> OrderStatusInfo:
        """
        NOTE ON FIELD NAMES: `state`, `cumulative_quantity`, and
        `average_price` are Robinhood's documented stock-order schema
        (matching robin_stocks.orders.get_stock_order_info()'s own
        examples), but this project has never exercised this call against
        a REAL Robinhood order response — this account trades on
        BROKER_MODE=alpaca_paper. Verify this mapping against one real
        order before trusting it to gate live (real-money) entries under
        BROKER_MODE=robinhood_live.
        """
        def _do() -> OrderStatusInfo:
            with self._lock:
                payload = self._rh.orders.get_stock_order_info(order_id)
            payload = self._check_payload(f"get_order_status {order_id}", payload)

            state = str(payload.get("state", "")).strip().lower()
            filled_qty = float(payload.get("cumulative_quantity") or 0)
            avg_raw = payload.get("average_price")
            filled_avg = (float(avg_raw)
                         if avg_raw not in (None, "None", "") else None)

            if state == "filled":
                status = "filled"
            elif state == "partially_filled":
                status = "partially_filled"
            elif state == "rejected":
                status = "rejected"
            elif state in ("canceled", "failed"):
                status = "canceled"
            elif state in ("unconfirmed", "queued", "confirmed"):
                status = "new"
            elif filled_qty > 0:
                status = "partially_filled"
            else:
                status = "unknown"

            return OrderStatusInfo(order_id=order_id, status=status,
                                   filled_qty=filled_qty,
                                   filled_avg_price=filled_avg)

        return self._resilient(f"get_order_status({order_id})", _do)


if __name__ == "__main__":  # pragma: no cover
    b = RobinhoodBroker()
    print("SPY:", b.get_price("SPY"))
