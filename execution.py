"""
Execution feasibility and position sizing.

Two questions this module answers, both of which the raw signal ignores:

1. Is the price still in a place where the trade is worth taking? A breakout
   signal stays "true" all the way up; a trade taken 3% above the trigger has
   the same target and a much wider stop, so its reward:risk is nothing like
   the one the backtest measured. That is what the ideal execution range is for.

2. How much? Sized off risk, not off conviction — the stop distance and the
   risk budget together determine the share count, and a hard position cap stops
   any single name dominating a small account.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from enum import Enum

from config import (
    RISK_PCT, MAX_POSITION_PCT, MIN_POSITION_USD, ALLOW_FRACTIONAL,
    STOP_PCT, TARGET_R, MAX_EXTENSION_PCT, APPROACH_PCT, SLIPPAGE_BPS,
)


class State(str, Enum):
    EXECUTE = "EXECUTE_NOW"      # inside the ideal range — take it
    APPROACHING = "APPROACHING"  # just below the trigger — arm and watch
    EXTENDED = "EXTENDED"        # past the range — chasing, stand down
    BELOW = "BELOW_SETUP"        # not near the trigger


@dataclass
class Plan:
    symbol: str
    state: State
    price: float
    trigger: float
    ideal_low: float
    ideal_high: float
    approach_low: float

    shares: float = 0.0
    dollars: float = 0.0
    stop: float = 0.0
    target: float = 0.0
    risk_dollars: float = 0.0
    reward_dollars: float = 0.0
    rr: float = 0.0
    pct_from_trigger: float = 0.0
    valuation_target: float | None = None
    size_note: str = ""

    @property
    def actionable(self) -> bool:
        return self.state is State.EXECUTE and self.shares > 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["state"] = self.state.value
        return d


def classify(price: float, trigger: float) -> tuple[State, float]:
    """Where price sits relative to the trigger, and by how much (fraction)."""
    pct = price / trigger - 1.0
    if 0.0 < pct <= MAX_EXTENSION_PCT:
        return State.EXECUTE, pct
    if pct > MAX_EXTENSION_PCT:
        return State.EXTENDED, pct
    if -APPROACH_PCT <= pct <= 0.0:
        return State.APPROACHING, pct
    return State.BELOW, pct


def size_position(equity: float, price: float, stop: float,
                  size_multiplier: float = 1.0,
                  max_position_pct: float | None = None,
                  risk_pct: float | None = None) -> tuple[float, str]:
    """
    Share count from the risk budget, then clamped by the position cap.

    `max_position_pct` must be no larger than 1/MAX_CONCURRENT or the book
    cannot actually hold that many positions: a 30% cap with 5 slots needs 150%
    of equity, so cash runs out after three names and every later signal is
    dropped for lack of funds rather than for any risk reason. That turns trade
    selection into a race, which is exactly what ranking candidates was meant to
    prevent.

    Returns (shares, note) where the note records which constraint bound — worth
    surfacing, because "risk-limited" and "cap-limited" mean different things
    when you are deciding whether the account is too small for the name.
    """
    risk_per_share = price - stop
    if risk_per_share <= 0 or price <= 0:
        return 0.0, "invalid stop"

    cap_pct = MAX_POSITION_PCT if max_position_pct is None else max_position_pct
    rp = RISK_PCT if risk_pct is None else risk_pct
    risk_budget = equity * rp * size_multiplier
    by_risk = risk_budget / risk_per_share
    by_cap = (equity * cap_pct) / price

    shares = min(by_risk, by_cap)
    note = "risk-limited" if by_risk <= by_cap else "position-cap-limited"

    if not ALLOW_FRACTIONAL:
        shares = math.floor(shares)
    else:
        shares = math.floor(shares * 1e4) / 1e4

    if shares <= 0 or shares * price < MIN_POSITION_USD:
        return 0.0, f"below ${MIN_POSITION_USD:.0f} minimum ({note})"

    return shares, note


def build_plan(symbol: str, price: float, trigger: float, equity: float,
               valuation_target: float | None = None,
               size_multiplier: float = 1.0) -> Plan:
    """Full execution plan: range, state, size, stop, target."""
    ideal_low = trigger
    ideal_high = trigger * (1.0 + MAX_EXTENSION_PCT)
    approach_low = trigger * (1.0 - APPROACH_PCT)

    state, pct = classify(price, trigger)

    stop = price * (1.0 - STOP_PCT)
    target = price + (price - stop) * TARGET_R

    plan = Plan(
        symbol=symbol, state=state, price=price, trigger=trigger,
        ideal_low=ideal_low, ideal_high=ideal_high, approach_low=approach_low,
        stop=round(stop, 2), target=round(target, 2),
        pct_from_trigger=pct, valuation_target=valuation_target,
    )

    if state in (State.EXECUTE, State.APPROACHING):
        # Size an APPROACHING name off the trigger, not the current price —
        # that is where the fill would actually happen if it breaks out.
        ref = price if state is State.EXECUTE else trigger
        ref_stop = ref * (1.0 - STOP_PCT)
        shares, note = size_position(equity, ref, ref_stop, size_multiplier)
        plan.shares = shares
        plan.dollars = round(shares * ref, 2)
        plan.risk_dollars = round(shares * (ref - ref_stop), 2)
        plan.reward_dollars = round(shares * (ref - ref_stop) * TARGET_R, 2)
        plan.rr = TARGET_R
        plan.size_note = note
        if state is State.APPROACHING:
            plan.stop = round(ref_stop, 2)
            plan.target = round(ref + (ref - ref_stop) * TARGET_R, 2)

    return plan


def apply_slippage(price: float, side: str) -> float:
    """Fills are worse than the signal price. Buys pay up, sells give up."""
    adj = price * (SLIPPAGE_BPS / 10_000.0)
    return price + adj if side == "buy" else price - adj
