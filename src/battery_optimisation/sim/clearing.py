"""The market as the simulator sees it: offers, price-taker clearing and the information set.

Clearing rules (CLAUDE.md), interval by interval, pay-as-clear at the realised USEP:
  - a DISCHARGE tranche clears in full if its price <= realised price
  - a CHARGE tranche clears in full if its price >= realised price
If both sides clear in the same interval the battery cannot do both: cleared quantities are
netted (same price, so revenue is unchanged).

``MarketView`` is what a strategy may know when offering for an interval: the realised prices
of earlier intervals only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Self

import polars as pl
import structlog
from pydantic import BaseModel, ConfigDict, NonNegativeFloat

from battery_optimisation.columns import Col
from battery_optimisation.config import MarketConfig

log = structlog.get_logger(__name__)


class Side(StrEnum):
    CHARGE = "charge"  # bid to buy energy
    DISCHARGE = "discharge"  # offer to sell energy


class Tranche(BaseModel):
    model_config = ConfigDict(frozen=True)

    side: Side
    price: float  # SGD/MWh
    quantity_mw: NonNegativeFloat


class IntervalOffers(BaseModel):
    """All energy tranches submitted for one interval (may be empty: no offer)."""

    model_config = ConfigDict(frozen=True)

    interval_start: datetime
    tranches: tuple[Tranche, ...] = ()

    @classmethod
    def self_schedule(
        cls, interval_start: datetime, charge_mw: float, discharge_mw: float, market: MarketConfig
    ) -> Self:
        """Price-taking offers that always clear: discharge at the floor, charge at the cap."""
        tranches = []
        if discharge_mw > 0:
            tranches.append(
                Tranche(side=Side.DISCHARGE, price=market.price_floor, quantity_mw=discharge_mw)
            )
        if charge_mw > 0:
            tranches.append(
                Tranche(side=Side.CHARGE, price=market.price_cap, quantity_mw=charge_mw)
            )
        return cls(interval_start=interval_start, tranches=tuple(tranches))

    def on(self, side: Side) -> list[Tranche]:
        return [t for t in self.tranches if t.side is side]


@dataclass(frozen=True)
class Cleared:
    interval_start: datetime
    price: float  # realised USEP, SGD/MWh
    charge_mw: float
    discharge_mw: float


def check_prices(prices: pl.DataFrame, market: MarketConfig) -> None:
    """Prices must be a contiguous grid of ``market.interval_h`` steps with no missing USEP."""
    if prices[Col.USEP].null_count():
        raise ValueError(f"{prices[Col.USEP].null_count()} missing prices")
    steps = prices[Col.INTERVAL_START].diff().drop_nulls().unique()
    if steps.len() > 1 or (steps.len() == 1 and steps.item() != timedelta(hours=market.interval_h)):
        raise ValueError(f"intervals must be contiguous steps of {market.interval_h} h")


class MarketView:
    """What a strategy knows when it must fix its offer for interval ``now``.

    Offers close ``gate`` intervals before ``now`` (``decided_at``); only realised prices strictly
    before ``decided_at`` are visible (including history from before the simulation window).
    """

    def __init__(self, prices: pl.DataFrame, index: int, gate: int = 0) -> None:
        # ``prices`` = pre-window history + window; ``index`` = position of ``now`` in it
        self._prices, self._index, self._gate = prices, index, gate

    @property
    def now(self) -> datetime:
        """Start of the interval being offered for."""
        return self._prices[self._index, Col.INTERVAL_START]

    @property
    def decided_at(self) -> datetime:
        """Gate closure: when the offer for ``now`` is fixed."""
        step = self._prices[1, Col.INTERVAL_START] - self._prices[0, Col.INTERVAL_START]
        return self.now - self._gate * step

    @property
    def end(self) -> datetime:
        """Start of the last interval of the simulation (planning horizon limit)."""
        return self._prices[-1, Col.INTERVAL_START]

    def history(self) -> pl.DataFrame:
        """Realised ``Col.INTERVAL_START``, ``Col.USEP`` strictly before ``decided_at``."""
        return self._prices.slice(0, max(self._index - self._gate, 0))


class Market:
    """Price-taker NEMS energy market over a window of realised prices.

    ``history``: realised prices before the window, visible to strategies but never traded.
    """

    def __init__(
        self, config: MarketConfig, prices: pl.DataFrame, history: pl.DataFrame | None = None
    ) -> None:
        prices = prices.select(Col.INTERVAL_START, Col.USEP).sort(Col.INTERVAL_START)
        check_prices(prices, config)
        before = (
            history.select(Col.INTERVAL_START, Col.USEP)
            .filter(pl.col(Col.INTERVAL_START) < prices[0, Col.INTERVAL_START])
            .sort(Col.INTERVAL_START)
            if history is not None
            else prices.clear()
        )
        self.config = config
        self.prices = prices
        self.intervals: list[datetime] = prices[Col.INTERVAL_START].to_list()
        self._price_at = dict(zip(self.intervals, prices[Col.USEP].to_list(), strict=True))
        self._known = pl.concat([before, prices])
        self._offset = before.height

    def view(self, index: int) -> MarketView:
        """View for offering on window interval ``index`` (gate closure from the config)."""
        return MarketView(self._known, self._offset + index, self.config.gate_closure_intervals)

    def validate(self, offers: IntervalOffers) -> None:
        cfg = self.config
        for side in Side:
            if len(offers.on(side)) > cfg.n_tranches:
                raise ValueError(f"more than {cfg.n_tranches} {side} tranches")
        for t in offers.tranches:
            if not cfg.price_floor <= t.price <= cfg.price_cap:
                raise ValueError(f"offer price {t.price} outside [floor, cap]")

    def clear(self, offers: IntervalOffers) -> Cleared:
        self.validate(offers)
        price = self._price_at[offers.interval_start]
        discharge = sum(t.quantity_mw for t in offers.on(Side.DISCHARGE) if t.price <= price)
        charge = sum(t.quantity_mw for t in offers.on(Side.CHARGE) if t.price >= price)
        if charge > 0 and discharge > 0:
            log.warning("clearing.netted", interval=str(offers.interval_start))
            net = discharge - charge
            discharge, charge = max(net, 0.0), max(-net, 0.0)
        return Cleared(offers.interval_start, price, float(charge), float(discharge))
