"""Interval-by-interval simulation: strategy -> market clearing -> battery step.

For each interval, in order:
  1. the strategy sees ``MarketView`` (realised prices of earlier intervals only) and the current
     SOC, and returns its offers for the interval;
  2. the market clears them against the realised price;
  3. the battery applies the cleared powers (raising ``LimitBreachError`` if infeasible);
  4. revenue and degradation are booked.
"""

from __future__ import annotations

from typing import Protocol

import polars as pl

from battery_optimisation.battery import Battery
from battery_optimisation.columns import Col
from battery_optimisation.sim.clearing import IntervalOffers, Market, MarketView
from battery_optimisation.sim.dispatch import Dispatch


class Strategy(Protocol):
    """A bidding policy: offers for one interval from what is known before it."""

    def offers(self, view: MarketView, soc_mwh: float) -> IntervalOffers: ...


class Simulator:
    def __init__(self, market: Market, battery: Battery) -> None:
        self.market = market
        self.battery = battery

    def run(self, strategy: Strategy) -> Dispatch:
        dt = self.market.config.interval_h
        rows = []
        for i, interval in enumerate(self.market.intervals):
            offers = strategy.offers(self.market.view(i), self.battery.soc_mwh)
            if offers.interval_start != interval:
                raise ValueError(f"strategy offered for {offers.interval_start}, not {interval}")
            cleared = self.market.clear(offers)
            step = self.battery.step(cleared.charge_mw, cleared.discharge_mw, dt)
            revenue = cleared.price * (step.discharge_mw - step.charge_mw) * dt
            rows.append(
                {
                    Col.INTERVAL_START: interval,
                    Col.USEP: cleared.price,
                    Col.CHARGE_MW: step.charge_mw,
                    Col.DISCHARGE_MW: step.discharge_mw,
                    Col.NET_MW: step.discharge_mw - step.charge_mw,
                    Col.SOC_MWH: step.soc_mwh,
                    Col.REVENUE_SGD: revenue,
                    Col.DEGRADATION_SGD: step.degradation_sgd,
                    Col.PROFIT_SGD: revenue - step.degradation_sgd,
                }
            )
        frame = pl.DataFrame(
            rows,
            schema_overrides={Col.INTERVAL_START: self.market.prices.schema[Col.INTERVAL_START]},
        )
        return Dispatch(frame=frame, battery=self.battery.config, market=self.market.config)
