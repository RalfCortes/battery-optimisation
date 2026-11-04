"""Interval-by-interval simulation: strategy -> market clearing -> battery step.

Gate closure G (``market.gate_closure_intervals``): the offer for interval t is fixed at the start
of t - G. At the start of each interval i, in order:
  1. the strategy is asked for its offers for interval i + G, seeing ``MarketView`` (realised
     prices before the decision time only) and the SOC at the start of i (it does not yet know
     how i .. i + G - 1 will clear); offers for the first G intervals are fixed at the window start;
  2. the market clears interval i's offers (fixed G intervals ago) against the realised price;
  3. the battery applies the cleared powers (raising ``LimitBreachError`` if infeasible);
  4. revenue and degradation are booked.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

import polars as pl

from battery_optimisation.battery import Battery
from battery_optimisation.columns import Col
from battery_optimisation.config import LOWER_PRODUCTS, RAISE_PRODUCTS, Product
from battery_optimisation.sim.clearing import IntervalOffers, Market, MarketView
from battery_optimisation.sim.dispatch import Dispatch


class Strategy(Protocol):
    """A bidding policy: offers for one interval from what is known before it."""

    def offers(self, view: MarketView, soc_mwh: float) -> IntervalOffers: ...


class ShortfallPolicy(StrEnum):
    """What happens when cleared offers exceed what the battery can physically do."""

    RAISE = "raise"  # LimitBreachError: offers were supposed to be feasible by construction
    PENALISE = "penalise"  # deliver what is possible; settle the rest at the shortfall penalty


class Simulator:
    def __init__(
        self,
        market: Market,
        battery: Battery,
        on_shortfall: ShortfallPolicy = ShortfallPolicy.RAISE,
    ) -> None:
        self.market = market
        self.battery = battery
        self.on_shortfall = on_shortfall

    def run(self, strategy: Strategy) -> Dispatch:
        dt = self.market.config.interval_h
        gate = self.market.config.gate_closure_intervals
        intervals = self.market.intervals
        pending: dict[int, IntervalOffers] = {}

        def submit(j: int) -> None:
            offers = strategy.offers(self.market.view(j), self.battery.soc_mwh)
            if offers.interval_start != intervals[j]:
                raise ValueError(
                    f"strategy offered for {offers.interval_start}, not {intervals[j]}"
                )
            pending[j] = offers

        for j in range(min(gate + 1, len(intervals))):  # gates closed at or before the start
            submit(j)
        rows = []
        for i, interval in enumerate(intervals):
            if i > 0 and i + gate < len(intervals):
                submit(i + gate)
            cleared = self.market.clear(pending.pop(i))
            charge, discharge = cleared.charge_mw, cleared.discharge_mw
            if self.on_shortfall is ShortfallPolicy.PENALISE:
                charge, discharge = self.battery.deliverable(charge, discharge, dt)
            shortfall = (cleared.charge_mw - charge) + (cleared.discharge_mw - discharge)
            m = self.market.config
            awards = cleared.awards
            raise_mw = sum(awards.get(p, 0.0) for p in RAISE_PRODUCTS)
            lower_mw = sum(awards.get(p, 0.0) for p in LOWER_PRODUCTS)
            if raise_mw or lower_mw:
                self.battery.check_reserves(
                    self.battery.soc_after(charge, discharge, dt),
                    charge,
                    discharge,
                    raise_mw,
                    lower_mw,
                    sum(awards.get(p, 0.0) * m.sustain_h(p) for p in RAISE_PRODUCTS),
                    sum(awards.get(p, 0.0) * m.sustain_h(p) for p in LOWER_PRODUCTS),
                )
            step = self.battery.step(charge, discharge, dt)
            ancillary = sum(awards.get(p, 0.0) * cleared.ancillary_prices[p] * dt for p in awards)
            regulation_wear = self.battery.degradation_sgd(
                0.0, m.ancillary.regulation_utilisation * awards.get(Product.REGULATION, 0.0), dt
            )
            penalty = shortfall * dt * self.market.config.shortfall_penalty_sgd_per_mwh
            revenue = cleared.price * (step.discharge_mw - step.charge_mw) * dt  # delivered only
            rows.append(
                {
                    Col.INTERVAL_START: interval,
                    Col.USEP: cleared.price,
                    Col.CHARGE_MW: step.charge_mw,
                    Col.DISCHARGE_MW: step.discharge_mw,
                    Col.NET_MW: step.discharge_mw - step.charge_mw,
                    Col.SOC_MWH: step.soc_mwh,
                    Col.REVENUE_SGD: revenue,
                    Col.REG_MW: awards.get(Product.REGULATION, 0.0),
                    Col.PRI_MW: awards.get(Product.PRIMARY_RESERVE, 0.0),
                    Col.CON_MW: awards.get(Product.CONTINGENCY_RESERVE, 0.0),
                    Col.ANCILLARY_SGD: ancillary,
                    Col.DEGRADATION_SGD: step.degradation_sgd + regulation_wear,
                    Col.SHORTFALL_MW: shortfall,
                    Col.PENALTY_SGD: penalty,
                    Col.PROFIT_SGD: revenue
                    + ancillary
                    - step.degradation_sgd
                    - regulation_wear
                    - penalty,
                }
            )
        frame = pl.DataFrame(
            rows,
            schema_overrides={Col.INTERVAL_START: self.market.prices.schema[Col.INTERVAL_START]},
        )
        return Dispatch(frame=frame, battery=self.battery.config, market=self.market.config)
