"""Interval-by-interval simulation: strategy -> market clearing -> battery step.

Gate closure G (``market.gate_closure_intervals``): the offer for interval t is fixed at the start
of t - G. At the start of each interval i, in order:
  1. the strategy is asked for its offers for interval i + G, seeing ``MarketView`` (realised
     prices before the decision time only) and the SOC at the start of i (it does not yet know
     how i .. i + G - 1 will clear); offers for the first G intervals are fixed at the window start;
  2. the market clears interval i's offers (fixed G intervals ago) against the realised price;
     with ``storage_rules.enforce_soc`` the engine then cuts the schedule to what the SoC at
     the start of i allows (NEMS RC386, ``sim.storage_rules``);
  3. the battery applies the cleared powers (raising ``LimitBreachError`` if infeasible, or
     settling the undeliverable part at the shortfall penalty with ``PENALISE``);
  4. revenue and degradation are booked.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

import polars as pl

from battery_optimisation.battery import Battery
from battery_optimisation.columns import Col
from battery_optimisation.config import Product
from battery_optimisation.sim.clearing import IntervalOffers, Market, MarketView, price_column
from battery_optimisation.sim.dispatch import Dispatch
from battery_optimisation.sim.storage_rules import check_storage_rules, cut_mw, enforce_soc


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
        rows, offered = [], []
        for i, interval in enumerate(intervals):
            if i > 0 and i + gate < len(intervals):
                submit(i + gate)
            m = self.market.config
            offers = pending.pop(i)
            offered.extend(_offer_rows(offers))
            cleared = self.market.clear(offers)
            cut = 0.0
            if m.storage_rules.enforce_soc:
                engine = enforce_soc(
                    cleared, offers, m.storage_rules, self.battery.config, self.battery.soc_mwh
                )
                cut, cleared = cut_mw(cleared, engine), engine
            charge, discharge = cleared.charge_mw, cleared.discharge_mw
            if self.on_shortfall is ShortfallPolicy.PENALISE:
                charge, discharge = self.battery.deliverable(charge, discharge, dt)
            shortfall = (cleared.charge_mw - charge) + (cleared.discharge_mw - discharge)
            awards = cleared.awards
            if any(mw > 0 for mw in awards.values()):
                check_storage_rules(
                    m.storage_rules,
                    self.battery.config,
                    self.battery.soc_mwh,
                    charge,
                    discharge,
                    awards,
                )
            step = self.battery.step(charge, discharge, dt)
            # energy moved by AGC around the schedule: metered, settled at the energy price
            reg, energy = awards.get(Product.REGULATION, 0.0), m.ancillary.regulation_energy
            reg_up, reg_down = energy.up * reg * dt, energy.down * reg * dt
            soc_end = self.battery.regulate(reg_up, reg_down) if reg else step.soc_mwh
            ancillary = sum(awards.get(p, 0.0) * cleared.ancillary_prices[p] * dt for p in awards)
            regulation_wear = self.battery.degradation_sgd(0.0, energy.throughput * reg, dt)
            penalty = shortfall * dt * m.shortfall_penalty.price
            delivered = (step.discharge_mw - step.charge_mw) * dt + reg_up - reg_down
            revenue = cleared.price * delivered  # delivered only
            rows.append(
                {
                    Col.INTERVAL_START: interval,
                    Col.USEP: cleared.price,
                    Col.CHARGE_MW: step.charge_mw,
                    Col.DISCHARGE_MW: step.discharge_mw,
                    Col.NET_MW: step.discharge_mw - step.charge_mw,
                    Col.SOC_MWH: soc_end,
                    Col.REVENUE_SGD: revenue,
                    Col.REG_MW: awards.get(Product.REGULATION, 0.0),
                    Col.PRI_MW: awards.get(Product.PRIMARY_RESERVE, 0.0),
                    Col.CON_MW: awards.get(Product.CONTINGENCY_RESERVE, 0.0),
                    Col.ANCILLARY_SGD: ancillary,
                    Col.DEGRADATION_SGD: step.degradation_sgd + regulation_wear,
                    Col.SHORTFALL_MW: shortfall,
                    Col.PENALTY_SGD: penalty,
                    Col.SOC_CUT_MW: cut,
                    Col.REG_NET_MWH: reg_up - reg_down,
                    # realised ancillary prices (whatever was awarded)
                    **{
                        price_column(p, m): cleared.ancillary_prices[p]
                        for p in self.market.products
                    },
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
        offer_frame = pl.DataFrame(
            offered,
            schema={
                Col.INTERVAL_START: self.market.prices.schema[Col.INTERVAL_START],
                Col.PRODUCT: pl.String,
                Col.SIDE: pl.String,
                Col.TRANCHE: pl.Int32,
                Col.OFFER_PRICE: pl.Float64,
                Col.OFFER_MW: pl.Float64,
            },
            orient="row",
        )
        return Dispatch(
            frame=frame,
            battery=self.battery.config,
            market=self.market.config,
            offers=offer_frame,
        )


def _offer_rows(offers: IntervalOffers) -> list[tuple]:
    """One row per submitted tranche (energy in submitted order, then ancillary)."""
    rows = [
        (offers.interval_start, str(Product.ENERGY), str(t.side), k, t.price, t.quantity_mw)
        for k, t in enumerate(offers.tranches)
    ]
    rows += [
        (offers.interval_start, str(t.product), None, k, t.price, t.quantity_mw)
        for k, t in enumerate(offers.ancillary)
    ]
    return rows
