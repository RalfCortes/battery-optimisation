"""Bidding strategies (implement ``sim.simulator.Strategy``).

``ScheduleStrategy``: at the start of each planning window, solve the MILP on the forecaster's
prices from the current SOC, then self-schedule the plan (offers that always clear). With a
``PerfectForecaster`` this is the perfect-foresight benchmark; with a real forecaster it is the
self-schedule baseline from CLAUDE.md.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from enum import StrEnum

import numpy as np
import polars as pl

from battery_optimisation.battery import Battery
from battery_optimisation.bidding.compress import compress
from battery_optimisation.bidding.feasibility import expected_soc_after, make_feasible
from battery_optimisation.bidding.water_value import (
    offer_curve,
    price_take_beyond_band,
    trim_to_band,
)
from battery_optimisation.columns import Col
from battery_optimisation.config import BatteryConfig, BiddingConfig, MarketConfig
from battery_optimisation.forecast.base import Forecaster
from battery_optimisation.optim.milp import MilpSolution, solve_schedule
from battery_optimisation.optim.sdp import ValueFunction, solve_sdp
from battery_optimisation.sim.clearing import IntervalOffers, MarketView
from battery_optimisation.timing import day_start


class Horizon(StrEnum):
    """Planning window of ``ScheduleStrategy``."""

    FULL = "full"  # one plan to the end of the simulation: with perfect forecast, the true optimum
    DAY = "day"  # re-plan at the start of each day, to the end of that day


class ScheduleStrategy:
    def __init__(
        self,
        forecaster: Forecaster,
        battery: BatteryConfig,
        market: MarketConfig,
        horizon: Horizon = Horizon.FULL,
    ) -> None:
        self.forecaster = forecaster
        self.battery = battery
        self.market = market
        self.horizon = horizon
        self.plans: list[MilpSolution] = []
        self._plan: dict[datetime, tuple[float, float]] = {}  # interval -> (charge, discharge)
        self._soc_end: dict[datetime, float] = {}  # interval -> planned SOC at its end

    @property
    def planned_profit_sgd(self) -> float:
        """Sum of the MILP objectives of all plans made so far (on forecast prices)."""
        return sum(p.objective_sgd for p in self.plans)

    @property
    def n_plans(self) -> int:
        return len(self.plans)

    @property
    def solve_time_s(self) -> float:
        return sum(p.solve_time_s for p in self.plans)

    def _window_end(self, view: MarketView) -> datetime:
        if self.horizon is Horizon.FULL:
            return view.end
        last_of_day = datetime.combine(view.now.date(), time(), view.now.tzinfo) + timedelta(
            days=1, hours=-self.market.interval_h
        )
        return min(last_of_day, view.end)

    def _start_soc(self, view: MarketView, soc_mwh: float) -> float:
        """SOC at the start of ``view.now``. With gate closure the intervals between the decision
        and ``now`` are already committed by the previous plan, which always clears (self-schedule):
        its planned SOC is exact. Otherwise (first plan) the SOC known at decision time."""
        previous = view.now - timedelta(hours=self.market.interval_h)
        return self._soc_end.get(previous, soc_mwh)

    def _replan(self, view: MarketView, soc_mwh: float) -> None:
        soc_mwh = self._start_soc(view, soc_mwh)
        prices = self.forecaster.forecast(view, self._window_end(view))
        # Each window must leave at least the initial SOC: no free energy taken from the end.
        plan = solve_schedule(
            prices,
            Battery(self.battery, soc_mwh=soc_mwh),
            self.market,
            soc_terminal_min=self.battery.soc_init,
        )
        self.plans.append(plan)
        s = plan.schedule
        stamps = s[Col.INTERVAL_START].to_list()
        self._plan = dict(
            zip(
                stamps,
                zip(s[Col.CHARGE_MW].to_list(), s[Col.DISCHARGE_MW].to_list(), strict=True),
                strict=True,
            )
        )
        self._soc_end.update(zip(stamps, s[Col.SOC_MWH].to_list(), strict=True))

    def offers(self, view: MarketView, soc_mwh: float) -> IntervalOffers:
        if view.now not in self._plan:
            self._replan(view, soc_mwh)
        charge, discharge = self._plan[view.now]
        return IntervalOffers.self_schedule(view.now, charge, discharge, self.market)


class CurveStep(StrEnum):
    """Steps that turn the water-value curve into submitted offers (``curve_steps``)."""

    RAW = "raw"  # micro-tranches from V_{t+1} around the expected SOC
    BAND = "band"  # trimmed to the interval's scenario price range
    COMPRESSED = "compressed"  # <= K tranches per side, clipped to [floor, cap]
    PRICE_TAKING = "price_taking"  # tranches beyond the band at floor / cap (certain)
    FEASIBLE = "feasible"  # capped from the worst-case SOC (= submitted)


class WaterValueStrategy:
    """Offer curves from the day's SDP water value (CLAUDE.md bidding layer).

    For each delivery day, at its first decision: scenarios issued by then (latest issue) give
    per-stage price distributions; ``solve_sdp`` gives V_t. For each interval t, decided at gate
    closure: the curve from V_{t+1} around the SOC *expected* at t (decision SOC + own pending
    offers cleared under the scenario prices), trimmed to tranches that can clear within the
    interval's scenario price range, compressed to K tranches per side, clipped to [floor, cap];
    tranches that clear at every plausible price are made price-taking (certain); finally capped
    from the *worst-case* SOC after the pending offers (``feasibility``).
    """

    def __init__(
        self,
        scenarios: pl.DataFrame,
        battery: BatteryConfig,
        market: MarketConfig,
        bidding: BiddingConfig,
    ) -> None:
        self.battery, self.market, self.bidding = battery, market, bidding
        days = scenarios.with_columns(pl.col(Col.INTERVAL_START).dt.date().alias("_day"))
        self._by_day: dict[date, pl.DataFrame] = {
            key[0]: part for key, part in days.partition_by("_day", as_dict=True).items()
        }
        self.value_functions: dict[date, ValueFunction] = {}
        self.expected_profit_sgd: dict[date, float] = {}
        self.submitted: dict[datetime, IntervalOffers] = {}
        self._stage_prices: dict[date, tuple[np.ndarray, np.ndarray]] = {}  # [T, K], [K]
        self.reference_soc: dict[datetime, float] = {}  # SOC each curve was built around
        self.decision_soc: dict[datetime, float] = {}  # SOC known at the decision
        self.pending_slots: dict[datetime, list[datetime]] = {}  # offers pending at the decision

    @property
    def planned_profit_sgd(self) -> float:
        """Sum over days of the SDP's expected value at the day's first decision."""
        return sum(self.expected_profit_sgd.values())

    @property
    def n_plans(self) -> int:
        return len(self.value_functions)

    @property
    def solve_time_s(self) -> float:
        return sum(v.solve_time_s for v in self.value_functions.values())

    def stage_prices(self, day: date) -> np.ndarray:
        """Scenario prices [T, K] the day's SDP used."""
        return self._stage_prices[day][0]

    @staticmethod
    def day_start_of(day: date) -> datetime:
        return day_start(day)

    def _stage(self, at: datetime) -> tuple[date, int]:
        day = at.date()
        return day, int((at - day_start(day)) / timedelta(hours=self.market.interval_h))

    def _value_function(self, day: date, view: MarketView, soc_mwh: float) -> ValueFunction:
        if day in self.value_functions:
            return self.value_functions[day]
        if day not in self._by_day:
            raise ValueError(f"no scenarios for delivery day {day}")
        known = self._by_day[day].filter(pl.col(Col.ISSUED_AT) <= view.decided_at)
        if known.is_empty():
            raise ValueError(f"no scenarios for {day} issued by {view.decided_at}")
        latest = known.filter(pl.col(Col.ISSUED_AT) == known[Col.ISSUED_AT].max())
        scenario_ids = sorted(latest[Col.SCENARIO].unique().to_list())
        wide = latest.pivot(on=Col.SCENARIO, index=Col.INTERVAL_START, values=Col.USEP_FORECAST)
        prices = wide.sort(Col.INTERVAL_START).select([str(k) for k in scenario_ids]).to_numpy()
        weights = (
            latest.unique(Col.SCENARIO).sort(Col.SCENARIO)[Col.WEIGHT].to_numpy().astype(float)
        )
        vf = solve_sdp(prices, weights, Battery(self.battery), self.market.interval_h, self.bidding)
        self.value_functions[day] = vf
        self._stage_prices[day] = (prices, weights / weights.sum())
        self.expected_profit_sgd[day] = vf.at(0, soc_mwh)
        return vf

    def _expected_soc(
        self, soc_mwh: float, pending: list[tuple[datetime, IntervalOffers]], here: Battery
    ) -> float:
        """SOC expected at the target: pending offers cleared under their day's scenarios."""
        if not pending:
            return soc_mwh
        rows, weights = [], None
        for at, _ in pending:
            day, t = self._stage(at)
            if day not in self._stage_prices:
                return soc_mwh
            prices, w = self._stage_prices[day]
            if weights is not None and len(w) != len(weights):
                return soc_mwh  # scenario sets of different sizes across midnight
            rows.append(prices[t].tolist())
            weights = w
        return expected_soc_after(
            soc_mwh, [o for _, o in pending], rows, weights.tolist(), here, self.market.interval_h
        )

    def curve_steps(
        self,
        at: datetime,
        soc_mwh: float,
        reference: float,
        pending: list[IntervalOffers],
    ) -> dict[CurveStep, IntervalOffers]:
        """The offers for interval ``at`` after each step of the pipeline (in order)."""
        dt, m = self.market.interval_h, self.market
        day, t = self._stage(at)
        vf = self.value_functions[day]
        here = Battery(self.battery, soc_mwh)
        stage_prices = self._stage_prices[day][0][t]
        margin = self.bidding.price_band_margin_sgd_per_mwh
        low, high = stage_prices.min() - margin, stage_prices.max() + margin

        def offers(discharge: list, charge: list) -> IntervalOffers:
            return IntervalOffers(interval_start=at, tranches=tuple(discharge + charge))

        def clip(tranches: list) -> list:
            return [
                tr.model_copy(update={"price": min(max(tr.price, m.price_floor), m.price_cap)})
                for tr in compress(tranches, m.n_tranches)
            ]

        steps: dict[CurveStep, IntervalOffers] = {}
        discharge, charge = offer_curve(vf, t, reference, here, dt)
        steps[CurveStep.RAW] = offers(discharge, charge)
        discharge, charge = trim_to_band(discharge, charge, low, high)
        steps[CurveStep.BAND] = offers(discharge, charge)
        discharge, charge = clip(discharge), clip(charge)
        steps[CurveStep.COMPRESSED] = offers(discharge, charge)
        discharge, charge = price_take_beyond_band(
            discharge, charge, low, high, m.price_floor, m.price_cap
        )
        steps[CurveStep.PRICE_TAKING] = offers(discharge, charge)
        steps[CurveStep.FEASIBLE] = make_feasible(
            steps[CurveStep.PRICE_TAKING], soc_mwh, pending, here, dt, m.price_floor, m.price_cap
        )
        return steps

    def offers(self, view: MarketView, soc_mwh: float) -> IntervalOffers:
        day, _ = self._stage(view.now)
        step = timedelta(hours=self.market.interval_h)
        gaps = round((view.now - view.decided_at) / step)  # = gate closure intervals
        slots = [view.now - k * step for k in range(gaps, 0, -1)]  # decided .. now, in order
        pending = [(at, self.submitted[at]) for at in slots if at in self.submitted]
        self._value_function(day, view, soc_mwh)
        reference = self._expected_soc(soc_mwh, pending, Battery(self.battery, soc_mwh))
        self.reference_soc[view.now] = reference
        self.decision_soc[view.now] = soc_mwh
        self.pending_slots[view.now] = [at for at, _ in pending]
        steps = self.curve_steps(view.now, soc_mwh, reference, [o for _, o in pending])
        offers = steps[CurveStep.FEASIBLE]
        self.submitted[view.now] = offers
        return offers
