"""Bidding strategies (implement ``sim.simulator.Strategy``).

``ScheduleStrategy``: at the start of each planning window, solve the MILP on the forecaster's
prices from the current SOC, then self-schedule the plan (offers that always clear). With a
``PerfectForecaster`` this is the perfect-foresight benchmark; with a real forecaster it is the
self-schedule baseline from CLAUDE.md.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from enum import StrEnum

from battery_optimisation.battery import Battery
from battery_optimisation.columns import Col
from battery_optimisation.config import BatteryConfig, MarketConfig
from battery_optimisation.forecast.base import Forecaster
from battery_optimisation.optim.milp import MilpSolution, solve_schedule
from battery_optimisation.sim.clearing import IntervalOffers, MarketView


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

    @property
    def planned_profit_sgd(self) -> float:
        """Sum of the MILP objectives of all plans made so far (on forecast prices)."""
        return sum(p.objective_sgd for p in self.plans)

    def _window_end(self, view: MarketView) -> datetime:
        if self.horizon is Horizon.FULL:
            return view.end
        last_of_day = datetime.combine(view.now.date(), time(), view.now.tzinfo) + timedelta(
            days=1, hours=-self.market.interval_h
        )
        return min(last_of_day, view.end)

    def _replan(self, view: MarketView, soc_mwh: float) -> None:
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
        self._plan = dict(
            zip(
                s[Col.INTERVAL_START].to_list(),
                zip(s[Col.CHARGE_MW].to_list(), s[Col.DISCHARGE_MW].to_list(), strict=True),
                strict=True,
            )
        )

    def offers(self, view: MarketView, soc_mwh: float) -> IntervalOffers:
        if view.now not in self._plan:
            self._replan(view, soc_mwh)
        charge, discharge = self._plan[view.now]
        return IntervalOffers.self_schedule(view.now, charge, discharge, self.market)
