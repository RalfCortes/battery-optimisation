"""Bidding strategies (implement ``sim.simulator.Strategy``).

``ScheduleStrategy``: at the start of each planning window, solve the MILP on the forecaster's
prices from the current SOC, then self-schedule the plan (offers that always clear). With a
``PerfectForecaster`` this is the perfect-foresight benchmark; with a real forecaster it is the
self-schedule baseline from CLAUDE.md.
"""

from __future__ import annotations

import bisect
from datetime import datetime, time, timedelta
from enum import StrEnum
from types import SimpleNamespace

import numpy as np
import polars as pl

from battery_optimisation.battery import TOL, Battery
from battery_optimisation.bidding.compress import compress
from battery_optimisation.bidding.feasibility import (
    caps_from_bounds,
    expected_soc_after,
    make_feasible,
    soc_quantiles_after,
    soc_range_after,
)
from battery_optimisation.bidding.stages import ScenarioStages, StageSource
from battery_optimisation.bidding.water_value import (
    offer_curve,
    price_take_beyond_band,
    trim_to_band,
)
from battery_optimisation.columns import Col
from battery_optimisation.config import (
    LOWER_PRODUCTS,
    RAISE_PRODUCTS,
    BatteryConfig,
    BiddingConfig,
    FeasibilityMode,
    MarketConfig,
    Product,
)
from battery_optimisation.forecast.base import Forecaster
from battery_optimisation.forecast.baselines import PersistenceForecaster
from battery_optimisation.optim.milp import MilpSolution, solve_schedule
from battery_optimisation.optim.sdp import StageLimits, ValueFunction, solve_sdp
from battery_optimisation.sim.clearing import (
    AWARD_COLUMN,
    AncillaryTranche,
    IntervalOffers,
    MarketView,
    Side,
    Tranche,
    price_column,
)


def AWARD_PRICE_COLUMNS(market: MarketConfig) -> list[str]:  # noqa: N802
    """Price columns of the enabled ancillary products."""
    return [price_column(p, market) for p in market.ancillary_products]


class Horizon(StrEnum):
    """Planning window of ``ScheduleStrategy``."""

    FULL = "full"  # one plan to the end of the simulation: with perfect forecast, the true optimum
    DAY = "day"  # re-plan at the start of each day, to the end of that day
    ROLLING = "rolling"  # re-plan at every decision, over the next day's worth of intervals


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
        self._awards: dict[datetime, dict[Product, float]] = {}  # interval -> ancillary MW
        self.committed: dict[datetime, tuple[float, float]] = {}  # offered (charge, discharge)

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
        if self.horizon is Horizon.ROLLING:
            span = (self.market.intervals_per_day - 1) * timedelta(hours=self.market.interval_h)
            return min(view.now + span, view.end)
        last_of_day = datetime.combine(view.now.date(), time(), view.now.tzinfo) + timedelta(
            days=1, hours=-self.market.interval_h
        )
        return min(last_of_day, view.end)

    def _start_soc(self, view: MarketView, soc_mwh: float) -> float:
        """SOC at the start of ``view.now``: the SOC known at the decision plus the intervals
        already committed in between (gate closure). Self-scheduled offers always clear, so
        this is exact."""
        b = Battery(self.battery, soc_mwh)
        dt = self.market.interval_h
        soc, at = soc_mwh, view.decided_at
        while at < view.now:
            charge, discharge = self.committed.get(at, (0.0, 0.0))
            soc += b.charge_gain(dt) * charge - b.discharge_loss(dt) * discharge
            at += timedelta(hours=dt)
        cfg = self.battery
        return min(max(soc, cfg.soc_min), cfg.soc_max)

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
        award_cols = {p: col for p, col in AWARD_COLUMN.items() if col in s.columns}
        self._awards = {
            at: {p: float(row[col]) for p, col in award_cols.items()}
            for at, row in zip(
                stamps, s.select(list(award_cols.values())).iter_rows(named=True), strict=True
            )
        }
        self._plan = dict(
            zip(
                stamps,
                zip(s[Col.CHARGE_MW].to_list(), s[Col.DISCHARGE_MW].to_list(), strict=True),
                strict=True,
            )
        )

    def offers(self, view: MarketView, soc_mwh: float) -> IntervalOffers:
        if self.horizon is Horizon.ROLLING or view.now not in self._plan:
            self._replan(view, soc_mwh)
        charge, discharge = self._plan[view.now]
        self.committed[view.now] = (charge, discharge)
        return IntervalOffers.self_schedule(
            view.now, charge, discharge, self.market, self._awards.get(view.now)
        )


class CurveStep(StrEnum):
    """Steps that turn the water-value curve into submitted offers (``curve_steps``)."""

    RAW = "raw"  # micro-tranches from V_{t+1} around the expected SOC
    BAND = "band"  # trimmed to the interval's scenario price range
    COMPRESSED = "compressed"  # <= K tranches per side, clipped to [floor, cap]
    PRICE_TAKING = "price_taking"  # tranches beyond the band at floor / cap (certain)
    FEASIBLE = "feasible"  # capped from the worst-case SOC (= submitted)


class WaterValueStrategy:
    """Offer curves from the water value of the latest forecast issue (CLAUDE.md bidding layer).

    For each forecast issue (the latest one issued by the decision time): its scenarios give
    per-stage price distributions over its horizon; ``solve_sdp`` gives V_t (terminal: below
    soc_init penalised at the end of the horizon). With daily issues this is one SDP per day;
    with intraday issues one per decision. For interval t, decided at gate closure: the curve
    from V_{t+1} around the SOC *expected* at t (decision SOC + own pending offers cleared under
    the scenario prices), trimmed to tranches that can clear within the interval's scenario
    price range, compressed to K tranches per side, clipped to [floor, cap]; tranches that clear
    at every plausible price are made price-taking (certain); finally capped from the
    *worst-case* SOC after the pending offers (``feasibility``).
    """

    def __init__(
        self,
        scenarios: pl.DataFrame | StageSource,
        battery: BatteryConfig,
        market: MarketConfig,
        bidding: BiddingConfig,
        ancillary: bool = False,
        ancillary_forecaster: Forecaster | None = None,
        firm_reserves: bool | None = None,
    ) -> None:
        self.battery, self.market, self.bidding = battery, market, bidding
        self.ancillary = ancillary  # commit ancillary services from a plan, energy in the rest
        # ancillary prices for the reserve plan (default: persistence)
        self.ancillary_forecaster = ancillary_forecaster
        self.ancillary_fallbacks = 0  # issues where it had no forecast: persistence used
        # firm: reserves fit beside any energy flow; else the charge a plan needs to hold more
        # than P of raise (or discharge for lower) is committed price-taking
        self.firm_reserves = bidding.firm_reserves if firm_reserves is None else firm_reserves
        self._plan_awards: dict[datetime, dict[Product, float]] = {}  # target -> planned MW
        # per-issue price stages: copula scenarios (frame) or any StageSource (Markov lattice)
        self._stages: StageSource = (
            ScenarioStages(scenarios) if isinstance(scenarios, pl.DataFrame) else scenarios
        )
        self._issues: list[datetime] = self._stages.issues
        self._transitions: dict[datetime, np.ndarray | None] = {}
        self.value_functions: dict[datetime, ValueFunction] = {}  # by issue time
        self._issue_prices: dict[datetime, tuple[list[datetime], np.ndarray, np.ndarray]] = {}
        self.expected_profit_sgd: dict[datetime, float] = {}
        self.submitted: dict[datetime, IntervalOffers] = {}
        self.issue_of: dict[datetime, tuple[datetime, int]] = {}  # target -> (issue, stage)
        self.reference_soc: dict[datetime, float] = {}  # SOC each curve was built around
        self.decision_soc: dict[datetime, float] = {}  # SOC known at the decision
        self.pending_slots: dict[datetime, list[datetime]] = {}  # offers pending at the decision

    @property
    def planned_profit_sgd(self) -> float:
        """Sum of the SDPs' expected values when each issue was first used."""
        return sum(self.expected_profit_sgd.values())

    @property
    def n_plans(self) -> int:
        return len(self.value_functions)

    @property
    def solve_time_s(self) -> float:
        return sum(v.solve_time_s for v in self.value_functions.values())

    def stage_prices(self, issue: datetime) -> np.ndarray:
        """Price states (scenarios) [T, K] the issue's SDP used."""
        return self._issue_prices[issue][1]

    def _load_issue(self, issue: datetime) -> tuple[list[datetime], np.ndarray, np.ndarray]:
        if issue not in self._issue_prices:
            st = self._stages.load(issue)
            self._issue_prices[issue] = (st.stamps, st.prices, st.weights)
            self._transitions[issue] = st.transitions
        return self._issue_prices[issue]

    def _issue_for(self, view: MarketView) -> tuple[datetime, int]:
        """(latest issue known at the decision that covers ``view.now``, stage of ``now``)."""
        i = bisect.bisect_right(self._issues, view.decided_at)
        while i > 0:
            issue = self._issues[i - 1]
            stamps, _, _ = self._load_issue(issue)
            if stamps[0] <= view.now <= stamps[-1]:
                return issue, stamps.index(view.now)
            if stamps[-1] < view.now:
                break
            i -= 1
        raise ValueError(f"no scenarios covering {view.now} issued by {view.decided_at}")

    def _ancillary_limits(
        self, issue: datetime, view: MarketView, soc_mwh: float
    ) -> StageLimits | None:
        """Ancillary awards over the issue's horizon from a MILP on the scenarios' mean energy
        price and forecast ancillary prices; the per-stage limits they leave to energy."""
        if not self.ancillary:
            return None
        stamps, prices, weights = self._load_issue(issue)
        known = SimpleNamespace(now=stamps[0], decided_at=view.decided_at, history=view.history)
        ancillary = self._ancillary_prices(known, stamps[-1])
        if not any(c in ancillary.columns for c in AWARD_PRICE_COLUMNS(self.market)):
            return None
        plan_prices = pl.DataFrame({Col.INTERVAL_START: stamps, Col.USEP: prices @ weights})
        plan_prices = plan_prices.with_columns(
            pl.col(Col.INTERVAL_START).dt.cast_time_unit("us")
        ).join(ancillary, on=Col.INTERVAL_START, how="left")
        start = min(max(soc_mwh, self.battery.soc_min), self.battery.soc_max)
        plan = solve_schedule(
            plan_prices,
            Battery(self.battery, start),
            self.market,
            soc_terminal_min=self.battery.soc_init,
            firm_reserves=self.firm_reserves,
        ).schedule
        m, b = self.market, self.battery
        rows = plan.select(Col.INTERVAL_START, *AWARD_COLUMN.values()).iter_rows(named=True)
        raise_mw, lower_mw, raise_e, lower_e = [], [], [], []
        for row in rows:
            awards = {p: float(row[c]) for p, c in AWARD_COLUMN.items()}
            self._plan_awards[row[Col.INTERVAL_START]] = awards
            raise_mw.append(sum(awards[p] for p in RAISE_PRODUCTS))
            lower_mw.append(sum(awards[p] for p in LOWER_PRODUCTS))
            raise_e.append(sum(awards[p] * m.sustain_h(p) for p in RAISE_PRODUCTS))
            lower_e.append(sum(awards[p] * m.sustain_h(p) for p in LOWER_PRODUCTS))
        return StageLimits(
            max_discharge_mw=np.maximum(b.p_max_mw - np.asarray(raise_mw), 0.0),
            max_charge_mw=np.maximum(b.p_max_mw - np.asarray(lower_mw), 0.0),
            soc_low=b.soc_min + np.asarray(raise_e) / b.eta_d,
            soc_high=b.soc_max - np.asarray(lower_e) * b.eta_c,
        )

    @property
    def planned_awards(self) -> dict[datetime, dict[Product, float]]:
        """Reserve plan per interval (latest issue's), before feasibility scaling."""
        return self._plan_awards

    def _ancillary_prices(self, known, end: datetime) -> pl.DataFrame:
        if self.ancillary_forecaster is not None:
            try:
                out = self.ancillary_forecaster.forecast(known, end)
                return out.drop([c for c in (Col.USEP,) if c in out.columns])
            except ValueError:
                self.ancillary_fallbacks += 1
        return PersistenceForecaster(strict=False).forecast(known, end).drop(Col.USEP)

    def _value_function(
        self, issue: datetime, stage: int, soc_mwh: float, view: MarketView | None = None
    ) -> ValueFunction:
        if issue not in self.value_functions:
            _, prices, weights = self._load_issue(issue)
            limits = self._ancillary_limits(issue, view, soc_mwh) if view is not None else None
            vf = solve_sdp(
                prices,
                weights,
                Battery(self.battery),
                self.market.interval_h,
                self.bidding,
                limits=limits,
                transitions=self._transitions[issue],
            )
            self.value_functions[issue] = vf
            self.expected_profit_sgd[issue] = vf.at(stage, soc_mwh)
        return self.value_functions[issue]

    def _expected_soc(
        self, soc_mwh: float, pending: list[tuple[datetime, IntervalOffers]], here: Battery
    ) -> float:
        """SOC expected at the target: pending offers cleared under the scenarios they were
        built with (their issue's stage prices)."""
        if not pending:
            return soc_mwh
        rows, weights = [], None
        for at, _ in pending:
            issue, stage = self.issue_of[at]
            _, prices, w = self._issue_prices[issue]
            if weights is not None and len(w) != len(weights):
                return soc_mwh  # scenario sets of different sizes
            rows.append(prices[stage].tolist())
            weights = w
        return expected_soc_after(
            soc_mwh, [o for _, o in pending], rows, weights.tolist(), here, self.market.interval_h
        )

    def curve_steps(
        self,
        at: datetime,
        soc_mwh: float,
        reference: float,
        pending: list[tuple[datetime, IntervalOffers]],
    ) -> dict[CurveStep, IntervalOffers]:
        """The offers for interval ``at`` after each step of the pipeline (in order)."""
        dt, m = self.market.interval_h, self.market
        issue, t = self.issue_of[at]
        vf = self.value_functions[issue]
        here = Battery(self.battery, soc_mwh)
        stage_prices = self.stage_prices(issue)[t]
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
        steps[CurveStep.FEASIBLE] = self._feasible(
            steps[CurveStep.PRICE_TAKING], soc_mwh, pending, here
        )
        return steps

    def _feasible(
        self,
        offers: IntervalOffers,
        soc_mwh: float,
        pending: list[tuple[datetime, IntervalOffers]],
        here: Battery,
    ) -> IntervalOffers:
        """Cap tranche sizes from the SOC bounds after the pending offers (``feasibility``)."""
        dt, m, f = self.market.interval_h, self.market, self.bidding.feasibility
        pending_offers = [o for _, o in pending]
        if self.ancillary and offers.interval_start in self._plan_awards:
            return self._feasible_with_ancillary(offers, soc_mwh, pending_offers, here)
        if f.mode is FeasibilityMode.WORST_CASE or not pending:
            return make_feasible(
                offers, soc_mwh, pending_offers, here, dt, m.price_floor, m.price_cap
            )
        worst_low, worst_high = soc_range_after(
            soc_mwh, pending_offers, here, dt, m.price_floor, m.price_cap
        )
        rows, weights = [], []
        for at, _ in pending:
            issue, stage = self.issue_of[at]
            _, prices, w = self._issue_prices[issue]
            rows.append(prices[stage])
            weights.append(w)
        q_low, q_high = soc_quantiles_after(
            soc_mwh, pending_offers, rows, weights, here, dt, f.soc_quantile
        )
        margin = f.margin_pct / 100 * self.battery.usable_mwh
        low = max(worst_low, q_low - margin)
        high = min(worst_high, q_high + margin)
        return caps_from_bounds(offers, low, high, here, dt)

    def _support_charge(self, awards: dict[Product, float], high: float, here: Battery) -> float:
        """Charge (MW) committed so raise reserves beyond P stay deliverable, limited by what the
        worst-case high SOC can absorb while keeping the lower reserves' room."""
        dt, m, b = self.market.interval_h, self.market, self.battery
        need = sum(awards[p] for p in RAISE_PRODUCTS) - b.p_max_mw
        if need <= TOL:
            return 0.0
        lower_e = sum(awards[p] * m.sustain_h(p) for p in LOWER_PRODUCTS)
        room = (b.soc_max - lower_e * b.eta_c - high) / here.charge_gain(dt)
        return max(0.0, min(need, room))

    def _feasible_with_ancillary(
        self,
        offers: IntervalOffers,
        soc_mwh: float,
        pending: list[IntervalOffers],
        here: Battery,
    ) -> IntervalOffers:
        """Worst-case joint feasibility: planned ancillary awards scaled to what the worst-case
        SOC after the pending offers can sustain, then energy capped to the headroom left.
        Awards are offered at price 0 (they always clear)."""
        dt, m, b = self.market.interval_h, self.market, self.battery
        low, high = soc_range_after(soc_mwh, pending, here, dt, m.price_floor, m.price_cap)
        awards = dict(self._plan_awards[offers.interval_start])
        raise_e = sum(awards[p] * m.sustain_h(p) for p in RAISE_PRODUCTS)
        lower_e = sum(awards[p] * m.sustain_h(p) for p in LOWER_PRODUCTS)
        f_raise = min(1.0, max(0.0, (low - b.soc_min) * b.eta_d / raise_e)) if raise_e else 1.0
        f_lower = min(1.0, max(0.0, (b.soc_max - high) / (b.eta_c * lower_e))) if lower_e else 1.0
        for p in awards:
            awards[p] *= min(f_raise, f_lower) if p in LOWER_PRODUCTS else f_raise
        support = self._support_charge(awards, high, here) if not self.firm_reserves else 0.0
        # reserves beyond what the power (and the committed support charge) can back are cut
        for group, room in (
            (RAISE_PRODUCTS, b.p_max_mw + support),
            (LOWER_PRODUCTS, b.p_max_mw - support),
        ):
            total = sum(awards[p] for p in group)
            if total > room:
                for p in group:
                    awards[p] *= max(room, 0.0) / total
        raise_mw = sum(awards[p] for p in RAISE_PRODUCTS)
        lower_mw = sum(awards[p] for p in LOWER_PRODUCTS)
        raise_e = sum(awards[p] * m.sustain_h(p) for p in RAISE_PRODUCTS)
        lower_e = sum(awards[p] * m.sustain_h(p) for p in LOWER_PRODUCTS)
        # the SOC at the end of this interval is the start of the next: keep what its planned
        # reserves need too, so energy here does not crowd them out
        following = self._plan_awards.get(offers.interval_start + timedelta(hours=dt))
        if following:
            raise_e = max(raise_e, sum(following[p] * m.sustain_h(p) for p in RAISE_PRODUCTS))
            lower_e = max(lower_e, sum(following[p] * m.sustain_h(p) for p in LOWER_PRODUCTS))
        if support > TOL:  # this interval: the committed charge only (certain, at the cap)
            energy = IntervalOffers(
                interval_start=offers.interval_start,
                tranches=(Tranche(side=Side.CHARGE, price=m.price_cap, quantity_mw=support),),
            )
        else:
            energy = caps_from_bounds(
                offers,
                low - raise_e / b.eta_d,
                high + lower_e * b.eta_c,
                here,
                dt,
                max_discharge_mw=b.p_max_mw - raise_mw,
                max_charge_mw=b.p_max_mw - lower_mw,
            )
        ancillary = tuple(
            AncillaryTranche(product=p, price=0.0, quantity_mw=mw)
            for p, mw in awards.items()
            if mw > TOL and p in m.ancillary_products
        )
        return energy.model_copy(update={"ancillary": ancillary})

    def offers(self, view: MarketView, soc_mwh: float) -> IntervalOffers:
        step = timedelta(hours=self.market.interval_h)
        gaps = round((view.now - view.decided_at) / step)  # = gate closure intervals
        slots = [view.now - k * step for k in range(gaps, 0, -1)]  # decided .. now, in order
        pending = [(at, self.submitted[at]) for at in slots if at in self.submitted]
        issue, stage = self._issue_for(view)
        self.issue_of[view.now] = (issue, stage)
        self._value_function(issue, stage, soc_mwh, view)
        reference = self._expected_soc(soc_mwh, pending, Battery(self.battery, soc_mwh))
        self.reference_soc[view.now] = reference
        self.decision_soc[view.now] = soc_mwh
        self.pending_slots[view.now] = [at for at, _ in pending]
        steps = self.curve_steps(view.now, soc_mwh, reference, pending)
        offers = steps[CurveStep.FEASIBLE]
        self.submitted[view.now] = offers
        return offers
