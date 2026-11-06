"""Bidding strategies (implement ``sim.simulator.Strategy``).

``ScheduleStrategy``: at the start of each planning window, solve the MILP on the forecaster's
prices from the current SOC, then self-schedule the plan (offers that always clear). With a
``PerfectForecaster`` this is the perfect-foresight benchmark; with a real forecaster it is the
self-schedule baseline from CLAUDE.md.
"""

from __future__ import annotations

import bisect
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, time, timedelta
from enum import StrEnum
from types import SimpleNamespace

import numpy as np
import polars as pl

from battery_optimisation.battery import TOL, Battery
from battery_optimisation.bidding.ancillary import curve_tranches, reserve_curve
from battery_optimisation.bidding.compress import compress
from battery_optimisation.bidding.feasibility import (
    caps_from_bounds,
    expected_soc_after,
    make_feasible,
    soc_quantiles_after,
    soc_range_after,
)
from battery_optimisation.bidding.stages import ReserveLattices, ScenarioStages, StageSource
from battery_optimisation.bidding.water_value import (
    offer_curve,
    price_take_beyond_band,
    trim_to_band,
)
from battery_optimisation.columns import Col
from battery_optimisation.config import (
    BatteryConfig,
    BiddingConfig,
    FeasibilityMode,
    MarketConfig,
    Product,
    ReserveOffers,
    ReservePlanner,
)
from battery_optimisation.forecast.base import Forecaster
from battery_optimisation.forecast.baselines import PersistenceForecaster
from battery_optimisation.optim.milp import MilpSolution, solve_reserve_plan, solve_schedule
from battery_optimisation.optim.reserves import (
    best_reserves,
    corners,
    pair_corners,
    reserve_set,
    stage_values,
)
from battery_optimisation.optim.sdp import (
    StageLimits,
    ValueFunction,
    expected_schedule,
    solve_sdp,
)
from battery_optimisation.sim.clearing import (
    AWARD_COLUMN,
    AncillaryTranche,
    IntervalOffers,
    MarketView,
    Side,
    Tranche,
    conform,
    price_column,
)
from battery_optimisation.sim.storage_rules import (
    lower_mw,
    raise_mw,
    reserve_draw_mwh,
    reserve_fill_mwh,
)


def AWARD_PRICE_COLUMNS(market: MarketConfig) -> list[str]:  # noqa: N802
    """Price columns of the enabled ancillary products."""
    return [price_column(p, market) for p in market.ancillary_products]


# plans.parquet: one row per decision x planned interval (values Float32: memory)
PLAN_COLUMNS = (
    Col.DECIDED_AT,
    Col.TARGET_START,
    Col.INTERVAL_START,
    Col.USEP_FORECAST,  # energy price the plan expected
    Col.NET_MW,  # planned (water value: expected) discharge - charge
    Col.SOC_MWH,  # end of interval (water value: expected)
    Col.SOC_P10_MWH,
    Col.SOC_P90_MWH,
    Col.REG_MW,
    Col.PRI_MW,
    Col.CON_MW,
)


def _plan_frame(
    decided_at: datetime, target: datetime, stamps: list[datetime], columns: dict
) -> pl.DataFrame:
    n = len(stamps)
    data = {
        Col.DECIDED_AT: [decided_at] * n,
        Col.TARGET_START: [target] * n,
        Col.INTERVAL_START: stamps,
        **{c: np.asarray(columns.get(c, np.zeros(n)), np.float32) for c in PLAN_COLUMNS[3:]},
    }
    return pl.DataFrame(data).with_columns(
        pl.col(Col.DECIDED_AT, Col.TARGET_START, Col.INTERVAL_START).dt.cast_time_unit("us")
    )


def collect_plans(frames: list[pl.DataFrame]) -> pl.DataFrame:
    if not frames:
        return pl.DataFrame(
            schema={
                **{c: pl.Datetime("us", "Asia/Singapore") for c in PLAN_COLUMNS[:3]},
                **{c: pl.Float32 for c in PLAN_COLUMNS[3:]},
            }
        )
    return pl.concat(frames)


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
        record_plans: bool = False,
    ) -> None:
        self.forecaster = forecaster
        self.battery = battery
        self.market = market
        self.horizon = horizon
        self.record_plans = record_plans  # keep every plan as a frame (``plans()``)
        self._plan_frames: list[pl.DataFrame] = []
        self.plans_made: list[MilpSolution] = []
        self._plan: dict[datetime, tuple[float, float]] = {}  # interval -> (charge, discharge)
        self._awards: dict[datetime, dict[Product, float]] = {}  # interval -> ancillary MW
        self.committed: dict[datetime, tuple[float, float]] = {}  # offered (charge, discharge)

    @property
    def planned_profit_sgd(self) -> float:
        """Sum of the MILP objectives of all plans made so far (on forecast prices)."""
        return sum(p.objective_sgd for p in self.plans_made)

    @property
    def n_plans(self) -> int:
        return len(self.plans_made)

    @property
    def solve_time_s(self) -> float:
        return sum(p.solve_time_s for p in self.plans_made)

    def plans(self) -> pl.DataFrame:
        """Every plan made (``PLAN_COLUMNS``): the MILP schedule from the decision on."""
        return collect_plans(self._plan_frames)

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
        this is exact (regulation energy: its expected value)."""
        b = Battery(self.battery, soc_mwh)
        dt, energy = self.market.interval_h, self.market.ancillary.regulation_energy
        drift = energy.down * self.battery.eta_c - energy.up / self.battery.eta_d
        soc, at = soc_mwh, view.decided_at
        while at < view.now:
            charge, discharge = self.committed.get(at, (0.0, 0.0))
            reg = self._awards.get(at, {}).get(Product.REGULATION, 0.0)
            soc += b.charge_gain(dt) * charge - b.discharge_loss(dt) * discharge
            soc += reg * dt * drift
            at += timedelta(hours=dt)
        cfg = self.battery
        return min(max(soc, cfg.soc_min), cfg.soc_max)

    def _replan(self, view: MarketView, soc_mwh: float) -> None:
        soc_mwh = self._start_soc(view, soc_mwh)
        prices = self.forecaster.forecast(view, self._window_end(view))
        # Each window must leave at least the initial SOC: no free energy taken from the end
        # (or as close as the window can charge: offers rounded to the MW step may leave a
        # short last window just below what it needs).
        here = Battery(self.battery, soc_mwh=soc_mwh)
        reachable = soc_mwh + prices.height * here.charge_gain(self.market.interval_h) * (
            self.battery.p_max_mw
        )
        plan = solve_schedule(
            prices,
            here,
            self.market,
            soc_terminal_min=min(self.battery.soc_init, reachable - 1e-6),
        )
        self.plans_made.append(plan)
        s = plan.schedule
        if self.record_plans:
            soc = s[Col.SOC_MWH]
            columns = {
                Col.USEP_FORECAST: s[Col.USEP],
                Col.NET_MW: s[Col.DISCHARGE_MW] - s[Col.CHARGE_MW],
                Col.SOC_MWH: soc,
                Col.SOC_P10_MWH: soc,
                Col.SOC_P90_MWH: soc,
                **{c: s[c] for c in AWARD_COLUMN.values() if c in s.columns},
            }
            self._plan_frames.append(
                _plan_frame(view.decided_at, view.now, s[Col.INTERVAL_START].to_list(), columns)
            )
        stamps = s[Col.INTERVAL_START].to_list()
        award_cols = {p: col for p, col in AWARD_COLUMN.items() if col in s.columns}
        self._awards |= {
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
        offers = IntervalOffers.self_schedule(
            view.now, charge, discharge, self.market, self._awards.get(view.now)
        )
        # what was offered (rounded to the market's MW step): exactly what will clear
        self.committed[view.now] = (
            sum(t.quantity_mw for t in offers.on(Side.CHARGE)),
            sum(t.quantity_mw for t in offers.on(Side.DISCHARGE)),
        )
        self._awards[view.now] = {t.product: t.quantity_mw for t in offers.ancillary}
        return offers


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
        reserve_lattices: ReserveLattices | None = None,
    ) -> None:
        self.battery, self.market, self.bidding = battery, market, bidding
        # reserve_prices markov: reserve price states per decision (else point forecasts)
        self.reserve_lattices = reserve_lattices
        self.reserve_lattice_fallbacks = 0  # issues with no lattice: point forecast used
        self._reserve_chain: dict[datetime, np.ndarray] = {}  # issue -> reserve transitions
        self._reserve_weights: dict[datetime, np.ndarray] = {}
        self.ancillary = ancillary  # commit ancillary services from a plan, energy in the rest
        # ancillary prices for the reserve plan (default: persistence)
        self.ancillary_forecaster = ancillary_forecaster
        self.ancillary_fallbacks = 0  # issues where it had no forecast: persistence used
        # firm: reserves fit beside any energy flow; else the charge a plan needs to hold more
        # than P of raise (or discharge for lower) is committed price-taking
        self.firm_reserves = bidding.firm_reserves if firm_reserves is None else firm_reserves
        self._plan_awards: dict[datetime, dict[Product, float]] = {}  # target -> planned MW
        # reserve_planner sdp: SGD per MW held, per issue [T, P] (products of ``_reserves``), or
        # [T, R, P] per reserve price state
        self._reserve_values: dict[datetime, np.ndarray] = {}
        self._reserves = (
            reserve_set(market, battery, self.firm_reserves)
            if ancillary and bidding.reserve_planner is ReservePlanner.SDP
            else None
        )
        # reserve_offers curve: priced reserve tranches per target interval
        self.priced_reserves = (
            self._reserves is not None and bidding.reserve_offers is ReserveOffers.CURVE
        )
        if self.priced_reserves and bidding.feasibility.mode is not FeasibilityMode.MARKET:
            raise ValueError(
                "reserve_offers curve needs feasibility: market (the engine settles reserves "
                "against energy and the SoC)"
            )
        self._reserve_offers: dict[datetime, tuple[AncillaryTranche, ...]] = {}
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
        self._plan_frames: list[pl.DataFrame] = []  # ``bidding.record_plans``

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
        if self.bidding.reserve_planner is ReservePlanner.STOCHASTIC_MILP:
            plan = solve_reserve_plan(
                plan_prices.drop(Col.USEP),
                self._price_paths(issue),
                np.ones(self.bidding.stochastic_paths),
                Battery(self.battery, start),
                self.market,
                soc_terminal_min=self.battery.soc_init,
                firm_reserves=self.firm_reserves,
                relax=self.bidding.stochastic_relax,
            ).schedule
        else:
            plan = solve_schedule(
                plan_prices,
                Battery(self.battery, start),
                self.market,
                soc_terminal_min=self.battery.soc_init,
                firm_reserves=self.firm_reserves,
            ).schedule
        rules, b = self.market.storage_rules, self.battery
        rows = plan.select(Col.INTERVAL_START, *AWARD_COLUMN.values()).iter_rows(named=True)
        up, down, draw, fill = [], [], [], []
        for row in rows:
            awards = {p: float(row[c]) for p, c in AWARD_COLUMN.items()}
            self._plan_awards[row[Col.INTERVAL_START]] = awards
            up.append(raise_mw(rules, awards))
            down.append(lower_mw(rules, awards))
            draw.append(reserve_draw_mwh(rules, b, awards))
            fill.append(reserve_fill_mwh(rules, b, awards))
        return StageLimits(
            max_discharge_mw=np.maximum(b.p_max_mw - np.asarray(up), 0.0),
            max_charge_mw=np.maximum(b.p_max_mw - np.asarray(down), 0.0),
            soc_low=b.soc_min + np.asarray(draw),
            soc_high=b.soc_max - np.asarray(fill),
        )

    def _price_paths(self, issue: datetime) -> np.ndarray:
        """``stochastic_paths`` energy price paths [S, T] from the issue's price stages: Markov
        chains through the states (or independent draws), seeded by the issue time."""
        _, prices, weights = self._load_issue(issue)
        transitions = self._transitions[issue]
        rng = np.random.default_rng(int(issue.timestamp()))
        n, (stages, states) = self.bidding.stochastic_paths, prices.shape
        state = rng.choice(states, size=n, p=weights)
        paths = np.empty((n, stages))
        for t in range(stages):
            if t > 0:
                rows = transitions[state] if transitions is not None else np.tile(weights, (n, 1))
                cum = rows.cumsum(axis=1)
                state = (rng.random(n)[:, None] > cum).sum(axis=1).clip(max=states - 1)
            paths[:, t] = prices[t, state]
        return paths

    def decisions(self) -> pl.DataFrame:
        """Decision trace, one row per offered interval: issue and stage used, SOC known at
        gate closure and expected at the interval, the SDP's price states, planned reserves."""
        rows = []
        for at in sorted(self.submitted):
            issue, stage = self.issue_of[at]
            plan = self._plan_awards.get(at, {})
            rows.append(
                {
                    Col.INTERVAL_START: at,
                    Col.ISSUED_AT: issue,
                    Col.STAGE: stage,
                    Col.DECISION_SOC_MWH: self.decision_soc.get(at),
                    Col.EXPECTED_SOC_MWH: self.reference_soc.get(at),
                    Col.STATE_PRICES: [float(p) for p in self.stage_prices(issue)[stage]],
                    Col.PLANNED_REG_MW: plan.get(Product.REGULATION, 0.0),
                    Col.PLANNED_PRI_MW: plan.get(Product.PRIMARY_RESERVE, 0.0),
                    Col.PLANNED_CON_MW: plan.get(Product.CONTINGENCY_RESERVE, 0.0),
                }
            )
        return pl.DataFrame(rows)

    def plans(self) -> pl.DataFrame:
        """What the SDP policy planned at each decision (``PLAN_COLUMNS``): rolled forward from
        the expected SOC at the offered interval through the price states (expected power and
        SOC, its 10-90% band, the reserves held: chosen in the SDP, or the reserve plan)."""
        return collect_plans(self._plan_frames)

    def _record_plan(self, view: MarketView, issue: datetime, stage: int, soc: float) -> None:
        vf = self.value_functions[issue]
        if vf.policy is None:
            return  # independent price stages: no policy kept
        mixes = None
        if self._reserves is not None:
            corners_ = pair_corners(
                self.market,
                self.battery,
                self.bidding.soc_levels,
                self.market.interval_h,
                self.firm_reserves,
            )
            mixes = corners_.mixes(self._reserve_values[issue][stage:], vf.policy[stage:])
        transitions = self._transitions[issue]
        if issue in self._reserve_chain:  # joint (energy, reserve) states, s = k R + r
            transitions = np.kron(transitions, self._reserve_chain[issue])
        plan = expected_schedule(
            vf,
            transitions,
            Battery(self.battery),
            self.market.interval_h,
            stage,
            soc,
            mixes,
        )
        stamps = self._issue_prices[issue][0][stage:]
        columns = {
            Col.USEP_FORECAST: plan.price,
            Col.NET_MW: plan.net_mw,
            Col.SOC_MWH: plan.soc_mwh,
            Col.SOC_P10_MWH: plan.soc_p10,
            Col.SOC_P90_MWH: plan.soc_p90,
        }
        if plan.reserves is not None:
            for i, p in enumerate(self._reserves.products):
                columns[AWARD_COLUMN[p]] = plan.reserves[:, i]
        elif self._plan_awards:
            for p, col in AWARD_COLUMN.items():
                columns[col] = [self._plan_awards.get(at, {}).get(p, 0.0) for at in stamps]
        self._plan_frames.append(_plan_frame(view.decided_at, view.now, stamps, columns))

    @property
    def planned_awards(self) -> dict[datetime, dict[Product, float]]:
        """Reserve plan per interval (latest issue's), before feasibility scaling."""
        return self._plan_awards

    def _reserve_reward(self, issue: datetime, view: MarketView) -> np.ndarray | None:
        """Reserve planner ``sdp``: best reserve reward per stage and SDP move [T, N, N] from
        forecast ancillary prices (and the stage mean energy price for regulation energy).
        ``reserve_prices: markov``: per reserve price state [T, R, N, N] (factor lattice of the
        ancillary quantiles; its transitions in ``_reserve_chain``)."""
        stamps, prices, weights = self._load_issue(issue)
        dt = self.market.interval_h
        products = self._reserves.products
        corners_ = pair_corners(
            self.market, self.battery, self.bidding.soc_levels, dt, self.firm_reserves
        )
        if self.reserve_lattices is not None:
            lattice = self.reserve_lattices.load(view.decided_at, stamps)
            if lattice is None:
                self.reserve_lattice_fallbacks += 1
            else:
                t_, r_ = lattice.prices.shape[:2]
                order = [lattice.products.index(p) for p in products]
                ancillary = lattice.prices[:, :, order].reshape(t_ * r_, len(products))
                energy = np.repeat(prices @ weights, r_)
                values = stage_values(
                    self.market, self.battery, products, ancillary, energy, dt
                ).reshape(t_, r_, len(products))
                self._reserve_values[issue] = values
                self._reserve_chain[issue] = lattice.transitions
                self._reserve_weights[issue] = lattice.weights
                return corners_.rewards(values)
        known = SimpleNamespace(now=stamps[0], decided_at=view.decided_at, history=view.history)
        forecast = self._ancillary_prices(known, stamps[-1])
        frame = (
            pl.DataFrame({Col.INTERVAL_START: stamps})
            .with_columns(pl.col(Col.INTERVAL_START).dt.cast_time_unit("us"))
            .join(forecast, on=Col.INTERVAL_START, how="left")
        )
        columns = [price_column(p, self.market) for p in products]
        ancillary = np.column_stack(
            [
                frame[c].fill_null(0.0).to_numpy() if c in frame.columns else np.zeros(len(stamps))
                for c in columns
            ]
        )
        values = stage_values(self.market, self.battery, products, ancillary, prices @ weights, dt)
        self._reserve_values[issue] = values
        return corners_.rewards(values)

    def _expected_reserve_value(self, vf: ValueFunction, issue: datetime, t: int) -> np.ndarray:
        """SGD per MW held in stage ``t`` [P]; with reserve price states, the expectation over
        their distribution at t (reserves are offered before the price is known)."""
        value = self._reserve_values[issue][t]
        if value.ndim == 2:
            occ = vf.joint_occupancy[t].reshape(-1, value.shape[0]).sum(axis=0)
            value = occ @ value
        return value

    def _energy_options(self, vf: ValueFunction, t: int, soc: float) -> SimpleNamespace:
        """The energy actions of stage ``t`` from ``soc`` (targets on the SOC grid and at the
        power limits) and their value per energy price state [K, targets]: cash now + value
        after t (which depends on the price), minus wear; ``fits(awards)``: the actions a
        reserve mix allows (storage rules at ``soc``)."""
        rs, dt, b = self._reserves, self.market.interval_h, self.battery
        here = Battery(b, soc)
        lo = soc - here.max_discharge_mw(dt) * here.discharge_loss(dt)
        hi = soc + here.max_charge_mw(dt) * here.charge_gain(dt)
        grid = vf.soc[(vf.soc > lo + TOL) & (vf.soc < hi - TOL)]
        target = np.unique(np.concatenate([[lo, soc, hi], grid]))
        discharge = np.where(target < soc, (soc - target) / here.discharge_loss(dt), 0.0)
        charge = np.where(target > soc, (target - soc) / here.charge_gain(dt), 0.0)
        net = discharge - charge
        states = vf.state_prices[t]
        after = vf.continuation_at(t, states)  # [K, N]
        position = np.interp(target, vf.soc, np.arange(vf.soc.size, dtype=float))
        left = np.minimum(np.floor(position).astype(int), vf.soc.size - 2)
        f = position - left
        energy = (
            states[:, None] * net[None, :] * dt
            + (1 - f) * after[:, left]
            + f * after[:, left + 1]
            - here.degradation_sgd(charge, discharge, dt)
        )  # [K, targets]
        weights = vf.occupancy[t] if vf.occupancy is not None else np.full(states.size, 1.0)

        def fits(awards: np.ndarray):
            def allowed(c: np.ndarray, d: np.ndarray) -> np.ndarray:
                z = np.stack([np.full(np.size(c), soc), c, d, np.ones(np.size(c))], axis=1)
                return (rs.a @ awards <= z @ rs.rhs.T + 1e-6).all(axis=1)

            return allowed

        return SimpleNamespace(
            target=target,
            charge=charge,
            discharge=discharge,
            net=net,
            energy=energy,
            weights=weights / weights.sum(),
            fits=fits,
        )

    def _choose_reserves(
        self, vf: ValueFunction, issue: datetime, t: int, soc: float
    ) -> tuple[dict[Product, float], Callable[[np.ndarray, np.ndarray], np.ndarray]]:
        """Reserves to offer for stage ``t`` from ``soc``: among the mixes that are best for some
        price state (jointly with its energy action), the one with the highest expected value
        over the price states when the energy action must then fit beside it. Returns the
        awards and the energy actions they allow."""
        rs = self._reserves
        value = self._expected_reserve_value(vf, issue, t)
        o = self._energy_options(vf, t, soc)
        x, ok = corners(rs, np.full(o.target.size, soc), o.charge, o.discharge)
        reward, mix = best_reserves(x, ok, value)  # [targets], [targets, P]
        best = (o.energy + reward[None, :]).argmax(axis=1)
        candidates = np.unique(np.vstack([mix[best], np.zeros(len(rs.products))]), axis=0)

        def expected(awards: np.ndarray) -> float:
            keep = o.fits(awards)(o.charge, o.discharge) | (np.abs(o.net) < TOL)
            if not keep.any():
                return -np.inf
            return float(o.weights @ np.where(keep, o.energy, -np.inf).max(axis=1) + value @ awards)

        chosen = max(candidates, key=expected)
        return dict(zip(rs.products, map(float, chosen), strict=True)), o.fits(chosen)

    RESERVE_GRID_POINTS = 241  # reserve prices swept per curve (0 .. 3 x the highest state)

    def _reserve_curves(
        self, vf: ValueFunction, issue: datetime, t: int, soc: float
    ) -> tuple[dict[Product, float], tuple[AncillaryTranche, ...], np.ndarray]:
        """``reserve_offers: curve``: for each product, the MW the gate-closure choice would hold
        at each of its prices (the others at their expected value), over every valid reserve mix
        at ``soc`` (corners for each energy action); each step priced at its opportunity cost
        (``bidding.ancillary``). Returns the mix chosen at the expected prices (the plan), the
        tranches, and the MW held at price 0 (certain: the energy curve must fit beside it)."""
        rs, dt, m = self._reserves, self.market.interval_h, self.market
        value = self._expected_reserve_value(vf, issue, t)
        o = self._energy_options(vf, t, soc)
        x, ok = corners(rs, np.full(o.target.size, soc), o.charge, o.discharge)
        mixes = np.unique(
            np.vstack([np.round(x[ok], 6), np.zeros((1, len(rs.products)))]), axis=0
        )  # [C, P]
        z = np.stack([np.full(o.target.size, soc), o.charge, o.discharge, np.ones(o.target.size)])
        room = z.T @ rs.rhs.T  # [targets, M]
        keep = (mixes @ rs.a.T)[:, None, :] <= room[None, :, :] + 1e-6  # [C, targets, M]
        keep = keep.all(axis=2) | (np.abs(o.net) < TOL)[None, :]
        best = np.where(keep[None], o.energy[:, None, :], -np.inf).max(axis=2)  # [K, C]
        base = o.weights @ best  # [C]
        _, prices, weights = self._issue_prices[issue]
        energy_mean = np.array([prices[t] @ weights])
        extra = stage_values(
            m, self.battery, rs.products, np.zeros((1, len(rs.products))), energy_mean, dt
        )[0]
        states = self._reserve_values[issue][t]
        states = states if states.ndim == 2 else states[None, :]
        highest = ((states - extra) / dt).max(axis=0)  # [P] SGD/MW/h
        chosen = mixes[np.argmax(base + mixes @ value)]
        tranches, certain = [], np.zeros(len(rs.products))
        for i, p in enumerate(rs.products):
            grid = np.linspace(0.0, 3.0 * max(highest[i], 1.0), self.RESERVE_GRID_POINTS)
            mw = reserve_curve(mixes, base, value, extra, i, grid, dt)
            certain[i] = mw[0]
            tranches += curve_tranches(grid, mw, p, m.n_tranches)
        awards = dict(zip(rs.products, map(float, chosen), strict=True))
        return awards, tuple(t for t in tranches if t.product in m.ancillary_products), certain

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
            limits, reward = None, None
            if self._reserves is not None and view is not None:
                reward = self._reserve_reward(issue, view)
            elif view is not None:
                limits = self._ancillary_limits(issue, view, soc_mwh)
            vf = solve_sdp(
                prices,
                weights,
                Battery(self.battery),
                self.market.interval_h,
                self.bidding,
                limits=limits,
                transitions=self._transitions[issue],
                reserve_reward=reward,
                reserve_transitions=self._reserve_chain.get(issue),
                reserve_weights=self._reserve_weights.get(issue),
            )
            self._keep_policies(issue)
            self.value_functions[issue] = vf
            self.expected_profit_sgd[issue] = vf.at(stage, soc_mwh)
        return self.value_functions[issue]

    POLICIES_KEPT = 3  # value functions keep their policy (plans) for the latest issues only

    def _keep_policies(self, new_issue: datetime) -> None:
        """Drop the policy (and per-state reserve data) of older issues: memory over a year."""
        old = [i for i, v in self.value_functions.items() if v.policy is not None]
        for issue in old[: max(0, len(old) - self.POLICIES_KEPT + 1)]:
            if issue != new_issue:
                self.value_functions[issue] = replace(self.value_functions[issue], policy=None)

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
        allowed = None
        if self.priced_reserves:
            awards, tranches, certain = self._reserve_curves(vf, issue, t, reference)
            self._plan_awards[at] = awards  # the plan: the mix best at the expected prices
            self._reserve_offers[at] = tranches
            # the energy curve may use whatever is not committed for certain (offered at 0);
            # if more reserves clear, the engine settles them against energy by surplus
            allowed = self._energy_options(vf, t, reference).fits(certain)
        elif self._reserves is not None:
            awards, allowed = self._choose_reserves(vf, issue, t, reference)
            self._plan_awards[at] = awards
        discharge, charge = offer_curve(vf, t, reference, here, dt, allowed)
        steps[CurveStep.RAW] = offers(discharge, charge)
        discharge, charge = trim_to_band(discharge, charge, low, high)
        steps[CurveStep.BAND] = offers(discharge, charge)
        discharge, charge = clip(discharge), clip(charge)
        steps[CurveStep.COMPRESSED] = offers(discharge, charge)
        discharge, charge = price_take_beyond_band(
            discharge, charge, low, high, m.price_floor, m.price_cap
        )
        steps[CurveStep.PRICE_TAKING] = offers(discharge, charge)
        steps[CurveStep.FEASIBLE] = conform(
            self._feasible(steps[CurveStep.PRICE_TAKING], soc_mwh, pending, here), m
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
        if offers.interval_start in self._reserve_offers:  # priced curves, feasibility market:
            # the engine cuts the schedule (energy and reserves) to the SoC
            return offers.model_copy(
                update={"ancillary": self._reserve_offers[offers.interval_start]}
            )
        if self.ancillary and offers.interval_start in self._plan_awards:
            return self._feasible_with_ancillary(offers, soc_mwh, pending_offers, here)
        if f.mode is FeasibilityMode.MARKET:
            return offers  # the engine cuts the schedule to the SoC
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
        dt, rules, b = self.market.interval_h, self.market.storage_rules, self.battery
        need = raise_mw(rules, awards) - b.p_max_mw
        if need <= TOL:
            return 0.0
        room = (b.soc_max - reserve_fill_mwh(rules, b, awards) - high) / here.charge_gain(dt)
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
        Awards are offered at price 0 (they always clear). Feasibility ``market``: the same from
        the *expected* SOC (what can reasonably be delivered); the engine cuts to the SoC."""
        dt, m, b = self.market.interval_h, self.market, self.battery
        rules = m.storage_rules
        awards = dict(self._plan_awards[offers.interval_start])
        if self.bidding.feasibility.mode is FeasibilityMode.MARKET:
            # what can reasonably be delivered (Ch. 6 5.2B.5-6, 5.3.7, 5.4.8): from the
            # expected SOC; the engine cuts to the real SoC (RC386)
            low = high = self.reference_soc.get(offers.interval_start, soc_mwh)
        else:
            low, high = soc_range_after(soc_mwh, pending, here, dt, m.price_floor, m.price_cap)
        draw = reserve_draw_mwh(rules, b, awards)
        fill = reserve_fill_mwh(rules, b, awards)
        f_raise = min(1.0, max(0.0, (low - b.soc_min) / draw)) if draw > TOL else 1.0
        f_lower = min(1.0, max(0.0, (b.soc_max - high) / fill)) if fill > TOL else 1.0
        for p in awards:
            awards[p] *= min(f_raise, f_lower) if p in rules.lower_products else f_raise
        support = self._support_charge(awards, high, here) if not self.firm_reserves else 0.0
        # reserves beyond what the power (and the committed support charge) can back are cut
        for need, group, room in (
            (raise_mw, rules.raise_products, b.p_max_mw + support),
            (lower_mw, rules.lower_products, b.p_max_mw - support),
        ):
            total = need(rules, awards)
            if total > room:
                for p in group:
                    awards[p] *= max(room, 0.0) / total
        up, down = raise_mw(rules, awards), lower_mw(rules, awards)
        draw = reserve_draw_mwh(rules, b, awards)
        fill = reserve_fill_mwh(rules, b, awards)
        # the SOC at the end of this interval is the start of the next: keep what its planned
        # reserves need too, so energy here does not crowd them out
        following = self._plan_awards.get(offers.interval_start + timedelta(hours=dt))
        if following:
            draw = max(draw, reserve_draw_mwh(rules, b, following))
            fill = max(fill, reserve_fill_mwh(rules, b, following))
        if support > TOL:  # this interval: the committed charge only (certain, at the cap)
            energy = IntervalOffers(
                interval_start=offers.interval_start,
                tranches=(Tranche(side=Side.CHARGE, price=m.price_cap, quantity_mw=support),),
            )
        else:
            energy = caps_from_bounds(
                offers,
                low - draw,
                high + fill,
                here,
                dt,
                max_discharge_mw=b.p_max_mw - up,
                max_charge_mw=b.p_max_mw - down,
            )
        return energy.model_copy(update={"ancillary": self._award_tranches(awards)})

    def _award_tranches(self, awards: dict[Product, float]) -> tuple[AncillaryTranche, ...]:
        """Awards offered at price 0 (they always clear)."""
        return tuple(
            AncillaryTranche(product=p, price=0.0, quantity_mw=mw)
            for p, mw in awards.items()
            if mw > TOL and p in self.market.ancillary_products
        )

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
        if self.bidding.record_plans:
            self._record_plan(view, issue, stage, reference)
        return offers
