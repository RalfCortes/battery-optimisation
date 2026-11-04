"""Bidding uplift: water-value offer curves vs the self-schedule baseline.

Both strategies see the same forecasts (issued at gate closure), the same simulator, gate closure
and end-of-day rule; only how the battery participates differs:
  - self-schedule (CLAUDE.md baseline): MILP plan on the median forecast, offered at floor/cap
  - water value: SDP on the copula scenarios -> K-tranche offer curves (``WaterValueStrategy``)
Uplift = profit(water value) - profit(self-schedule). Perfect foresight (daily horizon, same
end-of-day rule) gives the capture rates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
from pydantic import BaseModel

from battery_optimisation.bidding.strategy import Horizon, ScheduleStrategy, WaterValueStrategy
from battery_optimisation.bidding.sweep import StepGap, SweepStats
from battery_optimisation.columns import Col
from battery_optimisation.config import BatteryConfig, BiddingConfig, FeasibilityMode, MarketConfig
from battery_optimisation.data.nems import TZ
from battery_optimisation.forecast.base import Forecaster
from battery_optimisation.forecast.baselines import WithAncillaryPersistence
from battery_optimisation.forecast.perfect import PerfectForecaster
from battery_optimisation.forecast.stored import QuantileForecaster
from battery_optimisation.sim.backtest import RunSummary, StrategyName, run_strategy, save_run
from battery_optimisation.sim.clearing import Side
from battery_optimisation.sim.dispatch import Dispatch
from battery_optimisation.sim.simulator import ShortfallPolicy


class StrategyResult(BaseModel):
    strategy: StrategyName
    horizon: Horizon
    run: str
    profit_sgd: float
    cycles_per_day: float
    capture_vs_perfect_day: float


class MonthlyUplift(BaseModel):
    period_start: date
    self_schedule_sgd: float
    water_value_sgd: float
    uplift_sgd: float


class BiddingReport(BaseModel):
    created_at: datetime
    start: date
    end: date
    gate_closure_intervals: int
    n_tranches: int
    bidding: BiddingConfig
    results: list[StrategyResult]
    uplift_sgd: float  # water value - self-schedule
    uplift_pct: float  # of |self-schedule profit|
    monthly: list[MonthlyUplift]
    sdp_days: int
    sdp_solve_time_s: float
    mean_tranches_per_interval: float  # offered, both sides
    price_taking_share: float  # share of offered MW made price-taking (floor / cap)
    feasibility: str = "worst_case"  # FeasibilityMode used (+ margin, quantile in ``bidding``)
    shortfall_mwh: float = 0.0  # water value: cleared but not deliverable
    penalty_sgd: float = 0.0  # water value: shortfall penalties (already in its profit)
    sweep: SweepStats | None = None  # price-sweep sanity check of the submitted curves
    sweep_steps: list[StepGap] = []  # gap to the optimum after each pipeline step


@dataclass(frozen=True)
class BiddingEvaluation:
    report: BiddingReport
    runs: dict[StrategyName, tuple[RunSummary, Dispatch]]
    water_value: WaterValueStrategy


def _offer_stats(strategy: WaterValueStrategy, market: MarketConfig) -> tuple[float, float]:
    offers = list(strategy.submitted.values())
    n = sum(len(o.tranches) for o in offers) / max(len(offers), 1)
    total = sum(t.quantity_mw for o in offers for t in o.tranches)
    sure = sum(
        t.quantity_mw
        for o in offers
        for t in o.tranches
        if (t.side is Side.DISCHARGE and t.price <= market.price_floor)
        or (t.side is Side.CHARGE and t.price >= market.price_cap)
    )
    return n, sure / total if total else 0.0


def evaluate_bidding(
    scenarios: pl.DataFrame,
    quantiles: pl.DataFrame,
    realised: pl.DataFrame,
    battery: BatteryConfig,
    market: MarketConfig,
    bidding: BiddingConfig,
    horizon: Horizon = Horizon.DAY,
    history: pl.DataFrame | None = None,
    ancillary: bool = False,
) -> BiddingEvaluation:
    """Perfect foresight, self-schedule and water value over ``realised``'s window, all planned
    with ``horizon`` (DAY for daily forecasts, ROLLING for intraday ones)."""
    keep = [Col.INTERVAL_START, Col.USEP]
    if ancillary:  # realised ancillary prices are traded; forecasts use persistence for them
        keep += [c for c in realised.columns if c not in keep]
    prices = realised.select(keep).sort(Col.INTERVAL_START)
    days = horizon
    water_value = WaterValueStrategy(scenarios, battery, market, bidding, ancillary=ancillary)
    median: Forecaster = QuantileForecaster(quantiles, level=0.5)
    if ancillary:
        median = WithAncillaryPersistence(median)
    strategies = {
        StrategyName.PERFECT_FORESIGHT: ScheduleStrategy(
            PerfectForecaster(prices), battery, market, days
        ),
        StrategyName.CHRONOS_MEDIAN: ScheduleStrategy(median, battery, market, days),
        StrategyName.WATER_VALUE: water_value,
    }
    probabilistic = bidding.feasibility.mode is FeasibilityMode.PROBABILISTIC
    runs = {
        name: run_strategy(
            name,
            strategy,
            days,
            prices,
            battery,
            market,
            history=history,
            on_shortfall=ShortfallPolicy.PENALISE
            if name is StrategyName.WATER_VALUE and probabilistic
            else ShortfallPolicy.RAISE,
        )
        for name, strategy in strategies.items()
    }
    perfect = runs[StrategyName.PERFECT_FORESIGHT][0].total.profit_sgd
    results = [
        StrategyResult(
            strategy=name,
            horizon=days,
            run=summary.name,
            profit_sgd=summary.total.profit_sgd,
            cycles_per_day=summary.cycles_per_day,
            capture_vs_perfect_day=summary.total.profit_sgd / perfect,
        )
        for name, (summary, _) in runs.items()
    ]
    ss = runs[StrategyName.CHRONOS_MEDIAN][0]
    wv = runs[StrategyName.WATER_VALUE][0]
    uplift = wv.total.profit_sgd - ss.total.profit_sgd
    monthly = [
        MonthlyUplift(
            period_start=a.period_start,
            self_schedule_sgd=a.profit_sgd,
            water_value_sgd=b.profit_sgd,
            uplift_sgd=b.profit_sgd - a.profit_sgd,
        )
        for a, b in zip(ss.monthly, wv.monthly, strict=True)
    ]
    n_tranches, sure_share = _offer_stats(water_value, market)
    report = BiddingReport(
        created_at=datetime.now(ZoneInfo(TZ)),
        start=wv.start,
        end=wv.end,
        gate_closure_intervals=market.gate_closure_intervals,
        n_tranches=market.n_tranches,
        bidding=bidding,
        results=results,
        uplift_sgd=uplift,
        uplift_pct=uplift / abs(ss.total.profit_sgd) if ss.total.profit_sgd else float("nan"),
        monthly=monthly,
        sdp_days=water_value.n_plans,
        sdp_solve_time_s=water_value.solve_time_s,
        mean_tranches_per_interval=n_tranches,
        price_taking_share=sure_share,
        feasibility=str(bidding.feasibility.mode),
        shortfall_mwh=wv.total.shortfall_mwh,
        penalty_sgd=wv.total.penalty_sgd,
    )
    return BiddingEvaluation(report=report, runs=runs, water_value=water_value)


def save_bidding(ev: BiddingEvaluation, out_dir: Path) -> Path:
    """``report.json`` + one backtest folder per strategy under ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(ev.report.model_dump_json(indent=2))
    for summary, dispatch in ev.runs.values():
        save_run(summary, dispatch, out_dir / "backtests")
    return out_dir


def load_bidding_report(out_dir: Path) -> BiddingReport:
    return BiddingReport.model_validate_json((Path(out_dir) / "report.json").read_text())


# --------------------------------------------------------------------------- comparison


class ComparisonInput(BaseModel):
    strategy: StrategyName
    horizon: Horizon
    profit_sgd: float
    cycles_per_day: float


def comparison_table(rows: list[ComparisonInput], baseline: StrategyName) -> pl.DataFrame:
    """One row per (strategy, horizon): profit, cycles, capture vs both perfect-foresight
    benchmarks, and difference to the ``baseline`` strategy (SGD and share of its profit)."""

    def profit(name: StrategyName, horizon: Horizon | None = None) -> float | None:
        return next(
            (
                r.profit_sgd
                for r in rows
                if r.strategy is name and (horizon is None or r.horizon is horizon)
            ),
            None,
        )

    pf_full = profit(StrategyName.PERFECT_FORESIGHT, Horizon.FULL)
    pf_day = profit(StrategyName.PERFECT_FORESIGHT, Horizon.DAY)
    base = profit(baseline)
    return pl.DataFrame(
        [
            {
                "strategy": str(r.strategy),
                "horizon": str(r.horizon),
                "profit_sgd": r.profit_sgd,
                "cycles_per_day": r.cycles_per_day,
                "capture_vs_perfect_full": r.profit_sgd / pf_full if pf_full else None,
                "capture_vs_perfect_day": r.profit_sgd / pf_day if pf_day else None,
                "vs_baseline_sgd": r.profit_sgd - base,
                "vs_baseline_pct": (r.profit_sgd - base) / abs(base) if base else float("nan"),
            }
            for r in rows
        ]
    )


def to_markdown(table: pl.DataFrame) -> str:
    """Plain markdown rendering (money rounded, ratios as %)."""
    pct = {"capture_vs_perfect_full", "capture_vs_perfect_day", "vs_baseline_pct"}
    money = {"profit_sgd", "vs_baseline_sgd"}

    def cell(col: str, v) -> str:
        if isinstance(v, float) and col in pct:
            return f"{v:+.1%}" if col == "vs_baseline_pct" else f"{v:.1%}"
        if isinstance(v, float) and col in money:
            return f"{v:,.0f}"
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)

    head = "| " + " | ".join(table.columns) + " |"
    sep = "|" + "|".join("---" for _ in table.columns) + "|"
    body = [
        "| " + " | ".join(cell(c, r[c]) for c in table.columns) + " |" for r in table.to_dicts()
    ]
    return "\n".join([head, sep, *body]) + "\n"
