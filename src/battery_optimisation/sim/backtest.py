"""Backtest runs: a strategy through the simulator over a window, summarised and saved.

A run is saved as a folder (``RunFiles``):
  summary.json      ``RunSummary`` (pydantic): settings, totals, monthly and daily statistics
  dispatch.parquet  ``Dispatch.frame``: one row per interval (USEP, MW, SOC, revenue, profit)
Load it back with ``load_run``.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo

import polars as pl
import structlog
from pydantic import BaseModel, ConfigDict

from battery_optimisation.battery import Battery
from battery_optimisation.bidding.strategy import Horizon, ScheduleStrategy
from battery_optimisation.columns import Col
from battery_optimisation.config import BatteryConfig, MarketConfig
from battery_optimisation.data.nems import TZ, NemsSeries, load_nems
from battery_optimisation.forecast.base import Forecaster
from battery_optimisation.forecast.perfect import PerfectForecaster
from battery_optimisation.sim.clearing import Market, price_column
from battery_optimisation.sim.dispatch import Dispatch
from battery_optimisation.sim.metrics import Period, PeriodStats
from battery_optimisation.sim.simulator import ShortfallPolicy, Simulator, Strategy

log = structlog.get_logger(__name__)

DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[3] / "runs"


class StrategyName(StrEnum):
    """MILP self-schedule driven by different forecasts (``ScheduleStrategy`` + forecaster)."""

    PERFECT_FORESIGHT = "perfect_foresight"  # realised prices (oracle, benchmark)
    CHRONOS_MEDIAN = "chronos_median"  # Chronos-2 median forecast
    PERSISTENCE = "persistence"  # same interval yesterday
    WATER_VALUE = "water_value"  # SDP water-value offer curves on copula scenarios (bidding)


class ReportingStrategy(Strategy, Protocol):
    """A strategy that reports what it planned (for ``RunSummary``)."""

    @property
    def planned_profit_sgd(self) -> float: ...

    @property
    def n_plans(self) -> int: ...

    @property
    def solve_time_s(self) -> float: ...


class RunFiles(StrEnum):
    SUMMARY = "summary.json"
    DISPATCH = "dispatch.parquet"
    OFFERS = "offers.parquet"  # every submitted tranche
    PLANS = "plans.parquet"  # what the strategy planned at each decision (bidding runs)


class RunSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy: StrategyName
    horizon: Horizon
    start: date
    end: date  # inclusive
    days: int
    created_at: datetime
    battery: BatteryConfig
    interval_h: float
    planned_profit_sgd: float  # sum of MILP objectives (on forecast prices)
    plans: int  # number of MILP solves
    solve_time_s: float  # MILP time
    simulate_time_s: float  # whole simulation, including MILP time
    total: PeriodStats
    profit_per_day_sgd: float
    cycles_per_day: float
    monthly: list[PeriodStats]
    daily: list[PeriodStats]

    ancillary: bool = False  # regulation and reserves co-optimised with energy

    @property
    def name(self) -> str:
        suffix = "_as" if self.ancillary else ""
        return f"{self.strategy}_{self.horizon}_{self.start}_{self.end}{suffix}"


def last_months_window(months: int, end: date | None = None) -> tuple[date, date]:
    """``months`` whole months ending on ``end`` (default: last day with USEP history)."""
    if end is None:
        usep = load_nems(series=[NemsSeries.USEP]).drop_nulls(Col.USEP)
        end = usep[Col.INTERVAL_START].max().date()
    year, month = divmod(end.month - 1 - months, 12)
    year += end.year
    try:
        start = end.replace(year=year, month=month + 1)
    except ValueError:  # e.g. 31st in a shorter month
        start = end.replace(year=year, month=month + 2, day=1) - timedelta(days=1)
    return start + timedelta(days=1), end


def run_schedule(
    name: StrategyName,
    forecaster: Forecaster,
    prices: pl.DataFrame,
    battery: BatteryConfig,
    market: MarketConfig,
    horizon: Horizon = Horizon.FULL,
    history: pl.DataFrame | None = None,
) -> tuple[RunSummary, Dispatch]:
    """MILP self-schedule on ``forecaster``'s prices, cleared against realised ``prices``.

    ``history``: realised prices before the window (visible to forecasters, not traded).
    """
    strategy = ScheduleStrategy(forecaster, battery, market, horizon)
    return run_strategy(name, strategy, horizon, prices, battery, market, history)


def run_strategy(
    name: StrategyName,
    strategy: ReportingStrategy,
    horizon: Horizon,
    prices: pl.DataFrame,
    battery: BatteryConfig,
    market: MarketConfig,
    history: pl.DataFrame | None = None,
    on_shortfall: ShortfallPolicy = ShortfallPolicy.RAISE,
) -> tuple[RunSummary, Dispatch]:
    """Any strategy through the simulator over ``prices``; summarised."""
    t0 = time.perf_counter()
    simulator = Simulator(Market(market, prices, history), Battery(battery), on_shortfall)
    dispatch = simulator.run(strategy)
    simulate_time = time.perf_counter() - t0

    if name is StrategyName.PERFECT_FORESIGHT and horizon is not Horizon.ROLLING:
        # With a perfect forecast, the simulator must reproduce the plans (rolling plans
        # overlap: their objectives do not add up), up to offers rounded down to the market's
        # quantity step (0.1 MW): within 0.1% here.
        profit = dispatch.profit_sgd
        if abs(profit - strategy.planned_profit_sgd) > 1e-3 * max(1.0, abs(profit)):
            raise RuntimeError(
                f"simulated profit {profit:.2f} != planned {strategy.planned_profit_sgd:.2f}"
            )
    start = prices[Col.INTERVAL_START].min().date()
    end = prices[Col.INTERVAL_START].max().date()
    days = (end - start).days + 1
    total = dispatch.total()
    summary = RunSummary(
        strategy=name,
        horizon=horizon,
        start=start,
        end=end,
        days=days,
        created_at=datetime.now(ZoneInfo(TZ)),
        battery=battery,
        interval_h=market.interval_h,
        planned_profit_sgd=strategy.planned_profit_sgd,
        plans=strategy.n_plans,
        solve_time_s=strategy.solve_time_s,
        simulate_time_s=simulate_time,
        total=total,
        profit_per_day_sgd=total.profit_sgd / days,
        cycles_per_day=total.cycles / days,
        monthly=dispatch.stats(Period.MONTH),
        daily=dispatch.stats(Period.DAY),
    )
    log.info(
        "backtest.done",
        run=summary.name,
        profit_sgd=round(total.profit_sgd),
        cycles_per_day=round(summary.cycles_per_day, 2),
        simulate_time_s=round(simulate_time, 1),
    )
    return summary, dispatch


def load_prices(start: date, end: date, market: MarketConfig, ancillary: bool) -> pl.DataFrame:
    """Realised USEP (+ the enabled ancillary products' price columns) for [start, end]."""
    if not ancillary:
        return load_nems(start, end, series=[NemsSeries.USEP]).select(Col.INTERVAL_START, Col.USEP)
    frame = load_nems(start, end)
    cols = [price_column(p, market) for p in market.ancillary_products]
    return frame.select(Col.INTERVAL_START, Col.USEP, *cols)


def run_perfect_foresight(
    start: date,
    end: date,
    battery: BatteryConfig,
    market: MarketConfig,
    horizon: Horizon = Horizon.FULL,
    ancillary: bool = False,
) -> tuple[RunSummary, Dispatch]:
    """Perfect-foresight benchmark: MILP on realised prices (USEP, plus the enabled ancillary
    products when ``ancillary``), self-scheduled through the simulator."""
    prices = load_prices(start, end, market, ancillary)
    summary, dispatch = run_schedule(
        StrategyName.PERFECT_FORESIGHT, PerfectForecaster(prices), prices, battery, market, horizon
    )
    return summary.model_copy(update={"ancillary": ancillary}), dispatch


def save_run(summary: RunSummary, dispatch: Dispatch, runs_dir: Path = DEFAULT_RUNS_DIR) -> Path:
    run_dir = Path(runs_dir) / summary.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / RunFiles.SUMMARY).write_text(summary.model_dump_json(indent=2))
    dispatch.frame.write_parquet(run_dir / RunFiles.DISPATCH)
    dispatch.offers.write_parquet(run_dir / RunFiles.OFFERS)
    return run_dir


def load_run(run_dir: Path) -> tuple[RunSummary, pl.DataFrame]:
    run_dir = Path(run_dir)
    summary = RunSummary.model_validate_json((run_dir / RunFiles.SUMMARY).read_text())
    return summary, pl.read_parquet(run_dir / RunFiles.DISPATCH)


def list_runs(runs_dir: Path = DEFAULT_RUNS_DIR) -> list[Path]:
    """Run folders, most recently written first."""
    dirs = [p.parent for p in Path(runs_dir).glob(f"*/{RunFiles.SUMMARY}")]
    return sorted(dirs, key=lambda p: (p / RunFiles.SUMMARY).stat().st_mtime, reverse=True)
