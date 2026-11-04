"""Forecast evaluation over a test window: statistics, scenarios and capture rates.

``evaluate`` is pure (no I/O): give it quantile forecasts and realised prices. The CLI
(``uv run forecast run``) loads data, runs Chronos-2, calls ``evaluate`` and saves everything.

Capture rate = profit / perfect-foresight profit, against both benchmarks:
  - perfect foresight, full horizon: the true optimum (headline, CLAUDE.md)
  - perfect foresight, daily horizon: same structure as the forecast strategies (each day
    planned at 00:00 and ending at soc_init), so the gap is due to the forecast only
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
from pydantic import BaseModel

from battery_optimisation.bidding.strategy import Horizon
from battery_optimisation.columns import Col
from battery_optimisation.config import BatteryConfig, ForecastConfig, MarketConfig
from battery_optimisation.data.nems import PERIODS_PER_DAY, TZ
from battery_optimisation.data.weather import WeatherProcessingReport
from battery_optimisation.forecast.baselines import PersistenceForecaster
from battery_optimisation.forecast.perfect import PerfectForecaster
from battery_optimisation.forecast.stored import QuantileForecaster
from battery_optimisation.forecast.walkforward import (
    ForecastStats,
    ScenarioStats,
    calibration_pits,
    fit_month_copula,
    forecast_stats,
    lag1,
    month_scenarios,
    month_windows,
)
from battery_optimisation.sim.backtest import (
    DEFAULT_RUNS_DIR,
    RunSummary,
    StrategyName,
    list_runs,
    load_run,
    run_schedule,
    save_run,
)
from battery_optimisation.sim.dispatch import Dispatch
from battery_optimisation.sim.metrics import seasonal_naive_mae

_SGT = ZoneInfo(TZ)

RunKey = tuple[StrategyName, Horizon]


def schedule_horizon(cfg: ForecastConfig, market: MarketConfig) -> Horizon:
    """Re-plan with every forecast: rolling for intraday issues, per day for daily issues."""
    return Horizon.DAY if cfg.issue_every_intervals >= market.intervals_per_day else Horizon.ROLLING


def runs_for(horizon: Horizon) -> list[RunKey]:
    """Benchmarks (perfect foresight: full, daily, and the strategies' horizon) + strategies."""
    perfect = [Horizon.FULL, Horizon.DAY] + ([horizon] if horizon is Horizon.ROLLING else [])
    return [(StrategyName.PERFECT_FORESIGHT, h) for h in perfect] + [
        (StrategyName.CHRONOS_MEDIAN, horizon),
        (StrategyName.PERSISTENCE, horizon),
    ]


class CaptureStats(BaseModel):
    strategy: StrategyName
    horizon: Horizon
    run: str
    profit_sgd: float
    cycles_per_day: float
    capture_vs_perfect_full: float
    capture_vs_perfect_day: float
    capture_vs_perfect_same: float | None = None  # vs perfect foresight planned the same way


class ForecastReport(BaseModel):
    created_at: datetime
    config: ForecastConfig
    test_start: date
    test_end: date
    issues: int
    chronos_seconds: float | None = None
    spike_threshold: float  # SGD/MWh, realised USEP quantile over the test window
    mase_scale: float  # SGD/MWh, seasonal-naive (same interval yesterday) MAE before the test
    overall: ForecastStats
    monthly: list[ForecastStats]
    scenarios: list[ScenarioStats]
    capture: list[CaptureStats]
    weather: WeatherProcessingReport | None = None


@dataclass(frozen=True)
class Evaluation:
    report: ForecastReport
    scenarios: pl.DataFrame
    runs: dict[RunKey, tuple[RunSummary, Dispatch]]


def _midnight(d: date) -> datetime:
    return datetime.combine(d, datetime.min.time(), _SGT)


def evaluate(
    qf: pl.DataFrame,
    realised: pl.DataFrame,
    test_start: date,
    test_end: date,
    cfg: ForecastConfig,
    battery: BatteryConfig,
    market: MarketConfig,
) -> Evaluation:
    """Statistics, monthly scenarios and capture rates over [test_start, test_end].

    ``qf`` must also cover the ``calibration_days`` before ``test_start`` (copula warm-up);
    ``realised`` must cover everything ``qf`` covers plus the day before ``test_start``.
    """
    levels = cfg.quantile_levels
    t0, t1 = _midnight(test_start), _midnight(test_end + timedelta(days=1))
    in_test = pl.col(Col.INTERVAL_START).is_between(t0, t1, closed="left")
    window = realised.filter(in_test).select(Col.INTERVAL_START, Col.USEP).sort(Col.INTERVAL_START)
    history = realised.filter(pl.col(Col.INTERVAL_START) < t0)
    qf_test = qf.filter(in_test)  # forecasts of the delivery intervals in the test window
    threshold = float(window[Col.USEP].quantile(cfg.spike_quantile))
    # MASE scale: in-sample seasonal-naive MAE on the history available before the test window
    scale = seasonal_naive_mae(
        history.sort(Col.INTERVAL_START)[Col.USEP].to_numpy(), PERIODS_PER_DAY
    )

    monthly, scen_stats, scen_frames = [], [], []
    for m_start, m_end in month_windows(test_start, test_end):
        a, b = _midnight(m_start), _midnight(m_end + timedelta(days=1))
        month_qf = qf_test.filter(pl.col(Col.INTERVAL_START).is_between(a, b, closed="left"))
        monthly.append(
            forecast_stats(month_qf, realised, levels, threshold, scale, period_start=m_start)
        )

        calib = calibration_pits(qf, realised, a, cfg.scenarios.calibration_days, levels)
        copula = fit_month_copula(qf, realised, m_start, cfg)
        own = calibration_pits(qf, realised, b, (m_end - m_start).days + 1, levels)
        scen_stats.append(
            ScenarioStats(
                period_start=m_start,
                calibration_issues=int(calib.shape[0]),
                copula_lag1=float(np.diag(copula.corr, 1).mean()),
                realised_lag1=lag1(own),
            )
        )
        scen_frames.append(month_scenarios(qf, realised, m_start, m_end, cfg, copula))

    horizon = schedule_horizon(cfg, market)
    forecasters = {
        StrategyName.PERFECT_FORESIGHT: PerfectForecaster(window),
        StrategyName.CHRONOS_MEDIAN: QuantileForecaster(qf_test, level=0.5),
        StrategyName.PERSISTENCE: PersistenceForecaster(),
    }
    runs = {
        (name, h): run_schedule(
            name, forecasters[name], window, battery, market, h, history=history
        )
        for name, h in runs_for(horizon)
    }
    pf_full = runs[(StrategyName.PERFECT_FORESIGHT, Horizon.FULL)][0].total.profit_sgd
    pf_day = runs[(StrategyName.PERFECT_FORESIGHT, Horizon.DAY)][0].total.profit_sgd
    pf_same = {
        h: runs[(StrategyName.PERFECT_FORESIGHT, h)][0].total.profit_sgd
        for (name, h) in runs
        if name is StrategyName.PERFECT_FORESIGHT
    }
    capture = [
        CaptureStats(
            strategy=name,
            horizon=h,
            run=summary.name,
            profit_sgd=summary.total.profit_sgd,
            cycles_per_day=summary.cycles_per_day,
            capture_vs_perfect_full=summary.total.profit_sgd / pf_full,
            capture_vs_perfect_day=summary.total.profit_sgd / pf_day,
            capture_vs_perfect_same=summary.total.profit_sgd / pf_same[h],
        )
        for (name, h), (summary, _) in runs.items()
    ]
    report = ForecastReport(
        created_at=datetime.now(_SGT),
        config=cfg,
        test_start=test_start,
        test_end=test_end,
        issues=qf_test[Col.ISSUED_AT].n_unique(),
        spike_threshold=threshold,
        mase_scale=scale,
        overall=forecast_stats(
            qf_test, realised, levels, threshold, scale, period_start=test_start
        ),
        monthly=monthly,
        scenarios=scen_stats,
        capture=capture,
    )
    return Evaluation(report=report, scenarios=pl.concat(scen_frames), runs=runs)


# --------------------------------------------------------------------------- saved runs


class ForecastRunFiles(StrEnum):
    REPORT = "report.json"
    QUANTILES = "quantiles.parquet"
    SCENARIOS = "scenarios.parquet"
    BACKTESTS = "backtests"  # one backtest run folder per (strategy, horizon)


@dataclass(frozen=True)
class ForecastRun:
    path: Path
    report: ForecastReport
    quantiles: pl.DataFrame
    scenarios: pl.DataFrame
    backtests: dict[RunKey, tuple[RunSummary, pl.DataFrame]]


def save_forecast_run(
    ev: Evaluation, quantiles: pl.DataFrame, run_dir: Path, report: ForecastReport | None = None
) -> Path:
    """Write report, quantiles, scenarios and backtests to ``run_dir``."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / ForecastRunFiles.REPORT).write_text((report or ev.report).model_dump_json(indent=2))
    quantiles.write_parquet(run_dir / ForecastRunFiles.QUANTILES)
    ev.scenarios.write_parquet(run_dir / ForecastRunFiles.SCENARIOS)
    for summary, dispatch in ev.runs.values():
        save_run(summary, dispatch, run_dir / ForecastRunFiles.BACKTESTS)
    return run_dir


def load_forecast_run(run_dir: Path) -> ForecastRun:
    run_dir = Path(run_dir)
    backtests = {}
    for path in list_runs(run_dir / ForecastRunFiles.BACKTESTS):
        summary, dispatch = load_run(path)
        backtests[(summary.strategy, summary.horizon)] = (summary, dispatch)
    return ForecastRun(
        path=run_dir,
        report=ForecastReport.model_validate_json((run_dir / ForecastRunFiles.REPORT).read_text()),
        quantiles=pl.read_parquet(run_dir / ForecastRunFiles.QUANTILES),
        scenarios=pl.read_parquet(run_dir / ForecastRunFiles.SCENARIOS),
        backtests=backtests,
    )


def list_forecast_runs(runs_dir: Path = DEFAULT_RUNS_DIR) -> list[Path]:
    """Forecast run folders, most recently written first."""
    dirs = [p.parent for p in Path(runs_dir).glob(f"*/{ForecastRunFiles.REPORT}")]
    return sorted(dirs, key=lambda p: (p / ForecastRunFiles.REPORT).stat().st_mtime, reverse=True)
