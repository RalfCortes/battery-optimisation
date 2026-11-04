"""Walk-forward forecasting: issue schedule, features, monthly copula scenarios, statistics.

- One forecast per delivery day D, issued ``issue_lead_intervals`` before D 00:00 (gate closure
  of D's first interval) from data strictly before then (``timing``, ``forecast.chronos``).
- Scenarios are built month by month: the copula for month M is fitted on PITs of issues whose
  whole horizon was realised before M starts (the ``calibration_days`` preceding M), so nothing
  from M or later is used.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import structlog
from pydantic import BaseModel

from battery_optimisation.columns import Col
from battery_optimisation.config import Covariate, ForecastConfig
from battery_optimisation.data import weather
from battery_optimisation.data.nems import (
    PERIODS_PER_DAY,
    TZ,
    NemsSeries,
    load_nems,
    month_chunks,
)
from battery_optimisation.data.weather import WeatherProcessingReport
from battery_optimisation.forecast.covariates import add_known_covariates
from battery_optimisation.forecast.quantiles import pit
from battery_optimisation.forecast.scenarios import GaussianCopula, reduce_scenarios, sample_paths
from battery_optimisation.sim.metrics import coverage, crps_quantiles, mase, spike_rates
from battery_optimisation.timing import INTERVAL, day_start, issue_time

log = structlog.get_logger(__name__)
_SGT = ZoneInfo(TZ)


def issue_times(start: date, end: date, cfg: ForecastConfig) -> list[datetime]:
    """Issue times for the intervals of days ``start``..``end``: a forecast is issued at the gate
    closure of every ``issue_every_intervals``-th interval (from 00:00), for the next
    ``prediction_length`` intervals. With 48 this is one forecast per day, at D-1 23:00."""
    first = issue_time(start, cfg)  # gate closure of start 00:00
    n = ((end - start).days + 1) * PERIODS_PER_DAY
    return [first + k * INTERVAL for k in range(0, n, cfg.issue_every_intervals)]


def month_windows(start: date, end: date) -> list[tuple[date, date]]:
    return list(month_chunks(start, end))


def weather_columns(cfg: ForecastConfig) -> list[str]:
    return weather.feature_names(cfg.weather)


def check_weather_coverage(
    features: pl.DataFrame, cfg: ForecastConfig, first_needed: datetime
) -> None:
    """Fail loudly if the weather provider's archive starts after the first context interval."""
    cols = weather_columns(cfg)
    present = features.filter(pl.any_horizontal(pl.col(cols).is_not_null()))
    first = present[Col.INTERVAL_START].min() if present.height else None
    if first is None or first > first_needed:
        raise ValueError(
            f"{cfg.weather.model} weather starts {first} but the first forecast context needs "
            f"{first_needed}; start the test window later or pick another provider"
        )


def build_features(
    start: date, end: date, cfg: ForecastConfig
) -> tuple[pl.DataFrame, weather.WeatherTrajectories | None, WeatherProcessingReport | None]:
    """USEP target + covariates for issues ``start``..``end`` (context before, horizon after).

    Weather is processed in memory from the raw run files: daily as-of values in the frame
    (context), and per-run trajectories for each issue's horizon. The transform report is
    returned.
    """
    first = start - timedelta(days=cfg.context_days + 1)
    last = end + timedelta(days=cfg.prediction_length // 48 + 1)
    frame = load_nems(first, last, series=[NemsSeries.USEP]).select(
        Col.INTERVAL_START, Col.USEP, Col.DEMAND_MW
    )
    report, trajectories = None, None
    if Covariate.WEATHER in cfg.covariates:
        runs = weather.all_runs(first - timedelta(days=2), last, cfg.weather)
        raw = weather.load_runs(cfg.weather.model, runs)
        wx, report = weather.to_features(raw, cfg, first, last)
        wx = wx.drop(Col.WEATHER_LEAD_H)
        frame = frame.join(wx, on=Col.INTERVAL_START, how="left")
        check_weather_coverage(frame, cfg, datetime.combine(first, datetime.min.time(), _SGT))
        trajectories = weather.WeatherTrajectories.from_raw(raw, cfg)
    return add_known_covariates(frame), trajectories, report


# --------------------------------------------------------------------------- quantile arrays


def _arrays(qf: pl.DataFrame, n_levels: int) -> tuple[list[datetime], np.ndarray, np.ndarray]:
    """(issues, interval starts [I, P], quantile values [I, P, Q]) from a long quantile frame."""
    qf = qf.sort(Col.ISSUED_AT, Col.LEAD_PERIODS, Col.QUANTILE)
    issues = qf[Col.ISSUED_AT].unique(maintain_order=True).to_list()
    values = qf[Col.USEP_FORECAST].to_numpy().reshape(len(issues), -1, n_levels)
    stamps = (
        qf.filter(pl.col(Col.QUANTILE) == qf[0, Col.QUANTILE])[Col.INTERVAL_START]
        .to_numpy()
        .reshape(len(issues), -1)
    )
    return issues, stamps, values


def _realised_for(stamps: np.ndarray, realised: pl.DataFrame) -> np.ndarray:
    lookup = dict(
        zip(
            realised[Col.INTERVAL_START].to_numpy(),
            realised[Col.USEP].fill_null(float("nan")).to_numpy(),
            strict=True,
        )
    )
    return np.array([[lookup.get(t, np.nan) for t in row] for row in stamps])


def calibration_pits(
    qf: pl.DataFrame,
    realised: pl.DataFrame,
    before: datetime,
    days: int,
    levels: list[float],
) -> np.ndarray:
    """PITs [n_issues, P] of the forecasts whose whole delivery horizon lies in the ``days``
    before ``before`` (so it was realised by then)."""
    horizon = qf.group_by(Col.ISSUED_AT).agg(
        pl.col(Col.INTERVAL_START).min().alias("_first"),
        pl.col(Col.INTERVAL_START).max().alias("_last"),
    )
    keep = horizon.filter(
        (pl.col("_first") >= before - timedelta(days=days)) & (pl.col("_last") < before)
    )[Col.ISSUED_AT]
    window = qf.filter(pl.col(Col.ISSUED_AT).is_in(keep.implode()))
    if window.is_empty():
        return np.empty((0, qf[Col.LEAD_PERIODS].n_unique()))
    known = realised.filter(pl.col(Col.INTERVAL_START) < before)
    _, stamps, values = _arrays(window, len(levels))
    y = _realised_for(stamps, known)
    u = np.full(y.shape, np.nan)
    ok = ~np.isnan(y)
    u[ok] = pit(levels, values[ok], y[ok])
    return u


def fit_month_copula(
    qf: pl.DataFrame, realised: pl.DataFrame, month_start: date, cfg: ForecastConfig
) -> GaussianCopula:
    before = datetime.combine(month_start, datetime.min.time(), _SGT)
    u = calibration_pits(qf, realised, before, cfg.scenarios.calibration_days, cfg.quantile_levels)
    return GaussianCopula.fit(u, cfg.scenarios.shrinkage)


def month_scenarios(
    qf: pl.DataFrame,
    realised: pl.DataFrame,
    month_start: date,
    month_end: date,
    cfg: ForecastConfig,
    copula: GaussianCopula | None = None,
) -> pl.DataFrame:
    """Reduced scenarios for issues in [month_start, month_end] (long frame)."""
    sc = cfg.scenarios
    copula = copula or fit_month_copula(qf, realised, month_start, cfg)
    rng = np.random.default_rng([sc.seed, month_start.toordinal()])
    start, end = day_start(month_start), day_start(month_end + timedelta(days=1))
    month = qf.filter(pl.col(Col.INTERVAL_START).is_between(start, end, closed="left"))
    issues, stamps, values = _arrays(month, len(cfg.quantile_levels))
    frames = []
    for issued, ts, q in zip(issues, stamps, values, strict=True):
        paths = sample_paths(np.asarray(cfg.quantile_levels), q, copula, sc.n_samples, rng)
        scen = reduce_scenarios(paths, sc.n_scenarios, sc.n_spike, rng)
        k, horizon = scen.paths.shape
        frames.append(
            pl.DataFrame(
                {
                    Col.SCENARIO: np.repeat(np.arange(k, dtype=np.int32), horizon),
                    Col.WEIGHT: np.repeat(scen.weights, horizon),
                    Col.INTERVAL_START: np.tile(ts, k),
                    Col.USEP_FORECAST: scen.paths.reshape(-1),
                }
            ).with_columns(
                pl.lit(issued).dt.cast_time_unit("us").alias(Col.ISSUED_AT),
                pl.col(Col.INTERVAL_START).dt.replace_time_zone("UTC").dt.convert_time_zone(TZ),
            )
        )
    return pl.concat(frames).select(
        Col.ISSUED_AT, Col.SCENARIO, Col.WEIGHT, Col.INTERVAL_START, Col.USEP_FORECAST
    )


# --------------------------------------------------------------------------- statistics


class ForecastStats(BaseModel):
    """Quantile-forecast diagnostics over a period (SGD/MWh where applicable)."""

    period_start: date | None = None
    intervals: int
    crps: float
    mae_median: float
    mase: float  # MAE(median) / in-sample seasonal-naive MAE (report.mase_scale); < 1 is better
    coverage_10_90: float  # ideal 0.8
    spike_hit_rate: float | None  # share of spikes where q90 >= threshold; None: no spike
    spike_false_alarm: float | None  # share of non-spikes where q90 >= threshold


def forecast_stats(
    qf: pl.DataFrame,
    realised: pl.DataFrame,
    levels: list[float],
    spike_threshold: float,
    mase_scale: float,
    period_start: date | None = None,
) -> ForecastStats:
    _, stamps, values = _arrays(qf, len(levels))
    y = _realised_for(stamps, realised).reshape(-1)
    v = values.reshape(-1, len(levels))
    ok = ~np.isnan(y)
    y, v = y[ok], v[ok]
    lv = list(levels)
    q10, q50, q90 = v[:, lv.index(0.1)], v[:, lv.index(0.5)], v[:, lv.index(0.9)]
    hit, false_alarm = spike_rates(y, q90, spike_threshold)
    return ForecastStats(
        period_start=period_start,
        intervals=int(y.size),
        crps=float(crps_quantiles(y, np.asarray(levels), v).mean()),
        mae_median=float(np.abs(y - q50).mean()),
        mase=mase(y, q50, mase_scale),
        coverage_10_90=coverage(y, q10, q90),
        spike_hit_rate=None if np.isnan(hit) else hit,
        spike_false_alarm=None if np.isnan(false_alarm) else false_alarm,
    )


class ScenarioStats(BaseModel):
    """Copula diagnostics for one month (CLAUDE.md: sampled lag-1 corr ~ realised)."""

    period_start: date
    calibration_issues: int
    copula_lag1: float  # mean correlation of normal scores between consecutive intervals
    realised_lag1: float  # same, measured on the month's own PITs (out of sample)


def lag1(u: np.ndarray) -> float:
    """Mean correlation between normal scores of consecutive steps, across rows."""
    from scipy.special import ndtri

    u = u[~np.isnan(u).any(axis=1)]
    if u.shape[0] < 3:
        return float("nan")
    z = ndtri(u)
    return float(
        np.nanmean([np.corrcoef(z[:, t], z[:, t + 1])[0, 1] for t in range(z.shape[1] - 1)])
    )
