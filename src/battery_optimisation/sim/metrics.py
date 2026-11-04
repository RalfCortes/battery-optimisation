"""Evaluation metrics on a settled dispatch frame (see ``sim.dispatch.Dispatch``).

Equivalent full cycles = energy taken out of the cells (discharged MWh / eta_d) divided by the
usable capacity (soc_max - soc_min).
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict

from battery_optimisation.columns import Col
from battery_optimisation.config import BatteryConfig


class Period(StrEnum):
    """Aggregation periods (values are polars ``every`` strings)."""

    DAY = "1d"
    MONTH = "1mo"


class PeriodStats(BaseModel):
    """Statistics over one day, month or a whole run. Money in SGD, energy in MWh."""

    model_config = ConfigDict(frozen=True)

    period_start: date
    revenue_sgd: float
    degradation_sgd: float
    profit_sgd: float
    charged_mwh: float
    discharged_mwh: float
    cycles: float  # equivalent full cycles
    mean_usep: float


def _stats_exprs(battery: BatteryConfig, dt: float) -> list[pl.Expr]:
    return [
        pl.col(Col.REVENUE_SGD).sum(),
        pl.col(Col.DEGRADATION_SGD).sum(),
        pl.col(Col.PROFIT_SGD).sum(),
        (pl.col(Col.CHARGE_MW) * dt).sum().alias(Col.CHARGED_MWH),
        (pl.col(Col.DISCHARGE_MW) * dt).sum().alias(Col.DISCHARGED_MWH),
        ((pl.col(Col.DISCHARGE_MW) * dt / battery.eta_d).sum() / battery.usable_mwh).alias(
            Col.CYCLES
        ),
        pl.col(Col.USEP).mean().alias(Col.MEAN_USEP),
    ]


def aggregate(
    frame: pl.DataFrame, battery: BatteryConfig, dt: float, every: Period
) -> pl.DataFrame:
    """Per-period statistics; periods follow the Asia/Singapore calendar."""
    return (
        frame.sort(Col.INTERVAL_START)
        .group_by_dynamic(Col.INTERVAL_START, every=every)
        .agg(_stats_exprs(battery, dt))
        .with_columns(pl.col(Col.INTERVAL_START).dt.date().alias(Col.PERIOD_START))
        .drop(Col.INTERVAL_START)
        .select(Col.PERIOD_START, pl.exclude(Col.PERIOD_START))
    )


def total(frame: pl.DataFrame, battery: BatteryConfig, dt: float) -> PeriodStats:
    row = frame.select(_stats_exprs(battery, dt)).row(0, named=True)
    return PeriodStats(period_start=frame[Col.INTERVAL_START].min().date(), **row)


def capture_rate(profit_sgd: float, perfect_foresight_profit_sgd: float) -> float:
    """Realised profit / perfect-foresight profit (headline metric)."""
    return profit_sgd / perfect_foresight_profit_sgd


# --------------------------------------------------------------------------- forecast diagnostics


def _cell_weights(levels: np.ndarray) -> np.ndarray:
    """Width of the probability cell around each level (cells split at midpoints, cover [0, 1])."""
    edges = np.concatenate([[0.0], (levels[1:] + levels[:-1]) / 2, [1.0]])
    return np.diff(edges)


def crps_quantiles(y: np.ndarray, levels: np.ndarray, values: np.ndarray) -> np.ndarray:
    """CRPS per observation from quantile forecasts ``values`` [N, Q] at ``levels`` [Q].

    CRPS = 2 * integral over tau of the pinball loss, approximated on the quantile grid.
    Same units as ``y`` (SGD/MWh); lower is better; equals |y - c| for a point forecast c.
    """
    levels = np.asarray(levels, float)
    diff = np.asarray(y, float)[:, None] - np.asarray(values, float)
    pinball = np.maximum(levels * diff, (levels - 1) * diff)
    return 2 * pinball @ _cell_weights(levels)


def coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Share of observations inside [lower, upper]."""
    return float(np.mean((y >= lower) & (y <= upper)))


def spike_rates(y: np.ndarray, flag: np.ndarray, threshold: float) -> tuple[float, float]:
    """(hit rate, false-alarm rate) of ``flag >= threshold`` for spikes ``y >= threshold``."""
    spike, flagged = y >= threshold, flag >= threshold
    hit = float(flagged[spike].mean()) if spike.any() else float("nan")
    false_alarm = float(flagged[~spike].mean()) if (~spike).any() else float("nan")
    return hit, false_alarm


def seasonal_naive_mae(history: np.ndarray, season: int) -> float:
    """MAE of the seasonal-naive forecast y[t] = y[t - season] over ``history`` (MASE scale)."""
    history = np.asarray(history, float)
    if history.size <= season:
        raise ValueError(f"history shorter than one season ({season})")
    diff = np.abs(history[season:] - history[:-season])
    return float(np.nanmean(diff))


def mase(y: np.ndarray, forecast: np.ndarray, scale: float) -> float:
    """Mean absolute scaled error: MAE / ``scale`` (< 1: better than the in-sample naive)."""
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(forecast))) / scale)
