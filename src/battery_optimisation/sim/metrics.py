"""Evaluation metrics on a settled dispatch frame (see ``sim.dispatch.Dispatch``).

Equivalent full cycles = energy taken out of the cells (discharged MWh / eta_d) divided by the
usable capacity (soc_max - soc_min).
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

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
