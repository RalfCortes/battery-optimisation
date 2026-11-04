"""Covariates for the price forecast and how each may be used.

Two kinds (Chronos-2 vocabulary):
  PAST   observed up to the issue time only (given in the context window)
  KNOWN  also known over the forecast horizon (given in context and horizon)

Demand is PAST: EMC pre-dispatch demand forecasts are only archived from Nov 2026 on, and using
realised future demand would leak. Weather is KNOWN because we use day-ahead weather *forecasts*
(Open-Meteo ``previous_day1``), never observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum

import holidays
import numpy as np
import polars as pl

from battery_optimisation.columns import Col
from battery_optimisation.config import Covariate, load_forecast
from battery_optimisation.data.weather import feature_names

TPC_START = date(2023, 7, 1)  # temporary price cap policy start (EMA)


class CovariateKind(StrEnum):
    PAST = "past"
    KNOWN = "known"


@dataclass(frozen=True)
class CovariateGroup:
    kind: CovariateKind
    columns: tuple[str, ...]  # Col members, or weather grid features (data/weather.py)


COVARIATES: dict[Covariate, CovariateGroup] = {
    Covariate.CALENDAR: CovariateGroup(
        CovariateKind.KNOWN,
        (Col.HOUR_SIN, Col.HOUR_COS, Col.DOW_SIN, Col.DOW_COS, Col.IS_OFF_DAY),
    ),
    Covariate.REGIME: CovariateGroup(CovariateKind.KNOWN, (Col.TPC_REGIME,)),
    Covariate.DEMAND: CovariateGroup(CovariateKind.PAST, (Col.DEMAND_MW,)),
    # raw day-ahead forecasts at every grid point, one covariate per (variable, point)
    Covariate.WEATHER: CovariateGroup(
        CovariateKind.KNOWN, tuple(feature_names(load_forecast().weather))
    ),
}


def _off_days(days: pl.Series) -> pl.Series:
    years = sorted({d.year for d in days.to_list()})
    public = set(holidays.country_holidays("SG", years=years))
    return pl.Series([float(d.weekday() >= 5 or d in public) for d in days.to_list()])


def add_known_covariates(frame: pl.DataFrame) -> pl.DataFrame:
    """Add calendar and regime columns (deterministic functions of ``Col.INTERVAL_START``)."""
    ts = pl.col(Col.INTERVAL_START)
    minutes = ts.dt.hour().cast(pl.Int32) * 60 + ts.dt.minute().cast(pl.Int32)  # Int8 overflows
    day_frac = minutes / (24 * 60) * 2 * np.pi
    week_frac = (ts.dt.weekday() - 1) / 7 * 2 * np.pi  # Monday = 0
    days = frame[Col.INTERVAL_START].dt.date()
    unique_days = days.unique()
    off = dict(zip(unique_days.to_list(), _off_days(unique_days).to_list(), strict=True))
    return frame.with_columns(
        day_frac.sin().alias(Col.HOUR_SIN),
        day_frac.cos().alias(Col.HOUR_COS),
        week_frac.sin().alias(Col.DOW_SIN),
        week_frac.cos().alias(Col.DOW_COS),
        days.replace_strict(off, return_dtype=pl.Float64).alias(Col.IS_OFF_DAY),
        (ts.dt.date() >= TPC_START).cast(pl.Float64).alias(Col.TPC_REGIME),
    )
