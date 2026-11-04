"""Synthetic feature frames and quantile forecasts for forecasting tests."""

from datetime import date, datetime, timedelta
from statistics import NormalDist
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
from synthetic import daily_shape, make_prices

from battery_optimisation.columns import Col
from battery_optimisation.config import load_forecast
from battery_optimisation.data.weather import feature_names
from battery_optimisation.forecast.covariates import add_known_covariates

TZ = "Asia/Singapore"


def make_features(days: int, start: date = date(2026, 1, 1)) -> pl.DataFrame:
    """USEP + demand + weather + calendar/regime covariates on a half-hourly grid."""
    prices = make_prices(daily_shape(days, seed=0), start)
    n = prices.height
    wave = np.sin(np.arange(n) / 48 * 2 * np.pi)
    frame = prices.with_columns(
        pl.Series(Col.DEMAND_MW, 6000 + 1000 * wave),
        *[
            pl.Series(name, 30 + i + 5 * wave)
            for i, name in enumerate(feature_names(load_forecast().weather))
        ],
    )
    return add_known_covariates(frame)


def at(day: date, hour: int = 0) -> datetime:
    return datetime(day.year, day.month, day.day, hour, tzinfo=ZoneInfo(TZ))


def normal_quantiles(mu: np.ndarray, sigma: float, levels: list[float]) -> np.ndarray:
    """[len(mu), len(levels)] quantiles of N(mu, sigma)."""
    z = np.array([NormalDist().inv_cdf(q) for q in levels])
    return mu[:, None] + sigma * z[None, :]


def quantile_frame(
    issued_at: datetime, values: np.ndarray, levels: list[float], gap: int = 0
) -> pl.DataFrame:
    """Long quantile-forecast frame for one issue: values [48, Q] for the intervals starting
    ``gap`` intervals after the issue time."""
    rows = []
    for i in range(values.shape[0]):
        lead = gap + i
        for j, q in enumerate(levels):
            rows.append(
                {
                    Col.ISSUED_AT: issued_at,
                    Col.INTERVAL_START: issued_at + timedelta(minutes=30 * lead),
                    Col.LEAD_PERIODS: lead + 1,
                    Col.QUANTILE: q,
                    Col.USEP_FORECAST: float(values[i, j]),
                }
            )
    return pl.DataFrame(rows).with_columns(
        pl.col(Col.ISSUED_AT, Col.INTERVAL_START).dt.cast_time_unit("us"),
        pl.col(Col.LEAD_PERIODS).cast(pl.Int32),
    )
