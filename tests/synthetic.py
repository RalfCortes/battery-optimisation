"""Synthetic prices, forecasters and strategies for tests."""

from datetime import date, datetime, timedelta

import numpy as np
import polars as pl

from battery_optimisation.columns import Col
from battery_optimisation.config import MarketConfig
from battery_optimisation.sim.clearing import IntervalOffers, MarketView


def make_prices(values: np.ndarray | list[float], start: date = date(2026, 1, 1)) -> pl.DataFrame:
    """Half-hourly price frame (Col.INTERVAL_START, Col.USEP) starting at ``start`` 00:00 SGT."""
    t0 = datetime(start.year, start.month, start.day)
    stamps = [t0 + timedelta(minutes=30 * i) for i in range(len(values))]
    return pl.DataFrame(
        {Col.INTERVAL_START: stamps, Col.USEP: np.asarray(values, dtype=float)}
    ).with_columns(
        pl.col(Col.INTERVAL_START).dt.cast_time_unit("us").dt.replace_time_zone("Asia/Singapore")
    )


def daily_shape(days: int = 1, seed: int | None = None) -> np.ndarray:
    """Cheap nights, expensive evenings (SGD/MWh), optional noise."""
    hours = np.arange(48) / 2
    day = 150 + 100 * np.sin((hours - 10) / 24 * 2 * np.pi)
    prices = np.tile(day, days)
    if seed is not None:
        prices = prices + np.random.default_rng(seed).normal(0, 40, prices.size)
    return prices


class NoisyForecaster:
    """Realised prices plus Gaussian noise: a stand-in for an imperfect forecast model."""

    def __init__(self, prices: pl.DataFrame, sigma: float, seed: int) -> None:
        noise = np.random.default_rng(seed).normal(0, sigma, prices.height)
        self._prices = prices.with_columns(pl.col(Col.USEP) + noise)

    def forecast(self, view: MarketView, end: datetime) -> pl.DataFrame:
        return self._prices.filter(pl.col(Col.INTERVAL_START).is_between(view.now, end))


class FixedStrategy:
    """Self-schedules fixed charge/discharge MW per interval index (no decision making)."""

    def __init__(self, charge_mw: list[float], discharge_mw: list[float], market: MarketConfig):
        self._charge, self._discharge, self._market = charge_mw, discharge_mw, market
        self._i = 0

    def offers(self, view: MarketView, soc_mwh: float) -> IntervalOffers:
        i, self._i = self._i, self._i + 1
        return IntervalOffers.self_schedule(
            view.now, self._charge[i], self._discharge[i], self._market
        )
