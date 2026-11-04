"""Naive forecasters used as reference points."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from battery_optimisation.columns import Col
from battery_optimisation.forecast.base import Forecaster
from battery_optimisation.sim.clearing import MarketView


class PersistenceForecaster:
    """Seasonal naive: each interval's price = the latest known price of the same half-hour
    (yesterday, or the day before when yesterday's is not yet known at the decision time)."""

    max_days_back: int = 7

    def __init__(self, strict: bool = True) -> None:
        # strict: raise when a price is unknown; otherwise use the history's mean (0 if none)
        self.strict = strict

    def forecast(self, view: MarketView, end: datetime) -> pl.DataFrame:
        history = view.history()
        target = pl.datetime_range(
            view.now, end, timedelta(minutes=30), time_unit="us", eager=True
        ).alias(Col.INTERVAL_START)
        cols = [c for c in history.columns if c != Col.INTERVAL_START]  # USEP + ancillary prices
        out = target.to_frame().with_columns(pl.lit(None, pl.Float64).alias(c) for c in cols)
        for days in range(1, self.max_days_back + 1):  # nearest known day first
            lagged = history.with_columns(pl.col(Col.INTERVAL_START).dt.offset_by(f"{days}d"))
            out = out.join(lagged, on=Col.INTERVAL_START, how="left", suffix="_lag").select(
                Col.INTERVAL_START, *[pl.coalesce(c, f"{c}_lag").alias(c) for c in cols]
            )
        if out[Col.USEP].null_count():
            if self.strict:
                raise ValueError(f"persistence has no known price for some of {view.now}..{end}")
            out = out.with_columns(
                pl.col(c).fill_null(history[c].mean() if history.height else 0.0) for c in cols
            )
        return out


class WithAncillary:
    """Energy price from ``energy`` (e.g. the Chronos median), ancillary prices from
    ``ancillary`` (stored Chronos forecasts, persistence, ...)."""

    def __init__(self, energy: Forecaster, ancillary: Forecaster) -> None:
        self.energy, self.ancillary = energy, ancillary

    def forecast(self, view: MarketView, end: datetime) -> pl.DataFrame:
        energy = self.energy.forecast(view, end).select(Col.INTERVAL_START, Col.USEP)
        ancillary = self.ancillary.forecast(view, end)
        ancillary = ancillary.drop([c for c in (Col.USEP,) if c in ancillary.columns])
        return energy.join(ancillary, on=Col.INTERVAL_START, how="left")


class WithAncillaryPersistence(WithAncillary):
    """Energy from ``energy``, ancillary prices by persistence."""

    def __init__(self, energy: Forecaster) -> None:
        super().__init__(energy, PersistenceForecaster())
