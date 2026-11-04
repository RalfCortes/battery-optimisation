"""Naive forecasters used as reference points."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from battery_optimisation.columns import Col
from battery_optimisation.sim.clearing import MarketView


class PersistenceForecaster:
    """Seasonal naive: each interval's price = the latest known price of the same half-hour
    (yesterday, or the day before when yesterday's is not yet known at the decision time)."""

    max_days_back: int = 7

    def forecast(self, view: MarketView, end: datetime) -> pl.DataFrame:
        history = view.history()
        target = pl.datetime_range(
            view.now, end, timedelta(minutes=30), time_unit="us", eager=True
        ).alias(Col.INTERVAL_START)
        out = target.to_frame().with_columns(pl.lit(None, pl.Float64).alias(Col.USEP))
        for days in range(1, self.max_days_back + 1):  # nearest known day first
            lagged = history.with_columns(pl.col(Col.INTERVAL_START).dt.offset_by(f"{days}d"))
            out = out.join(lagged, on=Col.INTERVAL_START, how="left", suffix="_lag").select(
                Col.INTERVAL_START,
                pl.coalesce(Col.USEP, f"{Col.USEP}_lag").alias(Col.USEP),
            )
        if out[Col.USEP].null_count():
            raise ValueError(f"persistence has no known price for some of {view.now}..{end}")
        return out
