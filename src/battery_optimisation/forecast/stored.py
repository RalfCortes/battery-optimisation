"""Adapter: stored quantile forecasts -> ``Forecaster`` (one quantile level as a point forecast)."""

from __future__ import annotations

import bisect
from datetime import datetime, timedelta

import polars as pl

from battery_optimisation.columns import Col
from battery_optimisation.sim.clearing import MarketView

_LEVEL_TOL = 1e-9


class QuantileForecaster:
    """Uses the latest forecast issued at or before the decision time (never a later one)."""

    def __init__(self, quantiles: pl.DataFrame, level: float = 0.5) -> None:
        selected = quantiles.filter((pl.col(Col.QUANTILE) - level).abs() < _LEVEL_TOL)
        if selected.is_empty():
            raise ValueError(f"quantile level {level} not in the forecasts")
        self.level = level
        self._by_issue: dict[datetime, pl.DataFrame] = {
            key[0]: part.select(Col.INTERVAL_START, pl.col(Col.USEP_FORECAST).alias(Col.USEP)).sort(
                Col.INTERVAL_START
            )
            for key, part in selected.partition_by(Col.ISSUED_AT, as_dict=True).items()
        }
        self._issues = sorted(self._by_issue)

    def forecast(self, view: MarketView, end: datetime) -> pl.DataFrame:
        i = bisect.bisect_right(self._issues, view.decided_at)
        if i == 0:
            raise ValueError(f"no forecast issued at or before {view.decided_at}")
        issued = self._issues[i - 1]
        out = self._by_issue[issued].filter(pl.col(Col.INTERVAL_START).is_between(view.now, end))
        expected = int((end - view.now) / timedelta(minutes=30)) + 1
        if out.height != expected:
            raise ValueError(f"no forecast issued {issued} covering {view.now}..{end}")
        return out
