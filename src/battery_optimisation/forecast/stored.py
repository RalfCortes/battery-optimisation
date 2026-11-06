"""Adapter: stored quantile forecasts -> ``Forecaster`` (one quantile level, or the mean, as a
point forecast)."""

from __future__ import annotations

import bisect
from datetime import datetime, timedelta

import polars as pl

from battery_optimisation.columns import Col
from battery_optimisation.sim.clearing import MarketView

_LEVEL_TOL = 1e-9


def quantile_mean(quantiles: pl.DataFrame) -> pl.DataFrame:
    """Mean of each forecast distribution (per issue and interval) from its quantiles: the
    integral of the quantile function, trapezoids between levels, flat beyond the extremes."""
    keys = [Col.ISSUED_AT, Col.INTERVAL_START]
    q = quantiles.sort(*keys, Col.QUANTILE)
    u, v = pl.col(Col.QUANTILE), pl.col(Col.USEP_FORECAST)
    du, mid = u - u.shift(1).over(keys), (v + v.shift(1).over(keys)) / 2
    return q.group_by(keys).agg(
        ((du * mid).sum() + u.first() * v.first() + (1 - u.last()) * v.last()).alias(
            Col.USEP_FORECAST
        )
    )


class QuantileForecaster:
    """Uses the latest forecast issued at or before the decision time (never a later one).
    ``level``: the quantile used as the point forecast; ``None``: the mean."""

    def __init__(self, quantiles: pl.DataFrame, level: float | None = 0.5) -> None:
        if level is None:
            selected = quantile_mean(quantiles)
        else:
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
        """``Col.USEP`` = the forecast value (whatever the target)."""
        i = bisect.bisect_right(self._issues, view.decided_at)
        if i == 0:
            raise ValueError(f"no forecast issued at or before {view.decided_at}")
        issued = self._issues[i - 1]
        out = self._by_issue[issued].filter(pl.col(Col.INTERVAL_START).is_between(view.now, end))
        expected = int((end - view.now) / timedelta(minutes=30)) + 1
        if out.height != expected:
            raise ValueError(f"no forecast issued {issued} covering {view.now}..{end}")
        return out


class AncillaryQuantileForecaster:
    """Stored ancillary price forecasts (``Col.TARGET`` = price column): one column per target,
    from the latest issue at or before the decision time (``level``: quantile, ``None``: mean)."""

    def __init__(self, quantiles: pl.DataFrame, level: float | None = 0.5) -> None:
        self._by_target = {
            key[0]: QuantileForecaster(part, level)
            for key, part in quantiles.partition_by(Col.TARGET, as_dict=True).items()
        }

    def forecast(self, view: MarketView, end: datetime) -> pl.DataFrame:
        out = None
        for target, f in self._by_target.items():
            part = f.forecast(view, end).rename({Col.USEP: target})
            out = part if out is None else out.join(part, on=Col.INTERVAL_START, how="left")
        if out is None:
            raise ValueError("no ancillary forecasts")
        return out
