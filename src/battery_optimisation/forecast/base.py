"""Forecaster interface: price forecasts from the information available at offer time."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

import polars as pl

from battery_optimisation.sim.clearing import MarketView


class Forecaster(Protocol):
    def forecast(self, view: MarketView, end: datetime) -> pl.DataFrame:
        """USEP forecast for intervals ``view.now`` .. ``end`` (inclusive).

        Returns ``Col.INTERVAL_START``, ``Col.USEP``, one row per interval. Must only use what
        ``view`` exposes (plus external data published before ``view.now``).
        """
        ...
