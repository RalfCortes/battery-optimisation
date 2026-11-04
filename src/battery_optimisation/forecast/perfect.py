"""Perfect forecast: an oracle returning realised prices. Benchmark only.

It deliberately breaks the no-look-ahead rule (it is given the realised prices up front), which
is what makes the perfect-foresight benchmark an upper bound for every real strategy.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl

from battery_optimisation.columns import Col
from battery_optimisation.sim.clearing import MarketView


class PerfectForecaster:
    def __init__(self, realised: pl.DataFrame) -> None:
        self._prices = realised.select(Col.INTERVAL_START, Col.USEP).sort(Col.INTERVAL_START)

    def forecast(self, view: MarketView, end: datetime) -> pl.DataFrame:
        return self._prices.filter(pl.col(Col.INTERVAL_START).is_between(view.now, end))
