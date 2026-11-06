"""Result of a simulation: one row per interval, with settlement and metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from battery_optimisation.columns import Col
from battery_optimisation.config import BatteryConfig, MarketConfig
from battery_optimisation.sim import metrics
from battery_optimisation.sim.metrics import Period, PeriodStats


@dataclass(frozen=True)
class Dispatch:
    """Settled dispatch.

    ``frame`` columns (``Col``): INTERVAL_START, USEP, CHARGE_MW, DISCHARGE_MW, NET_MW,
    SOC_MWH (end of interval), REVENUE_SGD, DEGRADATION_SGD, PROFIT_SGD.
    """

    frame: pl.DataFrame
    battery: BatteryConfig
    market: MarketConfig
    # every submitted tranche: INTERVAL_START, PRODUCT, SIDE, TRANCHE, OFFER_PRICE, OFFER_MW
    offers: pl.DataFrame = field(default_factory=pl.DataFrame)

    @property
    def profit_sgd(self) -> float:
        return float(self.frame[Col.PROFIT_SGD].sum())

    def total(self) -> PeriodStats:
        return metrics.total(self.frame, self.battery, self.market.interval_h)

    def aggregate(self, every: Period) -> pl.DataFrame:
        """Per-period statistics as a frame (for plotting)."""
        return metrics.aggregate(self.frame, self.battery, self.market.interval_h, every)

    def stats(self, every: Period) -> list[PeriodStats]:
        return [PeriodStats.model_validate(r) for r in self.aggregate(every).to_dicts()]
