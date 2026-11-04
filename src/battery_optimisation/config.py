"""Typed configuration: pydantic models for ``config/battery.yaml`` and ``config/market.yaml``.

battery = load_battery()
battery.p_max_mw, battery.round_trip_efficiency
market = load_market()
market.enabled_products, market.reserve_sustain_h[Product.CONTINGENCY_RESERVE]
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


class Product(StrEnum):
    ENERGY = "energy"
    REGULATION = "regulation"
    PRIMARY_RESERVE = "primary_reserve"
    CONTINGENCY_RESERVE = "contingency_reserve"


class _Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class BatteryConfig(_Config):
    """Battery parameters. Energies in MWh, powers in MW."""

    p_max_mw: PositiveFloat
    e_max_mwh: PositiveFloat
    soc_min: NonNegativeFloat  # MWh
    soc_max: PositiveFloat  # MWh
    soc_init: NonNegativeFloat  # MWh
    eta_c: float = Field(gt=0, le=1)
    eta_d: float = Field(gt=0, le=1)
    kappa_sgd_per_mwh: NonNegativeFloat

    @model_validator(mode="after")
    def _soc_bounds(self) -> Self:
        if not self.soc_min <= self.soc_init <= self.soc_max <= self.e_max_mwh:
            raise ValueError("need soc_min <= soc_init <= soc_max <= e_max_mwh")
        return self

    @property
    def round_trip_efficiency(self) -> float:
        return self.eta_c * self.eta_d

    @property
    def usable_mwh(self) -> float:
        """SOC range available for cycling (soc_max - soc_min)."""
        return self.soc_max - self.soc_min


class MarketConfig(_Config):
    """Market parameters. Prices in SGD/MWh, durations in hours."""

    timezone: str
    interval_h: PositiveFloat
    intervals_per_day: PositiveInt
    n_tranches: PositiveInt
    price_floor: float
    price_cap: float
    products: dict[Product, bool]
    reserve_sustain_h: dict[Product, NonNegativeFloat | None]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if abs(self.interval_h * self.intervals_per_day - 24) > 1e-9:
            raise ValueError("interval_h * intervals_per_day must equal 24")
        if self.price_floor >= self.price_cap:
            raise ValueError("price_floor must be below price_cap")
        return self

    @property
    def enabled_products(self) -> list[Product]:
        return [p for p, on in self.products.items() if on]


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def load_battery(path: Path = CONFIG_DIR / "battery.yaml") -> BatteryConfig:
    return BatteryConfig.model_validate(_read_yaml(path))


def load_market(path: Path = CONFIG_DIR / "market.yaml") -> MarketConfig:
    return MarketConfig.model_validate(_read_yaml(path))
