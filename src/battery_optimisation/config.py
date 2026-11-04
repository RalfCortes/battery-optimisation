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
    NonNegativeInt,
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


ANCILLARY_PRODUCTS = (Product.REGULATION, Product.PRIMARY_RESERVE, Product.CONTINGENCY_RESERVE)
RAISE_PRODUCTS = ANCILLARY_PRODUCTS  # need upward power and stored energy
LOWER_PRODUCTS = (Product.REGULATION,)  # need downward power and free capacity


class ReserveGroup(StrEnum):
    """NEMS reserve groups (prices differ by group)."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"


class Covariate(StrEnum):
    """Covariate groups for the price forecast (columns: ``forecast.covariates.COVARIATES``)."""

    CALENDAR = "calendar"  # time of day, day of week, weekend / public holiday
    REGIME = "regime"  # temporary price cap policy in force
    DEMAND = "demand"  # system demand (past only: no archived demand forecasts)
    WEATHER = "weather"  # day-ahead temperature and irradiance forecasts


class Device(StrEnum):
    CUDA = "cuda"
    CPU = "cpu"


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


class AncillaryConfig(_Config):
    reserve_group: ReserveGroup  # whose reserve prices apply to the battery
    regulation_utilisation: float = Field(ge=0, le=1)  # share of regulation MW actually moved
    max_award_mw: dict[Product, NonNegativeFloat]  # cap per product (market size)


class MarketConfig(_Config):
    """Market parameters. Prices in SGD/MWh, durations in hours."""

    timezone: str
    interval_h: PositiveFloat
    intervals_per_day: PositiveInt
    gate_closure_intervals: NonNegativeInt  # offers for t are fixed at the start of t - this
    shortfall_penalty_sgd_per_mwh: NonNegativeFloat  # cleared energy the battery cannot deliver
    n_tranches: PositiveInt
    price_floor: float
    price_cap: float
    products: dict[Product, bool]
    reserve_sustain_h: dict[Product, NonNegativeFloat | None]
    ancillary: AncillaryConfig

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

    @property
    def ancillary_products(self) -> list[Product]:
        return [p for p in ANCILLARY_PRODUCTS if self.products.get(p, False)]

    def sustain_h(self, product: Product) -> float:
        return self.reserve_sustain_h.get(product) or 0.0


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def load_battery(path: Path = CONFIG_DIR / "battery.yaml") -> BatteryConfig:
    return BatteryConfig.model_validate(_read_yaml(path))


def load_market(path: Path = CONFIG_DIR / "market.yaml") -> MarketConfig:
    return MarketConfig.model_validate(_read_yaml(path))


# --------------------------------------------------------------------------- weather


class WeatherVar(StrEnum):
    """Weather forecast variables (value = Open-Meteo variable name)."""

    TEMPERATURE = "temperature_2m"  # cooling demand
    HUMIDITY = "relative_humidity_2m"  # cooling demand
    CLOUD_COVER = "cloud_cover"  # embedded solar
    IRRADIANCE = "shortwave_radiation"  # embedded solar
    PRECIPITATION = "precipitation"  # thunderstorms: solar drop-outs


class Aggregation(StrEnum):
    MEAN = "mean"
    SUM = "sum"


class GapFill(StrEnum):
    NONE = "none"
    LINEAR = "linear"


class GridConfig(_Config):
    latitudes: list[float]
    longitudes: list[float]


class GapFillConfig(_Config):
    method: GapFill
    max_gap_steps: NonNegativeInt  # longest run of missing 15-min steps that may be filled


class WeatherConfig(_Config):
    model: str  # provider used for the forecast features
    availability_delay_h: NonNegativeFloat  # run usable this long after its nominal start
    run_hours_utc: list[int]
    forecast_days: PositiveInt  # trajectory length kept per run
    archive_models: list[str]  # providers whose runs are archived (comparison / ensembles)
    grid: GridConfig
    variables: dict[WeatherVar, Aggregation]
    gap_fill: GapFillConfig

    @model_validator(mode="after")
    def _model_archived(self) -> Self:
        if self.model not in self.archive_models:
            raise ValueError(f"model {self.model!r} must be one of archive_models")
        return self

    @property
    def points(self) -> list[tuple[float, float]]:
        """Grid points (latitude, longitude), latitude-major."""
        return [(lat, lon) for lat in self.grid.latitudes for lon in self.grid.longitudes]


class ScenarioConfig(_Config):
    calibration_days: PositiveInt  # copula fitted on PITs of the preceding N days
    shrinkage: float = Field(ge=0, le=1)  # correlation shrinkage towards identity
    n_samples: PositiveInt
    n_scenarios: PositiveInt
    n_spike: NonNegativeInt
    seed: int

    @model_validator(mode="after")
    def _sizes(self) -> Self:
        if not self.n_spike < self.n_scenarios <= self.n_samples:
            raise ValueError("need n_spike < n_scenarios <= n_samples")
        return self


class ForecastConfig(_Config):
    model_id: str
    device: Device
    cuda_visible_devices: str | None = None  # applied unless CUDA_VISIBLE_DEVICES is already set
    batch_size: PositiveInt
    context_days: PositiveInt
    prediction_length: PositiveInt
    issue_lead_intervals: NonNegativeInt  # issued this many intervals before its first interval
    # 1 = re-forecast at every gate closure, 48 = once a day (default: runs saved before it existed)
    issue_every_intervals: PositiveInt = 48
    quantile_levels: list[float]
    covariates: list[Covariate]
    weather: WeatherConfig
    spike_quantile: float = Field(gt=0, lt=1)
    scenarios: ScenarioConfig

    @model_validator(mode="after")
    def _levels(self) -> Self:
        levels = self.quantile_levels
        if any(not 0 < q < 1 for q in levels) or levels != sorted(set(levels)):
            raise ValueError("quantile_levels must be strictly increasing in (0, 1)")
        missing = {0.1, 0.5, 0.9} - set(levels)
        if missing:
            raise ValueError(f"quantile_levels must include 0.1, 0.5 and 0.9 (missing {missing})")
        return self


def load_forecast(path: Path = CONFIG_DIR / "forecast.yaml") -> ForecastConfig:
    return ForecastConfig.model_validate(_read_yaml(path))


# --------------------------------------------------------------------------- bidding


class SweepConfig(_Config):
    price_min: float
    price_max: float
    price_step: PositiveFloat

    @property
    def prices(self) -> list[float]:
        n = int(round((self.price_max - self.price_min) / self.price_step)) + 1
        return [self.price_min + i * self.price_step for i in range(n)]


class FeasibilityMode(StrEnum):
    WORST_CASE = "worst_case"  # every pending tranche may clear: never infeasible
    PROBABILISTIC = "probabilistic"  # SOC quantiles over the scenarios + margin


class FeasibilityConfig(_Config):
    mode: FeasibilityMode
    soc_quantile: float = Field(gt=0, lt=0.5)
    margin_pct: NonNegativeFloat  # of usable capacity, widens the probabilistic bounds


class BiddingConfig(_Config):
    soc_levels: int = Field(ge=3)  # SOC grid points from soc_min to soc_max
    terminal_penalty_sgd_per_mwh: PositiveFloat  # per MWh below soc_init at the end of the day
    price_band_margin_sgd_per_mwh: NonNegativeFloat  # offers kept within scenario range +- this
    sweep: SweepConfig
    feasibility: FeasibilityConfig


def load_bidding(path: Path = CONFIG_DIR / "bidding.yaml") -> BiddingConfig:
    return BiddingConfig.model_validate(_read_yaml(path))
