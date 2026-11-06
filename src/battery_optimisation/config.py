"""Typed configuration: pydantic models for ``config/battery.yaml`` and ``config/market.yaml``.

battery = load_battery()
battery.p_max_mw, battery.round_trip_efficiency
market = load_market()
market.enabled_products, market.storage_rules.discharge_checks
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
LOWER_PRODUCTS = (Product.REGULATION,)  # the only ones that can push energy into the battery


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


class RegulationEnergy(_Config):
    """Energy AGC moves per MW of regulation held, as a share of MW x interval (expected):
    ``up`` injected (raise, paid the energy price), ``down`` withdrawn (lower, pays it). It is
    ordinary metered energy (Market Rules ch. 7): it moves the SoC and adds throughput (wear)."""

    up: float = Field(ge=0, le=1)
    down: float = Field(ge=0, le=1)

    @property
    def throughput(self) -> float:
        return self.up + self.down


class AncillaryConfig(_Config):
    reserve_group: ReserveGroup  # whose reserve prices apply to the battery
    regulation_energy: RegulationEnergy
    max_award_mw: dict[Product, NonNegativeFloat]  # cap per product (market size)


class StorageRules(_Config):
    """Limits the market engine puts on a storage facility's schedule (``sim.storage_rules``).

    Power, per group (NEMS: one per reserve class, RC383 D.17.2.4A / D.18.1.6-7):
      raise: discharge - charge + sum of the group's awards <= P
      lower: charge - discharge + sum of the group's awards <= P
    SoC, per check (RC386): hours each scheduled MW must be sustained from the SoC at the start
    of the interval (``energy`` = the energy schedule itself):
      discharge checks: drawn MWh <= SoC - soc_min      charge checks: added MWh <= soc_max - SoC
    ``enforce_soc``: the engine cuts schedules to what the SoC allows (NEMS since 12 Aug 2025);
    otherwise offers clear regardless and infeasible ones reach the battery.
    """

    enforce_soc: bool
    raise_power_groups: list[list[Product]]
    lower_power_groups: list[list[Product]]
    discharge_checks: list[dict[Product, NonNegativeFloat]]
    charge_checks: list[dict[Product, NonNegativeFloat]]

    @model_validator(mode="after")
    def _products(self) -> Self:
        for group in self.raise_power_groups + self.lower_power_groups:
            if not set(group) <= set(ANCILLARY_PRODUCTS):
                raise ValueError("power groups hold ancillary products only")
        for group in self.lower_power_groups:
            if not set(group) <= set(LOWER_PRODUCTS):
                raise ValueError("lower_power_groups: only regulation pushes energy in")
        for check in self.charge_checks:
            if not set(check) <= {Product.ENERGY, *LOWER_PRODUCTS}:
                raise ValueError("charge_checks: reserves only lower the SoC (not counted)")
        return self

    @property
    def raise_products(self) -> set[Product]:
        return {p for g in self.raise_power_groups for p in g}

    @property
    def lower_products(self) -> set[Product]:
        return {p for g in self.lower_power_groups for p in g}


class ShortfallPenalty(_Config):
    """Cleared energy the battery cannot deliver (safety net: with ``enforce_soc`` the engine
    already caps schedules to the SoC)."""

    enabled: bool
    sgd_per_mwh: NonNegativeFloat

    @property
    def price(self) -> float:
        return self.sgd_per_mwh if self.enabled else 0.0


class MarketConfig(_Config):
    """Market parameters. Prices in SGD/MWh, durations in hours."""

    timezone: str
    interval_h: PositiveFloat
    intervals_per_day: PositiveInt
    gate_closure_intervals: NonNegativeInt  # offers for t are fixed at the start of t - this
    shortfall_penalty: ShortfallPenalty
    n_tranches: PositiveInt  # per side (energy storage offer) and per ancillary product
    price_decimals: NonNegativeInt = 2  # offer prices, $/MWh (Market Rules ch. 6)
    quantity_decimals: NonNegativeInt = 1  # offer quantities, MW
    price_floor: float
    price_cap: float
    products: dict[Product, bool]
    storage_rules: StorageRules
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
    MARKET = "market"  # no SoC caps: the engine cuts to the SoC (storage_rules.enforce_soc)


class FeasibilityConfig(_Config):
    mode: FeasibilityMode
    soc_quantile: float = Field(gt=0, lt=0.5)
    margin_pct: NonNegativeFloat  # of usable capacity, widens the probabilistic bounds


class PriceModel(StrEnum):
    """How the water-value SDP sees prices across stages."""

    INDEPENDENT = "independent"  # stagewise marginals of the copula scenarios
    MARKOV = "markov"  # price states from the quantiles, persisting (forecast/markov.py)
    CONTINUOUS = "continuous"  # continuous score state: nodes + interpolation (node_lattice)


class MarkovConfig(_Config):
    states: int = Field(ge=2)  # price states per stage
    rho: float | None = Field(default=None, ge=0, lt=1)  # lag-1 correlation; None: copula's


class ContinuousConfig(_Config):
    """Continuous price state (``forecast.markov.node_lattice``): values at ``nodes`` normal
    scores on [-z_max, z_max], linear in between; rho as ``markov.rho``."""

    nodes: int = Field(default=21, ge=3)
    z_max: float = Field(default=3.0, gt=0)


class ReservePrices(StrEnum):
    """How the water value's SDP sees reserve prices (reserve planner ``sdp``)."""

    POINT = "point"  # one forecast per stage (``ancillary_price``: median or mean)
    MARKOV = "markov"  # Markov states of a common reserve factor (``reserve_markov``)


class ReserveMarkovConfig(_Config):
    """One-factor model of the reserve prices (``forecast.markov.factor_lattice``). Defaults
    measured on the 2025-26 Chronos forecast errors (normal scores): pairwise correlations
    reg/pri 0.28, reg/con 0.49, pri/con 0.35 -> loadings; lag-1 0.71-0.77."""

    states: int = Field(default=5, ge=2)
    rho: float = Field(default=0.75, ge=0, lt=1)
    loadings: dict[Product, float] = {
        Product.REGULATION: 0.63,
        Product.PRIMARY_RESERVE: 0.45,
        Product.CONTINGENCY_RESERVE: 0.78,
    }


class AncillaryPrice(StrEnum):
    """Point forecast of ancillary prices the water value plans reserves on."""

    MEDIAN = "median"  # quantile 0.5 (what the self-schedule uses)
    MEAN = "mean"  # mean of the forecast distribution: availability revenue is linear in price


class ReservePlanner(StrEnum):
    """Who chooses the ancillary awards of the water-value strategy."""

    MILP = "milp"  # a MILP plan on the mean stage price commits them; SDP gets what is left
    SDP = "sdp"  # the SDP chooses them per stage, SOC and energy action (``optim.reserves``)
    STOCHASTIC_MILP = "stochastic_milp"  # two-stage MILP: reserves vs sampled price paths


class BiddingConfig(_Config):
    soc_levels: int = Field(ge=3)  # SOC grid points from soc_min to soc_max
    terminal_penalty_sgd_per_mwh: PositiveFloat  # per MWh below soc_init at the end of the day
    price_band_margin_sgd_per_mwh: NonNegativeFloat  # offers kept within scenario range +- this
    sweep: SweepConfig
    feasibility: FeasibilityConfig
    firm_reserves: bool = True  # ancillary: reserves fit beside any energy flow (else committed
    # price-taking charge backs raise reserves beyond P, as in the plan)
    price_model: PriceModel = PriceModel.INDEPENDENT
    markov: MarkovConfig = MarkovConfig(states=11)
    continuous: ContinuousConfig = ContinuousConfig()
    reserve_planner: ReservePlanner = ReservePlanner.MILP
    stochastic_paths: int = Field(default=20, ge=1)  # price paths of the stochastic reserve plan
    ancillary_price: AncillaryPrice = AncillaryPrice.MEDIAN
    reserve_prices: ReservePrices = ReservePrices.POINT
    reserve_markov: ReserveMarkovConfig = ReserveMarkovConfig()
    record_plans: bool = True  # keep each decision's plan (plans.parquet) for troubleshooting
    stochastic_relax: bool = False  # LP relaxation of the stochastic plan (no binaries, faster)


def load_bidding(path: Path = CONFIG_DIR / "bidding.yaml") -> BiddingConfig:
    return BiddingConfig.model_validate(_read_yaml(path))
