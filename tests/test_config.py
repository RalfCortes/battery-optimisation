import pytest
from pydantic import ValidationError

from battery_optimisation.config import (
    BatteryConfig,
    MarketConfig,
    Product,
    load_battery,
    load_market,
)


def test_battery_yaml_loads() -> None:
    battery = load_battery()
    assert battery.p_max_mw == 100
    assert battery.e_max_mwh == 100
    assert battery.round_trip_efficiency == pytest.approx(0.85)


def test_market_yaml_loads() -> None:
    market = load_market()
    assert market.interval_h == 0.5
    assert set(market.enabled_products) == set(Product)
    assert market.reserve_sustain_h[Product.CONTINGENCY_RESERVE] == 0.5


def test_battery_rejects_inconsistent_soc() -> None:
    data = load_battery().model_dump() | {"soc_init": 150.0}
    with pytest.raises(ValidationError, match="soc_init"):
        BatteryConfig.model_validate(data)


def test_battery_rejects_unknown_key() -> None:
    data = load_battery().model_dump() | {"p_max": 10.0}
    with pytest.raises(ValidationError):
        BatteryConfig.model_validate(data)


def test_market_rejects_unknown_product() -> None:
    data = load_market().model_dump() | {"products": {"capacity": True}}
    with pytest.raises(ValidationError):
        MarketConfig.model_validate(data)


def test_forecast_yaml_loads() -> None:
    from battery_optimisation.config import Covariate, load_forecast

    cfg = load_forecast()
    assert cfg.prediction_length == 48
    assert {0.1, 0.5, 0.9} <= set(cfg.quantile_levels)
    assert Covariate.DEMAND in cfg.covariates
    assert 20 <= cfg.scenarios.n_scenarios <= 50


def test_forecast_rejects_levels_without_median() -> None:
    from battery_optimisation.config import ForecastConfig, load_forecast

    data = load_forecast().model_dump() | {"quantile_levels": [0.1, 0.9]}
    with pytest.raises(ValidationError, match="0.5"):
        ForecastConfig.model_validate(data)


def test_bidding_yaml_loads() -> None:
    from battery_optimisation.config import load_bidding

    cfg = load_bidding()
    assert cfg.soc_levels >= 51
    assert cfg.terminal_penalty_sgd_per_mwh > load_market().price_cap
    assert cfg.price_band_margin_sgd_per_mwh >= 0
    assert (
        cfg.sweep.prices[0] == cfg.sweep.price_min and cfg.sweep.prices[-1] == cfg.sweep.price_max
    )


def test_bidding_firm_reserves_flag() -> None:
    from battery_optimisation.config import load_bidding

    assert load_bidding().firm_reserves is False  # hybrid: better on 12 months
