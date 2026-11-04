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
