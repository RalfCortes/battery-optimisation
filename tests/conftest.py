import pytest

from battery_optimisation.config import BatteryConfig, MarketConfig, load_battery, load_market


@pytest.fixture
def battery_cfg() -> BatteryConfig:
    return load_battery()


@pytest.fixture
def market_cfg() -> MarketConfig:
    return load_market()
