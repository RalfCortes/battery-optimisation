import pytest
from synthetic import NoisyForecaster, daily_shape, make_prices

from battery_optimisation.battery import Battery
from battery_optimisation.bidding.strategy import Horizon, ScheduleStrategy
from battery_optimisation.forecast.perfect import PerfectForecaster
from battery_optimisation.sim.clearing import Market
from battery_optimisation.sim.simulator import Simulator


def run(strategy, prices, battery_cfg, market_cfg) -> float:
    return Simulator(Market(market_cfg, prices), Battery(battery_cfg)).run(strategy).profit_sgd


def test_daily_horizon_never_beats_full_horizon(battery_cfg, market_cfg) -> None:
    prices = make_prices(daily_shape(3, seed=1))
    profits = {
        h: run(
            ScheduleStrategy(PerfectForecaster(prices), battery_cfg, market_cfg, horizon=h),
            prices,
            battery_cfg,
            market_cfg,
        )
        for h in Horizon
    }
    assert profits[Horizon.DAY] <= profits[Horizon.FULL] + 1e-6


@pytest.mark.parametrize("seed", range(5))
def test_perfect_foresight_beats_forecast_based_strategy(battery_cfg, market_cfg, seed) -> None:
    """Any feasible strategy (here: MILP on a noisy forecast) earns at most perfect foresight."""
    prices = make_prices(daily_shape(2, seed=seed))
    perfect = ScheduleStrategy(PerfectForecaster(prices), battery_cfg, market_cfg)
    noisy = ScheduleStrategy(NoisyForecaster(prices, 60.0, seed), battery_cfg, market_cfg)
    assert (
        run(noisy, prices, battery_cfg, market_cfg)
        <= run(perfect, prices, battery_cfg, market_cfg) + 1e-6
    )
