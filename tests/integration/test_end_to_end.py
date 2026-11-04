import pytest
from synthetic import daily_shape, make_prices

from battery_optimisation.battery import Battery
from battery_optimisation.bidding.strategy import ScheduleStrategy
from battery_optimisation.columns import Col
from battery_optimisation.forecast.perfect import PerfectForecaster
from battery_optimisation.sim.clearing import Market
from battery_optimisation.sim.simulator import Simulator


def test_end_to_end_synthetic_day(market_cfg, battery_cfg) -> None:
    """Perfect foresight -> MILP plan -> self-schedule -> simulator reproduces the MILP."""
    prices = make_prices(daily_shape(1, seed=3))
    strategy = ScheduleStrategy(PerfectForecaster(prices), battery_cfg, market_cfg)
    dispatch = Simulator(Market(market_cfg, prices), Battery(battery_cfg)).run(strategy)

    frame = dispatch.frame
    assert frame.height == 48
    assert dispatch.profit_sgd == pytest.approx(strategy.planned_profit_sgd, rel=1e-9)
    assert dispatch.profit_sgd > 0
    assert frame[Col.SOC_MWH].min() >= battery_cfg.soc_min - 1e-6
    assert frame[Col.SOC_MWH].max() <= battery_cfg.soc_max + 1e-6
    assert frame[Col.SOC_MWH][-1] >= battery_cfg.soc_init - 1e-6


def test_daily_plans_survive_gate_closure(market_cfg, battery_cfg) -> None:
    """With offers fixed 2 intervals ahead, each daily plan must start from the SOC the previous
    plan will reach (not the SOC at decision time): the simulator still reproduces the plans."""
    from battery_optimisation.bidding.strategy import Horizon

    assert market_cfg.gate_closure_intervals == 2
    prices = make_prices(daily_shape(3, seed=5))
    strategy = ScheduleStrategy(PerfectForecaster(prices), battery_cfg, market_cfg, Horizon.DAY)
    dispatch = Simulator(Market(market_cfg, prices), Battery(battery_cfg)).run(strategy)
    assert len(strategy.plans) == 3
    assert dispatch.profit_sgd == pytest.approx(strategy.planned_profit_sgd, rel=1e-9)
