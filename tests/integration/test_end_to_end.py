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


def test_rolling_plans_survive_gate_closure(market_cfg, battery_cfg) -> None:
    """Re-planning at every decision over the next 48 intervals, from the SOC the committed
    offers will reach: with perfect information the simulator reproduces what was planned
    for each interval."""
    from battery_optimisation.bidding.strategy import Horizon

    prices = make_prices(daily_shape(3, seed=5))
    strategy = ScheduleStrategy(PerfectForecaster(prices), battery_cfg, market_cfg, Horizon.ROLLING)
    dispatch = Simulator(Market(market_cfg, prices), Battery(battery_cfg)).run(strategy)
    assert len(strategy.plans) == 3 * 48  # one plan per decision
    planned = [strategy.committed[t] for t in dispatch.frame[Col.INTERVAL_START]]
    assert dispatch.frame[Col.CHARGE_MW].to_list() == pytest.approx([c for c, _ in planned])
    assert dispatch.frame[Col.DISCHARGE_MW].to_list() == pytest.approx([d for _, d in planned])
    assert dispatch.frame[Col.SOC_MWH].min() >= battery_cfg.soc_min - 1e-6


def test_perfect_foresight_with_ancillary_services_reproduces_the_plan(
    market_cfg, battery_cfg
) -> None:
    """Energy + regulation + reserves co-optimised; the simulator books the same profit, and the
    reserve awards are sustainable at every interval (CLAUDE.md invariant: it raises if not)."""
    import numpy as np
    import polars as pl

    from battery_optimisation.bidding.strategy import Horizon
    from battery_optimisation.config import Product
    from battery_optimisation.sim.clearing import price_column

    rng = np.random.default_rng(3)
    prices = make_prices(daily_shape(2, seed=5)).with_columns(
        pl.Series(price_column(Product.REGULATION, market_cfg), rng.uniform(5, 40, 96)),
        pl.Series(price_column(Product.PRIMARY_RESERVE, market_cfg), rng.uniform(0, 20, 96)),
        pl.Series(price_column(Product.CONTINGENCY_RESERVE, market_cfg), rng.uniform(0, 30, 96)),
    )
    strategy = ScheduleStrategy(PerfectForecaster(prices), battery_cfg, market_cfg, Horizon.DAY)
    dispatch = Simulator(Market(market_cfg, prices), Battery(battery_cfg)).run(strategy)
    assert dispatch.profit_sgd == pytest.approx(strategy.planned_profit_sgd, rel=1e-6)
    assert dispatch.total().ancillary_sgd > 0
