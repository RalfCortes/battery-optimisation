import numpy as np
import polars as pl
import pytest
from synthetic import NoisyForecaster, daily_shape, make_prices

from battery_optimisation.battery import Battery
from battery_optimisation.bidding.strategy import Horizon, ScheduleStrategy
from battery_optimisation.columns import Col
from battery_optimisation.forecast.perfect import PerfectForecaster
from battery_optimisation.optim.milp import solve_schedule
from battery_optimisation.sim.clearing import Market
from battery_optimisation.sim.simulator import Simulator


def run(strategy, prices, battery_cfg, market_cfg) -> float:
    return Simulator(Market(market_cfg, prices), Battery(battery_cfg)).run(strategy).profit_sgd


def test_schedule_respects_limits_and_terminal_soc(battery_cfg, market_cfg) -> None:
    sol = solve_schedule(make_prices(daily_shape(2)), Battery(battery_cfg), market_cfg)
    s = sol.schedule
    assert sol.objective_sgd > 0
    assert s[Col.CHARGE_MW].max() <= battery_cfg.p_max_mw + 1e-6
    assert s[Col.DISCHARGE_MW].max() <= battery_cfg.p_max_mw + 1e-6
    assert s[Col.SOC_MWH].min() >= battery_cfg.soc_min - 1e-6
    assert s[Col.SOC_MWH].max() <= battery_cfg.soc_max + 1e-6
    assert s[Col.SOC_MWH][-1] >= battery_cfg.soc_init - 1e-6


def test_no_simultaneous_charge_and_discharge(battery_cfg, market_cfg) -> None:
    # Negative prices and a full battery: without the binary, charging and discharging at once
    # would burn grid energy through the losses and be paid for it.
    cfg = battery_cfg.model_copy(update={"kappa_sgd_per_mwh": 0.0})
    full = Battery(cfg, soc_mwh=cfg.soc_max)
    sol = solve_schedule(make_prices(np.full(48, -500.0)), full, market_cfg)
    both = sol.schedule.filter((pl.col(Col.CHARGE_MW) > 0) & (pl.col(Col.DISCHARGE_MW) > 0))
    assert both.height == 0


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


def test_rejects_gaps_and_missing_prices(battery_cfg, market_cfg) -> None:
    prices = make_prices(daily_shape(1))
    with_gap = prices.with_row_index().filter(pl.col("index") != 5).drop("index")
    with pytest.raises(ValueError, match="contiguous"):
        solve_schedule(with_gap, Battery(battery_cfg), market_cfg)
    missing = prices.with_columns(pl.lit(None, pl.Float64).alias(Col.USEP))
    with pytest.raises(ValueError, match="missing"):
        solve_schedule(missing, Battery(battery_cfg), market_cfg)
