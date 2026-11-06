import pytest
from synthetic import daily_shape, make_prices

from battery_optimisation.bidding.strategy import Horizon
from battery_optimisation.forecast.perfect import PerfectForecaster
from battery_optimisation.sim.backtest import StrategyName, run_schedule


@pytest.mark.parametrize("horizon", [Horizon.DAY, Horizon.ROLLING])
def test_perfect_foresight_runs_with_every_horizon(battery_cfg, market_cfg, horizon) -> None:
    """The plan-vs-simulation check applies to non-overlapping plans only (rolling plans
    overlap, their objectives do not add up to the realised profit)."""
    prices = make_prices(daily_shape(2, seed=1))
    summary, dispatch = run_schedule(
        StrategyName.PERFECT_FORESIGHT,
        PerfectForecaster(prices),
        prices,
        battery_cfg,
        market_cfg,
        horizon,
    )
    assert dispatch.profit_sgd > 0
    assert summary.horizon is horizon


def test_saved_runs_keep_the_offers(battery_cfg, market_cfg, tmp_path) -> None:
    import polars as pl

    from battery_optimisation.sim.backtest import RunFiles, save_run

    prices = make_prices(daily_shape(1, seed=2))
    summary, dispatch = run_schedule(
        StrategyName.PERFECT_FORESIGHT,
        PerfectForecaster(prices),
        prices,
        battery_cfg,
        market_cfg,
        Horizon.DAY,
    )
    run_dir = save_run(summary, dispatch, tmp_path)
    saved = pl.read_parquet(run_dir / RunFiles.OFFERS)
    assert saved.height == dispatch.offers.height > 0
