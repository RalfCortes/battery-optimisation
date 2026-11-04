"""Water-value bidding through the simulator with gate closure."""

from datetime import timedelta

import numpy as np
import polars as pl
import pytest
from synthetic import daily_shape, make_prices

from battery_optimisation.battery import Battery
from battery_optimisation.bidding.strategy import Horizon, ScheduleStrategy, WaterValueStrategy
from battery_optimisation.columns import Col
from battery_optimisation.config import load_bidding
from battery_optimisation.forecast.perfect import PerfectForecaster
from battery_optimisation.sim.clearing import Market
from battery_optimisation.sim.simulator import Simulator

DAYS = 3


def scenario_frame(prices: pl.DataFrame, n: int, noise: float, gap: int, seed: int) -> pl.DataFrame:
    """Per delivery day, issued at gate closure: realised + noise (n paths, equal weights)."""
    rng = np.random.default_rng(seed)
    frames = []
    for _, part in prices.group_by(pl.col(Col.INTERVAL_START).dt.date(), maintain_order=True):
        stamps, y = part[Col.INTERVAL_START], part[Col.USEP].to_numpy()
        issued = stamps[0] - timedelta(minutes=30 * gap)
        for k in range(n):
            frames.append(
                pl.DataFrame(
                    {
                        Col.ISSUED_AT: [issued] * len(y),
                        Col.SCENARIO: [k] * len(y),
                        Col.WEIGHT: [1 / n] * len(y),
                        Col.INTERVAL_START: stamps,
                        Col.USEP_FORECAST: y + rng.normal(0, noise, len(y)),
                    }
                )
            )
    return pl.concat(frames).with_columns(pl.col(Col.SCENARIO).cast(pl.Int32))


def run(strategy, prices, battery_cfg, market_cfg):
    return Simulator(Market(market_cfg, prices), Battery(battery_cfg)).run(strategy)


def test_offer_curves_never_breach_limits(battery_cfg, market_cfg) -> None:
    prices = make_prices(daily_shape(DAYS, seed=4))
    scen = scenario_frame(prices, n=20, noise=80.0, gap=market_cfg.gate_closure_intervals, seed=1)
    strategy = WaterValueStrategy(scen, battery_cfg, market_cfg, load_bidding())
    dispatch = run(strategy, prices, battery_cfg, market_cfg)  # raises on any breach
    assert dispatch.frame.height == DAYS * 48
    assert dispatch.frame[Col.SOC_MWH].min() >= battery_cfg.soc_min - 1e-6
    assert dispatch.profit_sgd > 0
    assert len(strategy.value_functions) == DAYS


def test_perfect_scenarios_approach_the_benchmark(battery_cfg, market_cfg) -> None:
    """One exact scenario: bidding the water value is close to the daily perfect-foresight MILP."""
    prices = make_prices(daily_shape(DAYS, seed=6))
    scen = scenario_frame(prices, n=1, noise=0.0, gap=market_cfg.gate_closure_intervals, seed=0)
    bid = run(
        WaterValueStrategy(scen, battery_cfg, market_cfg, load_bidding()),
        prices,
        battery_cfg,
        market_cfg,
    )
    pf = run(
        ScheduleStrategy(PerfectForecaster(prices), battery_cfg, market_cfg, Horizon.DAY),
        prices,
        battery_cfg,
        market_cfg,
    )
    assert bid.profit_sgd == pytest.approx(pf.profit_sgd, rel=0.05)


def test_offers_respect_tranche_limit_and_price_bounds(battery_cfg, market_cfg) -> None:
    prices = make_prices(daily_shape(1, seed=2))
    scen = scenario_frame(prices, n=20, noise=200.0, gap=market_cfg.gate_closure_intervals, seed=3)
    strategy = WaterValueStrategy(scen, battery_cfg, market_cfg, load_bidding())
    run(strategy, prices, battery_cfg, market_cfg)
    for offers in strategy.submitted.values():
        for side in (s for s in {t.side for t in offers.tranches}):
            assert len(offers.on(side)) <= market_cfg.n_tranches
        assert all(
            market_cfg.price_floor <= t.price <= market_cfg.price_cap for t in offers.tranches
        )


def test_bidding_report(battery_cfg, market_cfg) -> None:
    from synthetic_forecast import normal_quantiles, quantile_frame

    from battery_optimisation.bidding.report import evaluate_bidding
    from battery_optimisation.config import load_forecast
    from battery_optimisation.sim.backtest import StrategyName

    prices = make_prices(daily_shape(DAYS, seed=8))
    gap = market_cfg.gate_closure_intervals
    scen = scenario_frame(prices, n=20, noise=60.0, gap=gap, seed=2)
    levels = load_forecast().quantile_levels
    qf = pl.concat(
        quantile_frame(
            part[Col.INTERVAL_START][0] - timedelta(minutes=30 * gap),
            normal_quantiles(part[Col.USEP].to_numpy(), 60.0, levels),
            levels,
            gap=gap,
        )
        for _, part in prices.group_by(pl.col(Col.INTERVAL_START).dt.date(), maintain_order=True)
    )
    ev = evaluate_bidding(scen, qf, prices, battery_cfg, market_cfg, load_bidding())
    r = ev.report
    by_name = {c.strategy: c for c in r.results}
    assert set(by_name) == {
        StrategyName.PERFECT_FORESIGHT,
        StrategyName.CHRONOS_MEDIAN,
        StrategyName.WATER_VALUE,
    }
    wv, ss = by_name[StrategyName.WATER_VALUE], by_name[StrategyName.CHRONOS_MEDIAN]
    assert r.uplift_sgd == pytest.approx(wv.profit_sgd - ss.profit_sgd)
    assert r.uplift_pct == pytest.approx(r.uplift_sgd / abs(ss.profit_sgd))
    assert sum(m.uplift_sgd for m in r.monthly) == pytest.approx(r.uplift_sgd)
    assert by_name[StrategyName.PERFECT_FORESIGHT].capture_vs_perfect_day == pytest.approx(1.0)
    assert wv.capture_vs_perfect_day <= 1.0 + 1e-9
    assert r.sdp_days == DAYS


def test_sweep_and_comparison_table(battery_cfg, market_cfg) -> None:
    from battery_optimisation.bidding.report import ComparisonInput, comparison_table
    from battery_optimisation.bidding.sweep import sweep
    from battery_optimisation.sim.backtest import StrategyName

    prices = make_prices(daily_shape(2, seed=9))
    scen = scenario_frame(prices, n=20, noise=60.0, gap=market_cfg.gate_closure_intervals, seed=4)
    strategy = WaterValueStrategy(scen, battery_cfg, market_cfg, load_bidding())
    run(strategy, prices, battery_cfg, market_cfg)
    stats, per_interval = sweep(strategy, np.linspace(-100, 600, 71))
    assert stats.intervals == 2 * 48 == per_interval.height
    assert stats.prices == 71
    assert 0 <= stats.share_matching <= 1
    assert stats.max_gap_mw <= 2 * battery_cfg.p_max_mw

    pf, ss, wv = (
        StrategyName.PERFECT_FORESIGHT,
        StrategyName.CHRONOS_MEDIAN,
        StrategyName.WATER_VALUE,
    )
    table = comparison_table(
        [
            ComparisonInput(
                strategy=pf, horizon=Horizon.FULL, profit_sgd=110.0, cycles_per_day=2.0
            ),
            ComparisonInput(strategy=pf, horizon=Horizon.DAY, profit_sgd=100.0, cycles_per_day=2.0),
            ComparisonInput(strategy=ss, horizon=Horizon.DAY, profit_sgd=40.0, cycles_per_day=1.0),
            ComparisonInput(strategy=wv, horizon=Horizon.DAY, profit_sgd=55.0, cycles_per_day=1.0),
        ],
        baseline=ss,
    )
    row = table.filter(pl.col("strategy") == wv).to_dicts()[0]
    assert row["vs_baseline_sgd"] == pytest.approx(15.0)
    assert row["vs_baseline_pct"] == pytest.approx(0.375)
    assert row["capture_vs_perfect_day"] == pytest.approx(0.55)
    assert row["capture_vs_perfect_full"] == pytest.approx(0.5)


def test_sweep_breakdown_by_pipeline_step(battery_cfg, market_cfg) -> None:
    from battery_optimisation.bidding.sweep import CurveStep, sweep_breakdown

    prices = make_prices(daily_shape(2, seed=9))
    scen = scenario_frame(prices, n=20, noise=60.0, gap=market_cfg.gate_closure_intervals, seed=4)
    strategy = WaterValueStrategy(scen, battery_cfg, market_cfg, load_bidding())
    run(strategy, prices, battery_cfg, market_cfg)
    steps = sweep_breakdown(strategy, np.linspace(-100, 600, 71))
    assert [s.step for s in steps] == list(CurveStep)
    gaps = {s.step: s.mean_abs_gap_in_band_mw for s in steps}
    assert gaps[CurveStep.RAW] == pytest.approx(0.0, abs=1e-6)  # the curve is the optimum
    stats, _ = __import__("battery_optimisation.bidding.sweep", fromlist=["sweep"]).sweep(
        strategy, np.linspace(-100, 600, 71)
    )
    # replaying every step reproduces exactly what was submitted
    assert gaps[CurveStep.FEASIBLE] == pytest.approx(stats.mean_abs_gap_in_band_mw)
