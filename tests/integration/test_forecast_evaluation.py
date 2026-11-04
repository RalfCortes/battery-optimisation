"""End to end on synthetic data: quantiles -> scenarios -> stats -> capture rates."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from synthetic import make_prices
from synthetic_forecast import at, normal_quantiles, quantile_frame

from battery_optimisation.bidding.strategy import Horizon
from battery_optimisation.columns import Col
from battery_optimisation.config import load_battery, load_forecast, load_market
from battery_optimisation.forecast.report import evaluate
from battery_optimisation.sim.backtest import StrategyName

DAYS, TEST_DAYS = 20, 10
START = date(2026, 1, 1)


@pytest.fixture(scope="module")
def report_and_runs():
    rng = np.random.default_rng(7)
    cfg = load_forecast().model_copy(update={"issue_every_intervals": 48})  # daily issues
    cfg = cfg.model_copy(
        update={
            "scenarios": cfg.scenarios.model_copy(
                update={"calibration_days": 7, "n_samples": 200, "n_scenarios": 20}
            )
        }
    )
    hours = np.arange(48) / 2
    mu = 150 + 100 * np.sin((hours - 10) / 24 * 2 * np.pi)
    realised = make_prices(
        np.concatenate([mu + 25 * rng.standard_normal(48) for _ in range(DAYS)]), START
    )
    gap = cfg.issue_lead_intervals  # issued at gate closure before each delivery day
    qf = pl.concat(
        quantile_frame(
            at(START + timedelta(days=d)) - timedelta(minutes=30 * gap),
            normal_quantiles(mu, 25.0, cfg.quantile_levels),
            cfg.quantile_levels,
            gap=gap,
        )
        for d in range(DAYS)
    )
    test_start = START + timedelta(days=DAYS - TEST_DAYS)
    return evaluate(
        qf=qf,
        realised=realised,
        test_start=test_start,
        test_end=START + timedelta(days=DAYS - 1),
        cfg=cfg,
        battery=load_battery(),
        market=load_market(),
    )


def test_forecast_and_scenario_statistics(report_and_runs) -> None:
    report = report_and_runs.report
    assert report.overall.intervals == TEST_DAYS * 48
    assert 0.6 < report.overall.coverage_10_90 < 0.95
    # median MAE ~ 25 * 0.80; seasonal-naive MAE ~ 25 * sqrt(2) * 0.80 -> MASE ~ 0.71
    assert 0.55 < report.overall.mase < 0.85
    assert report.mase_scale > 0
    assert all(m.mase > 0 for m in report.monthly)
    assert len(report.monthly) == 1  # all in January
    scen = report.scenarios[0]
    assert scen.calibration_issues == 7
    assert report_and_runs.scenarios[Col.ISSUED_AT].n_unique() == TEST_DAYS


def test_capture_rates(report_and_runs) -> None:
    capture = {(c.strategy, c.horizon): c for c in report_and_runs.report.capture}
    assert set(capture) == {
        (StrategyName.PERFECT_FORESIGHT, Horizon.FULL),
        (StrategyName.PERFECT_FORESIGHT, Horizon.DAY),
        (StrategyName.CHRONOS_MEDIAN, Horizon.DAY),
        (StrategyName.PERSISTENCE, Horizon.DAY),
    }
    assert capture[
        (StrategyName.PERFECT_FORESIGHT, Horizon.FULL)
    ].capture_vs_perfect_full == pytest.approx(1.0)
    for key in [
        (StrategyName.CHRONOS_MEDIAN, Horizon.DAY),
        (StrategyName.PERSISTENCE, Horizon.DAY),
    ]:
        assert capture[key].capture_vs_perfect_day <= 1.0 + 1e-9
    # the median of a calibrated forecast beats yesterday's noisy prices
    assert (
        capture[(StrategyName.CHRONOS_MEDIAN, Horizon.DAY)].profit_sgd
        > capture[(StrategyName.PERSISTENCE, Horizon.DAY)].profit_sgd
    )
    assert set(report_and_runs.runs) == set(capture)


def test_forecast_run_round_trip(report_and_runs, tmp_path) -> None:
    from battery_optimisation.forecast.report import (
        list_forecast_runs,
        load_forecast_run,
        save_forecast_run,
    )

    qf = pl.DataFrame({Col.ISSUED_AT: [], Col.QUANTILE: []})  # content irrelevant here
    run_dir = save_forecast_run(report_and_runs, qf, tmp_path / "forecast_x")
    assert list_forecast_runs(tmp_path) == [run_dir]

    run = load_forecast_run(run_dir)
    assert run.report == report_and_runs.report
    assert run.scenarios.shape == report_and_runs.scenarios.shape
    assert set(run.backtests) == set(report_and_runs.runs)
    summary, dispatch = run.backtests[(StrategyName.CHRONOS_MEDIAN, Horizon.DAY)]
    assert dispatch.height == TEST_DAYS * 48
