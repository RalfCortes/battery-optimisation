from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from synthetic import make_prices
from synthetic_forecast import at, normal_quantiles, quantile_frame

from battery_optimisation.columns import Col
from battery_optimisation.config import load_forecast
from battery_optimisation.forecast import walkforward

CFG = load_forecast()
LEVELS = CFG.quantile_levels
DAYS = 10


def synthetic(days: int = DAYS, seed: int = 0) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Calibrated N(mu, 20) quantile forecasts issued daily at 00:00 + realised prices."""
    rng = np.random.default_rng(seed)
    mu = 150 + 50 * np.sin(np.arange(48) / 48 * 2 * np.pi)
    qf = pl.concat(
        quantile_frame(
            at(date(2026, 1, 1) + timedelta(days=d)), normal_quantiles(mu, 20.0, LEVELS), LEVELS
        )
        for d in range(days)
    )
    realised = make_prices(np.concatenate([mu + 20 * rng.standard_normal(48) for _ in range(days)]))
    return qf, realised


def test_issue_times_one_per_delivery_day_before_it_starts() -> None:
    issues = walkforward.issue_times(date(2026, 1, 1), date(2026, 1, 3), CFG)
    assert issues == [
        at(date(2025, 12, 31), 23),
        at(date(2026, 1, 1), 23),
        at(date(2026, 1, 2), 23),
    ]


def test_month_windows_split_calendar_months() -> None:
    assert walkforward.month_windows(date(2025, 10, 28), date(2025, 12, 5)) == [
        (date(2025, 10, 28), date(2025, 10, 31)),
        (date(2025, 11, 1), date(2025, 11, 30)),
        (date(2025, 12, 1), date(2025, 12, 5)),
    ]


def test_calibration_uses_only_fully_realised_past_issues() -> None:
    qf, realised = synthetic()
    cut = at(date(2026, 1, 6))
    u = walkforward.calibration_pits(qf, realised, before=cut, days=30, levels=LEVELS)
    assert u.shape == (5, 48)  # issues of Jan 1..5; Jan 6 is not yet realised

    poisoned = realised.with_columns(
        pl.when(pl.col(Col.INTERVAL_START) >= cut)
        .then(1e9)
        .otherwise(pl.col(Col.USEP))
        .alias(Col.USEP)
    )
    again = walkforward.calibration_pits(qf, poisoned, before=cut, days=30, levels=LEVELS)
    assert np.array_equal(u, again)  # nothing at or after the cut is used

    recent = walkforward.calibration_pits(qf, realised, before=cut, days=2, levels=LEVELS)
    assert recent.shape == (2, 48)


def test_month_scenarios_layout() -> None:
    qf, realised = synthetic()
    scen = CFG.scenarios.model_copy(
        update={"calibration_days": 5, "n_samples": 200, "n_scenarios": 20}
    )
    cfg = CFG.model_copy(update={"scenarios": scen})
    out = walkforward.month_scenarios(qf, realised, date(2026, 1, 7), date(2026, 1, 10), cfg)

    assert set(out.columns) == {
        Col.ISSUED_AT,
        Col.SCENARIO,
        Col.WEIGHT,
        Col.INTERVAL_START,
        Col.USEP_FORECAST,
    }
    assert out[Col.ISSUED_AT].n_unique() == 4  # Jan 7..10
    per_issue = out.group_by(Col.ISSUED_AT).agg(
        pl.col(Col.SCENARIO).n_unique().alias("k"), pl.len().alias("rows")
    )
    assert per_issue["k"].to_list() == [20] * 4
    assert per_issue["rows"].to_list() == [20 * 48] * 4
    weights = (
        out.unique([Col.ISSUED_AT, Col.SCENARIO])
        .group_by(Col.ISSUED_AT)
        .agg(pl.col(Col.WEIGHT).sum())
    )
    assert weights[Col.WEIGHT].to_list() == pytest.approx([1.0] * 4)


def test_forecast_stats_on_calibrated_forecasts() -> None:
    qf, realised = synthetic(days=60, seed=3)
    stats = walkforward.forecast_stats(qf, realised, LEVELS, spike_threshold=1e9, mase_scale=1.0)
    assert stats.intervals == 60 * 48
    assert stats.coverage_10_90 == pytest.approx(0.8, abs=0.04)
    # CRPS of N(mu, 20) for a calibrated forecast = 20 * 0.5642 (=1/sqrt(pi)) ~ 11.3
    assert stats.crps == pytest.approx(20 / np.sqrt(np.pi), rel=0.08)


def test_weather_must_cover_the_first_context() -> None:
    from synthetic_forecast import make_features

    features = make_features(10)  # weather from 2026-01-01
    walkforward.check_weather_coverage(features, CFG, first_needed=at(date(2026, 1, 1)))
    late = features.with_columns(
        pl.when(pl.col(Col.INTERVAL_START) < at(date(2026, 1, 4)))
        .then(None)
        .otherwise(pl.col(c))
        .alias(c)
        for c in walkforward.weather_columns(CFG)
    )
    with pytest.raises(ValueError, match="2026-01-04"):
        walkforward.check_weather_coverage(late, CFG, first_needed=at(date(2026, 1, 2)))


def test_months_without_spikes_round_trip_through_json() -> None:
    qf, realised = synthetic(days=3)
    stats = walkforward.forecast_stats(qf, realised, LEVELS, spike_threshold=1e9, mase_scale=1.0)
    assert stats.spike_hit_rate is None  # no spike: undefined, not NaN
    again = walkforward.ForecastStats.model_validate_json(stats.model_dump_json())
    assert again == stats
