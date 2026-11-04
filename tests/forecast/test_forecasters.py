from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from synthetic import make_prices
from synthetic_forecast import at, normal_quantiles, quantile_frame

from battery_optimisation.columns import Col
from battery_optimisation.config import load_forecast, load_market
from battery_optimisation.forecast.baselines import PersistenceForecaster
from battery_optimisation.forecast.stored import QuantileForecaster
from battery_optimisation.sim.clearing import Market

LEVELS = load_forecast().quantile_levels


def two_days_market(day2: list[float] | None = None) -> Market:
    day1 = [float(i) for i in range(48)]
    day2 = day2 or [1e6] * 48  # unknowable at day-2 00:00
    return Market(load_market(), make_prices(day1 + day2))


def day2_forecast(issued, value, gap: int) -> pl.DataFrame:
    """Forecast frame covering 2026-01-02 00:00 .. 23:30."""
    return quantile_frame(issued, np.full((48, len(LEVELS)), value), LEVELS, gap=gap)


def test_quantile_forecaster_returns_requested_level() -> None:
    issued = at(date(2026, 1, 1), 23)  # = decision time of Jan 2 00:00 (gate closure 2)
    mu = np.arange(48, dtype=float) + 100
    qf = quantile_frame(issued, normal_quantiles(mu, 10.0, LEVELS), LEVELS, gap=2)
    view = two_days_market().view(48)  # now = 2026-01-02 00:00
    out = QuantileForecaster(qf, level=0.5).forecast(view, view.end)
    assert out.columns == [Col.INTERVAL_START, Col.USEP]
    assert out[Col.USEP].to_numpy() == pytest.approx(mu)


def test_quantile_forecaster_never_uses_later_issues() -> None:
    early = day2_forecast(at(date(2026, 1, 1), 22), 1.0, gap=4)
    late = day2_forecast(at(date(2026, 1, 1), 23) + timedelta(minutes=1), 2.0, gap=2)
    view = two_days_market().view(48)  # decided at 23:00: the 23:01 issue is in the future
    out = QuantileForecaster(pl.concat([early, late]), level=0.5).forecast(view, view.end)
    assert out[Col.USEP].unique().to_list() == [1.0]

    before_any_issue = two_days_market().view(45)  # 22:30, decided 21:30: nothing issued yet
    with pytest.raises(ValueError, match="no forecast"):
        QuantileForecaster(early, level=0.5).forecast(before_any_issue, before_any_issue.end)


def test_persistence_uses_latest_known_same_half_hour() -> None:
    """Deciding at 23:00, yesterday's 23:00 and 23:30 are not known yet: use the day before."""
    day0 = [1000.0 + i for i in range(48)]
    day1 = [float(i) for i in range(48)]
    day2 = [1e6] * 48  # unknowable
    market = Market(load_market(), make_prices(day0 + day1 + day2))
    view = market.view(96)  # target day 2 00:00, decided day 1 23:00
    out = PersistenceForecaster().forecast(view, view.end)
    assert out[Col.USEP].to_list() == day1[:46] + day0[46:]


def test_quantile_forecaster_uses_issues_known_at_gate_closure() -> None:
    issued = at(date(2026, 1, 1), 23)  # forecast for Jan 2, issued at its first gate closure
    rows = quantile_frame(issued, np.full((50, len(LEVELS)), 7.0), LEVELS).filter(
        pl.col(Col.LEAD_PERIODS) > 2
    )
    market = two_days_market()
    view = market.view(48)  # target Jan 2 00:00, decided at 23:00 (gate closure 2)
    assert view.decided_at == issued
    out = QuantileForecaster(rows, level=0.5).forecast(view, view.end)
    assert out[Col.USEP].unique().to_list() == [7.0]

    late = load_market().model_copy(update={"gate_closure_intervals": 3})
    early_view = Market(late, market.prices).view(48)  # decided 22:30: forecast not issued yet
    with pytest.raises(ValueError, match="no forecast"):
        QuantileForecaster(rows, level=0.5).forecast(early_view, early_view.end)
