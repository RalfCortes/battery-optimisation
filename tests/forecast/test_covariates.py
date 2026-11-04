from datetime import date

import polars as pl
import pytest
from synthetic import make_prices

from battery_optimisation.columns import Col
from battery_optimisation.config import Covariate
from battery_optimisation.forecast.covariates import (
    COVARIATES,
    CovariateKind,
    add_known_covariates,
)


def test_calendar_marks_weekends_and_singapore_holidays() -> None:
    # 2026-01-01 Thu = New Year's Day, 2026-01-03 Sat, 2026-01-05 Mon
    days = {date(2026, 1, 1): 1, date(2026, 1, 3): 1, date(2026, 1, 5): 0}
    for day, expected in days.items():
        frame = add_known_covariates(make_prices([100.0] * 48, day))
        assert frame[Col.IS_OFF_DAY].unique().to_list() == [expected]


def test_time_of_day_is_cyclic() -> None:
    frame = add_known_covariates(make_prices([100.0] * 48))
    row = {
        r[Col.INTERVAL_START].hour * 2 + r[Col.INTERVAL_START].minute // 30: r
        for r in frame.to_dicts()
    }
    assert row[0][Col.HOUR_COS] == pytest.approx(1.0)  # 00:00
    assert row[12][Col.HOUR_SIN] == pytest.approx(1.0)  # 06:00
    assert row[24][Col.HOUR_COS] == pytest.approx(-1.0)  # 12:00


def test_regime_flag_switches_on_with_the_price_cap() -> None:
    before = add_known_covariates(make_prices([100.0] * 48, date(2023, 6, 30)))
    after = add_known_covariates(make_prices([100.0] * 48, date(2023, 7, 1)))
    assert before[Col.TPC_REGIME].unique().to_list() == [0.0]
    assert after[Col.TPC_REGIME].unique().to_list() == [1.0]


def test_covariate_kinds() -> None:
    # No archived demand forecasts exist: demand may only be used as a past covariate.
    assert COVARIATES[Covariate.DEMAND].kind is CovariateKind.PAST
    for group in (Covariate.CALENDAR, Covariate.REGIME, Covariate.WEATHER):
        assert COVARIATES[group].kind is CovariateKind.KNOWN
    assert set(COVARIATES) == set(Covariate)


def test_known_covariates_do_not_touch_prices() -> None:
    prices = make_prices([float(i) for i in range(48)])
    frame = add_known_covariates(prices)
    assert frame[Col.USEP].to_list() == prices[Col.USEP].to_list()
    assert frame.select(pl.all().is_null().sum()).row(0) == tuple([0] * frame.width)
