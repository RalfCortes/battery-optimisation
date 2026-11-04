from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
import torch
from synthetic_forecast import at, make_features

from battery_optimisation.columns import Col
from battery_optimisation.config import Covariate, load_forecast
from battery_optimisation.data.weather import feature_names
from battery_optimisation.forecast.chronos import ChronosModel, build_input
from battery_optimisation.forecast.covariates import COVARIATES, CovariateKind

POISON = 1e9


@pytest.fixture
def cfg():
    return load_forecast().model_copy(update={"context_days": 3})


def poisoned(features: pl.DataFrame, issued_at) -> pl.DataFrame:
    """Overwrite everything that is not known at issue time with an obvious marker."""
    future = pl.col(Col.INTERVAL_START) >= issued_at
    unknown = [Col.USEP, Col.DEMAND_MW]
    return features.with_columns(
        pl.when(future).then(POISON).otherwise(pl.col(c)).alias(c) for c in unknown
    )


class FrameWeather:
    """Stand-in weather source: values from a frame (the as-of run in production)."""

    def __init__(self, frame: pl.DataFrame, offset: float = 0.0) -> None:
        self.frame, self.offset = frame, offset
        self.calls: list = []

    def for_issue(self, issued_at, stamps):
        self.calls.append(issued_at)
        cols = feature_names(load_forecast().weather)
        rows = self.frame.filter(pl.col(Col.INTERVAL_START).is_in(list(stamps)))
        return rows.select(Col.INTERVAL_START, *[pl.col(c) + self.offset for c in cols])


def test_input_uses_only_information_before_issue(cfg) -> None:
    issued = at(date(2026, 1, 5))
    clean = make_features(7)
    features = poisoned(clean, issued)
    wx_cols = feature_names(load_forecast().weather)
    features = features.with_columns(  # future weather in the frame is NOT the as-of forecast
        pl.when(pl.col(Col.INTERVAL_START) >= issued).then(POISON).otherwise(pl.col(c)).alias(c)
        for c in wx_cols
    )
    source = FrameWeather(clean, offset=0.5)
    inp = build_input(features, issued, cfg, source)
    assert source.calls == [issued]
    assert np.all(
        inp.future_covariates[wx_cols[0]]
        == clean.filter(pl.col(Col.INTERVAL_START) >= issued)[wx_cols[0]]
        .head(len(inp.future_covariates[wx_cols[0]]))
        .to_numpy()
        + 0.5
    )

    assert inp.issued_at == issued
    assert len(inp.target) == cfg.context_days * 48
    assert POISON not in inp.target
    for values in inp.past_covariates.values():
        assert len(values) == len(inp.target) and POISON not in values
    known = {
        c
        for g in cfg.covariates
        if COVARIATES[g].kind is CovariateKind.KNOWN
        for c in COVARIATES[g].columns
    }
    assert set(inp.future_covariates) == known  # demand never in the future
    assert Col.DEMAND_MW in inp.past_covariates
    for values in inp.future_covariates.values():  # from the issue time to the end of the day
        assert len(values) == cfg.issue_lead_intervals + cfg.prediction_length


def test_input_respects_covariate_selection(cfg) -> None:
    issued = at(date(2026, 1, 5))
    only_calendar = cfg.model_copy(update={"covariates": [Covariate.CALENDAR]})
    inp = build_input(make_features(7), issued, only_calendar, None)
    assert set(inp.past_covariates) == set(COVARIATES[Covariate.CALENDAR].columns)


def test_input_needs_full_history_and_horizon(cfg) -> None:
    features = make_features(7)
    source = FrameWeather(features)
    with pytest.raises(ValueError, match="history"):
        build_input(features, at(date(2026, 1, 2)), cfg, source)  # only 1 day of context
    with pytest.raises(ValueError, match="horizon"):
        build_input(features, at(date(2026, 1, 7), 12), cfg, source)  # runs past the data
    with pytest.raises(ValueError, match="weather"):
        build_input(features, at(date(2026, 1, 5)), cfg, None)  # weather selected, no source


class FakePipeline:
    """Mimics Chronos2Pipeline.predict_quantiles: per input a tensor [1, horizon, Q]."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def predict_quantiles(self, inputs, prediction_length, quantile_levels):
        self.calls.append(len(inputs))
        out = []
        for inp in inputs:
            last = float(inp["target"][-1])
            base = np.full((prediction_length, 1), last) + np.arange(len(quantile_levels))[None, :]
            out.append(torch.tensor(base[None], dtype=torch.float32))
        return out, [o[:, :, 0] for o in out]


def test_forecast_frame_layout_and_batching(cfg) -> None:
    features = make_features(10)
    issues = [at(date(2026, 1, d)) for d in (5, 6, 7)]
    fake = FakePipeline()
    model = ChronosModel(cfg.model_copy(update={"batch_size": 2}), pipeline=fake)
    qf = model.forecast_quantiles(features, issues, FrameWeather(features))

    assert fake.calls == [2, 1]
    q = len(cfg.quantile_levels)
    assert qf.height == len(issues) * 48 * q
    assert set(qf.columns) == {
        Col.ISSUED_AT,
        Col.INTERVAL_START,
        Col.LEAD_PERIODS,
        Col.QUANTILE,
        Col.USEP_FORECAST,
    }
    gap = cfg.issue_lead_intervals
    first = qf.filter(pl.col(Col.ISSUED_AT) == issues[0])
    # only the delivery day is kept: from issue + gap, 48 intervals
    assert first[Col.INTERVAL_START].min() == issues[0] + timedelta(minutes=30 * gap)
    assert first[Col.INTERVAL_START].max() == issues[0] + timedelta(minutes=30 * (gap + 47))
    assert first[Col.LEAD_PERIODS].min() == gap + 1
    assert first[Col.LEAD_PERIODS].max() == gap + 48
    # quantiles non-decreasing in level for every interval
    crossing = (
        qf.sort(Col.ISSUED_AT, Col.INTERVAL_START, Col.QUANTILE)
        .group_by(Col.ISSUED_AT, Col.INTERVAL_START)
        .agg((pl.col(Col.USEP_FORECAST).diff() < 0).any().alias("x"))
    )
    assert not crossing["x"].any()
