"""Chronos-2 quantile forecasts of USEP with covariates.

For each issue time t: the context is the ``context_days`` of half-hours strictly before t
(target USEP + all selected covariates). The model predicts from t through the end of the delivery
day (``issue_lead_intervals`` gap + ``prediction_length``), with KNOWN covariates over that span
(see ``forecast.covariates``); only the delivery day's ``prediction_length`` intervals are kept.
Output: long frame of quantiles (``Col.ISSUED_AT``, INTERVAL_START, LEAD_PERIODS, QUANTILE,
USEP_FORECAST), non-crossing.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Protocol

import numpy as np
import polars as pl
import structlog

from battery_optimisation.columns import Col
from battery_optimisation.config import Covariate, Device, ForecastConfig
from battery_optimisation.data.nems import PERIODS_PER_DAY
from battery_optimisation.forecast.covariates import COVARIATES, CovariateKind
from battery_optimisation.forecast.quantiles import rearrange

log = structlog.get_logger(__name__)

STEP = timedelta(days=1) / PERIODS_PER_DAY


@dataclass(frozen=True)
class ForecastInput:
    issued_at: datetime
    target: np.ndarray  # context USEP
    past_covariates: dict[str, np.ndarray]  # context values of every selected covariate
    future_covariates: dict[str, np.ndarray]  # horizon values of KNOWN covariates only

    def to_chronos(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "past_covariates": {str(k): v for k, v in self.past_covariates.items()},
            "future_covariates": {str(k): v for k, v in self.future_covariates.items()},
        }


def covariate_columns(cfg: ForecastConfig) -> tuple[list[str], list[str]]:
    """(all selected covariate columns, those known over the horizon)."""
    groups = [COVARIATES[c] for c in cfg.covariates]
    past = [c for g in groups for c in g.columns]
    known = [c for g in groups if g.kind is CovariateKind.KNOWN for c in g.columns]
    return past, known


def _values(frame: pl.DataFrame, col: str) -> np.ndarray:
    return frame[col].cast(pl.Float64).fill_null(float("nan")).to_numpy()


class WeatherSource(Protocol):
    """Horizon weather for an issue (``data.weather.WeatherTrajectories``: the as-of run)."""

    def for_issue(self, issued_at: datetime, stamps: list[datetime]) -> pl.DataFrame: ...


def build_input(
    features: pl.DataFrame,
    issued_at: datetime,
    cfg: ForecastConfig,
    weather: WeatherSource | None,
    target: str = Col.USEP,
) -> ForecastInput:
    """Model input for one issue time, from data strictly before it (plus KNOWN covariates).

    Weather over the horizon comes from ``weather`` (the run usable at the issue time), never
    from ``features``, whose weather columns are the daily as-of values (context only)."""
    n_ctx = cfg.context_days * PERIODS_PER_DAY
    horizon = cfg.issue_lead_intervals + cfg.prediction_length  # issue time -> end of the day
    ts = pl.col(Col.INTERVAL_START)
    context = features.filter(ts < issued_at).sort(Col.INTERVAL_START).tail(n_ctx)
    if context.height < n_ctx or context[0, Col.INTERVAL_START] != issued_at - n_ctx * STEP:
        raise ValueError(f"not enough contiguous history before {issued_at}")
    future = features.filter((ts >= issued_at) & (ts < issued_at + horizon * STEP))
    if future.height != horizon:
        raise ValueError(f"forecast horizon from {issued_at} not covered by the features")
    past, known = covariate_columns(cfg)
    future = future.sort(Col.INTERVAL_START)
    if Covariate.WEATHER in cfg.covariates:
        if weather is None:
            raise ValueError("weather covariates selected but no weather source for the horizon")
        wx = weather.for_issue(issued_at, future[Col.INTERVAL_START].to_list())
        future = future.drop(wx.columns[1:], strict=False).join(wx, on=Col.INTERVAL_START)
    return ForecastInput(
        issued_at=issued_at,
        target=_values(context, target),
        past_covariates={c: _values(context, c) for c in past},
        future_covariates={c: _values(future, c) for c in known},
    )


class QuantilePipeline(Protocol):
    """The part of ``chronos.Chronos2Pipeline`` we use."""

    def predict_quantiles(self, inputs, prediction_length, quantile_levels): ...


class ChronosModel:
    def __init__(self, cfg: ForecastConfig, pipeline: QuantilePipeline | None = None) -> None:
        self.cfg = cfg
        self._pipeline = pipeline

    @property
    def pipeline(self) -> QuantilePipeline:
        if self._pipeline is None:
            cfg = self.cfg
            if cfg.device is Device.CUDA and cfg.cuda_visible_devices is not None:
                # must happen before CUDA is first initialised in this process
                os.environ.setdefault("CUDA_VISIBLE_DEVICES", cfg.cuda_visible_devices)
            from chronos import Chronos2Pipeline

            self._pipeline = Chronos2Pipeline.from_pretrained(cfg.model_id, device_map=cfg.device)
            log.info(
                "chronos.loaded",
                model=cfg.model_id,
                device=str(cfg.device),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            )
        return self._pipeline

    def forecast_quantiles(
        self,
        features: pl.DataFrame,
        issues: list[datetime],
        weather: WeatherSource | None = None,
        target: str = Col.USEP,
    ) -> pl.DataFrame:
        """Quantile forecasts of ``target`` (USEP, or an ancillary price column)."""
        cfg = self.cfg
        levels = np.asarray(cfg.quantile_levels)
        frames = []
        t0 = time.perf_counter()
        for b in range(0, len(issues), cfg.batch_size):
            batch = [
                build_input(features, t, cfg, weather, target)
                for t in issues[b : b + cfg.batch_size]
            ]
            gap = cfg.issue_lead_intervals
            quantiles, _ = self.pipeline.predict_quantiles(
                [inp.to_chronos() for inp in batch],
                prediction_length=gap + cfg.prediction_length,
                quantile_levels=list(cfg.quantile_levels),
            )
            for inp, q in zip(batch, quantiles, strict=True):
                values = q[0].detach().cpu().numpy().astype(float)[gap:]  # delivery day only
                frames.append(_long_frame(inp.issued_at, rearrange(values), levels, gap))
            log.info("chronos.batch", done=min(b + cfg.batch_size, len(issues)), total=len(issues))
        log.info("chronos.done", issues=len(issues), seconds=round(time.perf_counter() - t0, 1))
        return pl.concat(frames)


def _long_frame(
    issued_at: datetime, values: np.ndarray, levels: np.ndarray, gap: int
) -> pl.DataFrame:
    """Rows for intervals issue + gap .. issue + gap + horizon - 1 (lead = gap + 1 ..)."""
    horizon, n_q = values.shape
    lead = np.repeat(np.arange(gap + 1, gap + horizon + 1, dtype=np.int32), n_q)
    return pl.DataFrame(
        {
            Col.LEAD_PERIODS: lead,
            Col.QUANTILE: np.tile(levels, horizon),
            Col.USEP_FORECAST: values.reshape(-1),
        }
    ).select(
        pl.lit(issued_at).dt.cast_time_unit("us").alias(Col.ISSUED_AT),
        (pl.lit(issued_at).dt.cast_time_unit("us") + (pl.col(Col.LEAD_PERIODS) - 1) * STEP).alias(
            Col.INTERVAL_START
        ),
        Col.LEAD_PERIODS,
        Col.QUANTILE,
        Col.USEP_FORECAST,
    )
