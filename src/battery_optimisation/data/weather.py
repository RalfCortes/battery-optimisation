"""As-of weather forecasts on a grid over Singapore (Open-Meteo single-runs API).

Settings: ``ForecastConfig.weather`` (``config/forecast.yaml``, section ``weather``).

As-of rule: at time T the usable forecast is the latest run r (nominal start, UTC, one of
``run_hours_utc``) with r + ``availability_delay_h`` <= T. For each delivery day D, the weather
features of D's intervals come from the run usable at D's forecast issue time (``asof_run``),
so context and horizon covariates both mean "the forecast we had when we bid". Nothing published
after the issue time is used. Each value carries its lead time (``Col.WEATHER_LEAD_H``).

One fixed provider for the whole history: Open-Meteo's "best match" blend switches models over
time. Global models are natively hourly; we keep Open-Meteo's 15-minute series
(sun-position-aware radiation) and aggregate to our 30-minute intervals.

Processing (``to_features``), each step recorded in a ``WeatherProcessingReport``:
  raw        15-min values per run and grid point, stamped at the END of their step
  regrid     complete 15-min grid per run and point (missing stamps inserted as nulls)
  gap_fill   per config: linear in time, only runs of <= max_gap_steps missing steps
  aggregate  interval [t, t+30) = steps stamped t+15 and t+30 (mean or sum per variable)
  asof       per day, the run usable at the day's issue time; days without it stay empty
  pivot      one column per (variable, point): ``feature_name``; no spatial averaging

Storage: ``data/raw/weather_runs/<model>/<YYYY-MM>/<YYYYMMDDTHH>.parquet`` (one file per run).
Command line: ``uv run weather download`` / ``uv run weather process``.
"""

from __future__ import annotations

import bisect
import json
import time
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import requests
import structlog
from pydantic import BaseModel

from battery_optimisation.columns import Col
from battery_optimisation.config import (
    Aggregation,
    ForecastConfig,
    GapFill,
    WeatherConfig,
    WeatherVar,
)
from battery_optimisation.data.nems import TZ
from battery_optimisation.timing import issue_time

__all__ = ["WeatherVar", "asof_run", "feature_name", "feature_names", "to_features"]

log = structlog.get_logger(__name__)

SINGLE_RUNS_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
FREQ = "minutely_15"
STEP = timedelta(minutes=15)
INTERVAL = timedelta(minutes=30)
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = ROOT / "data" / "raw" / "weather_runs"
DEFAULT_PROCESSED_DIR = ROOT / "data" / "processed" / "weather"
_SGT = ZoneInfo(TZ)

Point = tuple[float, float]  # (latitude, longitude)

COLUMN: dict[WeatherVar, Col] = {
    WeatherVar.TEMPERATURE: Col.TEMPERATURE_C,
    WeatherVar.HUMIDITY: Col.HUMIDITY_PCT,
    WeatherVar.CLOUD_COVER: Col.CLOUD_COVER_PCT,
    WeatherVar.IRRADIANCE: Col.IRRADIANCE_WM2,
    WeatherVar.PRECIPITATION: Col.PRECIPITATION_MM,
}


class ProcessedFiles(StrEnum):
    FEATURES = "features.parquet"
    REPORT = "report.json"


def feature_name(var: WeatherVar, point: Point) -> str:
    """Column of one variable at one grid point, e.g. ``temperature_c@1.250,103.650``."""
    return f"{COLUMN[var]}@{point[0]:.3f},{point[1]:.3f}"


def feature_names(cfg: WeatherConfig) -> list[str]:
    return [feature_name(v, p) for v in cfg.variables for p in cfg.points]


# --------------------------------------------------------------------------- as-of rule


def asof_run(t: datetime, cfg: WeatherConfig) -> datetime:
    """Latest run (UTC) with run + availability_delay_h <= t."""
    cutoff = t.astimezone(UTC) - timedelta(hours=cfg.availability_delay_h)
    day = cutoff.date()
    while True:
        for hour in sorted(cfg.run_hours_utc, reverse=True):
            run = datetime(day.year, day.month, day.day, hour, tzinfo=UTC)
            if run <= cutoff:
                return run
        day -= timedelta(days=1)


def needed_runs(start: date, end: date, fcfg: ForecastConfig) -> list[datetime]:
    """Runs the daily features of [start, end] are taken from (one per day)."""
    days = [start + timedelta(days=d) for d in range((end - start).days + 1)]
    return sorted({asof_run(issue_time(d, fcfg), fcfg.weather) for d in days})


def all_runs(start: date, end: date, cfg: WeatherConfig) -> list[datetime]:
    """Every run of the configured run hours from ``start`` to ``end`` (UTC days)."""
    days = [start + timedelta(days=d) for d in range((end - start).days + 1)]
    return [
        datetime(d.year, d.month, d.day, h, tzinfo=UTC)
        for d in days
        for h in sorted(cfg.run_hours_utc)
    ]


# --------------------------------------------------------------------------- report


class Transform(StrEnum):
    RAW = "raw"
    REGRID = "regrid"
    GAP_FILL = "gap_fill"
    AGGREGATE = "aggregate"
    ASOF = "asof"
    PIVOT = "pivot"


class TransformStep(BaseModel):
    name: Transform
    detail: str
    rows_in: int
    rows_out: int
    timestamps_inserted: int = 0
    values_filled: dict[WeatherVar, int] = {}
    values_missing_after: dict[WeatherVar, int] = {}
    intervals_dropped: int = 0
    aggregation: dict[WeatherVar, Aggregation] = {}
    runs_used: list[datetime] = []
    runs_dropped: list[datetime] = []  # came back (mostly) empty: treated as missing
    days_without_run: list[date] = []
    days_on_older_run: int = 0  # the as-of run was missing: an older usable run was used
    lead_h_min: float | None = None
    lead_h_max: float | None = None


class WeatherProcessingReport(BaseModel):
    created_at: datetime
    config: WeatherConfig
    start: date
    end: date
    steps: list[TransformStep] = []

    def step(self, name: Transform | str) -> TransformStep:
        return next(s for s in self.steps if s.name == name)

    def add(self, step: TransformStep) -> None:
        self.steps.append(step)
        extra: dict = {}
        if step.timestamps_inserted:
            extra["timestamps_inserted"] = step.timestamps_inserted
        if step.values_filled:
            extra["values_filled"] = {str(k): v for k, v in step.values_filled.items()}
        if step.intervals_dropped:
            extra["intervals_dropped"] = step.intervals_dropped
        if step.runs_used:
            extra["runs_used"] = len(step.runs_used)
        if step.days_without_run:
            extra["days_without_run"] = [str(d) for d in step.days_without_run]
        if step.lead_h_min is not None:
            extra["lead_h"] = f"{step.lead_h_min:g}-{step.lead_h_max:g}"
        missing = {str(k): v for k, v in step.values_missing_after.items() if v}
        if missing:
            extra["still_missing"] = missing
        log.info(
            "weather.transform",
            step=str(step.name),
            detail=step.detail,
            rows_in=step.rows_in,
            rows_out=step.rows_out,
            **extra,
        )


# --------------------------------------------------------------------------- download / parse


def request_params(run: datetime, cfg: WeatherConfig, model: str | None = None) -> dict[str, str]:
    return {
        "latitude": ",".join(str(p[0]) for p in cfg.points),
        "longitude": ",".join(str(p[1]) for p in cfg.points),
        FREQ: ",".join(str(v) for v in cfg.variables),
        "models": model or cfg.model,
        "run": run.astimezone(UTC).strftime("%Y-%m-%dT%H:%M"),
        "forecast_days": str(cfg.forecast_days),
        "timezone": TZ,
    }


def run_path(run: datetime, model: str, raw_dir: Path = DEFAULT_RAW_DIR) -> Path:
    run = run.astimezone(UTC)
    return Path(raw_dir) / model / f"{run:%Y-%m}" / f"{run:%Y%m%dT%H}.parquet"


def parse_run(text: str, run: datetime, points: list[Point], cfg: WeatherConfig) -> pl.DataFrame:
    """Long 15-min frame of one run. Locations come back in request order and are labelled with
    the requested point (Open-Meteo reports the model cell centre instead)."""
    payload = json.loads(text)
    locations = payload if isinstance(payload, list) else [payload]
    if len(locations) != len(points):
        raise ValueError(f"expected {len(points)} locations, got {len(locations)}")
    frames = []
    for (lat, lon), loc in zip(points, locations, strict=True):
        series = loc[FREQ]
        frames.append(
            pl.DataFrame(
                {
                    Col.INTERVAL_START: series["time"],
                    **{COLUMN[v]: series[str(v)] for v in cfg.variables},
                },
                schema_overrides={COLUMN[v]: pl.Float64 for v in cfg.variables},
            ).with_columns(pl.lit(lat).alias(Col.LATITUDE), pl.lit(lon).alias(Col.LONGITUDE))
        )
    return pl.concat(frames).select(
        pl.lit(run.astimezone(UTC)).dt.cast_time_unit("us").alias(Col.RUN_TIME),
        pl.col(Col.INTERVAL_START)
        .str.to_datetime("%Y-%m-%dT%H:%M", time_unit="us")
        .dt.replace_time_zone(TZ),
        Col.LATITUDE,
        Col.LONGITUDE,
        *[COLUMN[v] for v in cfg.variables],
    )


MIN_RUN_COMPLETENESS = 0.5  # a run with less of every variable is treated as missing
# (one missing variable stays NaN: Chronos-2 handles missing covariate values)
RATE_LIMITED = 429
BACKOFF_S = (60.0, 300.0, 900.0, 3600.0)  # waits after successive 429 answers


SERVER_RETRIES_S = (10.0, 60.0)  # waits after 5xx answers before giving up on a run


def _get_with_backoff(session: requests.Session, params: dict[str, str]) -> requests.Response:
    """GET, waiting longer after each 429 (Open-Meteo limits per minute, hour and day) and
    retrying a few times on 5xx."""
    server_waits = list(SERVER_RETRIES_S)
    for wait in (*BACKOFF_S, None):
        resp = session.get(SINGLE_RUNS_URL, params=params, timeout=180)
        while resp.status_code >= 500 and server_waits:
            time.sleep(server_waits.pop(0))
            resp = session.get(SINGLE_RUNS_URL, params=params, timeout=180)
        if resp.status_code != RATE_LIMITED or wait is None:
            return resp
        log.warning("weather.rate_limited", run=params.get("run"), wait_s=wait)
        time.sleep(wait)
    return resp


def download_runs(
    runs: list[datetime],
    cfg: WeatherConfig,
    model: str,
    raw_dir: Path = DEFAULT_RAW_DIR,
    overwrite: bool = False,
    pause_s: float = 1.2,
) -> list[Path]:
    """One file per run; existing files are skipped (resumable). Unavailable runs are logged.

    ``pause_s`` keeps a 9-point request (counted as 9 calls) under 600 calls per minute."""
    written = []
    with requests.Session() as session:
        for run in runs:
            path = run_path(run, model, raw_dir)
            if path.exists() and not overwrite:
                continue
            resp = _get_with_backoff(session, request_params(run, cfg, model))
            if resp.status_code == 400 and "not available" in resp.text:
                log.warning("weather.run_unavailable", model=model, run=run.isoformat())
                continue
            if resp.status_code >= 500:  # still failing after retries: skip, a later run retries
                log.error(
                    "weather.run_failed", model=model, run=run.isoformat(), status=resp.status_code
                )
                continue
            resp.raise_for_status()
            df = parse_run(resp.text, run, cfg.points, cfg)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(path)
            written.append(path)
            log.info("weather.run_downloaded", model=model, run=run.isoformat(), rows=df.height)
            time.sleep(pause_s)
    return written


def load_runs(model: str, runs: list[datetime], raw_dir: Path = DEFAULT_RAW_DIR) -> pl.DataFrame:
    """Raw frames of the given runs that exist on disk (missing ones are reported later)."""
    files = [run_path(r, model, raw_dir) for r in runs]
    present = [f for f in files if f.exists()]
    if not present:
        raise FileNotFoundError(
            f"no {model} runs under {Path(raw_dir) / model}; run `uv run weather download`"
        )
    return pl.concat([pl.read_parquet(f) for f in present])


# --------------------------------------------------------------------------- processing


def _null_counts(df: pl.DataFrame, cfg: WeatherConfig) -> dict[WeatherVar, int]:
    return {v: int(df[COLUMN[v]].null_count()) for v in cfg.variables}


def _fill_short_gaps(s: pl.Series, max_gap: int) -> pl.Series:
    """Linear interpolation of interior null runs of length <= max_gap (longer runs stay null)."""
    runs = (
        s.to_frame()
        .with_columns(pl.col(s.name).is_null().alias("_missing"))
        .with_columns(
            (pl.col("_missing") != pl.col("_missing").shift(1, fill_value=False))
            .cum_sum()
            .alias("_run")
        )
        .with_columns(pl.len().over("_run").alias("_len"))
    )
    fillable = runs["_missing"] & (runs["_len"] <= max_gap)
    return runs.select(
        pl.when(fillable).then(pl.col(s.name).interpolate()).otherwise(pl.col(s.name)).alias(s.name)
    )[s.name]


def _process_runs(
    raw: pl.DataFrame, fcfg: ForecastConfig, start: date, end: date
) -> tuple[pl.DataFrame, WeatherProcessingReport]:
    """Steps raw -> regrid -> gap_fill -> aggregate: long 30-min values per run and point."""
    cfg = fcfg.weather
    keys = [Col.RUN_TIME, Col.LATITUDE, Col.LONGITUDE]
    cols = [COLUMN[v] for v in cfg.variables]
    report = WeatherProcessingReport(
        created_at=datetime.now(_SGT), config=cfg, start=start, end=end
    )
    raw = raw.sort(*keys, Col.INTERVAL_START)
    report.add(
        TransformStep(
            name=Transform.RAW,
            detail=f"{cfg.model}: {raw[Col.RUN_TIME].n_unique()} runs, "
            f"{raw.select(Col.LATITUDE, Col.LONGITUDE).n_unique()} points, 15-min",
            rows_in=raw.height,
            rows_out=raw.height,
            values_missing_after=_null_counts(raw, cfg),
        )
    )

    span = raw.group_by(keys).agg(
        pl.col(Col.INTERVAL_START).min().alias("_t0"), pl.col(Col.INTERVAL_START).max().alias("_t1")
    )
    grid = (
        span.with_columns(
            pl.datetime_ranges("_t0", "_t1", interval="15m").alias(Col.INTERVAL_START)
        )
        .explode(Col.INTERVAL_START)
        .drop("_t0", "_t1")
    )
    regridded = grid.join(raw, on=[*keys, Col.INTERVAL_START], how="left").sort(
        *keys, Col.INTERVAL_START
    )
    report.add(
        TransformStep(
            name=Transform.REGRID,
            detail="complete 15-min grid per run and point; missing stamps inserted as nulls",
            rows_in=raw.height,
            rows_out=regridded.height,
            timestamps_inserted=regridded.height - raw.height,
            values_missing_after=_null_counts(regridded, cfg),
        )
    )

    before = _null_counts(regridded, cfg)
    filled = regridded
    if cfg.gap_fill.method is GapFill.LINEAR:
        max_gap = cfg.gap_fill.max_gap_steps
        filled = regridded.with_columns(
            pl.col(c).map_batches(lambda s: _fill_short_gaps(s, max_gap)).over(keys) for c in cols
        )
    after = _null_counts(filled, cfg)
    report.add(
        TransformStep(
            name=Transform.GAP_FILL,
            detail=f"{cfg.gap_fill.method}, gaps of <= {cfg.gap_fill.max_gap_steps} 15-min steps",
            rows_in=regridded.height,
            rows_out=filled.height,
            values_filled={v: before[v] - after[v] for v in cfg.variables if before[v] > after[v]},
            values_missing_after=after,
        )
    )

    sums = [COLUMN[v] for v, a in cfg.variables.items() if a is Aggregation.SUM]
    agg = [
        pl.col(COLUMN[v]).sum() if a is Aggregation.SUM else pl.col(COLUMN[v]).mean()
        for v, a in cfg.variables.items()
    ]
    aggregated = (
        filled.with_columns(pl.col(Col.INTERVAL_START) - STEP)  # stamped at step end -> step start
        .sort(*keys, Col.INTERVAL_START)
        .group_by_dynamic(Col.INTERVAL_START, every="30m", group_by=keys)
        .agg(
            *agg,
            pl.len().alias("_steps"),
            *[pl.col(c).is_null().all().alias(f"_none_{c}") for c in sums],
        )
    )
    complete = (
        aggregated.filter(pl.col("_steps") == 2)
        # a sum over only-null steps is 0 in polars: keep it missing instead
        .with_columns(
            pl.when(pl.col(f"_none_{c}")).then(None).otherwise(pl.col(c)).alias(c) for c in sums
        )
        .drop("_steps", *[f"_none_{c}" for c in sums])
    )
    report.add(
        TransformStep(
            name=Transform.AGGREGATE,
            detail="interval [t, t+30) = steps stamped t+15 and t+30; incomplete intervals dropped",
            rows_in=filled.height,
            rows_out=complete.height,
            intervals_dropped=aggregated.height - complete.height,
            aggregation=dict(cfg.variables),
            values_missing_after=_null_counts(complete, cfg),
        )
    )
    share = complete.group_by(Col.RUN_TIME).agg(
        pl.max_horizontal(pl.col(c).is_not_null().mean() for c in cols).alias("_ok")
    )
    blank = share.filter(pl.col("_ok") < MIN_RUN_COMPLETENESS)[Col.RUN_TIME].sort().to_list()
    if blank:
        complete = complete.filter(~pl.col(Col.RUN_TIME).is_in(blank))
        report.steps[-1] = report.steps[-1].model_copy(update={"runs_dropped": blank})
        log.warning("weather.blank_runs", runs=[str(r) for r in blank])
    return complete, report


def to_features(
    raw: pl.DataFrame, fcfg: ForecastConfig, start: date, end: date
) -> tuple[pl.DataFrame, WeatherProcessingReport]:
    """Raw run trajectories -> as-of 30-min features for the days ``start``..``end`` (each day
    from the run usable at its daily forecast issue time). Context covariates."""
    cfg = fcfg.weather
    cols = [COLUMN[v] for v in cfg.variables]
    complete, report = _process_runs(raw, fcfg, start, end)
    days = [start + timedelta(days=d) for d in range((end - start).days + 1)]
    available = sorted(complete[Col.RUN_TIME].unique().to_list())
    chosen, older = [], 0
    for d in days:
        target = asof_run(issue_time(d, fcfg), cfg)
        k = bisect.bisect_right(available, target)
        chosen.append(available[k - 1] if k else None)
        older += bool(k) and available[k - 1] != target
    choice = pl.DataFrame(
        {"_day": days, Col.RUN_TIME: chosen},
        schema={"_day": pl.Date, Col.RUN_TIME: pl.Datetime("us", "UTC")},
    ).drop_nulls()
    selected = (
        complete.with_columns(pl.col(Col.INTERVAL_START).dt.date().alias("_day"))
        .join(choice, on=["_day", Col.RUN_TIME], how="inner")
        .with_columns(
            (
                (
                    pl.col(Col.INTERVAL_START) - pl.col(Col.RUN_TIME).dt.convert_time_zone(TZ)
                ).dt.total_minutes()
                / 60
            ).alias(Col.WEATHER_LEAD_H)
        )
    )
    used = selected[Col.RUN_TIME].unique().sort().to_list()
    missing_days = sorted(set(days) - set(selected["_day"].unique().to_list()))
    report.add(
        TransformStep(
            name=Transform.ASOF,
            detail=f"day D <- latest run usable at D's forecast issue time "
            f"(D 00:00 - {fcfg.issue_lead_intervals} intervals; run + "
            f"{cfg.availability_delay_h:g} h <= issue time)",
            rows_in=complete.height,
            rows_out=selected.height,
            runs_used=used,
            days_without_run=missing_days,
            days_on_older_run=older,
            lead_h_min=selected[Col.WEATHER_LEAD_H].min() if selected.height else None,
            lead_h_max=selected[Col.WEATHER_LEAD_H].max() if selected.height else None,
            values_missing_after=_null_counts(selected, cfg),
        )
    )

    grid30 = pl.datetime_range(
        datetime.combine(start, datetime.min.time(), _SGT),
        datetime.combine(end + timedelta(days=1), datetime.min.time(), _SGT),
        INTERVAL,
        closed="left",
        time_unit="us",
        time_zone=TZ,
        eager=True,
    ).alias(Col.INTERVAL_START)
    wide = grid30.to_frame().join(
        selected.group_by(Col.INTERVAL_START).agg(pl.col(Col.WEATHER_LEAD_H).first()),
        on=Col.INTERVAL_START,
        how="left",
    )
    for (lat, lon), part in selected.group_by(Col.LATITUDE, Col.LONGITUDE, maintain_order=True):
        part = part.select(Col.INTERVAL_START, *cols).rename(
            {COLUMN[v]: feature_name(v, (lat, lon)) for v in cfg.variables}
        )
        wide = wide.join(part, on=Col.INTERVAL_START, how="left")
    for name in feature_names(cfg):  # points absent from the data still get their column
        if name not in wide.columns:
            wide = wide.with_columns(pl.lit(None, pl.Float64).alias(name))
    wide = wide.select(Col.INTERVAL_START, Col.WEATHER_LEAD_H, *feature_names(cfg)).sort(
        Col.INTERVAL_START
    )
    report.add(
        TransformStep(
            name=Transform.PIVOT,
            detail=f"one column per (variable, point): {len(feature_names(cfg))} features, "
            "no spatial averaging",
            rows_in=selected.height,
            rows_out=wide.height,
        )
    )
    return wide, report


def process(
    fcfg: ForecastConfig,
    start: date,
    end: date,
    raw_dir: Path = DEFAULT_RAW_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> tuple[pl.DataFrame, WeatherProcessingReport]:
    """Build as-of features for [start, end] and save features + report (for inspection; the
    forecast pipeline runs ``to_features`` in memory)."""
    raw = load_runs(fcfg.weather.model, needed_runs(start, end, fcfg), raw_dir)
    features, report = to_features(raw, fcfg, start, end)
    out = Path(processed_dir)
    out.mkdir(parents=True, exist_ok=True)
    features.write_parquet(out / ProcessedFiles.FEATURES)
    (out / ProcessedFiles.REPORT).write_text(report.model_dump_json(indent=2))
    return features, report


def load_weather(processed_dir: Path = DEFAULT_PROCESSED_DIR) -> pl.DataFrame:
    path = Path(processed_dir) / ProcessedFiles.FEATURES
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `uv run weather process`")
    return pl.read_parquet(path)


class WeatherTrajectories:
    """Each run's processed 30-min trajectory (wide, one column per variable and point), to give
    a forecast issued at T the weather of the run usable at T (``asof_run``): horizon covariates
    for intraday issues."""

    def __init__(self, per_run: dict[datetime, pl.DataFrame], cfg: WeatherConfig) -> None:
        self._per_run, self.cfg = per_run, cfg
        self._sorted = sorted(per_run)
        self.fallbacks = 0  # issues served by an older run because the as-of one is missing

    @classmethod
    def from_raw(cls, raw: pl.DataFrame, fcfg: ForecastConfig) -> WeatherTrajectories:
        cfg = fcfg.weather
        stamps = raw[Col.INTERVAL_START]
        complete, _ = _process_runs(raw, fcfg, stamps.min().date(), stamps.max().date())
        per_run: dict[datetime, pl.DataFrame] = {}
        for (run,), part in complete.group_by(Col.RUN_TIME, maintain_order=True):
            wide = None
            for (lat, lon), point in part.group_by(
                Col.LATITUDE, Col.LONGITUDE, maintain_order=True
            ):
                point = point.select(
                    Col.INTERVAL_START, *[COLUMN[v] for v in cfg.variables]
                ).rename({COLUMN[v]: feature_name(v, (lat, lon)) for v in cfg.variables})
                wide = (
                    point
                    if wide is None
                    else wide.join(point, on=Col.INTERVAL_START, how="full", coalesce=True)
                )
            per_run[run] = wide.sort(Col.INTERVAL_START)
        return cls(per_run, cfg)

    @property
    def runs(self) -> list[datetime]:
        return sorted(self._per_run)

    def for_issue(self, issued_at: datetime, stamps: list[datetime]) -> pl.DataFrame:
        """Values of the run usable at ``issued_at`` for ``stamps`` (``Col.INTERVAL_START`` +
        feature columns). Raises if that run is missing or does not cover the stamps."""
        run = asof_run(issued_at, self.cfg)
        # latest downloaded run that was usable at the issue time (normally the as-of run)
        i = bisect.bisect_right(self._sorted, run)
        if i == 0:
            raise ValueError(f"no weather run usable at {issued_at} (as-of run {run})")
        key = self._sorted[i - 1]
        if key != run:
            self.fallbacks += 1
        rows = (
            pl.DataFrame({Col.INTERVAL_START: stamps})
            .with_columns(
                pl.col(Col.INTERVAL_START).dt.cast_time_unit("us").dt.convert_time_zone(TZ)
            )
            .join(self._per_run[key], on=Col.INTERVAL_START, how="left")
        )
        if rows.height != len(stamps) or rows.select(
            pl.exclude(Col.INTERVAL_START)
        ).null_count().sum_horizontal().item() == rows.height * (rows.width - 1):
            raise ValueError(f"weather run {run} does not cover {stamps[0]}..{stamps[-1]}")
        for name in feature_names(self.cfg):
            if name not in rows.columns:
                rows = rows.with_columns(pl.lit(None, pl.Float64).alias(name))
        return rows.select(Col.INTERVAL_START, *feature_names(self.cfg))
