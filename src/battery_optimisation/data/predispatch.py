"""EMC pre-dispatch snapshots: archive the rolling real-time/72-period files, one per run.

EMC publishes a rolling schedule on https://www.nems.emcsg.com/nems-prices, refreshed every
30 minutes about 1-2 minutes after the period starts (the "Last Updated" stamp). Only the latest
run is available, so past forecasts cannot be downloaded later and must be archived as they appear.
The same DataDownload endpoint as the price history serves it (form ``value`` ids):

  Feed.ENERGY:    10 = View.RT48 (today's 48 periods), 12 = View.RT72 (72-period window)
  Feed.ANCILLARY: 11 = View.RT48,                      13 = View.RT72

Each file mixes realised intervals (up to the current period) with pre-dispatch forecasts for the
rest. Both views are fetched and merged on ``Col.INTERVAL_START``.

Columns added:
  Col.PUBLISHED_AT  "Last Updated" stamp of the pre-dispatch run (Asia/Singapore)
  Col.FETCHED_AT    when we fetched it (Asia/Singapore)
  Col.LEAD_PERIODS  (interval_start - start of the period containing published_at) / 30 min;
                    <= 0 realised/current dispatch, >= 1 pre-dispatch forecast

Storage: ``data/raw/predispatch/<feed>/<YYYY-MM-DD>/<published_at:%Y%m%dT%H%M%S>.parquet``.
A run already on disk is not rewritten, so overlapping or repeated cron runs are harmless.
"""

from __future__ import annotations

import io
import os
from collections.abc import Iterable
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import requests
import structlog

from battery_optimisation.columns import Col
from battery_optimisation.data.nems import TZ, get_csv

log = structlog.get_logger(__name__)

DEFAULT_RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "predispatch"

# Patient retries: a run is lost for good if every attempt fails (waits 10+20+30+40 s).
RETRIES = 5
BACKOFF_S = 10.0


class Feed(StrEnum):
    """Pre-dispatch feeds. The value is also the storage sub-folder."""

    ENERGY = "energy"  # USEP, demand, solar, TCL, RUSEP, ...
    ANCILLARY = "ancillary"  # primary, contingency, regulation prices


class View(StrEnum):
    """The two rolling files EMC publishes per feed."""

    RT48 = "rt48"  # today's 48 periods
    RT72 = "rt72"  # 72-period window reaching into tomorrow


# Form ``value`` id of each (feed, view) on the download endpoint. RT72 last: preferred on overlap.
FORM_ID: dict[Feed, dict[View, str]] = {
    Feed.ENERGY: {View.RT48: "10", View.RT72: "12"},
    Feed.ANCILLARY: {View.RT48: "11", View.RT72: "13"},
}

# Source CSV header -> column. Keys are EMC's spelling (external format).
_HEADERS: dict[Feed, dict[str, Col]] = {
    Feed.ENERGY: {
        "Demand (MW)": Col.DEMAND_MW,
        "Solar (MW)": Col.SOLAR_MW,
        "TCL (MW)": Col.TCL_MW,
        "USEP ($/MWh)": Col.USEP,
        "EHEUR ($/MWh)": Col.EHEUR,
        "LCP ($/MWh)": Col.LCP,
        "RUSEP ($/MWh)": Col.RUSEP,
        "MAP ($/MWh)": Col.MAP,
        "MAPT ($/MWh)": Col.MAPT,
        "TPC Applied": Col.TPC_APPLIED,
    },
    Feed.ANCILLARY: {
        "Primary Reserve ($/MWh)": Col.PRIMARY,
        "Reference Primary Reserve ($/MWh)": Col.PRIMARY_REF,
        "Contingency Reserve ($/MWh)": Col.CONTINGENCY,
        "Reference Contingency Reserve ($/MWh)": Col.CONTINGENCY_REF,
        "Regulation ($/MWh)": Col.REGULATION,
        "Reference Regulation ($/MWh)": Col.REGULATION_REF,
    },
}
_PERIOD_LABEL = "_period_label"  # "HH:MM-HH:MM", parsed then dropped
_META_HEADERS: dict[str, str] = {
    "Date": Col.DATE,
    "Period": _PERIOD_LABEL,
    "Last Updated": Col.PUBLISHED_AT,
}
_KEYS = [Col.INTERVAL_START, Col.PUBLISHED_AT, Col.FETCHED_AT, Col.LEAD_PERIODS]


def parse_snapshot(feed: Feed, text: str, fetched_at: datetime) -> pl.DataFrame:
    """Parse one rolling CSV (either view) into typed ``Col`` columns."""
    feed = Feed(feed)
    df = pl.read_csv(io.StringIO(text), infer_schema=False, null_values=["-"])
    mapping = {**_META_HEADERS, **_HEADERS[feed]}
    unknown = {c.strip() for c in df.columns} - mapping.keys()
    if unknown:
        raise ValueError(f"unexpected {feed} columns: {sorted(unknown)}")
    df = df.rename({c: mapping[c.strip()] for c in df.columns})
    df = df.with_columns(pl.col(pl.String).str.strip_chars())

    values = [c for c in _HEADERS[feed].values() if c is not Col.TPC_APPLIED]
    df = df.with_columns(
        (
            pl.col(Col.DATE).str.to_date("%d-%b-%Y").cast(pl.Datetime("us"))
            + pl.col(_PERIOD_LABEL).str.slice(0, 5).str.to_time("%H:%M").cast(pl.Duration("us"))
        )
        .dt.replace_time_zone(TZ)
        .alias(Col.INTERVAL_START),
        pl.col(Col.PUBLISHED_AT)
        .str.to_datetime("%d %b %Y %I:%M:%S %p", time_unit="us")
        .dt.replace_time_zone(TZ),
        pl.lit(fetched_at).dt.convert_time_zone(TZ).dt.cast_time_unit("us").alias(Col.FETCHED_AT),
        pl.col(values).cast(pl.Float64),
    )
    if Col.TPC_APPLIED in df.columns:
        df = df.with_columns(
            pl.col(Col.TPC_APPLIED)
            .str.to_lowercase()
            .replace_strict({"yes": True, "no": False}, return_dtype=pl.Boolean)
        )
    df = df.with_columns(
        (pl.col(Col.INTERVAL_START) - pl.col(Col.PUBLISHED_AT).dt.truncate("30m"))
        .dt.total_minutes()
        .floordiv(30)
        .cast(pl.Int32)
        .alias(Col.LEAD_PERIODS)
    )
    return df.select(*_KEYS, *_HEADERS[feed].values())


def fetch_snapshot(feed: Feed, session: requests.Session | None = None) -> pl.DataFrame:
    """Fetch both views of a feed and merge them into one frame."""
    feed = Feed(feed)
    fetched_at = datetime.now(ZoneInfo(TZ))
    frames = []
    for view, form_id in FORM_ID[feed].items():
        text = get_csv(
            form_id,
            fetched_at.date(),
            fetched_at.date(),
            session,
            retries=RETRIES,
            backoff_s=BACKOFF_S,
        )
        if text is None:
            log.warning("predispatch.empty", feed=feed, view=view)
            continue
        frames.append(parse_snapshot(feed, text, fetched_at))
    if not frames:
        raise RuntimeError(f"no {feed} pre-dispatch data returned")
    # RT72 comes last in FORM_ID; prefer it where both views cover an interval.
    df = pl.concat(frames).unique(
        subset=[Col.PUBLISHED_AT, Col.INTERVAL_START], keep="last", maintain_order=True
    )
    return df.sort(Col.PUBLISHED_AT, Col.INTERVAL_START)


def _snapshot_path(raw_dir: Path, feed: Feed, published_at: datetime) -> Path:
    day_dir = Path(raw_dir) / feed / f"{published_at:%Y-%m-%d}"
    return day_dir / f"{published_at:%Y%m%dT%H%M%S}.parquet"


def archive(
    feeds: Iterable[Feed] = tuple(Feed),
    raw_dir: Path = DEFAULT_RAW_DIR,
    overwrite: bool = False,
) -> list[Path]:
    """Fetch the current pre-dispatch run for each feed and write it if not already stored.

    A failing feed does not stop the others; errors are logged and re-raised at the end.
    """
    written: list[Path] = []
    failed: list[Feed] = []
    with requests.Session() as session:
        for feed in map(Feed, feeds):
            try:
                df = fetch_snapshot(feed, session)
            except Exception:
                log.exception("predispatch.failed", feed=feed)
                failed.append(feed)
                continue
            # Usually one run; two if EMC published between our RT48 and RT72 requests.
            for (published_at,), part in df.group_by(Col.PUBLISHED_AT, maintain_order=True):
                path = _snapshot_path(raw_dir, feed, published_at)
                if path.exists() and not overwrite:
                    log.info("predispatch.exists", feed=feed, published_at=str(published_at))
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".parquet.tmp")
                part.write_parquet(tmp)
                os.replace(tmp, path)  # atomic: readers never see a partial file
                written.append(path)
                log.info(
                    "predispatch.saved",
                    feed=feed,
                    published_at=str(published_at),
                    rows=part.height,
                    forecast_rows=part.filter(pl.col(Col.LEAD_PERIODS) >= 1).height,
                    path=str(path),
                )
    if failed:
        raise RuntimeError(f"pre-dispatch fetch failed for: {', '.join(failed)}")
    return written


def load_predispatch(feed: Feed, raw_dir: Path = DEFAULT_RAW_DIR) -> pl.DataFrame:
    """All archived runs of a feed, sorted by (published_at, interval_start)."""
    feed = Feed(feed)
    files = sorted((Path(raw_dir) / feed).glob("*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no snapshots in {Path(raw_dir) / feed}")
    df = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
    return df.sort(Col.PUBLISHED_AT, Col.INTERVAL_START)
