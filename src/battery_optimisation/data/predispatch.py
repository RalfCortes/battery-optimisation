"""EMC pre-dispatch snapshots: archive the rolling real-time/72-period files, one per run.

EMC publishes a rolling schedule on https://www.nems.emcsg.com/nems-prices, refreshed every
30 minutes about 1-2 minutes after the period starts (the "Last Updated" stamp). Only the latest
run is available, so past forecasts cannot be downloaded later and must be archived as they appear.
The same DataDownload endpoint as the price history serves it (form ``value`` ids):

  energy:    10 = today's 48 periods, 12 = 72-period window    (USEP, demand, solar, TCL, ...)
  ancillary: 11 = today's 48 periods, 13 = 72-period window    (primary, contingency, regulation)

Each file mixes realised intervals (up to the current period) with pre-dispatch forecasts for the
rest. Both views are fetched and merged on ``interval_start``.

Columns added:
  published_at  "Last Updated" stamp of the pre-dispatch run (Asia/Singapore)
  fetched_at    when we fetched it (Asia/Singapore)
  lead_periods  (interval_start - start of the period containing published_at) / 30 min;
                <= 0 realised/current dispatch, >= 1 pre-dispatch forecast

Storage: ``data/raw/predispatch/<feed>/<YYYY-MM-DD>/<published_at:%Y%m%dT%H%M%S>.parquet``.
A run already on disk is not rewritten, so overlapping or repeated cron runs are harmless.
"""

from __future__ import annotations

import io
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import requests
import structlog

from battery_optimisation.data.nems import TZ, get_csv

log = structlog.get_logger(__name__)

DEFAULT_RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "predispatch"

# Patient retries: a run is lost for good if every attempt fails (waits 10+20+30+40 s).
RETRIES = 5
BACKOFF_S = 10.0

FEEDS: dict[str, dict[str, str]] = {
    "energy": {"rt48": "10", "rt72": "12"},
    "ancillary": {"rt48": "11", "rt72": "13"},
}

_COLUMNS: dict[str, dict[str, str]] = {
    "energy": {
        "Demand (MW)": "demand_mw",
        "Solar (MW)": "solar_mw",
        "TCL (MW)": "tcl_mw",
        "USEP ($/MWh)": "usep",
        "EHEUR ($/MWh)": "eheur",
        "LCP ($/MWh)": "lcp",
        "RUSEP ($/MWh)": "rusep",
        "MAP ($/MWh)": "map",
        "MAPT ($/MWh)": "mapt",
        "TPC Applied": "tpc_applied",
    },
    "ancillary": {
        "Primary Reserve ($/MWh)": "primary",
        "Reference Primary Reserve ($/MWh)": "primary_ref",
        "Contingency Reserve ($/MWh)": "contingency",
        "Reference Contingency Reserve ($/MWh)": "contingency_ref",
        "Regulation ($/MWh)": "regulation",
        "Reference Regulation ($/MWh)": "regulation_ref",
    },
}
_META = {"Date": "date", "Period": "period", "Last Updated": "published_at"}


def parse_snapshot(feed: str, text: str, fetched_at: datetime) -> pl.DataFrame:
    """Parse one rolling CSV (48- or 72-period view) into typed columns."""
    df = pl.read_csv(io.StringIO(text), infer_schema=False, null_values=["-"])
    mapping = {**_META, **_COLUMNS[feed]}
    unknown = set(c.strip() for c in df.columns) - mapping.keys()
    if unknown:
        raise ValueError(f"unexpected {feed} columns: {sorted(unknown)}")
    df = df.rename({c: mapping[c.strip()] for c in df.columns})
    df = df.with_columns(pl.col(pl.String).str.strip_chars())

    value_cols = [c for c in _COLUMNS[feed].values() if c != "tpc_applied"]
    df = df.with_columns(
        (
            pl.col("date").str.to_date("%d-%b-%Y").cast(pl.Datetime("us"))
            + pl.col("period").str.slice(0, 5).str.to_time("%H:%M").cast(pl.Duration("us"))
        )
        .dt.replace_time_zone(TZ)
        .alias("interval_start"),
        pl.col("published_at")
        .str.to_datetime("%d %b %Y %I:%M:%S %p", time_unit="us")
        .dt.replace_time_zone(TZ),
        pl.lit(fetched_at).dt.convert_time_zone(TZ).dt.cast_time_unit("us").alias("fetched_at"),
        pl.col(value_cols).cast(pl.Float64),
    )
    if "tpc_applied" in df.columns:
        df = df.with_columns(
            pl.col("tpc_applied")
            .str.to_lowercase()
            .replace_strict({"yes": True, "no": False}, return_dtype=pl.Boolean)
        )
    df = df.with_columns(
        (
            (pl.col("interval_start") - pl.col("published_at").dt.truncate("30m"))
            .dt.total_minutes()
            .floordiv(30)
        )
        .cast(pl.Int32)
        .alias("lead_periods")
    )
    return df.select(
        "interval_start", "published_at", "fetched_at", "lead_periods", *_COLUMNS[feed].values()
    )


def fetch_snapshot(feed: str, session: requests.Session | None = None) -> pl.DataFrame:
    """Fetch the 48- and 72-period views of a feed and merge them into one frame."""
    fetched_at = datetime.now(ZoneInfo(TZ))
    frames = []
    for view, value in FEEDS[feed].items():
        text = get_csv(
            value,
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
    # The 72-period view comes last; prefer it where both views cover an interval.
    df = pl.concat(frames).unique(
        subset=["published_at", "interval_start"], keep="last", maintain_order=True
    )
    return df.sort("published_at", "interval_start")


def _snapshot_path(raw_dir: Path, feed: str, published_at: datetime) -> Path:
    return (
        Path(raw_dir) / feed / f"{published_at:%Y-%m-%d}" / f"{published_at:%Y%m%dT%H%M%S}.parquet"
    )


def archive(
    feeds: list[str] | None = None, raw_dir: Path = DEFAULT_RAW_DIR, overwrite: bool = False
) -> list[Path]:
    """Fetch the current pre-dispatch run for each feed and write it if not already stored.

    A failing feed does not stop the others; errors are logged and re-raised at the end.
    """
    written: list[Path] = []
    failed: list[str] = []
    with requests.Session() as session:
        for feed in feeds or list(FEEDS):
            try:
                df = fetch_snapshot(feed, session)
            except Exception:
                log.exception("predispatch.failed", feed=feed)
                failed.append(feed)
                continue
            # Usually one run; two if EMC published between our 48- and 72-period requests.
            for (published_at,), part in df.group_by("published_at", maintain_order=True):
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
                    forecast_rows=part.filter(pl.col("lead_periods") >= 1).height,
                    path=str(path),
                )
    if failed:
        raise RuntimeError(f"pre-dispatch fetch failed for: {', '.join(failed)}")
    return written


def load_predispatch(feed: str, raw_dir: Path = DEFAULT_RAW_DIR) -> pl.DataFrame:
    """All archived runs of a feed, sorted by (published_at, interval_start)."""
    files = sorted((Path(raw_dir) / feed).glob("*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no snapshots in {Path(raw_dir) / feed}")
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed").sort(
        "published_at", "interval_start"
    )
