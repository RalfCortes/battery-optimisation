"""EMC NEMS price history: download from nems.emcsg.com and load as one half-hourly frame.

Source
------
The "Download Historical Data" form on https://www.nems.emcsg.com/nems-prices calls

    GET /api/sitecore/DataSync/DataDownload
        ?value=<id>&fromDate=YYYY-MM-DD&toDate=YYYY-MM-DD&tpcValue=1

and returns a CSV (HTTP 204 when there is no data). The web UI caps a request at 31 days and
offers history back to 1 January five calendar years ago; we fetch one calendar month per request.
Data appear with roughly a one-week publication lag. ``tpcValue`` does not change the content.

Series used here (form ``value`` id -> CSV columns):
  1 "USEP and Demand Forecast": USEP, LCP, DEMAND, SOLAR, TCL, RUSEP, MAP, MAPT, TPC Applied.
    Despite the name, history contains realised values only (no pre-dispatch forecasts).
    When the temporary price cap (TPC) applies, USEP is capped at MAPT and RUSEP holds the
    uncapped price.
  3 "Reserve Prices" (MRP): one price per reserve group PRIRESA..E and CONRESA..E.
  4 "Regulation Prices" (MFP): regulation price and a reference price.

Raw storage: ``data/raw/nems/<series>/<YYYY-MM>.parquet``, one month per file, columns as
published (renamed to snake_case, typed, plus ``interval_start``).

Command line: ``uv run nems download --help`` (see ``battery_optimisation.cli.nems``).
"""

from __future__ import annotations

import io
import re
import time
from collections.abc import Iterable, Iterator
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import requests
import structlog

log = structlog.get_logger(__name__)

# Properties of the source data format (not tunable market parameters).
TZ = "Asia/Singapore"
PERIODS_PER_DAY = 48

BASE_URL = "https://www.nems.emcsg.com/api/sitecore/DataSync/DataDownload"
USER_AGENT = "battery-optimisation-data-loader/0.1"
DEFAULT_RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "nems"

SERIES: dict[str, str] = {"usep": "1", "reserve": "3", "regulation": "4"}

# Months that ended less than this long ago are re-downloaded (late data may still arrive).
REFRESH_WINDOW = timedelta(days=21)

_COLUMNS: dict[str, dict[str, str]] = {
    "usep": {
        "USEP ($/MWh)": "usep",
        "LCP ($/MWh)": "lcp",
        "DEMAND (MW)": "demand_mw",
        "SOLAR(MW)": "solar_mw",
        "TCL (MW)": "tcl_mw",
        "RUSEP ($/MWh)": "rusep",
        "MAP ($/MWh)": "map",
        "MAPT ($/MWh)": "mapt",
        "TPC Applied": "tpc_applied",
    },
    "reserve": {"RESERVE GROUP": "reserve_group", "PRICE ($/MWh)": "price"},
    "regulation": {"PRICE ($/MWh)": "regulation", "REFERENCE PRICE ($/MWh)": "regulation_ref"},
}
_KEY_COLUMNS = {"date", "period", "interval_start"}
_TEXT_COLUMNS = {"reserve_group", "tpc_applied"}


# --------------------------------------------------------------------------- download


def _next_month(d: date) -> date:
    return (d.replace(day=1) + timedelta(days=32)).replace(day=1)


def month_chunks(start: date, end: date) -> Iterator[tuple[date, date]]:
    """Yield (first, last) day pairs covering [start, end], split at calendar-month boundaries."""
    cur = start
    while cur <= end:
        last = min(_next_month(cur) - timedelta(days=1), end)
        yield cur, last
        cur = last + timedelta(days=1)


def fetch_csv(
    series: str, start: date, end: date, session: requests.Session | None = None
) -> str | None:
    """Fetch one date range of a series as CSV text, or None if the server has no data."""
    if (end - start).days > 30:
        raise ValueError("NEMS download range is limited to 31 days")
    return get_csv(SERIES[series], start, end, session)


def get_csv(
    value: str,
    start: date,
    end: date,
    session: requests.Session | None = None,
    retries: int = 3,
    backoff_s: float = 5.0,
) -> str | None:
    """GET the DataDownload endpoint for form ``value``; CSV text, or None on HTTP 204.

    Transient failures (network errors, 5xx) are retried with linear backoff.
    """
    params = {
        "value": value,
        "fromDate": start.isoformat(),
        "toDate": end.isoformat(),
        "tpcValue": "1",
    }
    http = session or requests
    for attempt in range(1, retries + 1):
        try:
            resp = http.get(BASE_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=60)
            if resp.status_code < 500:
                break
            err: Exception = requests.HTTPError(f"HTTP {resp.status_code}")
        except requests.RequestException as exc:
            err = exc
        if attempt == retries:
            raise err
        log.warning("nems.retry", value=value, attempt=attempt, error=str(err))
        time.sleep(backoff_s * attempt)
    if resp.status_code == 204:
        return None
    resp.raise_for_status()
    # The server answers unknown requests with a 200 HTML "404" page.
    if "csv" not in resp.headers.get("Content-Type", ""):
        raise RuntimeError(f"unexpected content type {resp.headers.get('Content-Type')!r}")
    return resp.text


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def parse_csv(series: str, text: str) -> pl.DataFrame:
    """Parse a NEMS CSV into typed columns plus a tz-aware ``interval_start``."""
    df = pl.read_csv(io.StringIO(text), infer_schema=False, null_values=["-"])
    mapping = {"DATE": "date", "PERIOD": "period", **_COLUMNS[series]}
    df = df.rename({c: mapping.get(c.strip(), _slug(c)) for c in df.columns})
    df = df.drop("information_type", strict=False)
    df = df.with_columns(pl.col(pl.String).str.strip_chars())

    df = df.with_columns(pl.col("period").cast(pl.Int64))
    bad = df.filter(~pl.col("period").is_between(1, PERIODS_PER_DAY))
    if bad.height:
        raise ValueError(f"period outside 1..{PERIODS_PER_DAY}: {bad['period'].unique().to_list()}")

    numeric = [c for c in df.columns if c not in _KEY_COLUMNS | _TEXT_COLUMNS]
    df = df.with_columns(
        (
            pl.col("date").str.to_date("%d-%b-%Y").cast(pl.Datetime("us"))
            + pl.duration(minutes=(pl.col("period") - 1) * 30)
        )
        .dt.replace_time_zone(TZ)
        .alias("interval_start"),
        pl.col(numeric).cast(pl.Float64),
    )
    if "tpc_applied" in df.columns:
        df = df.with_columns(
            pl.col("tpc_applied")
            .str.to_lowercase()
            .replace_strict({"yes": True, "no": False}, return_dtype=pl.Boolean)
        )
    return df.select("interval_start", "period", pl.exclude(_KEY_COLUMNS))


def download(
    series: Iterable[str] = tuple(SERIES),
    start: date | None = None,
    end: date | None = None,
    raw_dir: Path = DEFAULT_RAW_DIR,
    overwrite: bool = False,
    pause_s: float = 1.0,
) -> list[Path]:
    """Download monthly parquet files for each series into ``raw_dir/<series>/``.

    Defaults to 1 January five years ago through today. Existing complete months are skipped
    unless ``overwrite``; months ending within ``REFRESH_WINDOW`` are always re-fetched.
    """
    today = date.today()
    start = start or date(today.year - 5, 1, 1)
    end = min(end or today, today)
    written: list[Path] = []
    with requests.Session() as session:
        for name in series:
            out_dir = Path(raw_dir) / name
            out_dir.mkdir(parents=True, exist_ok=True)
            for first, last in month_chunks(start, end):
                path = out_dir / f"{first:%Y-%m}.parquet"
                settled = today - _next_month(first) > REFRESH_WINDOW
                if path.exists() and settled and not overwrite:
                    continue
                text = fetch_csv(name, first, last, session)
                time.sleep(pause_s)
                if text is None:
                    log.warning("nems.no_data", series=name, start=str(first), end=str(last))
                    continue
                df = parse_csv(name, text)
                df.write_parquet(path)
                written.append(path)
                log.info(
                    "nems.downloaded",
                    series=name,
                    start=str(first),
                    end=str(last),
                    rows=df.height,
                    path=str(path),
                )
    return written


# --------------------------------------------------------------------------- load


def _read_series(raw_dir: Path, name: str) -> pl.DataFrame:
    files = sorted((Path(raw_dir) / name).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files in {Path(raw_dir) / name}; run download()")
    df = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
    keys = ["interval_start", "reserve_group"] if name == "reserve" else ["interval_start"]
    return df.unique(subset=keys, keep="last", maintain_order=True).drop("period")


def _reserve_group_name(group: str) -> str:
    kind = {"PRIRES": "primary", "CONRES": "contingency"}.get(group[:6], _slug(group[:-1]))
    return f"{kind}_{group[-1].lower()}"


def _reserve_wide(df: pl.DataFrame) -> pl.DataFrame:
    wide = df.pivot(on="reserve_group", index="interval_start", values="price")
    wide = wide.rename({c: _reserve_group_name(c) for c in wide.columns if c != "interval_start"})
    return wide.select("interval_start", *sorted(c for c in wide.columns if c != "interval_start"))


def _to_date(d: date | str) -> date:
    return date.fromisoformat(d) if isinstance(d, str) else d


def load_nems(
    start: date | str | None = None,
    end: date | str | None = None,
    raw_dir: Path = DEFAULT_RAW_DIR,
    series: Iterable[str] = tuple(SERIES),
) -> pl.DataFrame:
    """Load downloaded NEMS series as one frame keyed by ``interval_start`` (Asia/Singapore).

    ``interval_start`` is a sorted, complete 30-minute grid, 48 rows per day, from ``start``
    00:00 to ``end`` 23:30 inclusive (defaults: span of the data). Missing intervals are null.

    Columns (SGD/MWh unless noted): usep, lcp, demand_mw, solar_mw, tcl_mw, rusep, map, mapt,
    tpc_applied (bool), regulation, regulation_ref, primary_a..e, contingency_a..e.
    """
    frames = []
    for name in series:
        df = _read_series(raw_dir, name)
        frames.append(_reserve_wide(df) if name == "reserve" else df)

    stamps = pl.concat([f.select("interval_start") for f in frames])["interval_start"]
    first = _to_date(start) if start else stamps.min().date()
    last = _to_date(end) if end else stamps.max().date()
    grid = pl.datetime_range(
        datetime.combine(first, datetime.min.time()),
        datetime.combine(last + timedelta(days=1), datetime.min.time()),
        interval="30m",
        closed="left",
        time_unit="us",
        time_zone=TZ,
        eager=True,
    ).alias("interval_start")

    out = grid.to_frame()
    for f in frames:
        out = out.join(f, on="interval_start", how="left")
    out = out.set_sorted("interval_start")

    n_missing = out.select(pl.all_horizontal(pl.exclude("interval_start").is_null()).sum()).item()
    if n_missing:
        log.warning("nems.missing_intervals", missing=n_missing, total=out.height)
    return out
