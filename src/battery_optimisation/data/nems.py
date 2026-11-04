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
published (renamed to ``Col`` names, typed, plus ``interval_start``).

Typed names: series are ``NemsSeries``, reserve groups ``ReserveGroup``, columns ``Col``.

Command line: ``uv run nems download --help`` (see ``battery_optimisation.cli.nems``).
"""

from __future__ import annotations

import io
import time
from collections.abc import Iterable, Iterator
from datetime import date, datetime, timedelta
from enum import StrEnum
from pathlib import Path

import polars as pl
import requests
import structlog

from battery_optimisation.columns import Col

log = structlog.get_logger(__name__)

# Properties of the source data format (not tunable market parameters).
TZ = "Asia/Singapore"
PERIODS_PER_DAY = 48

BASE_URL = "https://www.nems.emcsg.com/api/sitecore/DataSync/DataDownload"
USER_AGENT = "battery-optimisation-data-loader/0.1"
DEFAULT_RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "nems"


class NemsSeries(StrEnum):
    """Historical series offered by the NEMS download form."""

    USEP = "usep"  # USEP, demand, solar, TCL, RUSEP, MAP, MAPT, TPC flag
    RESERVE = "reserve"  # primary and contingency reserve prices per reserve group
    REGULATION = "regulation"  # regulation price


class ReserveGroup(StrEnum):
    """Reserve groups in the reserve price files (PRIRES = primary, CONRES = contingency)."""

    PRIRESA = "PRIRESA"
    PRIRESB = "PRIRESB"
    PRIRESC = "PRIRESC"
    PRIRESD = "PRIRESD"
    PRIRESE = "PRIRESE"
    CONRESA = "CONRESA"
    CONRESB = "CONRESB"
    CONRESC = "CONRESC"
    CONRESD = "CONRESD"
    CONRESE = "CONRESE"


# Form ``value`` id of each series on the download endpoint.
FORM_ID: dict[NemsSeries, str] = {
    NemsSeries.USEP: "1",
    NemsSeries.RESERVE: "3",
    NemsSeries.REGULATION: "4",
}

# Wide column of each reserve group in ``load_nems``.
RESERVE_COLUMN: dict[ReserveGroup, Col] = {
    ReserveGroup.PRIRESA: Col.PRIMARY_A,
    ReserveGroup.PRIRESB: Col.PRIMARY_B,
    ReserveGroup.PRIRESC: Col.PRIMARY_C,
    ReserveGroup.PRIRESD: Col.PRIMARY_D,
    ReserveGroup.PRIRESE: Col.PRIMARY_E,
    ReserveGroup.CONRESA: Col.CONTINGENCY_A,
    ReserveGroup.CONRESB: Col.CONTINGENCY_B,
    ReserveGroup.CONRESC: Col.CONTINGENCY_C,
    ReserveGroup.CONRESD: Col.CONTINGENCY_D,
    ReserveGroup.CONRESE: Col.CONTINGENCY_E,
}

# Months that ended less than this long ago are re-downloaded (late data may still arrive).
REFRESH_WINDOW = timedelta(days=21)

# Source CSV header -> column, per series. Keys are EMC's spelling (external format).
_INFO_TYPE = "INFORMATION TYPE"  # constant per file ("USEP", "MRP", "MFP"); dropped
_HEADERS: dict[NemsSeries, dict[str, Col]] = {
    NemsSeries.USEP: {
        "USEP ($/MWh)": Col.USEP,
        "LCP ($/MWh)": Col.LCP,
        "DEMAND (MW)": Col.DEMAND_MW,
        "SOLAR(MW)": Col.SOLAR_MW,
        "TCL (MW)": Col.TCL_MW,
        "RUSEP ($/MWh)": Col.RUSEP,
        "MAP ($/MWh)": Col.MAP,
        "MAPT ($/MWh)": Col.MAPT,
        "TPC Applied": Col.TPC_APPLIED,
    },
    NemsSeries.RESERVE: {"RESERVE GROUP": Col.RESERVE_GROUP, "PRICE ($/MWh)": Col.PRICE},
    NemsSeries.REGULATION: {
        "PRICE ($/MWh)": Col.REGULATION,
        "REFERENCE PRICE ($/MWh)": Col.REGULATION_REF,
    },
}
_KEY_HEADERS = {"DATE": Col.DATE, "PERIOD": Col.PERIOD}
_KEY_COLUMNS = {Col.DATE, Col.PERIOD, Col.INTERVAL_START}
_TEXT_COLUMNS = {Col.RESERVE_GROUP, Col.TPC_APPLIED}


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
    series: NemsSeries, start: date, end: date, session: requests.Session | None = None
) -> str | None:
    """Fetch one date range of a series as CSV text, or None if the server has no data."""
    if (end - start).days > 30:
        raise ValueError("NEMS download range is limited to 31 days")
    return get_csv(FORM_ID[NemsSeries(series)], start, end, session)


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


def parse_csv(series: NemsSeries, text: str) -> pl.DataFrame:
    """Parse a NEMS CSV into typed ``Col`` columns plus a tz-aware ``interval_start``.

    Raises ``ValueError`` on unknown headers, so a change in EMC's format is noticed.
    """
    series = NemsSeries(series)
    df = pl.read_csv(io.StringIO(text), infer_schema=False, null_values=["-"])
    df = df.drop([c for c in df.columns if c.strip() == _INFO_TYPE])
    mapping = {**_KEY_HEADERS, **_HEADERS[series]}
    unknown = {c.strip() for c in df.columns} - mapping.keys()
    if unknown:
        raise ValueError(f"unexpected {series} columns: {sorted(unknown)}")
    df = df.rename({c: mapping[c.strip()] for c in df.columns})
    df = df.with_columns(pl.col(pl.String).str.strip_chars())

    df = df.with_columns(pl.col(Col.PERIOD).cast(pl.Int64))
    bad = df.filter(~pl.col(Col.PERIOD).is_between(1, PERIODS_PER_DAY))
    if bad.height:
        periods = bad[Col.PERIOD].unique().to_list()
        raise ValueError(f"period outside 1..{PERIODS_PER_DAY}: {periods}")

    numeric = [c for c in df.columns if c not in _KEY_COLUMNS | _TEXT_COLUMNS]
    df = df.with_columns(
        (
            pl.col(Col.DATE).str.to_date("%d-%b-%Y").cast(pl.Datetime("us"))
            + pl.duration(minutes=(pl.col(Col.PERIOD) - 1) * 30)
        )
        .dt.replace_time_zone(TZ)
        .alias(Col.INTERVAL_START),
        pl.col(numeric).cast(pl.Float64),
    )
    if Col.TPC_APPLIED in df.columns:
        df = df.with_columns(
            pl.col(Col.TPC_APPLIED)
            .str.to_lowercase()
            .replace_strict({"yes": True, "no": False}, return_dtype=pl.Boolean)
        )
    if Col.RESERVE_GROUP in df.columns:
        unknown_groups = set(df[Col.RESERVE_GROUP].unique()) - set(ReserveGroup)
        if unknown_groups:
            raise ValueError(f"unexpected reserve groups: {sorted(unknown_groups)}")
    return df.select(Col.INTERVAL_START, Col.PERIOD, pl.exclude(_KEY_COLUMNS))


def download(
    series: Iterable[NemsSeries] = tuple(NemsSeries),
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
        for name in map(NemsSeries, series):
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


def _read_series(raw_dir: Path, series: NemsSeries) -> pl.DataFrame:
    files = sorted((Path(raw_dir) / series).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files in {Path(raw_dir) / series}; run download()")
    df = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
    keys = [Col.INTERVAL_START]
    if series is NemsSeries.RESERVE:
        keys.append(Col.RESERVE_GROUP)
    return df.unique(subset=keys, keep="last", maintain_order=True).drop(Col.PERIOD)


def _reserve_wide(df: pl.DataFrame) -> pl.DataFrame:
    wide = df.pivot(on=Col.RESERVE_GROUP, index=Col.INTERVAL_START, values=Col.PRICE)
    groups = [ReserveGroup(c) for c in wide.columns if c != Col.INTERVAL_START]
    wide = wide.rename({g: RESERVE_COLUMN[g] for g in groups})
    present = [c for c in RESERVE_COLUMN.values() if c in wide.columns]
    return wide.select(Col.INTERVAL_START, *present)


def _to_date(d: date | str) -> date:
    return date.fromisoformat(d) if isinstance(d, str) else d


def load_nems(
    start: date | str | None = None,
    end: date | str | None = None,
    raw_dir: Path = DEFAULT_RAW_DIR,
    series: Iterable[NemsSeries] = tuple(NemsSeries),
) -> pl.DataFrame:
    """Load downloaded NEMS series as one frame keyed by ``interval_start`` (Asia/Singapore).

    ``interval_start`` is a sorted, complete 30-minute grid, 48 rows per day, from ``start``
    00:00 to ``end`` 23:30 inclusive (defaults: span of the data). Missing intervals are null.

    Columns (``Col``; SGD/MWh unless noted): USEP, LCP, DEMAND_MW, SOLAR_MW, TCL_MW, RUSEP,
    MAP, MAPT, TPC_APPLIED (bool), REGULATION, REGULATION_REF, PRIMARY_A..E, CONTINGENCY_A..E.
    """
    frames = []
    for name in map(NemsSeries, series):
        df = _read_series(raw_dir, name)
        frames.append(_reserve_wide(df) if name is NemsSeries.RESERVE else df)

    stamps = pl.concat([f.select(Col.INTERVAL_START) for f in frames])[Col.INTERVAL_START]
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
    ).alias(Col.INTERVAL_START)

    out = grid.to_frame()
    for f in frames:
        out = out.join(f, on=Col.INTERVAL_START, how="left")
    out = out.set_sorted(Col.INTERVAL_START)

    all_null = pl.all_horizontal(pl.exclude(Col.INTERVAL_START).is_null())
    n_missing = out.select(all_null.sum()).item()
    if n_missing:
        log.warning("nems.missing_intervals", missing=n_missing, total=out.height)
    return out
