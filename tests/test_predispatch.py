from datetime import datetime
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from battery_optimisation.data import predispatch

SGT = ZoneInfo("Asia/Singapore")
HEADER = (
    '"Date","Period","Demand (MW)","Solar (MW)","TCL (MW)","USEP ($/MWh)","EHEUR ($/MWh)",'
    '"LCP ($/MWh)","RUSEP ($/MWh)","MAP ($/MWh)","MAPT ($/MWh)","TPC Applied","Last Updated"\r\n'
)
ROW = (
    '"{d}","{p}","7000.0","0.000","0.000","{usep}","-1.0","0.00","{rusep}","-","-","{tpc}",'
    '"{u}"\r\n'
)
UPDATED = "04 Nov 2026 10:31:30 PM"


def _energy_csv(rows: list[tuple[str, str, str]], updated: str = UPDATED) -> str:
    """First row looks realised (RUSEP, TPC filled), the rest like forecasts ("-")."""
    lines = []
    for i, (d, p, usep) in enumerate(rows):
        rusep, tpc = (usep, "No") if i == 0 else ("-", "-")
        lines.append(ROW.format(d=d, p=p, usep=usep, rusep=rusep, tpc=tpc, u=updated))
    return HEADER + "".join(lines)


RT48 = _energy_csv(
    [("04-Nov-2026", "22:30-23:00", "250.0"), ("04-Nov-2026", "23:00-23:30", "260.0")]
)
RT72 = _energy_csv(
    [("04-Nov-2026", "23:00-23:30", "260.0"), ("05-Nov-2026", "00:00-00:30", "210.0")]
)
FETCHED = datetime(2026, 11, 4, 22, 34, 5, tzinfo=SGT)


def test_parse_snapshot_times_and_lead() -> None:
    df = predispatch.parse_snapshot("energy", RT48, FETCHED)
    assert df.columns[:4] == ["interval_start", "published_at", "fetched_at", "lead_periods"]
    for col in ("interval_start", "published_at", "fetched_at"):
        assert df.schema[col] == pl.Datetime("us", "Asia/Singapore")
    assert df["interval_start"][0] == datetime(2026, 11, 4, 22, 30, tzinfo=SGT)
    assert df["published_at"][0] == datetime(2026, 11, 4, 22, 31, 30, tzinfo=SGT)
    assert df["fetched_at"][0] == FETCHED
    assert df["lead_periods"].to_list() == [0, 1]  # current period, then forecast
    assert df["rusep"][1] is None and df["tpc_applied"][1] is None


def test_parse_snapshot_lead_across_midnight() -> None:
    df = predispatch.parse_snapshot("energy", RT72, FETCHED)
    assert df["lead_periods"].to_list() == [1, 3]


def test_parse_snapshot_rejects_unknown_columns() -> None:
    with pytest.raises(ValueError, match="unexpected"):
        predispatch.parse_snapshot("energy", RT48.replace("EHEUR", "NEWCOL"), FETCHED)


def test_archive_merges_views_and_is_idempotent(tmp_path, monkeypatch) -> None:
    responses = {"10": RT48, "12": RT72}
    monkeypatch.setattr(predispatch, "get_csv", lambda value, *a, **k: responses[value])

    first = predispatch.archive(["energy"], raw_dir=tmp_path)
    assert [p.relative_to(tmp_path).as_posix() for p in first] == [
        "energy/2026-11-04/20261104T223130.parquet"
    ]
    df = pl.read_parquet(first[0])
    assert df.height == 3  # 22:30, 23:00 (in both views, kept once), 00:00
    assert df["interval_start"].is_sorted()

    assert predispatch.archive(["energy"], raw_dir=tmp_path) == []  # same run: not rewritten
    assert predispatch.load_predispatch("energy", raw_dir=tmp_path).height == 3


def test_archive_splits_when_run_changes_between_views(tmp_path, monkeypatch) -> None:
    later = RT72.replace(UPDATED, "04 Nov 2026 11:01:30 PM")
    responses = {"10": RT48, "12": later}
    monkeypatch.setattr(predispatch, "get_csv", lambda value, *a, **k: responses[value])
    paths = predispatch.archive(["energy"], raw_dir=tmp_path)
    assert [p.name for p in paths] == ["20261104T223130.parquet", "20261104T230130.parquet"]


def test_archive_continues_after_failed_feed(tmp_path, monkeypatch) -> None:
    def fake_get_csv(value, *a, **k):
        if value in {"11", "13"}:
            raise ConnectionResetError("reset by peer")
        return {"10": RT48, "12": RT72}[value]

    monkeypatch.setattr(predispatch, "get_csv", fake_get_csv)
    with pytest.raises(RuntimeError, match="ancillary"):
        predispatch.archive(["ancillary", "energy"], raw_dir=tmp_path)
    assert len(list((tmp_path / "energy").glob("*/*.parquet"))) == 1  # energy still saved
