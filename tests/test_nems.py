from datetime import date, timedelta

import polars as pl
import pytest

from battery_optimisation.data import nems

USEP_CSV = (
    '"INFORMATION TYPE","DATE","PERIOD","USEP ($/MWh)","LCP ($/MWh)","DEMAND (MW)",'
    '"SOLAR(MW)","TCL (MW)","RUSEP ($/MWh)","MAP ($/MWh)","MAPT ($/MWh)","TPC Applied"\r\n'
    '"USEP","05-Jul-2023","1","100.50","0.00","6031.860","-","0.000","100.50","86.37","500.85","No"\r\n'
    '"USEP","05-Jul-2023","48","500.85","132.00","7277.197","-","6.000","2502.98","504.44","500.85","Yes"\r\n'
)
RESERVE_CSV = (
    '"INFORMATION TYPE","RESERVE GROUP","DATE","PERIOD","PRICE ($/MWh)"\r\n'
    '"MRP","CONRESA","05-Jul-2023","1","12.77"\r\n'
    '"MRP","PRIRESA","05-Jul-2023","1","10.00"\r\n'
    '"MRP","PRIRESB","05-Jul-2023","2","8.50"\r\n'
)
REGULATION_CSV = (
    '"INFORMATION TYPE","DATE","PERIOD","PRICE ($/MWh)","REFERENCE PRICE ($/MWh)"\r\n'
    '"MFP","05-Jul-2023","1","16.18","-"\r\n'
    '"MFP","06-Jul-2023","1","10.92","-"\r\n'
)


def test_month_chunks_split_at_month_boundaries() -> None:
    chunks = list(nems.month_chunks(date(2024, 1, 15), date(2024, 3, 2)))
    assert chunks == [
        (date(2024, 1, 15), date(2024, 1, 31)),
        (date(2024, 2, 1), date(2024, 2, 29)),
        (date(2024, 3, 1), date(2024, 3, 2)),
    ]
    assert all((last - first).days <= 30 for first, last in chunks)


def test_parse_usep_types_and_interval_start() -> None:
    df = nems.parse_csv("usep", USEP_CSV)
    assert df.columns[:2] == ["interval_start", "period"]
    assert df.schema["interval_start"] == pl.Datetime("us", "Asia/Singapore")
    assert df["interval_start"].dt.strftime("%Y-%m-%d %H:%M").to_list() == [
        "2023-07-05 00:00",
        "2023-07-05 23:30",
    ]
    assert df["tpc_applied"].to_list() == [False, True]
    assert df["solar_mw"].null_count() == 2  # "-" -> null
    assert df["rusep"][1] == pytest.approx(2502.98)


def test_parse_rejects_period_out_of_range() -> None:
    bad = USEP_CSV.replace('"48"', '"49"')
    with pytest.raises(ValueError, match="period"):
        nems.parse_csv("usep", bad)


def test_load_nems_full_grid_and_wide_reserves(tmp_path) -> None:
    for series, text in [
        ("usep", USEP_CSV),
        ("reserve", RESERVE_CSV),
        ("regulation", REGULATION_CSV),
    ]:
        (tmp_path / series).mkdir()
        nems.parse_csv(series, text).write_parquet(tmp_path / series / "2023-07.parquet")

    df = nems.load_nems(raw_dir=tmp_path)

    assert df.height == 2 * 48  # 5 and 6 July, complete grid
    assert df["interval_start"].is_sorted()
    steps = df["interval_start"].diff().drop_nulls().unique()
    assert steps.to_list() == [timedelta(minutes=30)]
    assert {"primary_a", "primary_b", "contingency_a"} <= set(df.columns)
    first = df.row(0, named=True)
    assert first["usep"] == pytest.approx(100.5)
    assert first["primary_a"] == pytest.approx(10.0)
    assert first["regulation"] == pytest.approx(16.18)
    assert df.row(1, named=True)["primary_b"] == pytest.approx(8.5)
    assert df.row(2, named=True)["usep"] is None  # missing interval kept as null


def test_load_nems_clips_to_requested_dates(tmp_path) -> None:
    (tmp_path / "regulation").mkdir()
    nems.parse_csv("regulation", REGULATION_CSV).write_parquet(
        tmp_path / "regulation" / "2023-07.parquet"
    )
    df = nems.load_nems("2023-07-06", "2023-07-06", raw_dir=tmp_path, series=["regulation"])
    assert df.height == 48
    assert df["regulation"][0] == pytest.approx(10.92)
