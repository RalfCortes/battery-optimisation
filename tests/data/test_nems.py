from datetime import date, timedelta

import polars as pl
import pytest

from battery_optimisation.columns import Col
from battery_optimisation.data import nems
from battery_optimisation.data.nems import NemsSeries

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
    df = nems.parse_csv(NemsSeries.USEP, USEP_CSV)
    assert df.columns[:2] == [Col.INTERVAL_START, Col.PERIOD]
    assert df.schema[Col.INTERVAL_START] == pl.Datetime("us", "Asia/Singapore")
    assert df[Col.INTERVAL_START].dt.strftime("%Y-%m-%d %H:%M").to_list() == [
        "2023-07-05 00:00",
        "2023-07-05 23:30",
    ]
    assert df[Col.TPC_APPLIED].to_list() == [False, True]
    assert df[Col.SOLAR_MW].null_count() == 2  # "-" -> null
    assert df[Col.RUSEP][1] == pytest.approx(2502.98)


def test_parse_rejects_period_out_of_range() -> None:
    bad = USEP_CSV.replace('"48"', '"49"')
    with pytest.raises(ValueError, match="period"):
        nems.parse_csv(NemsSeries.USEP, bad)


def test_load_nems_full_grid_and_wide_reserves(tmp_path) -> None:
    for series, text in [
        (NemsSeries.USEP, USEP_CSV),
        (NemsSeries.RESERVE, RESERVE_CSV),
        (NemsSeries.REGULATION, REGULATION_CSV),
    ]:
        (tmp_path / series).mkdir()
        nems.parse_csv(series, text).write_parquet(tmp_path / series / "2023-07.parquet")

    df = nems.load_nems(raw_dir=tmp_path)

    assert df.height == 2 * 48  # 5 and 6 July, complete grid
    assert df[Col.INTERVAL_START].is_sorted()
    steps = df[Col.INTERVAL_START].diff().drop_nulls().unique()
    assert steps.to_list() == [timedelta(minutes=30)]
    assert {Col.PRIMARY_A, Col.PRIMARY_B, Col.CONTINGENCY_A} <= set(df.columns)
    first = df.row(0, named=True)
    assert first[Col.USEP] == pytest.approx(100.5)
    assert first[Col.PRIMARY_A] == pytest.approx(10.0)
    assert first[Col.REGULATION] == pytest.approx(16.18)
    assert df.row(1, named=True)[Col.PRIMARY_B] == pytest.approx(8.5)
    assert df.row(2, named=True)[Col.USEP] is None  # missing interval kept as null


def test_load_nems_clips_to_requested_dates(tmp_path) -> None:
    (tmp_path / NemsSeries.REGULATION).mkdir()
    nems.parse_csv(NemsSeries.REGULATION, REGULATION_CSV).write_parquet(
        tmp_path / NemsSeries.REGULATION / "2023-07.parquet"
    )
    df = nems.load_nems(
        "2023-07-06", "2023-07-06", raw_dir=tmp_path, series=[NemsSeries.REGULATION]
    )
    assert df.height == 48
    assert df[Col.REGULATION][0] == pytest.approx(10.92)


def test_parse_rejects_unknown_header_and_reserve_group() -> None:
    with pytest.raises(ValueError, match="columns"):
        nems.parse_csv(NemsSeries.USEP, USEP_CSV.replace("LCP ($/MWh)", "NEW ($/MWh)"))
    with pytest.raises(ValueError, match="reserve groups"):
        nems.parse_csv(NemsSeries.RESERVE, RESERVE_CSV.replace("CONRESA", "CONRESX"))


def test_series_must_be_known() -> None:
    with pytest.raises(ValueError):
        nems.parse_csv("usepp", USEP_CSV)  # typo is caught, not silently accepted
