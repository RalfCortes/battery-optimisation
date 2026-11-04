"""Column names of every frame in the project (data, optimisation, simulation).

Always refer to columns through ``Col`` (``pl.col(Col.USEP)``, ``df[Col.INTERVAL_START]``),
never bare strings. ``Col`` is a ``StrEnum``, so members are plain ``str`` for polars, and the
values are the names stored in parquet files.
"""

from __future__ import annotations

from enum import StrEnum


class Col(StrEnum):
    # Time keys
    INTERVAL_START = "interval_start"  # start of the 30-min interval, Asia/Singapore
    PERIOD = "period"  # 1..48 within the day (raw history files only)
    DATE = "date"  # trading date (raw CSV only, dropped after parsing)
    PUBLISHED_AT = "published_at"  # pre-dispatch run "Last Updated" stamp
    FETCHED_AT = "fetched_at"  # when we fetched a pre-dispatch run
    LEAD_PERIODS = "lead_periods"  # pre-dispatch: >= 1 forecast, <= 0 realised/current

    # Energy (SGD/MWh unless noted)
    USEP = "usep"  # Uniform Singapore Energy Price (capped when TPC applies)
    RUSEP = "rusep"  # USEP before the temporary price cap
    LCP = "lcp"  # load currency price
    EHEUR = "eheur"  # energy hourly uplift rate (pre-dispatch files only)
    MAP = "map"  # TPC moving average price
    MAPT = "mapt"  # TPC moving average price threshold (USEP cap when TPC applies)
    TPC_APPLIED = "tpc_applied"  # bool
    DEMAND_MW = "demand_mw"
    SOLAR_MW = "solar_mw"
    TCL_MW = "tcl_mw"  # transmission-connected load

    # Regulation
    REGULATION = "regulation"
    REGULATION_REF = "regulation_ref"

    # Reserves, pre-dispatch files (one price per product)
    PRIMARY = "primary"
    PRIMARY_REF = "primary_ref"
    CONTINGENCY = "contingency"
    CONTINGENCY_REF = "contingency_ref"

    # Reserves, history (one price per reserve group A..E)
    PRIMARY_A = "primary_a"
    PRIMARY_B = "primary_b"
    PRIMARY_C = "primary_c"
    PRIMARY_D = "primary_d"
    PRIMARY_E = "primary_e"
    CONTINGENCY_A = "contingency_a"
    CONTINGENCY_B = "contingency_b"
    CONTINGENCY_C = "contingency_c"
    CONTINGENCY_D = "contingency_d"
    CONTINGENCY_E = "contingency_e"

    # Raw reserve files (long format, pivoted to PRIMARY_A.. by the loader)
    RESERVE_GROUP = "reserve_group"
    PRICE = "price"

    # Battery dispatch (MW at the grid connection; SOC in MWh at the END of the interval)
    CHARGE_MW = "charge_mw"
    DISCHARGE_MW = "discharge_mw"
    NET_MW = "net_mw"  # discharge - charge; positive = export
    SOC_MWH = "soc_mwh"

    # Settlement, per interval or aggregated (SGD)
    REVENUE_SGD = "revenue_sgd"  # USEP * net energy
    DEGRADATION_SGD = "degradation_sgd"  # kappa * throughput
    PROFIT_SGD = "profit_sgd"  # revenue - degradation

    # Aggregates (sim/metrics.py)
    PERIOD_START = "period_start"  # first day of the aggregation period (date)
    CHARGED_MWH = "charged_mwh"  # grid-side energy bought
    DISCHARGED_MWH = "discharged_mwh"  # grid-side energy sold
    CYCLES = "cycles"  # equivalent full cycles
    MEAN_USEP = "mean_usep"
