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

    # Ancillary awards (MW of availability per interval)
    REG_MW = "reg_mw"  # regulation (up and down)
    PRI_MW = "pri_mw"  # primary reserve (raise)
    CON_MW = "con_mw"  # contingency reserve (raise)

    # Settlement, per interval or aggregated (SGD)
    REVENUE_SGD = "revenue_sgd"  # USEP * net energy
    DEGRADATION_SGD = "degradation_sgd"  # kappa * throughput
    ANCILLARY_SGD = "ancillary_sgd"  # availability payments (pay-as-clear), in profit
    PROFIT_SGD = "profit_sgd"  # revenue + ancillary - degradation - penalty
    SHORTFALL_MW = "shortfall_mw"  # cleared but not deliverable (physical limits), MW
    PENALTY_SGD = "penalty_sgd"  # shortfall settled at the shortfall penalty price
    REG_NET_MWH = "reg_net_mwh"  # grid energy moved by regulation (+ injected), in revenue
    SOC_CUT_MW = "soc_cut_mw"
    # Submitted offers (offers.parquet, one row per tranche)
    PRODUCT = "product"  # Product: energy or an ancillary product
    SIDE = "side"  # energy: charge / discharge; ancillary: null
    TRANCHE = "tranche"  # position in the submitted offer (0 = first)
    OFFER_PRICE = "offer_price"  # SGD/MWh
    OFFER_MW = "offer_mw"
    # Water-value decision trace (decisions.parquet, one row per offered interval)
    STAGE = "stage"  # position of the interval in its issue's SDP
    DECISION_SOC_MWH = "decision_soc_mwh"  # SOC known at gate closure
    EXPECTED_SOC_MWH = "expected_soc_mwh"  # SOC expected at the start of the interval
    STATE_PRICES = "state_prices"  # the SDP's price states for the interval (list, SGD/MWh)
    PLANNED_REG_MW = "planned_reg_mw"  # reserves planned before feasibility scaling
    PLANNED_PRI_MW = "planned_pri_mw"
    PLANNED_CON_MW = "planned_con_mw"  # cleared MW (energy + ancillary) the engine cut for the SoC

    # Aggregates (sim/metrics.py)
    PERIOD_START = "period_start"  # first day of the aggregation period (date)
    CHARGED_MWH = "charged_mwh"  # grid-side energy bought
    DISCHARGED_MWH = "discharged_mwh"  # grid-side energy sold
    CYCLES = "cycles"  # equivalent full cycles
    SHORTFALL_MWH = "shortfall_mwh"
    SOC_CUT_MWH = "soc_cut_mwh"
    MEAN_USEP = "mean_usep"

    # Forecasts (long format: one row per issue time x interval x quantile)
    ISSUED_AT = "issued_at"  # when the forecast was made; uses data strictly before it
    QUANTILE = "quantile"  # quantile level in (0, 1)
    USEP_FORECAST = "usep_forecast"  # SGD/MWh (the forecast value, whatever the target)
    TARGET = "target"  # forecast price column (ancillary quantile frames)

    # Plans recorded at each decision (plans.parquet: what the strategy expected to do)
    DECIDED_AT = "decided_at"  # gate closure of the offered interval: when the plan was made
    TARGET_START = "target_start"  # the interval being offered (first of the plan)
    SOC_P10_MWH = "soc_p10_mwh"  # 10% / 90% quantiles of the planned SOC (water value)
    SOC_P90_MWH = "soc_p90_mwh"
    SCENARIO = "scenario"  # scenario id 0..K-1
    WEIGHT = "weight"  # scenario probability

    # Covariates
    HOUR_SIN = "hour_sin"  # time of day, cyclic
    HOUR_COS = "hour_cos"
    DOW_SIN = "dow_sin"  # day of week, cyclic
    DOW_COS = "dow_cos"
    IS_OFF_DAY = "is_off_day"  # weekend or Singapore public holiday (0/1)
    TPC_REGIME = "tpc_regime"  # temporary price cap policy in force (0/1)
    TEMPERATURE_C = "temperature_c"  # weather forecast, 2 m air temperature
    HUMIDITY_PCT = "humidity_pct"  # weather forecast, 2 m relative humidity
    CLOUD_COVER_PCT = "cloud_cover_pct"  # weather forecast, total cloud cover
    IRRADIANCE_WM2 = "irradiance_wm2"  # weather forecast, shortwave radiation
    PRECIPITATION_MM = "precipitation_mm"  # weather forecast, precipitation in the hour
    RUN_TIME = "run_time"  # weather model run (nominal start, UTC) a value comes from
    WEATHER_LEAD_H = "weather_lead_h"  # hours between the run and the valid time
    LATITUDE = "latitude"  # weather grid point
    LONGITUDE = "longitude"

    # Forecast evaluation (per month or overall)
    CRPS = "crps"  # SGD/MWh, lower is better
    COVERAGE_10_90 = "coverage_10_90"  # share of realised values inside [q10, q90]; ideal 0.8
    SPIKE_HIT_RATE = "spike_hit_rate"  # share of spikes where q90 >= spike threshold
    SPIKE_FALSE_ALARM = "spike_false_alarm"  # share of non-spikes where q90 >= threshold
