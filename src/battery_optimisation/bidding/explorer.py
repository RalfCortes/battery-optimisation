"""Settlement explorer: a self-contained HTML page to dig into one saved bidding run.

``explorer_data(run_dir)`` gathers, per settlement interval, what every strategy offered
(``offers.parquet``), what cleared and was settled (``dispatch.parquet``), the realised prices
and, for the water value, the decision trace (``decisions.parquet``: expected SOC, the SDP's
price states, planned reserves). ``write_explorer`` embeds it in ``explorer.html`` (template
next to this module) and writes ``explore.html`` into the run folder.
"""

from __future__ import annotations

import base64
import json
import math
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from battery_optimisation.bidding.report import load_bidding_report
from battery_optimisation.columns import Col
from battery_optimisation.config import ANCILLARY_PRODUCTS, Product, load_market
from battery_optimisation.sim.backtest import RunFiles, StrategyName, load_prices, load_run
from battery_optimisation.sim.clearing import price_column

TEMPLATE = Path(__file__).with_name("explorer.html")
OUTPUT = "explore.html"
PLAN_DIR = "explore_plans"  # beside explore.html: weekly int16 chunks (JSON, base64), on demand
MISSING = -32768
LEVELS = (0.1, 0.5, 0.9)
_Q = ("q10", "q50", "q90")
# what was known at each decision: forecast fan of the issue used, then each strategy's plan
# (from the offered interval on); value = int16 / scale
PLAN_FIELDS: list[tuple[str, float]] = [
    *[(f"usep_{q}", 1.0) for q in _Q],
    *[(f"{p}_{q}", 10.0) for p in ANCILLARY_PRODUCTS for q in _Q],
    *[
        (f"wv_{k}", 1.0 if k == "price" else 10.0)
        for k in ("price", "net", "soc", "p10", "p90", "reg", "pri", "con")
    ],
    *[
        (f"ss_{k}", 1.0 if k == "price" else 10.0)
        for k in ("price", "net", "soc", "reg", "pri", "con")
    ],
]
PLAN_SOURCE = {
    "price": Col.USEP_FORECAST,
    "net": Col.NET_MW,
    "soc": Col.SOC_MWH,
    "p10": Col.SOC_P10_MWH,
    "p90": Col.SOC_P90_MWH,
    "reg": Col.REG_MW,
    "pri": Col.PRI_MW,
    "con": Col.CON_MW,
}
PLAN_PREFIX = {StrategyName.WATER_VALUE: "wv", StrategyName.CHRONOS_MEDIAN: "ss"}

LABELS = {
    StrategyName.WATER_VALUE: "Water value",
    StrategyName.CHRONOS_MEDIAN: "Self-schedule",
    StrategyName.PERFECT_FORESIGHT: "Perfect foresight",
}
PRODUCT_CODE = {Product.ENERGY: 0, **{p: i + 1 for i, p in enumerate(ANCILLARY_PRODUCTS)}}
SIDE_CODE = {"discharge": 1, "charge": -1}


def _round(values, digits: int) -> list:
    return [
        None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), digits)
        for v in values
    ]


def _index(stamps: pl.Series, start, step: timedelta) -> list[int]:
    return [round((t - start) / step) for t in stamps.to_list()]


PriceLoader = Callable[[date, date], pl.DataFrame]


def _realised_prices(market) -> PriceLoader:
    return lambda start, end: load_prices(start, end, market, ancillary=True)


def _forecast_dir(run_dir: Path) -> Path | None:
    for parent in run_dir.parents[:3]:
        if (parent / "quantiles.parquet").exists():
            return parent
    return None


def _plan_strategies(run_dir: Path) -> dict[StrategyName, Path]:
    out = {}
    for summary_file in (run_dir / "backtests").glob(f"*/{RunFiles.SUMMARY}"):
        plans = summary_file.parent / RunFiles.PLANS
        name = StrategyName(json.loads(summary_file.read_text())["strategy"])
        if plans.exists() and name in PLAN_PREFIX:
            out[name] = plans
    return out


def _index_expr(col: str, start, step: timedelta) -> pl.Expr:
    return (
        ((pl.col(col) - pl.lit(start)).dt.total_seconds() / step.total_seconds())
        .round()
        .cast(pl.Int64)
    )


def plan_cube(
    run_dir: Path,
    quantiles: pl.DataFrame | None = None,
    ancillary_quantiles: pl.DataFrame | None = None,
) -> np.ndarray | None:
    """[intervals, fields, horizon] int16 (``PLAN_FIELDS``): for each offered interval i, step k
    = interval i + k. Forecast: the latest issue at or before the decision (gate closure);
    plans: ``plans.parquet`` of each strategy. ``quantiles`` default: the forecast run's."""
    run_dir = Path(run_dir).resolve()  # "latest" -> the run folder (its name is the run)
    plans = _plan_strategies(run_dir)
    if not plans:
        return None
    market = load_market()
    step = timedelta(hours=market.interval_h)
    _, frame = load_run(next(iter(plans.values())).parent)
    stamps = frame.sort(Col.INTERVAL_START)[Col.INTERVAL_START]
    start, n, h = stamps[0], stamps.len(), market.intervals_per_day
    fields = [f for f, _ in PLAN_FIELDS]
    scale = dict(PLAN_FIELDS)
    cube = np.full((n, len(fields), h), MISSING, dtype=np.int16)

    def fill(part: pl.DataFrame, field: str, value: str) -> None:
        i, k = part["i"].to_numpy(), part["k"].to_numpy()
        v = part[value].cast(pl.Float64).to_numpy()
        ok = (i >= 0) & (i < n) & (k >= 0) & (k < h) & np.isfinite(v)
        cube[i[ok], fields.index(field), k[ok]] = np.clip(
            np.round(v[ok] * scale[field]), -32767, 32767
        )

    for name, path in plans.items():
        frame = pl.read_parquet(path).with_columns(
            _index_expr(Col.TARGET_START, start, step).alias("i"),
            (
                _index_expr(Col.INTERVAL_START, start, step)
                - _index_expr(Col.TARGET_START, start, step)
            ).alias("k"),
        )
        for key, col in PLAN_SOURCE.items():
            if f"{PLAN_PREFIX[name]}_{key}" in scale:
                fill(frame, f"{PLAN_PREFIX[name]}_{key}", col)

    gate = market.gate_closure_intervals
    targets = pl.DataFrame({"i": np.arange(n), Col.DECIDED_AT: stamps - gate * step}).sort(
        Col.DECIDED_AT
    )
    forecast_dir = _forecast_dir(run_dir)

    def scan(given: pl.DataFrame | None, file: str) -> pl.LazyFrame | None:
        if given is not None:
            return given.lazy()
        if forecast_dir is not None and (forecast_dir / file).exists():
            return pl.scan_parquet(forecast_dir / file)
        return None

    def fan(lf: pl.LazyFrame | None, prefix_of) -> None:
        if lf is None:
            return
        lo, hi = targets[Col.DECIDED_AT].min(), targets[Col.DECIDED_AT].max()
        rows = lf.filter(
            pl.col(Col.ISSUED_AT).is_between(lo - timedelta(days=2), hi)
            & pl.col(Col.QUANTILE).round(3).is_in(list(LEVELS))
        ).collect()
        if rows.is_empty():
            return
        issues = rows.select(pl.col(Col.ISSUED_AT).unique().sort())
        used = targets.join_asof(
            issues, left_on=Col.DECIDED_AT, right_on=Col.ISSUED_AT, strategy="backward"
        ).drop_nulls(Col.ISSUED_AT)
        joined = used.join(rows, on=Col.ISSUED_AT).with_columns(
            (_index_expr(Col.INTERVAL_START, start, step) - pl.col("i")).alias("k")
        )
        for (key, level), part in joined.group_by(
            prefix_of, pl.col(Col.QUANTILE).round(3), maintain_order=True
        ):
            if key is not None:
                fill(part, f"{key}_{_Q[LEVELS.index(level)]}", Col.USEP_FORECAST)

    fan(scan(quantiles, "quantiles.parquet"), pl.lit("usep"))
    by_column = {str(price_column(p, market)): str(p) for p in ANCILLARY_PRODUCTS}
    fan(
        scan(ancillary_quantiles, "ancillary_quantiles.parquet"),
        pl.col(Col.TARGET).replace_strict(by_column, default=None),
    )
    return cube


def explorer_data(
    run_dir: Path,
    realised: PriceLoader | None = None,
    quantiles: pl.DataFrame | None = None,
) -> dict:
    """Everything the explorer shows, as plain lists (one value per interval).

    Reserve prices come from the dispatch frame; runs saved before it kept them take them from
    ``realised(start, end)`` (default: the NEMS price data)."""
    run_dir = Path(run_dir).resolve()  # "latest" -> the run folder (its name is the run)
    report = load_bidding_report(run_dir)
    market = load_market()
    step = timedelta(hours=market.interval_h)
    runs = {}
    for summary_file in sorted((run_dir / "backtests").glob(f"*/{RunFiles.SUMMARY}")):
        summary, frame = load_run(summary_file.parent)
        offers_file = summary_file.parent / RunFiles.OFFERS
        offers = pl.read_parquet(offers_file) if offers_file.exists() else None
        runs[summary.strategy] = (summary, frame.sort(Col.INTERVAL_START), offers)
    order = [s for s in LABELS if s in runs]
    wv_summary, wv_frame, _ = runs[order[0]]
    start = wv_frame[0, Col.INTERVAL_START]
    n = wv_frame.height

    cols = {p: price_column(p, market) for p in ANCILLARY_PRODUCTS}
    reserves = any(
        frame[c].abs().sum() > 0
        for _, frame, _ in runs.values()
        for c in (Col.REG_MW, Col.PRI_MW, Col.CON_MW)
    )
    source = wv_frame.select(
        Col.INTERVAL_START, *(c for c in cols.values() if c in wv_frame.columns)
    )
    if reserves and len(source.columns) < len(cols) + 1:
        loader = realised or _realised_prices(market)
        stamps = wv_frame[Col.INTERVAL_START]
        loaded = loader(stamps.min().date(), stamps.max().date())
        source = wv_frame.select(Col.INTERVAL_START).join(
            loaded.select(Col.INTERVAL_START, *(c for c in cols.values() if c in loaded.columns)),
            on=Col.INTERVAL_START,
            how="left",
        )
    prices = {str(p): _round(source[c], 2) for p, c in cols.items() if c in source.columns}

    dispatch, offers_out, totals = {}, {}, {}
    for strategy in order:
        summary, frame, offers = runs[strategy]
        dispatch[str(strategy)] = {
            "charge": _round(frame[Col.CHARGE_MW], 1),
            "discharge": _round(frame[Col.DISCHARGE_MW], 1),
            "soc": _round(frame[Col.SOC_MWH], 2),
            "reg": _round(frame[Col.REG_MW], 1),
            "pri": _round(frame[Col.PRI_MW], 1),
            "con": _round(frame[Col.CON_MW], 1),
            "revenue": _round(frame[Col.REVENUE_SGD], 0),
            "ancillary": _round(frame[Col.ANCILLARY_SGD], 0),
            "degradation": _round(frame[Col.DEGRADATION_SGD], 0),
            "penalty": _round(frame[Col.PENALTY_SGD], 0),
            "profit": _round(frame[Col.PROFIT_SGD], 0),
            "cut": _round(
                frame[Col.SOC_CUT_MW] if Col.SOC_CUT_MW in frame.columns else [0.0] * n, 1
            ),
            "reg_net": _round(
                frame[Col.REG_NET_MWH] if Col.REG_NET_MWH in frame.columns else [0.0] * n, 2
            ),
        }
        t = summary.total
        totals[str(strategy)] = {
            "profit": round(t.profit_sgd),
            "energy": round(t.revenue_sgd - t.degradation_sgd),
            "reserves": round(t.ancillary_sgd),
            "cycles_per_day": round(summary.cycles_per_day, 2),
            "cut_mwh": round(t.soc_cut_mwh),
        }
        if offers is not None and offers.height:
            offers_out[str(strategy)] = {
                "i": _index(offers[Col.INTERVAL_START], start, step),
                "product": [PRODUCT_CODE[Product(p)] for p in offers[Col.PRODUCT].to_list()],
                "side": [SIDE_CODE.get(s, 0) for s in offers[Col.SIDE].to_list()],
                "price": _round(offers[Col.OFFER_PRICE], 2),
                "mw": _round(offers[Col.OFFER_MW], 1),
            }

    decisions_file = run_dir / "decisions.parquet"
    decisions, states = {}, 0
    if decisions_file.exists():
        trace = pl.read_parquet(decisions_file).sort(Col.INTERVAL_START)
        idx = _index(trace[Col.INTERVAL_START], start, step)
        states = int(trace[Col.STATE_PRICES].list.len().max())
        full = {k: [None] * n for k in ("decision_soc", "expected_soc", "preg", "ppri", "pcon")}
        issue = [None] * n
        flat = [None] * (n * states)
        rows = trace.iter_rows(named=True)
        for i, row in zip(idx, rows, strict=True):
            if not 0 <= i < n:
                continue
            full["decision_soc"][i] = round(row[Col.DECISION_SOC_MWH], 2)
            full["expected_soc"][i] = round(row[Col.EXPECTED_SOC_MWH], 2)
            full["preg"][i] = round(row[Col.PLANNED_REG_MW], 1)
            full["ppri"][i] = round(row[Col.PLANNED_PRI_MW], 1)
            full["pcon"][i] = round(row[Col.PLANNED_CON_MW], 1)
            issue[i] = row[Col.ISSUED_AT].strftime("%Y-%m-%d %H:%M")
            for k, p in enumerate(row[Col.STATE_PRICES]):
                flat[i * states + k] = round(p, 2)
        decisions = {**full, "issue": issue, "states": flat}

    return {
        "meta": {
            "run": run_dir.name,
            "forecast_run": run_dir.parents[2].name if len(run_dir.parents) > 2 else "",
            "start": start.strftime("%Y-%m-%d"),
            "intervals": n,
            "per_day": market.intervals_per_day,
            "interval_h": market.interval_h,
            "soc_init": wv_summary.battery.soc_init,
            "soc_max": wv_summary.battery.soc_max,
            "p_max": wv_summary.battery.p_max_mw,
            "price_floor": market.price_floor,
            "price_cap": market.price_cap,
            "states": states,
            "feasibility": report.feasibility,
            "reserve_planner": str(report.bidding.reserve_planner),
            "strategies": [{"key": str(s), "label": LABELS[s]} for s in order],
            "totals": totals,
            "gate": market.gate_closure_intervals,
            "plans": {
                "fields": [f for f, _ in PLAN_FIELDS],
                "scales": [v for _, v in PLAN_FIELDS],
                "horizon": market.intervals_per_day,
                "chunk": 7 * market.intervals_per_day,
                "dir": PLAN_DIR,
                "files": math.ceil(n / (7 * market.intervals_per_day)),
            }
            if _plan_strategies(run_dir)
            else None,
        },
        "usep": _round(wv_frame[Col.USEP], 2),
        "prices": prices,
        "dispatch": dispatch,
        "offers": offers_out,
        "decisions": decisions,
    }


def plan_chunk_name(k: int) -> str:
    return f"w{k:03d}.json"


def encode_chunk(part: np.ndarray) -> str:
    """``{"i16": base64 of little-endian int16}``: JSON so web hosts serve it (not raw binary)."""
    raw = np.ascontiguousarray(part).astype("<i2").tobytes()
    return json.dumps({"i16": base64.b64encode(raw).decode("ascii")})


def decode_chunk(text: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(json.loads(text)["i16"]), dtype="<i2")


def write_explorer(
    run_dir: Path,
    out: Path | None = None,
    quantiles: pl.DataFrame | None = None,
    ancillary_quantiles: pl.DataFrame | None = None,
) -> Path:
    """``explore.html`` (data embedded) in the run folder, or ``out``; with saved plans, the
    decision view's chunks in ``explore_plans/`` beside it (fetched: serve the folder over
    HTTP to use them locally)."""
    data = explorer_data(run_dir, quantiles=quantiles)
    text = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("/*__DATA__*/", text)
    out = Path(out) if out else Path(run_dir) / OUTPUT
    out.write_text(html)
    plans = data["meta"]["plans"]
    if plans:
        cube = plan_cube(run_dir, quantiles, ancillary_quantiles)
        folder = out.parent / PLAN_DIR
        folder.mkdir(exist_ok=True)
        size = plans["chunk"]
        for k in range(plans["files"]):
            (folder / plan_chunk_name(k)).write_text(encode_chunk(cube[k * size : (k + 1) * size]))
    return out
