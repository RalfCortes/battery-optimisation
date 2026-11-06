"""Settlement explorer: one self-contained HTML page from a saved bidding run."""

import json
import re

import numpy as np
import polars as pl
import pytest
from synthetic import daily_shape, make_prices

from battery_optimisation.columns import Col
from battery_optimisation.config import Product
from battery_optimisation.sim.clearing import price_column


def _saved_run(tmp_path, battery_cfg, market_cfg):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parents[1] / "integration"))
    from test_bidding import markov_bidding, quantiles_for

    from battery_optimisation.bidding.report import evaluate_bidding, save_bidding

    days = 2
    rng = np.random.default_rng(0)
    lead = 8  # history days (persistence forecasts of reserve prices look a week back)
    base = make_prices(daily_shape(days + lead, seed=4))
    prices = base.with_columns(
        pl.Series(price_column(p, market_cfg), rng.uniform(5, 30, base.height))
        for p in (Product.REGULATION, Product.PRIMARY_RESERVE, Product.CONTINGENCY_RESERVE)
    )
    history, window = prices.slice(0, 48 * lead), prices.slice(48 * lead)
    gap = market_cfg.gate_closure_intervals
    qf = quantiles_for(window.select(Col.INTERVAL_START, Col.USEP), 40.0, gap)
    qf.write_parquet(tmp_path / "quantiles.parquet")
    ev = evaluate_bidding(
        qf.head(0),
        qf,
        window,
        battery_cfg,
        market_cfg,
        markov_bidding(),
        history=history,
        ancillary=True,
    )
    return save_bidding(ev, tmp_path / "run"), window.height


def test_explorer_embeds_every_interval_offer_and_decision(tmp_path, battery_cfg, market_cfg):
    from battery_optimisation.bidding.explorer import explorer_data, write_explorer

    run_dir, n = _saved_run(tmp_path, battery_cfg, market_cfg)
    data = explorer_data(run_dir)
    assert data["meta"]["intervals"] == n
    assert len(data["usep"]) == n
    assert set(data["prices"]) == {
        "regulation",
        "primary_reserve",
        "contingency_reserve",
    }
    keys = [s["key"] for s in data["meta"]["strategies"]]
    assert keys == ["water_value", "chronos_median", "perfect_foresight"]
    for key in keys:
        assert len(data["dispatch"][key]["soc"]) == n
        offers = data["offers"][key]
        assert len(offers["i"]) == len(offers["price"]) == len(offers["mw"]) > 0
        assert max(offers["i"]) < n
    k = data["meta"]["states"]
    assert len(data["decisions"]["states"]) == n * k
    html = write_explorer(run_dir).read_text()
    assert "<title>" in html
    embedded = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S)
    assert json.loads(embedded.group(1))["meta"]["intervals"] == n


def test_explorer_fills_reserve_prices_missing_from_older_dispatch(
    tmp_path, battery_cfg, market_cfg
):
    """Runs saved before the dispatch kept realised reserve prices: take them from the prices."""
    from battery_optimisation.bidding.explorer import explorer_data

    run_dir, n = _saved_run(tmp_path, battery_cfg, market_cfg)
    cols = [
        price_column(p, market_cfg)
        for p in (Product.REGULATION, Product.PRIMARY_RESERVE, Product.CONTINGENCY_RESERVE)
    ]
    realised = None
    for f in (run_dir / "backtests").glob("*/dispatch.parquet"):
        frame = pl.read_parquet(f)
        realised = frame.select(Col.INTERVAL_START, *cols)
        frame.drop(cols).write_parquet(f)
    assert explorer_data(run_dir, realised=lambda start, end: realised)["prices"] == {
        "regulation": [round(v, 2) for v in realised[cols[0]]],
        "primary_reserve": [round(v, 2) for v in realised[cols[1]]],
        "contingency_reserve": [round(v, 2) for v in realised[cols[2]]],
    }
    assert len(realised) == n


def test_saved_run_keeps_every_strategy_plans(tmp_path, battery_cfg, market_cfg):
    from battery_optimisation.sim.backtest import RunFiles

    run_dir, n = _saved_run(tmp_path, battery_cfg, market_cfg)
    folders = sorted((run_dir / "backtests").iterdir())
    assert len(folders) == 3
    for folder in folders:
        plans = pl.read_parquet(folder / RunFiles.PLANS)
        assert plans[Col.INTERVAL_START].n_unique() == n  # daily horizon: one plan a day
        if folder.name.startswith("water_value"):
            assert plans[Col.TARGET_START].n_unique() == n  # one per decision


def test_explorer_ships_each_decisions_forecast_and_plans(tmp_path, battery_cfg, market_cfg):
    """Per offered interval, what was known at gate closure: the forecast fan of the issue used
    and every strategy's plan from that interval on, as weekly int16 chunks loaded on demand."""
    from battery_optimisation.bidding.explorer import (
        PLAN_DIR,
        decode_chunk,
        explorer_data,
        write_explorer,
    )
    from battery_optimisation.sim.backtest import RunFiles

    run_dir, n = _saved_run(tmp_path, battery_cfg, market_cfg)
    qf = pl.read_parquet(tmp_path / "quantiles.parquet")
    data = explorer_data(run_dir, quantiles=qf)
    plans = data["meta"]["plans"]
    fields, scales, h = plans["fields"], plans["scales"], plans["horizon"]
    out = write_explorer(run_dir, quantiles=qf)
    chunks = sorted((out.parent / PLAN_DIR).glob("*.json"))
    cube = np.concatenate([decode_chunk(c.read_text()) for c in chunks])
    cube = cube.reshape(n, len(fields), h)

    def value(i, field, k):
        raw = cube[i, fields.index(field), k]
        return None if raw == -32768 else raw / scales[fields.index(field)]

    wv = pl.read_parquet(next((run_dir / "backtests").glob("water_value*")) / RunFiles.PLANS)
    row = wv.filter(pl.col(Col.TARGET_START) == pl.col(Col.TARGET_START).min()).row(1, named=True)
    assert value(0, "wv_net", 1) == pytest.approx(row[Col.NET_MW], abs=0.06)
    assert value(0, "wv_soc", 1) == pytest.approx(row[Col.SOC_MWH], abs=0.06)
    # forecast median of the first interval, from the issue the decision used
    first = qf.filter(
        (pl.col(Col.INTERVAL_START) == pl.col(Col.INTERVAL_START).min())
        & ((pl.col(Col.QUANTILE) - 0.5).abs() < 1e-9)
    )[Col.USEP_FORECAST][0]
    assert value(0, "usep_q50", 0) == pytest.approx(first, abs=0.6)
    assert value(n - 1, "ss_net", h - 1) is None  # beyond the run
    assert data["meta"]["gate"] == market_cfg.gate_closure_intervals


def test_adding_the_sweep_keeps_the_plans(tmp_path, battery_cfg, market_cfg):
    """Regression: the CLI rebuilt the evaluation after the sweep and dropped the plans."""
    import sys
    from pathlib import Path

    from battery_optimisation.bidding.report import with_sweep

    sys.path.insert(0, str(Path(__file__).parents[1] / "integration"))
    from test_bidding import markov_bidding, quantiles_for

    from battery_optimisation.bidding.report import evaluate_bidding

    prices = make_prices(daily_shape(1, seed=4))
    qf = quantiles_for(prices, 40.0, market_cfg.gate_closure_intervals)
    ev = evaluate_bidding(qf.head(0), qf, prices, battery_cfg, market_cfg, markov_bidding())
    assert ev.plans
    swept = with_sweep(ev, None, [])
    assert swept.plans is ev.plans and swept.report.sweep_steps == []
