"""Bidding commands: water-value offer curves vs the self-schedule baseline.

Examples::

    uv run bidding run                         # on the latest forecast run (runs/forecast_*)
    uv run bidding run --forecast-run runs/forecast_chronos2_...
    uv run limited --mem-gb 100 --cpus 8 -- uv run bidding run   # under resource limits
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import typer

from battery_optimisation.bidding.explorer import PLAN_DIR, write_explorer
from battery_optimisation.bidding.report import (
    ComparisonInput,
    comparison_table,
    evaluate_bidding,
    list_bidding_runs,
    new_bidding_dir,
    save_bidding,
    to_markdown,
    with_sweep,
)
from battery_optimisation.bidding.sweep import sweep, sweep_breakdown
from battery_optimisation.columns import Col
from battery_optimisation.config import (
    AncillaryPrice,
    BiddingConfig,
    FeasibilityMode,
    PriceModel,
    ReserveOffers,
    ReservePlanner,
    ReservePrices,
    load_battery,
    load_bidding,
    load_market,
)
from battery_optimisation.forecast.report import (
    list_forecast_runs,
    load_forecast_run,
    schedule_horizon,
)
from battery_optimisation.logs import configure_logging
from battery_optimisation.sim.backtest import StrategyName, load_prices

app = typer.Typer(
    help="Water-value bidding backtests.", no_args_is_help=True, pretty_exceptions_enable=False
)


@app.callback()
def _root() -> None:
    """Water-value bidding backtests."""


def bidding_tag(bidding: BiddingConfig, ancillary: bool) -> str:
    """Output folder suffix: feasibility rule, ``_markov`` (+ ``_r<rho>`` if fixed) for Markov
    prices, then ``_as`` (+ ``_hybrid``) with ancillary."""
    f = bidding.feasibility
    tag = (
        f"prob_q{f.soc_quantile:g}_m{f.margin_pct:g}"
        if f.mode is FeasibilityMode.PROBABILISTIC
        else str(f.mode)
    )
    rho = "" if bidding.markov.rho is None else f"_r{bidding.markov.rho:g}"
    if bidding.price_model is PriceModel.MARKOV:
        tag += "_markov" + rho
    elif bidding.price_model is PriceModel.CONTINUOUS:
        tag += f"_continuous{bidding.continuous.nodes}" + rho
    if ancillary:
        tag += "_as" + ("" if bidding.firm_reserves else "_hybrid")
        if bidding.reserve_planner is ReservePlanner.SDP:
            tag += "_sdpres"
        elif bidding.reserve_planner is ReservePlanner.STOCHASTIC_MILP:
            tag += f"_stoch{bidding.stochastic_paths}" + ("lp" if bidding.stochastic_relax else "")
        if bidding.reserve_prices is ReservePrices.MARKOV:
            tag += f"_resmkv{bidding.reserve_markov.states}"
        elif bidding.ancillary_price is not AncillaryPrice.MEDIAN:
            tag += f"_anc{bidding.ancillary_price}"
        if bidding.reserve_offers is ReserveOffers.CURVE:
            tag += "_rcurve"
    return tag


@app.command()
def run(
    forecast_run: Annotated[
        Path | None, typer.Option(help="Forecast run folder; default: the latest.")
    ] = None,
    feasibility: Annotated[FeasibilityMode | None, typer.Option(help="Default: config.")] = None,
    margin_pct: Annotated[
        float | None, typer.Option(help="Probabilistic margin, % usable.")
    ] = None,
    soc_quantile: Annotated[float | None, typer.Option(help="Probabilistic SOC quantile.")] = None,
    ancillary: Annotated[
        bool, typer.Option(help="Regulation and reserves too (plan + energy curves).")
    ] = False,
    firm_reserves: Annotated[
        bool | None,
        typer.Option(help="Ancillary: reserves beside any energy flow. Default: config."),
    ] = None,
    price_model: Annotated[PriceModel | None, typer.Option(help="Default: config.")] = None,
    rho: Annotated[
        float | None, typer.Option(help="Markov lag-1 correlation. Default: config, else copula.")
    ] = None,
    reserve_planner: Annotated[
        ReservePlanner | None,
        typer.Option(help="Ancillary: who picks the reserves. Default: config."),
    ] = None,
    stochastic_paths: Annotated[
        int | None, typer.Option(help="Stochastic reserve plan: price paths. Default: config.")
    ] = None,
    stochastic_relax: Annotated[
        bool | None, typer.Option(help="Stochastic reserve plan: LP relaxation. Default: config.")
    ] = None,
    reserve_offers: Annotated[
        ReserveOffers | None,
        typer.Option(help="Water value: reserves at 0 (fixed) or priced curves. Default: config."),
    ] = None,
    reserve_prices: Annotated[
        ReservePrices | None,
        typer.Option(help="Water value: reserve prices as point forecasts or Markov states."),
    ] = None,
    ancillary_price: Annotated[
        AncillaryPrice | None,
        typer.Option(help="Water value: reserve price forecast (median or mean). Default: config."),
    ] = None,
    market_config: Annotated[
        Path | None,
        typer.Option(
            "--market", help="Market YAML (e.g. config/market_pooled.yaml). Default: market.yaml."
        ),
    ] = None,
) -> None:
    """Water-value offers vs self-schedule, price sweep check, comparison of all strategies."""
    configure_logging()
    data = load_bidding().model_dump()  # overrides re-validated like the YAML
    for key, value in (
        ("mode", feasibility),
        ("margin_pct", margin_pct),
        ("soc_quantile", soc_quantile),
    ):
        if value is not None:
            data["feasibility"][key] = value
    if firm_reserves is not None:
        data["firm_reserves"] = firm_reserves
    if price_model is not None:
        data["price_model"] = price_model
    if rho is not None:
        data["markov"]["rho"] = rho
    if reserve_planner is not None:
        data["reserve_planner"] = reserve_planner
    if stochastic_paths is not None:
        data["stochastic_paths"] = stochastic_paths
    if stochastic_relax is not None:
        data["stochastic_relax"] = stochastic_relax
    if ancillary_price is not None:
        data["ancillary_price"] = ancillary_price
    if reserve_prices is not None:
        data["reserve_prices"] = reserve_prices
    if reserve_offers is not None:
        data["reserve_offers"] = reserve_offers
    bidding = BiddingConfig.model_validate(data)
    fr = load_forecast_run(forecast_run or list_forecast_runs()[0])
    market = load_market(market_config) if market_config else load_market()
    start = fr.report.test_start
    prices = load_prices(start - timedelta(days=3), fr.report.test_end, market, ancillary)
    realised = prices.filter(pl.col(Col.INTERVAL_START).dt.date() >= start)
    history = prices.filter(pl.col(Col.INTERVAL_START).dt.date() < start)
    horizon = schedule_horizon(fr.report.config, market)
    ev = evaluate_bidding(
        fr.scenarios,
        fr.quantiles,
        realised,
        load_battery(),
        market,
        bidding,
        horizon,
        history=history,
        ancillary=ancillary,
        ancillary_quantiles=fr.ancillary_quantiles,
        copula_lag1={s.period_start: s.copula_lag1 for s in fr.report.scenarios},
    )
    if ancillary:
        source = "Chronos-2" if fr.ancillary_quantiles is not None else "persistence"
        typer.echo(f"Ancillary prices forecast by {source}")
    m = bidding.markov
    n_states = m.states if bidding.price_model is PriceModel.MARKOV else bidding.continuous.nodes
    typer.echo(
        f"Water value prices: {bidding.price_model}"
        + (
            f" ({n_states} states, rho "
            + ("walk-forward copula" if m.rho is None else f"{m.rho:g}")
            + ")"
            if bidding.price_model is not PriceModel.INDEPENDENT
            else ""
        )
    )
    grid = np.asarray(bidding.sweep.prices)
    stats, per_interval = sweep(ev.water_value, grid)
    steps = sweep_breakdown(ev.water_value, grid)
    ev = with_sweep(ev, stats, steps)
    tag = bidding_tag(bidding, ancillary)
    if market_config and market_config.stem != "market":
        tag += f"_{market_config.stem}"
    out = save_bidding(ev, new_bidding_dir(fr.path, tag))
    per_interval.write_parquet(out / "sweep.parquet")

    # every strategy over the same window: forecast-run backtests + the bidding run
    # (with ancillary services: this run's own benchmark and strategies only)
    rows = (
        []
        if ancillary
        else [
            ComparisonInput(
                strategy=c.strategy,
                horizon=c.horizon,
                profit_sgd=c.profit_sgd,
                cycles_per_day=c.cycles_per_day,
            )
            for c in fr.report.capture
        ]
    )
    if ancillary:
        rows += [
            ComparisonInput(
                strategy=name,
                horizon=summary.horizon,
                profit_sgd=summary.total.profit_sgd,
                cycles_per_day=summary.cycles_per_day,
            )
            for name, (summary, _) in ev.runs.items()
            if name is not StrategyName.WATER_VALUE
        ]
    wv = ev.runs[StrategyName.WATER_VALUE][0]
    rows.append(
        ComparisonInput(
            strategy=StrategyName.WATER_VALUE,
            horizon=horizon,
            profit_sgd=wv.total.profit_sgd,
            cycles_per_day=wv.cycles_per_day,
        )
    )
    table = comparison_table(rows, baseline=StrategyName.CHRONOS_MEDIAN)
    markdown = to_markdown(table)
    (out / "comparison.md").write_text(markdown)
    table.write_csv(out / "comparison.csv")

    r = ev.report
    typer.echo(f"\nBidding backtest {r.start} -> {r.end} (forecast run {fr.path.name})")
    typer.echo(
        f"gate closure {r.gate_closure_intervals} intervals, K = {r.n_tranches} tranches/side, "
        f"{r.sdp_days} daily SDPs in {r.sdp_solve_time_s:.0f} s\n"
    )
    typer.echo(markdown)
    with pl.Config(tbl_rows=-1, tbl_hide_dataframe_shape=True, thousands_separator=True):
        typer.echo(
            pl.DataFrame([m.model_dump() for m in r.monthly]).select(
                "period_start",
                *[
                    pl.col(c).round(0).cast(pl.Int64)
                    for c in ("self_schedule_sgd", "water_value_sgd", "uplift_sgd")
                ],
            )
        )
    typer.echo(
        f"\nBidding uplift {r.uplift_sgd:,.0f} SGD ({r.uplift_pct:+.1%} vs self-schedule)"
        f"\nFeasibility {r.feasibility}: shortfall {r.shortfall_mwh:,.1f} MWh, "
        f"penalty {r.penalty_sgd:,.0f} SGD (in profit)"
        f"\nOffers: {r.mean_tranches_per_interval:.1f} tranches/interval, "
        f"{r.price_taking_share:.0%} of offered MW price-taking"
        f"\nPrice sweep ({stats.prices} prices {stats.price_min:g}..{stats.price_max:g}, "
        f"{stats.intervals} intervals): mean |offered - optimal| {stats.mean_abs_gap_mw:.1f} MW "
        f"(within scenario band {stats.mean_abs_gap_in_band_mw:.1f} MW), "
        f"{stats.share_matching:.0%} within 1 MW, max {stats.max_gap_mw:.0f} MW"
    )
    typer.echo("\nGap to the optimal response after each step (mean MW, all prices / in band):")
    for s in steps:
        typer.echo(f"  {s.step:<13} {s.mean_abs_gap_mw:6.1f} / {s.mean_abs_gap_in_band_mw:6.1f}")
    typer.echo(f"\nSaved to {out}")


if __name__ == "__main__":
    app()


@app.command()
def explore(
    run: Annotated[
        Path | None,
        typer.Option(help="Saved bidding run folder; default: the newest one."),
    ] = None,
) -> None:
    """Settlement explorer: explore.html in the bidding run folder (+ explore_plans/: what was
    known at each gate closure, loaded on demand; serve the folder over HTTP to see it)."""
    configure_logging()
    if run is None:
        runs = list_bidding_runs(list_forecast_runs()[0])
        if not runs:
            raise typer.BadParameter("no saved bidding run yet: run `bidding run` first")
        run = runs[0]
    out = write_explorer(run)
    typer.echo(f"Explorer: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    plans = out.parent / PLAN_DIR
    if plans.exists():
        size = sum(f.stat().st_size for f in plans.iterdir()) / 1e6
        typer.echo(
            f"Decision view: {plans} ({size:.1f} MB); to open locally: "
            f"python -m http.server -d {out.parent} then http://localhost:8000/{out.name}"
        )
