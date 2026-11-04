"""Parametric price sweep: does the offer curve give the optimal response? (CLAUDE.md check)

For an interval t, fix its price to each value on a grid and re-solve the stage: with
stagewise-independent prices V_{t+1} does not depend on the price at t, so the optimum is

    max over reachable s'  [ price * sold(s -> s') - degradation(s -> s') + V_{t+1}(s') ]

(V_{t+1} is piecewise linear, so the optimum is at a grid point or a power limit). Compare with
the net MW the submitted offers clear at that price (market rules). Uncompressed curves match
exactly; gaps on real offers measure compression to K tranches, the price-band trim and the
gate-closure feasibility caps.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from pydantic import BaseModel

from battery_optimisation.battery import TOL, Battery
from battery_optimisation.bidding.strategy import CurveStep, WaterValueStrategy
from battery_optimisation.columns import Col
from battery_optimisation.optim.sdp import ValueFunction
from battery_optimisation.sim.clearing import IntervalOffers, Side

MATCH_MW = 1.0  # |offered - optimal| below this counts as matching


def optimal_net_mw(
    vf: ValueFunction, t: int, soc: float, battery: Battery, dt: float, prices: np.ndarray
) -> np.ndarray:
    """Optimal net MW (discharge - charge) in interval ``t`` from ``soc`` for each price."""
    here = Battery(battery.config, soc)
    lo = soc - here.max_discharge_mw(dt) * here.discharge_loss(dt)
    hi = soc + here.max_charge_mw(dt) * here.charge_gain(dt)
    grid = vf.soc[(vf.soc > lo + TOL) & (vf.soc < hi - TOL)]
    target = np.unique(np.concatenate([[lo, soc, hi], grid]))
    discharge = np.where(target < soc, (soc - target) / here.discharge_loss(dt), 0.0)
    charge = np.where(target > soc, (target - soc) / here.charge_gain(dt), 0.0)
    net = discharge - charge
    base = np.interp(target, vf.soc, vf.values[t + 1]) - here.degradation_sgd(charge, discharge, dt)
    value = np.asarray(prices)[:, None] * net[None, :] * dt + base[None, :]  # [prices, targets]
    return net[value.argmax(axis=1)]


def offered_net_mw(offers: IntervalOffers, prices: np.ndarray) -> np.ndarray:
    """Net MW the offers clear at each price (discharge if offer <= price, charge if bid >=)."""
    prices = np.asarray(prices)
    out = np.zeros(prices.shape)
    for t in offers.on(Side.DISCHARGE):
        out += np.where(t.price <= prices, t.quantity_mw, 0.0)
    for t in offers.on(Side.CHARGE):
        out -= np.where(t.price >= prices, t.quantity_mw, 0.0)
    return out


class SweepStats(BaseModel):
    intervals: int
    prices: int
    price_min: float
    price_max: float
    mean_abs_gap_mw: float  # |offered - optimal|, all intervals x prices
    mean_abs_gap_in_band_mw: float  # same, prices inside each interval's scenario range only
    share_matching: float  # share of (interval, price) with gap < MATCH_MW
    max_gap_mw: float


def sweep(strategy: WaterValueStrategy, prices: np.ndarray) -> tuple[SweepStats, pl.DataFrame]:
    """Sweep every interval the strategy offered for. Per-interval frame: mean gaps."""
    dt = strategy.market.interval_h
    battery = Battery(strategy.battery)
    margin = strategy.bidding.price_band_margin_sgd_per_mwh
    rows, all_gaps, band_gaps = [], [], []
    for at, offers in sorted(strategy.submitted.items()):
        issue, t = strategy.issue_of[at]
        vf = strategy.value_functions[issue]
        soc = strategy.reference_soc[at]
        gap = np.abs(
            offered_net_mw(offers, prices) - optimal_net_mw(vf, t, soc, battery, dt, prices)
        )
        stage = strategy.stage_prices(issue)[t]
        in_band = (prices >= stage.min() - margin) & (prices <= stage.max() + margin)
        all_gaps.append(gap)
        band_gaps.append(gap[in_band])
        rows.append(
            {
                Col.INTERVAL_START: at,
                "mean_gap_mw": float(gap.mean()),
                "mean_gap_in_band_mw": float(gap[in_band].mean()) if in_band.any() else 0.0,
            }
        )
    gaps = np.concatenate(all_gaps)
    in_band = np.concatenate(band_gaps)
    stats = SweepStats(
        intervals=len(rows),
        prices=len(prices),
        price_min=float(np.min(prices)),
        price_max=float(np.max(prices)),
        mean_abs_gap_mw=float(gaps.mean()),
        mean_abs_gap_in_band_mw=float(in_band.mean()) if in_band.size else 0.0,
        share_matching=float((gaps < MATCH_MW).mean()),
        max_gap_mw=float(gaps.max()),
    )
    return stats, pl.DataFrame(rows)


class StepGap(BaseModel):
    step: CurveStep
    mean_abs_gap_mw: float
    mean_abs_gap_in_band_mw: float


def sweep_breakdown(strategy: WaterValueStrategy, prices: np.ndarray) -> list[StepGap]:
    """Gap to the optimal response after each pipeline step, replaying every interval with the
    same inputs it was offered with (decision SOC, reference SOC, pending offers)."""
    dt = strategy.market.interval_h
    battery = Battery(strategy.battery)
    margin = strategy.bidding.price_band_margin_sgd_per_mwh
    gaps: dict[CurveStep, list[np.ndarray]] = {s: [] for s in CurveStep}
    band: dict[CurveStep, list[np.ndarray]] = {s: [] for s in CurveStep}
    for at in sorted(strategy.submitted):
        issue, t = strategy.issue_of[at]
        reference = strategy.reference_soc[at]
        pending = [(p, strategy.submitted[p]) for p in strategy.pending_slots[at]]
        steps = strategy.curve_steps(at, strategy.decision_soc[at], reference, pending)
        best = optimal_net_mw(strategy.value_functions[issue], t, reference, battery, dt, prices)
        stage = strategy.stage_prices(issue)[t]
        in_band = (prices >= stage.min() - margin) & (prices <= stage.max() + margin)
        for step, offers in steps.items():
            gap = np.abs(offered_net_mw(offers, prices) - best)
            gaps[step].append(gap)
            band[step].append(gap[in_band])
    return [
        StepGap(
            step=step,
            mean_abs_gap_mw=float(np.concatenate(gaps[step]).mean()),
            mean_abs_gap_in_band_mw=float(np.concatenate(band[step]).mean()),
        )
        for step in CurveStep
    ]
