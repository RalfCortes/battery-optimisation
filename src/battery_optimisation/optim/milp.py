"""Battery dispatch MILP (energy only, for now), solved with HiGHS.

Used by ``bidding.strategy.ScheduleStrategy`` (perfect-foresight benchmark, self-schedule
baseline). Solves exactly the window it is given; planning windows are the strategy's business.

Variables per interval t (Δ = market.interval_h):
  c_t, d_t ∈ [0, P]       charge / discharge power at the grid connection (MW)
  u_t ∈ {0, 1}            1 = charging allowed, 0 = discharging allowed
  s_t ∈ [S_min, S_max]    state of charge at the END of interval t (MWh)

Constraints (coefficients from ``Battery``, the single source of the physics):
  c_t <= P u_t,  d_t <= P (1 - u_t)                         no simultaneous charge and discharge
  s_t = s_{t-1} + charge_gain(Δ) c_t - discharge_loss(Δ) d_t  s_{-1} = battery.soc_mwh
  s_{T-1} >= soc_terminal_min                               (default: battery.soc_mwh)

Objective (SGD), maximised:
  Σ_t price_t (d_t - c_t) Δ  -  κ (c_t + d_t) Δ  +  terminal_value · s_{T-1}
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import highspy
import numpy as np
import polars as pl

from battery_optimisation.battery import TOL, Battery
from battery_optimisation.columns import Col
from battery_optimisation.config import (
    ANCILLARY_PRODUCTS,
    LOWER_PRODUCTS,
    RAISE_PRODUCTS,
    MarketConfig,
    Product,
)
from battery_optimisation.sim.clearing import AWARD_COLUMN, check_prices, price_column


@dataclass(frozen=True)
class MilpSolution:
    schedule: pl.DataFrame  # Col.INTERVAL_START, USEP, CHARGE_MW, DISCHARGE_MW, NET_MW, SOC_MWH
    objective_sgd: float  # includes terminal_value * final SOC
    solve_time_s: float


def solve_schedule(
    prices: pl.DataFrame,
    battery: Battery,
    market: MarketConfig,
    soc_terminal_min: float | None = None,
    terminal_value: float = 0.0,
) -> MilpSolution:
    """Optimal schedule for ``prices`` (``Col.INTERVAL_START``, ``Col.USEP`` and, when present,
    the ancillary price columns of the enabled products), starting from ``battery.soc_mwh``.

    Ancillary services (co-optimised with energy) add per interval availability r_p in
    [0, max_award_p] paid price_p x r_p x dt, with
      raise power:   d - c + sum_p r_p <= P          lower power:  c - d + r_reg <= P
      sustain:       SOC at the start and end of the interval - soc_min >= sum_p r_p h_p / eta_d
                     soc_max - SOC >= r_reg h_reg eta_c
      wear:          kappa x regulation_utilisation x r_reg x dt
    """
    check_prices(prices, market)
    products = [p for p in market.ancillary_products if price_column(p, market) in prices.columns]
    prices = prices.select(
        Col.INTERVAL_START, Col.USEP, *[price_column(p, market) for p in products]
    ).sort(Col.INTERVAL_START)
    cfg, dt = battery.config, market.interval_h
    price = prices[Col.USEP].to_numpy()
    n, p_max = len(price), cfg.p_max_mw
    soc_end = battery.soc_mwh if soc_terminal_min is None else soc_terminal_min
    deg = battery.degradation_sgd(1.0, 0.0, dt)  # SGD per MW over one interval

    t0 = time.perf_counter()
    h = highspy.Highs()
    h.silent()
    c = h.addVariables(n, lb=0, ub=p_max, out_array=True)
    d = h.addVariables(n, lb=0, ub=p_max, out_array=True)
    s = h.addVariables(n, lb=cfg.soc_min, ub=cfg.soc_max, out_array=True)
    u = h.addBinaries(n, out_array=True)
    r = {
        p: h.addVariables(
            n, lb=0, ub=min(p_max, market.ancillary.max_award_mw.get(p, p_max)), out_array=True
        )
        for p in products
    }

    h.addConstrs(c <= p_max * u)
    h.addConstrs(d <= p_max - p_max * u)
    gain, loss = battery.charge_gain(dt), battery.discharge_loss(dt)
    h.addConstr(s[0] == battery.soc_mwh + gain * c[0] - loss * d[0])
    if n > 1:
        h.addConstrs(s[1:] == s[:-1] + gain * c[1:] - loss * d[1:])
    h.addConstr(s[n - 1] >= soc_end)

    objective = (
        h.qsum((price * dt - deg) * d) + h.qsum((-price * dt - deg) * c) + terminal_value * s[n - 1]
    )
    if products:
        raises = [p for p in products if p in RAISE_PRODUCTS]
        lowers = [p for p in products if p in LOWER_PRODUCTS]
        raise_mw = sum((r[p] for p in raises), start=0 * d)
        h.addConstrs(d - c + raise_mw <= p_max)
        if lowers:
            h.addConstrs(c - d + sum((r[p] for p in lowers), start=0 * c) <= p_max)
        raise_energy = sum((r[p] * (market.sustain_h(p) / cfg.eta_d) for p in raises), start=0 * d)
        lower_energy = sum((r[p] * (market.sustain_h(p) * cfg.eta_c) for p in lowers), start=0 * d)
        h.addConstrs(s - raise_energy >= cfg.soc_min)  # end of each interval
        h.addConstrs(s + lower_energy <= cfg.soc_max)
        h.addConstr(raise_energy[0] <= battery.soc_mwh - cfg.soc_min)  # start of the first
        h.addConstr(lower_energy[0] <= cfg.soc_max - battery.soc_mwh)
        if n > 1:  # start of the others = end of the previous
            h.addConstrs(s[:-1] - raise_energy[1:] >= cfg.soc_min)
            h.addConstrs(s[:-1] + lower_energy[1:] <= cfg.soc_max)
        wear = battery.degradation_sgd(0.0, market.ancillary.regulation_utilisation, dt)
        for p in products:
            pay = prices[price_column(p, market)].to_numpy() * dt
            objective = objective + h.qsum(
                (pay - (wear if p is Product.REGULATION else 0.0)) * r[p]
            )
    h.maximize(objective)
    status = h.getModelStatus()
    if status != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"MILP not optimal: {h.modelStatusToString(status)}")
    solve_time = time.perf_counter() - t0

    # Clean solver noise so downstream checks see exact zeros.
    cv, dv = h.vals(c), h.vals(d)
    cv, dv = np.where(cv > TOL, cv, 0.0), np.where(dv > TOL, dv, 0.0)
    awards = {
        AWARD_COLUMN[p]: np.where(h.vals(r[p]) > TOL, h.vals(r[p]), 0.0)
        for p in ANCILLARY_PRODUCTS
        if p in r
    }
    schedule = prices.with_columns(
        pl.Series(Col.CHARGE_MW, cv),
        pl.Series(Col.DISCHARGE_MW, dv),
        pl.Series(Col.NET_MW, dv - cv),
        pl.Series(Col.SOC_MWH, h.vals(s)),
        *[pl.Series(col, v) for col, v in awards.items()],
        *[pl.lit(0.0).alias(AWARD_COLUMN[p]) for p in ANCILLARY_PRODUCTS if p not in r],
    )
    return MilpSolution(schedule, h.getObjectiveValue(), solve_time)
