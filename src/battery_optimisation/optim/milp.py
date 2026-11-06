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
    MarketConfig,
    Product,
)
from battery_optimisation.sim.clearing import AWARD_COLUMN, check_prices, price_column


def _reserve_variables(h, n: int, products, battery: Battery, market: MarketConfig) -> dict:
    p_max = battery.config.p_max_mw
    return {
        p: h.addVariables(
            n, lb=0, ub=min(p_max, market.ancillary.max_award_mw.get(p, p_max)), out_array=True
        )
        for p in products
    }


def _path(
    h,
    price: np.ndarray,
    battery: Battery,
    market: MarketConfig,
    r: dict,
    firm_reserves: bool,
    soc_end: float,
    terminal_value: float,
    binaries: bool = True,
):
    """Energy variables and constraints for one price path beside the reserves ``r`` (shared
    across paths in the stochastic plan). Returns (objective, c, d, s): energy revenue,
    degradation, regulation energy settled on this path and terminal value."""
    cfg, dt = battery.config, market.interval_h
    n, p_max = len(price), cfg.p_max_mw
    deg = battery.degradation_sgd(1.0, 0.0, dt)  # SGD per MW over one interval
    c = h.addVariables(n, lb=0, ub=p_max, out_array=True)
    d = h.addVariables(n, lb=0, ub=p_max, out_array=True)
    s = h.addVariables(n, lb=cfg.soc_min, ub=cfg.soc_max, out_array=True)
    # relaxed (stochastic plan option): u in [0, 1] allows charging while discharging
    if binaries:
        u = h.addBinaries(n, out_array=True)
    else:
        u = h.addVariables(n, lb=0, ub=1, out_array=True)
    h.addConstrs(c <= p_max * u)
    h.addConstrs(d <= p_max - p_max * u)
    gain, loss = battery.charge_gain(dt), battery.discharge_loss(dt)
    reg_energy = market.ancillary.regulation_energy
    move = gain * c - loss * d
    if Product.REGULATION in r:  # expected energy AGC moves around the schedule
        drift = reg_energy.down * cfg.eta_c - reg_energy.up / cfg.eta_d
        move = move + r[Product.REGULATION] * (drift * dt)
    h.addConstr(s[0] == battery.soc_mwh + move[0])
    if n > 1:
        h.addConstrs(s[1:] == s[:-1] + move[1:])
    h.addConstr(s[n - 1] >= soc_end)
    objective = (
        h.qsum((price * dt - deg) * d) + h.qsum((-price * dt - deg) * c) + terminal_value * s[n - 1]
    )
    if not r:
        return objective, c, d, s
    rules = market.storage_rules

    def group_mw(group: list[Product]):
        return sum((r[p] for p in group if p in r), start=0 * d)

    up, down = (d, c) if firm_reserves else (d - c, c - d)
    for group in rules.raise_power_groups:
        if any(p in r for p in group):
            h.addConstrs(up + group_mw(group) <= p_max)
    for group in rules.lower_power_groups:
        if any(p in r for p in group):
            h.addConstrs(down + group_mw(group) <= p_max)
    start = battery.soc_mwh
    for check in rules.discharge_checks:
        he = check.get(Product.ENERGY, 0.0)
        held = sum((r[p] * check.get(p, 0.0) for p in r), start=0 * d)
        draw = (d * he + held) * (1 / cfg.eta_d)
        if not firm_reserves:  # charging refills while the reserves are held
            draw = draw - c * (he * cfg.eta_c)
        h.addConstr(draw[0] <= start - cfg.soc_min)
        if n > 1:
            h.addConstrs(s[:-1] - draw[1:] >= cfg.soc_min)
    for check in rules.charge_checks:
        he = check.get(Product.ENERGY, 0.0)
        held = sum((r[p] * check.get(p, 0.0) for p in r), start=0 * d)
        fill = (c * he + held) * cfg.eta_c
        if not firm_reserves:
            fill = fill - d * (he / cfg.eta_d)
        h.addConstr(fill[0] <= cfg.soc_max - start)
        if n > 1:
            h.addConstrs(s[:-1] + fill[1:] <= cfg.soc_max)
    if Product.REGULATION in r:  # regulation energy settled at this path's price
        settled = price * dt * (reg_energy.up - reg_energy.down)
        objective = objective + h.qsum(settled * r[Product.REGULATION])
    return objective, c, d, s


def _availability(h, prices: pl.DataFrame, r: dict, battery: Battery, market: MarketConfig):
    """Availability payments (minus regulation wear) of the reserves ``r``."""
    dt = market.interval_h
    wear = battery.degradation_sgd(0.0, market.ancillary.regulation_energy.throughput, dt)
    objective = 0.0
    for p, v in r.items():
        pay = prices[price_column(p, market)].to_numpy() * dt
        if p is Product.REGULATION:
            pay = pay - wear
        objective = objective + h.qsum(pay * v)
    return objective


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
    firm_reserves: bool = False,
) -> MilpSolution:
    """Optimal schedule for ``prices`` (``Col.INTERVAL_START``, ``Col.USEP`` and, when present,
    the ancillary price columns of the enabled products), starting from ``battery.soc_mwh``.

    Ancillary services (co-optimised with energy) add per interval availability r_p in
    [0, max_award_p] paid price_p x r_p x dt, within ``market.storage_rules``
    (``sim.storage_rules``; NEMS: power per reserve class, RC386 SoC checks from the SOC at the
    start of each interval):
      raise power, per group g:  d - c + sum_{p in g} r_p <= P
      lower power, per group g:  c - d + sum_{p in g} r_p <= P
      discharge checks k:        s_{t-1} - (d h_Ek + sum_p r_p h_pk) / eta_d + c h_Ek eta_c >= S_min
      charge checks k:           s_{t-1} + (c h_Ek + sum_p r_p h_pk) eta_c - d h_Ek / eta_d <= S_max
      regulation energy (``ancillary.regulation_energy``, expected): SoC moves by
        r_reg dt (down eta_c - up / eta_d), revenue price r_reg dt (up - down), wear
        kappa (up + down) r_reg dt
    ``firm_reserves``: reserves must not rely on the opposite energy flow (when the energy part
    is bid, not scheduled): raise: d + sum_g r_p <= P, lower: c + sum_g r_p <= P, and the SoC
    checks give no credit for the opposite flow (no -c term in draws, no -d term in fills).
    """
    check_prices(prices, market)
    products = [p for p in market.ancillary_products if price_column(p, market) in prices.columns]
    prices = prices.select(
        Col.INTERVAL_START, Col.USEP, *[price_column(p, market) for p in products]
    ).sort(Col.INTERVAL_START)
    price = prices[Col.USEP].to_numpy()
    n = len(price)
    soc_end = battery.soc_mwh if soc_terminal_min is None else soc_terminal_min

    t0 = time.perf_counter()
    h = highspy.Highs()
    h.silent()
    r = _reserve_variables(h, n, products, battery, market)
    objective, c, d, s = _path(h, price, battery, market, r, firm_reserves, soc_end, terminal_value)
    objective = objective + _availability(h, prices, r, battery, market)
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


def solve_reserve_plan(
    prices: pl.DataFrame,
    paths: np.ndarray,
    weights: np.ndarray,
    battery: Battery,
    market: MarketConfig,
    soc_terminal_min: float | None = None,
    firm_reserves: bool = False,
    relax: bool = False,
) -> MilpSolution:
    """Two-stage stochastic plan: reserves r (one per interval and product, decided now) shared
    by every energy price path [S, T] (``paths``, probability ``weights``), each with its own
    energy schedule (anticipative within the path). Maximises the expected profit. ``prices``:
    ``Col.INTERVAL_START`` and the ancillary price columns (energy comes from ``paths``).
    Schedule: the reserves, and the expected price, flows and SOC over the paths. ``relax``: no
    binaries (LP, ~6x faster; only the reserves are used: the paths' energy is never
    dispatched, but negative prices could make charging while discharging look worth it)."""
    products = [p for p in market.ancillary_products if price_column(p, market) in prices.columns]
    prices = prices.sort(Col.INTERVAL_START)
    paths = np.asarray(paths, float)
    weights = np.asarray(weights, float) / np.sum(weights)
    n = prices.height
    if paths.shape[1] != n:
        raise ValueError(f"paths cover {paths.shape[1]} intervals, prices {n}")
    soc_end = battery.soc_mwh if soc_terminal_min is None else soc_terminal_min

    t0 = time.perf_counter()
    h = highspy.Highs()
    h.silent()
    r = _reserve_variables(h, n, products, battery, market)
    objective = _availability(h, prices, r, battery, market)
    flows = []
    for w, path in zip(weights, paths, strict=True):
        obj, c, d, s = _path(
            h, path, battery, market, r, firm_reserves, soc_end, 0.0, binaries=not relax
        )
        objective = objective + w * obj
        flows.append((w, c, d, s))
    h.maximize(objective)
    status = h.getModelStatus()
    if status != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"stochastic MILP not optimal: {h.modelStatusToString(status)}")
    solve_time = time.perf_counter() - t0

    charge = sum(w * h.vals(c) for w, c, _, _ in flows)
    discharge = sum(w * h.vals(d) for w, _, d, _ in flows)
    soc = sum(w * h.vals(s) for w, _, _, s in flows)
    awards = {
        AWARD_COLUMN[p]: np.where(h.vals(r[p]) > TOL, h.vals(r[p]), 0.0)
        for p in ANCILLARY_PRODUCTS
        if p in r
    }
    schedule = prices.select(Col.INTERVAL_START).with_columns(
        pl.Series(Col.USEP, weights @ paths),
        pl.Series(Col.CHARGE_MW, charge),
        pl.Series(Col.DISCHARGE_MW, discharge),
        pl.Series(Col.NET_MW, discharge - charge),
        pl.Series(Col.SOC_MWH, soc),
        *[pl.Series(col, v) for col, v in awards.items()],
        *[pl.lit(0.0).alias(AWARD_COLUMN[p]) for p in ANCILLARY_PRODUCTS if p not in r],
    )
    return MilpSolution(schedule, h.getObjectiveValue(), solve_time)
