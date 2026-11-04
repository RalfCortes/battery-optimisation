"""Stochastic dynamic programming over a SOC grid: value function and water value.

Stages t = 0..T-1 are the intervals of a delivery day. Before interval t the price is unknown;
it is drawn from that step's scenario values (weighted), independently of earlier steps
(stagewise marginals). The battery observes the price and moves between SOC grid points
(one net action per stage) within its power limits:

    V_t(s) = sum_k w_k max_{s'} [price_{t,k} sold(s -> s') - degradation(s -> s') + V_{t+1}(s')]

Terminal value (same rule as the self-schedule baseline, which must end the day at >= soc_init):
    V_T(s) = -terminal_penalty * max(0, soc_target - s)

The water value at t is dV_t/ds (SGD per MWh of SOC); offer prices come from V_{t+1}
(``bidding.water_value``). Physics (efficiencies, degradation, limits) come from ``Battery``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from battery_optimisation.battery import TOL, Battery
from battery_optimisation.config import BiddingConfig


@dataclass(frozen=True)
class ValueFunction:
    soc: np.ndarray  # [N] grid, MWh
    values: (
        np.ndarray
    )  # [T + 1, N]: values[t] = V_t (before acting in interval t); values[T] terminal
    solve_time_s: float

    def at(self, t: int, soc: float) -> float:
        return float(np.interp(soc, self.soc, self.values[t]))

    def water_value(self, t: int) -> np.ndarray:
        """Slope of V_t between consecutive grid points [N - 1] (SGD/MWh of SOC)."""
        return np.diff(self.values[t]) / np.diff(self.soc)


def terminal_values(soc: np.ndarray, target: float, penalty: float) -> np.ndarray:
    return -penalty * np.maximum(target - soc, 0.0)


def solve_sdp(
    prices: np.ndarray,
    weights: np.ndarray,
    battery: Battery,
    dt: float,
    cfg: BiddingConfig,
    soc_target: float | None = None,
) -> ValueFunction:
    """Backward recursion. ``prices`` [T, K] scenario values per stage, ``weights`` [K]."""
    t0 = time.perf_counter()
    b = battery.config
    weights = np.asarray(weights, float) / np.sum(weights)
    soc = np.linspace(b.soc_min, b.soc_max, cfg.soc_levels)
    move = soc[None, :] - soc[:, None]  # [from, to] MWh of SOC
    charge = np.where(move > 0, move / battery.charge_gain(dt), 0.0)  # MW
    discharge = np.where(move < 0, -move / battery.discharge_loss(dt), 0.0)  # MW
    feasible = (charge <= b.p_max_mw + TOL) & (discharge <= b.p_max_mw + TOL)
    sold = (discharge - charge) * dt  # MWh delivered to the grid (negative = bought)
    degradation = battery.degradation_sgd(charge, discharge, dt)

    n_stages = prices.shape[0]
    values = np.empty((n_stages + 1, soc.size))
    target = b.soc_init if soc_target is None else soc_target
    values[-1] = terminal_values(soc, target, cfg.terminal_penalty_sgd_per_mwh)
    for t in range(n_stages - 1, -1, -1):
        base = np.where(feasible, values[t + 1][None, :] - degradation, -np.inf)  # [N, N]
        best = (prices[t][:, None, None] * sold[None] + base[None]).max(axis=2)  # [K, N]
        values[t] = weights @ best
    return ValueFunction(soc=soc, values=values, solve_time_s=time.perf_counter() - t0)
