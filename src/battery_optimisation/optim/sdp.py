"""Stochastic dynamic programming over a SOC grid: value function and water value.

Stages t = 0..T-1 are the intervals of a delivery day. Before interval t the price is unknown;
it is one of the step's K price states. The battery observes it and moves between SOC grid
points (one net action per stage) within its power limits. Per state k at stage t:

    V_t(s, k) = max_{s'} [price_{t,k} sold(s -> s') - degradation(s -> s') + C_t(s', k)]
    C_t(s', k) = sum_j P(j | k) V_{t+1}(s', j)          (continuation; C_{T-1} = terminal)

Without ``transitions`` the states are drawn independently each stage with ``weights``
(stagewise marginals: P(j | k) = w_j, so C_t(s', k) = V_{t+1}(s') for every k). With a
``transitions`` matrix the price persists (Markov, ``forecast.markov``): a high price now means
high prices next, which removes the fake spreads of independent stages. The reported
V_t(s) = sum_k pi_t(k) V_t(s, k), pi_0 = weights, pi_{t+1} = pi_t P.

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
    # Markov prices only: state prices [T, K] (increasing in k) and C_t(s', k) [T, K, N]
    state_prices: np.ndarray | None = None
    continuation: np.ndarray | None = None

    def continuation_at(self, t: int, prices: np.ndarray) -> np.ndarray:
        """Value after interval ``t`` [P, N] if its price turns out to be each of ``prices``:
        V_{t+1} for independent stages; for Markov prices, C_t interpolated between the states
        by price (constant beyond the extreme states)."""
        prices = np.atleast_1d(np.asarray(prices, float))
        if self.continuation is None:
            return np.broadcast_to(self.values[t + 1], (prices.size, self.soc.size))
        states = self.state_prices[t] + 1e-9 * np.arange(self.state_prices.shape[1])
        position = np.interp(prices, states, np.arange(states.size, dtype=float))
        lo = np.floor(position).astype(int)
        hi = np.minimum(lo + 1, states.size - 1)
        f = (position - lo)[:, None]
        cont = self.continuation[t]
        return (1 - f) * cont[lo] + f * cont[hi]

    def at(self, t: int, soc: float) -> float:
        return float(np.interp(soc, self.soc, self.values[t]))

    def water_value(self, t: int) -> np.ndarray:
        """Slope of V_t between consecutive grid points [N - 1] (SGD/MWh of SOC)."""
        return np.diff(self.values[t]) / np.diff(self.soc)


@dataclass(frozen=True)
class StageLimits:
    """Per-stage limits left by ancillary commitments (arrays of length T)."""

    max_discharge_mw: np.ndarray  # P - raise reserves held
    max_charge_mw: np.ndarray  # P - lower reserves held
    soc_low: np.ndarray  # SOC to hold at the start and end of the stage (raise energy)
    soc_high: np.ndarray  # (lower energy)


def terminal_values(soc: np.ndarray, target: float, penalty: float) -> np.ndarray:
    return -penalty * np.maximum(target - soc, 0.0)


def solve_sdp(
    prices: np.ndarray,
    weights: np.ndarray,
    battery: Battery,
    dt: float,
    cfg: BiddingConfig,
    soc_target: float | None = None,
    limits: StageLimits | None = None,
    transitions: np.ndarray | None = None,
) -> ValueFunction:
    """Backward recursion. ``prices`` [T, K] state (scenario) values per stage, ``weights`` [K];
    ``transitions`` [K, K] (rows: from state) for Markov prices, else stagewise independent.

    ``limits``: per-stage power caps (hard) and SOC band (penalised like the terminal rule, at
    the start of the stage and at its end) left by ancillary commitments."""
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
    penalty = cfg.terminal_penalty_sgd_per_mwh

    if transitions is not None:  # per-state values, state occupancy pi_t
        transitions = np.asarray(transitions, float)
        state_values = np.broadcast_to(values[-1], (prices.shape[1], soc.size))
        continuation = np.empty((n_stages, prices.shape[1], soc.size))
        occupancy = np.empty((n_stages, prices.shape[1]))
        occupancy[0] = weights
        for t in range(1, n_stages):
            occupancy[t] = occupancy[t - 1] @ transitions

    def outside(lo: float, hi: float) -> np.ndarray:
        return -penalty * (np.maximum(lo - soc, 0.0) + np.maximum(soc - hi, 0.0))

    for t in range(n_stages - 1, -1, -1):
        ok, end_band, start_band = feasible, 0.0, 0.0
        if limits is not None:
            ok = (
                feasible
                & (discharge <= limits.max_discharge_mw[t] + TOL)
                & (charge <= limits.max_charge_mw[t] + TOL)
            )
            if not ok.any(axis=1).all():  # keep at least "stay" (penalised by the band)
                ok = ok | np.eye(soc.size, dtype=bool)
            lo, hi = limits.soc_low[t], limits.soc_high[t]
            end_band = outside(lo, hi)[None, :]  # end of stage t
            start_band = outside(lo, hi)  # start of stage t
        if transitions is None:
            base = np.where(ok, values[t + 1][None, :] - degradation + end_band, -np.inf)  # [N, N]
            best = (prices[t][:, None, None] * sold[None] + base[None]).max(axis=2)  # [K, N]
            values[t] = weights @ best + start_band
            continue
        cont = transitions @ state_values if t < n_stages - 1 else state_values  # [K, N]
        continuation[t] = cont
        base = np.where(ok[None], cont[:, None, :] - degradation[None] + end_band, -np.inf)
        state_values = (prices[t][:, None, None] * sold[None] + base).max(axis=2) + start_band
        values[t] = occupancy[t] @ state_values
    if transitions is None:
        return ValueFunction(soc=soc, values=values, solve_time_s=time.perf_counter() - t0)
    return ValueFunction(
        soc=soc,
        values=values,
        solve_time_s=time.perf_counter() - t0,
        state_prices=np.asarray(prices, float),
        continuation=continuation,
    )
