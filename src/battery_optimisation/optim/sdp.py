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

With ``reserve_reward`` [T, N, N] (SGD for holding the best reserve mix during stage t when
moving s -> s', ``optim.reserves``) the stage reward adds it: reserves are chosen inside the
stage, so V values the SOC they need.

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
from battery_optimisation.optim.kernels import stage_max


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
    occupancy: np.ndarray | None = None  # [T, K] probability of each price state per stage
    # [T, S, N] grid index of the SOC chosen in stage t, state s, from SOC grid point n; S = K
    # price states, or K x R joint (energy, reserve) states, s = k R + r
    policy: np.ndarray | None = None
    joint_occupancy: np.ndarray | None = None  # [T, S] with reserve states (else ``occupancy``)
    joint_prices: np.ndarray | None = None  # [T, S] energy price of each joint state

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
    reserve_reward: np.ndarray | None = None,
    reserve_transitions: np.ndarray | None = None,
    reserve_weights: np.ndarray | None = None,
) -> ValueFunction:
    """Backward recursion. ``prices`` [T, K] state (scenario) values per stage, ``weights`` [K];
    ``transitions`` [K, K] (rows: from state) for Markov prices, else stagewise independent.

    ``limits``: per-stage power caps (hard) and SOC band (penalised like the terminal rule, at
    the start of the stage and at its end) left by ancillary commitments.

    ``reserve_reward`` [T, R, N, N] with ``reserve_transitions`` [R, R]: reserve prices follow
    their own Markov chain (``forecast.markov.factor_lattice``), independent of the energy one;
    the state is (energy k, reserve r). ``continuation`` stays [T, K, N]: averaged over the
    reserve state's distribution (what an offer, made before the prices are known, faces)."""
    if reserve_reward is not None and np.ndim(reserve_reward) == 4:
        return _solve_joint(
            prices,
            weights,
            battery,
            dt,
            cfg,
            soc_target,
            limits,
            transitions,
            np.asarray(reserve_reward, float),
            reserve_transitions,
            reserve_weights,
        )
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
    if reserve_reward is not None:
        reserve_reward = np.asarray(reserve_reward, float)

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
        policy = np.empty((n_stages, prices.shape[1], soc.size), dtype=np.int16)
        occupancy[0] = weights
        for t in range(1, n_stages):
            occupancy[t] = occupancy[t - 1] @ transitions

    def outside(lo: float, hi: float) -> np.ndarray:
        return -penalty * (np.maximum(lo - soc, 0.0) + np.maximum(soc - hi, 0.0))

    for t in range(n_stages - 1, -1, -1):
        ok, end_band, start_band = feasible, 0.0, 0.0
        cost = degradation if reserve_reward is None else degradation - reserve_reward[t]
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
            base = np.where(ok, values[t + 1][None, :] - cost + end_band, -np.inf)  # [N, N]
            best = (prices[t][:, None, None] * sold[None] + base[None]).max(axis=2)  # [K, N]
            values[t] = weights @ best + start_band
            continue
        cont = transitions @ state_values if t < n_stages - 1 else state_values  # [K, N]
        continuation[t] = cont
        base = np.where(ok[None], cont[:, None, :] - cost[None] + end_band, -np.inf)
        total = prices[t][:, None, None] * sold[None] + base  # [K, N, N]
        choice = total.argmax(axis=2)
        policy[t] = choice
        state_values = np.take_along_axis(total, choice[..., None], axis=2)[..., 0] + start_band
        values[t] = occupancy[t] @ state_values
    if transitions is None:
        return ValueFunction(soc=soc, values=values, solve_time_s=time.perf_counter() - t0)
    return ValueFunction(
        soc=soc,
        values=values,
        solve_time_s=time.perf_counter() - t0,
        state_prices=np.asarray(prices, float),
        continuation=continuation,
        occupancy=occupancy,
        policy=policy,
    )


def _solve_joint(
    prices, weights, battery, dt, cfg, soc_target, limits, transitions, reward, rt, rw
) -> ValueFunction:
    """``solve_sdp`` over joint (energy, reserve) price states."""
    if transitions is None or rt is None:
        raise ValueError("reserve states need Markov energy prices and reserve transitions")
    t0 = time.perf_counter()
    b = battery.config
    n_stages, k_states = prices.shape
    r_states = reward.shape[1]
    weights = np.asarray(weights, float) / np.sum(weights)
    rt = np.asarray(rt, float)
    rw = np.full(r_states, 1.0 / r_states) if rw is None else np.asarray(rw, float) / np.sum(rw)
    transitions = np.asarray(transitions, float)
    soc = np.linspace(b.soc_min, b.soc_max, cfg.soc_levels)
    move = soc[None, :] - soc[:, None]
    charge = np.where(move > 0, move / battery.charge_gain(dt), 0.0)
    discharge = np.where(move < 0, -move / battery.discharge_loss(dt), 0.0)
    feasible = (charge <= b.p_max_mw + TOL) & (discharge <= b.p_max_mw + TOL)
    sold = (discharge - charge) * dt
    degradation = battery.degradation_sgd(charge, discharge, dt)
    penalty = cfg.terminal_penalty_sgd_per_mwh
    target = b.soc_init if soc_target is None else soc_target
    values = np.empty((n_stages + 1, soc.size))
    values[-1] = terminal_values(soc, target, penalty)
    occ_e = np.empty((n_stages, k_states))
    occ_r = np.empty((n_stages, r_states))
    occ_e[0], occ_r[0] = weights, rw
    for t in range(1, n_stages):
        occ_e[t], occ_r[t] = occ_e[t - 1] @ transitions, occ_r[t - 1] @ rt
    state_values = np.broadcast_to(values[-1], (k_states, r_states, soc.size))
    continuation = np.empty((n_stages, k_states, soc.size))
    policy = np.empty((n_stages, k_states * r_states, soc.size), dtype=np.int16)

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
            if not ok.any(axis=1).all():
                ok = ok | np.eye(soc.size, dtype=bool)
            end_band = outside(limits.soc_low[t], limits.soc_high[t])[None, :]
            start_band = outside(limits.soc_low[t], limits.soc_high[t])
        if t < n_stages - 1:  # E[V_{t+1} | k, r] = sum_j P(j|k) sum_q P(q|r) V_{t+1}(j, q)
            cont = np.einsum("kj,jrn->krn", transitions, np.einsum("rq,jqn->jrn", rt, state_values))
        else:
            cont = state_values
        continuation[t] = np.einsum("r,krn->kn", occ_r[t], cont)
        extra = reward[t] + np.where(ok, -degradation + end_band, -np.inf)[None]  # [R, N, N]
        best, choice = stage_max(  # numba: max over the next SOC for every (k, r, n)
            np.ascontiguousarray(prices[t], dtype=float), sold, extra, np.ascontiguousarray(cont)
        )
        policy[t] = choice.reshape(k_states * r_states, soc.size)
        state_values = best + start_band
        values[t] = np.einsum("k,r,krn->n", occ_e[t], occ_r[t], state_values)
    return ValueFunction(
        soc=soc,
        values=values,
        solve_time_s=time.perf_counter() - t0,
        state_prices=np.asarray(prices, float),
        continuation=continuation,
        occupancy=occ_e,
        policy=policy,
        joint_occupancy=(occ_e[:, :, None] * occ_r[:, None, :]).reshape(n_stages, -1),
        joint_prices=np.repeat(np.asarray(prices, float), r_states, axis=1),
    )


@dataclass(frozen=True)
class ExpectedSchedule:
    """What the SDP policy plans from one stage on, over its price states (stage by stage)."""

    price: np.ndarray  # expected energy price, SGD/MWh
    net_mw: np.ndarray  # expected discharge - charge
    soc_mwh: np.ndarray  # expected SOC at the end of the stage
    soc_p10: np.ndarray  # 10% / 90% quantiles of that SOC
    soc_p90: np.ndarray
    reserves: np.ndarray | None = None  # [stages, P] expected MW held (``mixes``)


def _quantile(cdf: np.ndarray, soc: np.ndarray, q: float) -> float:
    return float(soc[min(np.searchsorted(cdf, q - 1e-9), soc.size - 1)])


def expected_schedule(
    vf: ValueFunction,
    transitions: np.ndarray,
    battery: Battery,
    dt: float,
    stage: int,
    soc_mwh: float,
    mixes: np.ndarray | None = None,
) -> ExpectedSchedule:
    """Roll the policy forward from ``soc_mwh`` at ``stage`` (split between its two grid
    neighbours) through the Markov price states (state probabilities: occupancy at ``stage``):
    exact distribution over (state, SOC grid point) at every later stage. ``mixes``
    [T - stage, K, N, P]: reserves held in stage ``stage + i``, state k, from grid point n
    (on the policy's move)."""
    if vf.policy is None or vf.occupancy is None:
        raise ValueError("expected_schedule needs a Markov SDP (policy and occupancy)")
    soc, policy = vf.soc, vf.policy
    occupancy = vf.occupancy if vf.joint_occupancy is None else vf.joint_occupancy
    state_prices = vf.state_prices if vf.joint_prices is None else vf.joint_prices
    n_states, n = policy.shape[1], soc.size
    move = soc[None, :] - soc[:, None]
    net = np.where(move < 0, -move / battery.discharge_loss(dt), -move / battery.charge_gain(dt))
    x = float(np.clip(soc_mwh, soc[0], soc[-1]))
    j = min(int(np.searchsorted(soc, x, side="right")) - 1, n - 2)
    f = (x - soc[j]) / (soc[j + 1] - soc[j])
    start = np.zeros(n)
    start[j], start[j + 1] = 1 - f, f
    p = occupancy[stage][:, None] * start[None, :]  # [S, N]
    rows = np.arange(n)[None, :]
    steps = policy.shape[0] - stage
    out = {k: np.empty(steps) for k in ("price", "net", "soc", "p10", "p90")}
    held = np.empty((steps, mixes.shape[-1])) if mixes is not None else None
    for i, t in enumerate(range(stage, policy.shape[0])):
        to = policy[t].astype(np.intp)  # [K, N]
        out["price"][i] = occupancy[t] @ state_prices[t]
        out["net"][i] = (p * net[rows, to]).sum()
        if held is not None:
            held[i] = (p[..., None] * mixes[i]).sum(axis=(0, 1))
        flat = np.bincount(
            (np.arange(n_states)[:, None] * n + to).ravel(), p.ravel(), n_states * n
        ).reshape(n_states, n)
        marginal = flat.sum(axis=0)
        cdf = np.cumsum(marginal) / marginal.sum()
        out["soc"][i] = marginal @ soc / marginal.sum()
        out["p10"][i], out["p90"][i] = _quantile(cdf, soc, 0.1), _quantile(cdf, soc, 0.9)
        p = transitions.T @ flat
    return ExpectedSchedule(
        price=out["price"],
        net_mw=out["net"],
        soc_mwh=out["soc"],
        soc_p10=out["p10"],
        soc_p90=out["p90"],
        reserves=held,
    )
