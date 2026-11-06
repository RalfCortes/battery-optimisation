"""Compiled inner loops (numba) of the SDP with reserve price states.

Only the joint (energy, reserve) path uses them, so the default SDP keeps its numpy arithmetic
(and its results to the last digit). Threads: numba's pool, capped to the CPUs this process may
use (``limited --cpus`` sets the affinity; numba would otherwise start one thread per core).
"""

from __future__ import annotations

import os

import numba as nb
import numpy as np

nb.set_num_threads(max(1, min(nb.config.NUMBA_NUM_THREADS, len(os.sched_getaffinity(0)))))


@nb.njit(parallel=True, cache=True)
def stage_max(
    price: np.ndarray, sold: np.ndarray, extra: np.ndarray, cont: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """One SDP stage over joint states: for energy state k, reserve state r and SOC n,
    max over the next SOC m of price[k] sold[n, m] + extra[r, n, m] + cont[k, r, m]
    (``extra``: reserve reward - wear, -inf where infeasible). Returns (values [K, R, N],
    first argmax [K, R, N] as int16, like numpy's argmax)."""
    k_states, r_states, n = price.shape[0], extra.shape[0], sold.shape[0]
    values = np.empty((k_states, r_states, n))
    choice = np.empty((k_states, r_states, n), dtype=np.int16)
    for kr in nb.prange(k_states * r_states):
        k, r = kr // r_states, kr % r_states
        for i in range(n):
            best, arg = -np.inf, 0
            for m in range(n):
                v = price[k] * sold[i, m] + extra[r, i, m] + cont[k, r, m]
                if v > best:
                    best, arg = v, m
            values[k, r, i] = best
            choice[k, r, i] = arg
    return values, choice


@nb.njit(parallel=True, cache=True)
def pair_rewards(x: np.ndarray, ok: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Best reserve reward [T, N, N] over the valid corners x [N, N, V, P] for ``value``
    [T, P] (SGD/MW), at least 0 (holding nothing)."""
    n, corners, products = x.shape[0], x.shape[2], x.shape[3]
    stages = value.shape[0]
    out = np.zeros((stages, n, n))
    for i in nb.prange(n):
        for j in range(n):
            for t in range(stages):
                best = 0.0
                for v in range(corners):
                    if ok[i, j, v]:
                        s = 0.0
                        for p in range(products):
                            s += x[i, j, v, p] * value[t, p]
                        if s > best:
                            best = s
                out[t, i, j] = best
    return out


@nb.njit(parallel=True, cache=True)
def policy_mixes(
    x: np.ndarray, ok: np.ndarray, value: np.ndarray, policy: np.ndarray
) -> np.ndarray:
    """Reserves held [T, S, N, P] on the moves of ``policy`` [T, S, N] (joint state s uses
    reserve state s % R of ``value`` [T, R, P]): the best valid corner, nothing if it is not
    worth more than 0 (as ``reserves.best_reserves``)."""
    stages, states, n = policy.shape
    r_states, products, corners = value.shape[1], x.shape[3], x.shape[2]
    out = np.zeros((stages, states, n, products))
    for ts in nb.prange(stages * states):
        t, s = ts // states, ts % states
        r = s % r_states
        for i in range(n):
            j = policy[t, s, i]
            best, arg = -np.inf, -1
            for v in range(corners):
                if ok[i, j, v]:
                    total = 0.0
                    for p in range(products):
                        total += x[i, j, v, p] * value[t, r, p]
                    if total > best:
                        best, arg = total, v
            if arg >= 0 and best > 1e-6:  # battery.TOL, as best_reserves
                for p in range(products):
                    out[t, s, i, p] = x[i, j, arg, p]
    return out
