"""Markov price states from quantile forecasts: a Tauchen lattice in normal-score space.

The SDP needs, per stage, a distribution of the price and how it carries over to the next
stage. Stagewise-independent draws (the scenario marginals) ignore that prices persist: the SDP
then sees spreads between independent high and low draws that correlated prices never offer
(CLAUDE.md: "never sample steps independently"). Here:

- each stage's price is split into ``states`` equal-probability bins of its quantile forecast
  (bins in normal-score space z = Phi^-1(u)); a state's price is the mean price in its bin
  (tails extrapolated as in ``forecast.quantiles``, so the top state keeps the spike mass);
- z follows an AR(1) with lag-1 correlation ``rho`` (the Gaussian copula's), so
  P(state k at t+1 | state j at t) = P(z' in bin k | z = node_j), z' ~ N(rho node_j, 1 - rho^2),
  node_j = mean of z in bin j.

rho = 0 gives back stagewise independence (every row = the state weights).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr, ndtri

from battery_optimisation.forecast.quantiles import inverse_cdf

SUB_POINTS = 16  # quantile points averaged per state (bin mean)


@dataclass(frozen=True)
class PriceLattice:
    prices: np.ndarray  # [T, J] mean price of each state, increasing in j
    weights: np.ndarray  # [J] state probabilities (equal for bins)
    transitions: np.ndarray  # [J, J] P(state k at t + 1 | state j at t), rows sum to 1
    scores: np.ndarray | None = None  # [J] normal-score nodes (``node_lattice``)


def price_lattice(
    levels: np.ndarray, quantiles: np.ndarray, states: int, rho: float
) -> PriceLattice:
    """Lattice for one issue: ``quantiles`` [T, Q] at ``levels`` [Q]."""
    edges = ndtri(np.linspace(0.0, 1.0, states + 1))  # -inf .. inf
    density = np.exp(-(edges**2) / 2) / np.sqrt(2 * np.pi)
    nodes = (density[:-1] - density[1:]) * states  # E[z | bin]
    u = (np.arange(states)[:, None] + (np.arange(SUB_POINTS)[None, :] + 0.5) / SUB_POINTS) / states
    quantiles = np.sort(np.asarray(quantiles, float), axis=1)
    prices = np.stack([inverse_cdf(levels, q, u).mean(axis=1) for q in quantiles])
    sd = np.sqrt(1.0 - rho**2)
    upper = ndtr((edges[None, 1:] - rho * nodes[:, None]) / sd)
    lower = ndtr((edges[None, :-1] - rho * nodes[:, None]) / sd)
    transitions = upper - lower
    return PriceLattice(
        prices=prices,
        weights=np.full(states, 1.0 / states),
        transitions=transitions / transitions.sum(axis=1, keepdims=True),
    )


FINE_POINTS = 4096  # equal-probability score points for the node prices (local means)


def hat_weights(nodes: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Linear-interpolation weights [len(z), J] of each z on the increasing ``nodes`` (clamped
    beyond the ends): B_l(z), non-negative, summing to 1."""
    z = np.clip(np.asarray(z, float), nodes[0], nodes[-1])
    j = np.clip(np.searchsorted(nodes, z, side="right") - 1, 0, nodes.size - 2)
    f = (z - nodes[j]) / (nodes[j + 1] - nodes[j])
    out = np.zeros((z.size, nodes.size))
    rows = np.arange(z.size)
    out[rows, j] = 1.0 - f
    out[rows, j + 1] += f
    return out


def hat_expectations(nodes: np.ndarray, means: np.ndarray, sd: float) -> np.ndarray:
    """E[B_l(Z)] [M, J] for Z ~ N(means[m], sd^2), exact. Each hat is a combination of ramps
    (z - z_l)^+, whose expectation G(x) = E[(Z - x)^+] = (m - x) Phi((m - x)/sd) + sd phi(.).
    R_k = E[ramp rising from 0 to 1 over segment k] = (G(z_k) - G(z_k+1)) / h_k; then
    B_0 = 1 - R_0, B_l = R_l-1 - R_l, B_last = R_last-1 (clamped ends; rows sum to 1)."""
    nodes = np.asarray(nodes, float)
    gap = np.asarray(means, float)[:, None] - nodes[None, :]  # m - x
    if sd <= 0:
        g = np.maximum(gap, 0.0)
    else:
        a = gap / sd
        g = gap * ndtr(a) + sd * np.exp(-(a**2) / 2) / np.sqrt(2 * np.pi)
    ramps = (g[:, :-1] - g[:, 1:]) / np.diff(nodes)[None, :]  # [M, J - 1]
    out = np.empty_like(g)
    out[:, 0] = 1.0 - ramps[:, 0]
    out[:, 1:-1] = ramps[:, :-1] - ramps[:, 1:]
    out[:, -1] = ramps[:, -1]
    return np.clip(out, 0.0, None) / np.clip(out, 0.0, None).sum(axis=1, keepdims=True)


def node_lattice(
    levels: np.ndarray, quantiles: np.ndarray, nodes: int, z_max: float, rho: float
) -> PriceLattice:
    """Continuous price state (fitted value iteration, linear interpolation in the score).

    The state is the normal score z of the price, z_t = Phi^-1(F_t(price_t)); values live at
    ``nodes`` equally spaced scores on [-z_max, z_max] and are interpolated linearly between
    them (hat functions B_l). Then E[V(Z') | z_j] = sum_l T[j, l] V(z_l) with
    T[j, l] = E[B_l(Z') | z_j]: a stochastic matrix, so the lattice SDP runs unchanged.
    - transitions: AR(1) Z' ~ N(rho z_j, sd^2), exact hat expectations (``hat_expectations``);
      sd^2 = 1 - rho^2 - rho^2 h^2/6 offsets the variance the interpolation adds (h^2/6 per
      step, h = node spacing), so the projected N(0, 1) stays the stationary distribution;
    - weights: N(0, 1) projected on the nodes;
    - prices: local mean under each hat, E[price B_l(Z)] / E[B_l(Z)], so the stage mean is
      the forecast's (revenue is linear in price).
    Other dependence models only change ``transitions`` (e.g. kernel regression on paths)."""
    scores = np.linspace(-z_max, z_max, nodes)
    h = scores[1] - scores[0]
    sd = float(np.sqrt(max(1.0 - rho**2 - rho**2 * h**2 / 6.0, 0.0)))
    transitions = hat_expectations(scores, rho * scores, sd)
    weights = hat_expectations(scores, np.zeros(1), 1.0)[0]
    u = (np.arange(FINE_POINTS) + 0.5) / FINE_POINTS
    hats = hat_weights(scores, ndtri(u))  # [FINE, J]
    quantiles = np.sort(np.asarray(quantiles, float), axis=1)
    prices = np.stack([inverse_cdf(levels, q, u) for q in quantiles]) @ hats / hats.sum(axis=0)
    return PriceLattice(prices=prices, weights=weights, transitions=transitions, scores=scores)


HERMITE_POINTS = 16  # quadrature over each product's own (idiosyncratic) part


@dataclass(frozen=True)
class FactorLattice:
    """Markov states of a common factor driving several prices (reserve products)."""

    prices: np.ndarray  # [T, R, P] mean price of product p given the factor in state r
    weights: np.ndarray  # [R] (equal)
    transitions: np.ndarray  # [R, R]
    products: tuple  # [P] keys, in the order of the last axis


def _transitions(edges: np.ndarray, nodes: np.ndarray, rho: float) -> np.ndarray:
    sd = np.sqrt(1.0 - rho**2)
    upper = ndtr((edges[None, 1:] - rho * nodes[:, None]) / sd)
    lower = ndtr((edges[None, :-1] - rho * nodes[:, None]) / sd)
    t = upper - lower
    return t / t.sum(axis=1, keepdims=True)


def factor_lattice(
    levels: np.ndarray,
    quantiles: dict,
    loadings: dict,
    states: int,
    rho: float,
) -> FactorLattice:
    """One-factor model in normal-score space: z_p = lambda_p F + sqrt(1 - lambda_p^2) e_p, with
    F an AR(1) (lag-1 ``rho``) split into ``states`` equal-probability bins (Tauchen, as
    ``price_lattice``) and e_p independent N(0, 1). State r's price of product p = E[price_p |
    F in bin r] (``quantiles[p]`` [T, Q] at ``levels``; mean over the bin and over e_p). Revenue
    is linear in price, so this conditional mean is all a reserve decision needs.
    Loading 0: every state is the product's mean; 1: the product's own Markov lattice."""
    edges = ndtri(np.linspace(0.0, 1.0, states + 1))
    density = np.exp(-(edges**2) / 2) / np.sqrt(2 * np.pi)
    nodes = (density[:-1] - density[1:]) * states
    u = (np.arange(states)[:, None] + (np.arange(SUB_POINTS)[None, :] + 0.5) / SUB_POINTS) / states
    factor = ndtri(u)  # [R, S] points of F inside each bin
    e, w = np.polynomial.hermite_e.hermegauss(HERMITE_POINTS)
    w = w / w.sum()
    products = tuple(quantiles)
    out = np.empty((np.asarray(quantiles[products[0]]).shape[0], states, len(products)))
    for j, p in enumerate(products):
        lam = float(np.clip(loadings[p], 0.0, 1.0))
        z = lam * factor[..., None] + np.sqrt(1.0 - lam**2) * e[None, None, :]  # [R, S, H]
        uu = ndtr(z)
        q = np.sort(np.asarray(quantiles[p], float), axis=1)
        for t, row in enumerate(q):
            out[t, :, j] = (inverse_cdf(levels, row, uu) * w).sum(axis=2).mean(axis=1)
    return FactorLattice(
        prices=out,
        weights=np.full(states, 1.0 / states),
        transitions=_transitions(edges, nodes, rho),
        products=products,
    )
