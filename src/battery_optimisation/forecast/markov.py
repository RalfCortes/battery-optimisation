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
    weights: np.ndarray  # [J] state probabilities (equal)
    transitions: np.ndarray  # [J, J] P(state k at t + 1 | state j at t), rows sum to 1


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
