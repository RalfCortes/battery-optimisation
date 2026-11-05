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
