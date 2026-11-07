"""Regulation and reserve offer pricing: reserve offer curves from the water value.

At gate closure the SDP chooses the reserve mix x (MW per product) that is best in expectation
over the price states, with the energy action then restricted to what fits beside it:

    G(x) = E_k[ max_{energy a fitting beside x} energy_value_k(a) ] + sum_p v_p x_p

(``WaterValueStrategy._choose_reserves``; v_p = SGD per MW held in the interval: price x dt,
plus regulation's AGC energy and wear). Offering that x at price 0 holds it whatever the
reserve price turns out to be (open loop in the reserve price).

A reserve offer *curve* instead answers: how many MW of product p would the same choice hold
if p's price were pi (the other products at their expected value)? Sweeping pi over a grid,

    q_p(pi) = x*_p(pi),   x*(pi) = argmax_x  base(x) + sum_{q != p} v_q x_q + (pi dt + e_p) x_p

with base(x) = E_k[max energy value beside x] and e_p the non-price value per MW. q_p rises
with pi (the objective has increasing differences in (x_p, pi)), and each step of q_p sits at
the price where holding those MW starts to pay for the energy value they displace and the SoC
headroom they take: the opportunity cost. Each step becomes one tranche (price, MW), and the
curve is compressed to the offer format's K pairs per product.

The energy curve keeps room for the mix chosen at the expected prices; reserves that clear
beyond it are settled against energy and the SoC by the market engine (surplus, RC386). That
mix therefore costs nothing more to hold: it is offered at 0, as fixed offers are, and only the
MW beyond it are priced (``upside_tranches``). Pricing the whole curve lost 0.27 M SGD over
the year (point reserve prices): below the expected price the reserves did not clear, yet the
energy curve had kept their room. Room for the certain part only, or for the whole curve, did
worse on real weeks (``WaterValueStrategy._reserve_curves``).
"""

from __future__ import annotations

import numpy as np

from battery_optimisation.battery import TOL
from battery_optimisation.bidding.compress import compress
from battery_optimisation.config import Product
from battery_optimisation.sim.clearing import AncillaryTranche


def reserve_curve(
    candidates: np.ndarray,
    base: np.ndarray,
    value: np.ndarray,
    extra: np.ndarray,
    product: int,
    grid: np.ndarray,
    dt: float,
) -> np.ndarray:
    """MW of ``product`` (column of ``candidates`` [C, P]) chosen at each price of ``grid``
    [G] (SGD/MW/h), rising. ``base`` [C]: expected energy value beside each candidate mix;
    ``value`` [P]: SGD per MW of each product at its expected price; ``extra`` [P]: the part of
    it that does not scale with the price (regulation energy and wear). Ties go to the larger
    hold of ``product``."""
    others = np.delete(np.arange(candidates.shape[1]), product)
    fixed = base + candidates[:, others] @ value[others]  # [C]
    mine = candidates[:, product]  # [C]
    paid = np.asarray(grid, float)[:, None] * dt + extra[product]  # [G, 1]
    total = fixed[None, :] + paid * mine[None, :] + 1e-9 * mine[None, :]
    mw = mine[np.argmax(total, axis=1)]
    return np.maximum.accumulate(mw)


def curve_tranches(
    grid: np.ndarray, mw: np.ndarray, product: Product, k: int
) -> list[AncillaryTranche]:
    """Tranches of a rising reserve curve ``mw`` on ``grid``: each increase offered at the
    lowest price where it is held, compressed to at most ``k`` pairs (increasing price)."""
    steps = np.diff(np.asarray(mw, float), prepend=0.0)
    tranches = [
        AncillaryTranche(product=product, price=float(p), quantity_mw=float(q))
        for p, q in zip(grid, steps, strict=True)
        if q > TOL
    ]
    return compress(tranches, k)


def upside_tranches(
    grid: np.ndarray, mw: np.ndarray, held: float, product: Product, k: int
) -> list[AncillaryTranche]:
    """The planned ``held`` MW at price 0 (the energy curve keeps room for them, so they cost
    nothing more to hold and always clear), then the rest of the rising curve ``mw`` beyond
    ``held``, priced at its opportunity cost (at most ``k`` pairs in all)."""
    if held <= TOL:
        return curve_tranches(grid, mw, product, k)
    base = AncillaryTranche(product=product, price=0.0, quantity_mw=float(held))
    extra = np.maximum(np.asarray(mw, float) - held, 0.0)
    return [base, *curve_tranches(grid, extra, product, k - 1)]
