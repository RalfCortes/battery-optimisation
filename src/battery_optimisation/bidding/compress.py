"""Compress a fine offer curve to at most K tranches (CLAUDE.md).

Breakpoints go where the price jumps most between consecutive tranches; each group becomes one
tranche with the summed quantity and the MW-weighted mean price. Order (and so monotonicity) is
kept, so a rising discharge curve stays rising and a falling charge curve stays falling.
"""

from __future__ import annotations

import numpy as np

from battery_optimisation.sim.clearing import AncillaryTranche, Tranche


def compress[T: (Tranche, AncillaryTranche)](tranches: list[T], k: int) -> list[T]:
    """``tranches``: one side (or one ancillary product), in cumulative order."""
    if len(tranches) <= k:
        return list(tranches)
    prices = np.array([t.price for t in tranches])
    jumps = np.abs(np.diff(prices))
    cuts = np.sort(np.argsort(-jumps, kind="stable")[: k - 1]) + 1  # group starts
    bounds = [0, *cuts.tolist(), len(tranches)]
    out = []
    for a, b in zip(bounds, bounds[1:], strict=False):
        group = tranches[a:b]
        mw = sum(t.quantity_mw for t in group)
        price = sum(t.price * t.quantity_mw for t in group) / mw
        out.append(group[0].model_copy(update={"price": price, "quantity_mw": mw}))
    return out
