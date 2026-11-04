"""Make offers feasible whatever clears before them (CLAUDE.md: worst-case SOC at gate closure).

At the decision for interval t, the offers for the intervals between the decision and t are
already fixed (``pending``) but not cleared. Worst cases: every pending discharge tranche clears
(lowest SOC) or every pending charge tranche clears (highest SOC); price-taking tranches (at the
floor / cap) surely clear. Discharge at t is capped from the lowest, charge from the highest;
tranches are truncated in curve order.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from battery_optimisation.battery import TOL, Battery
from battery_optimisation.sim.clearing import IntervalOffers, Side, Tranche


def soc_range_after(
    soc_mwh: float,
    pending: Iterable[IntervalOffers],
    battery: Battery,
    dt: float,
    floor: float | None = None,
    cap: float | None = None,
) -> tuple[float, float]:
    """(lowest, highest) SOC possible after the pending offers.

    Every tranche may clear, except that price-taking tranches (discharge at ``floor``, charge at
    ``cap``) surely clear. Per interval: lowest = all discharge + only sure charge (netted),
    highest = all charge + only sure discharge.
    """
    low = high = soc_mwh

    def apply(soc: float, net_discharge: float) -> float:
        if net_discharge >= 0:
            return soc - net_discharge * battery.discharge_loss(dt)
        return soc - net_discharge * battery.charge_gain(dt)

    for o in pending:
        discharge = sum(t.quantity_mw for t in o.on(Side.DISCHARGE))
        charge = sum(t.quantity_mw for t in o.on(Side.CHARGE))
        sure_discharge = sum(
            t.quantity_mw
            for t in o.on(Side.DISCHARGE)
            if floor is not None and t.price <= floor + TOL
        )
        sure_charge = sum(
            t.quantity_mw for t in o.on(Side.CHARGE) if cap is not None and t.price >= cap - TOL
        )
        low = apply(low, discharge - sure_charge)
        high = apply(high, sure_discharge - charge)
    return low, high


def _cap(tranches: list[Tranche], max_mw: float) -> list[Tranche]:
    out, left = [], max_mw
    for t in tranches:
        q = min(t.quantity_mw, left)
        if q <= TOL:
            break
        out.append(t if q == t.quantity_mw else t.model_copy(update={"quantity_mw": q}))
        left -= q
    return out


def caps_from_bounds(
    offers: IntervalOffers,
    low: float,
    high: float,
    battery: Battery,
    dt: float,
    max_discharge_mw: float | None = None,
    max_charge_mw: float | None = None,
) -> IntervalOffers:
    """Cap discharge so SOC ``low`` can deliver it and charge so SOC ``high`` can absorb it
    (and by the power limits given, e.g. what reserves leave); truncated in curve order."""
    cfg = battery.config
    p_dis = cfg.p_max_mw if max_discharge_mw is None else max_discharge_mw
    p_ch = cfg.p_max_mw if max_charge_mw is None else max_charge_mw
    max_discharge = max(0.0, min(p_dis, (low - cfg.soc_min) / battery.discharge_loss(dt)))
    max_charge = max(0.0, min(p_ch, (cfg.soc_max - high) / battery.charge_gain(dt)))
    tranches = _cap(offers.on(Side.DISCHARGE), max_discharge) + _cap(
        offers.on(Side.CHARGE), max_charge
    )
    return offers.model_copy(update={"tranches": tuple(tranches)})


def make_feasible(
    offers: IntervalOffers,
    soc_mwh: float,
    pending: Iterable[IntervalOffers],
    battery: Battery,
    dt: float,
    floor: float | None = None,
    cap: float | None = None,
) -> IntervalOffers:
    """Worst-case feasibility: whatever the pending offers clear, this one can be delivered."""
    low, high = soc_range_after(soc_mwh, pending, battery, dt, floor, cap)
    return caps_from_bounds(offers, low, high, battery, dt)


def soc_quantiles_after(
    soc_mwh: float,
    pending: list[IntervalOffers],
    prices: list[np.ndarray],
    weights: list[np.ndarray],
    battery: Battery,
    dt: float,
    q: float,
) -> tuple[float, float]:
    """(q, 1 - q) weighted quantiles of the SOC after the pending offers clear at their scenario
    prices (``prices[i]``, ``weights[i]`` for pending interval i; intervals combined
    independently, every combination enumerated)."""
    socs, probs = np.array([soc_mwh]), np.array([1.0])
    for offers, p, w in zip(pending, prices, weights, strict=True):
        moves = []
        for price in p:
            charge, discharge = _cleared(offers, price)
            moves.append(charge * battery.charge_gain(dt) - discharge * battery.discharge_loss(dt))
        socs = (socs[:, None] + np.asarray(moves)[None, :]).ravel()
        probs = (probs[:, None] * np.asarray(w, float)[None, :]).ravel()
    order = np.argsort(socs)
    socs, cum = socs[order], np.cumsum(probs[order]) / probs.sum()
    low = socs[np.searchsorted(cum, q - 1e-12)]
    high = socs[min(np.searchsorted(cum, 1 - q - 1e-12), len(socs) - 1)]
    return float(low), float(high)


def _cleared(offers: IntervalOffers, price: float) -> tuple[float, float]:
    """(charge, discharge) MW cleared at ``price`` (same rules as the market, netted)."""
    discharge = sum(t.quantity_mw for t in offers.on(Side.DISCHARGE) if t.price <= price)
    charge = sum(t.quantity_mw for t in offers.on(Side.CHARGE) if t.price >= price)
    net = discharge - charge
    return max(-net, 0.0), max(net, 0.0)


def expected_soc_after(
    soc_mwh: float,
    pending: list[IntervalOffers],
    prices: list[list[float]],
    weights: list[float],
    battery: Battery,
    dt: float,
) -> float:
    """SOC expected after the pending offers clear, under scenario prices [pending, K].

    Used as the reference SOC of the next offer curve (exact when one scenario is exact);
    feasibility still uses the worst case (``soc_range_after``).
    """
    cfg = battery.config
    soc = soc_mwh
    for offers, row in zip(pending, prices, strict=True):
        cleared = [_cleared(offers, p) for p in row]
        charge = sum(w * c for w, (c, _) in zip(weights, cleared, strict=True))
        discharge = sum(w * d for w, (_, d) in zip(weights, cleared, strict=True))
        soc += charge * battery.charge_gain(dt) - discharge * battery.discharge_loss(dt)
    return min(max(soc, cfg.soc_min), cfg.soc_max)
