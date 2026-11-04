"""Offer curves from the water value (CLAUDE.md).

For interval t and reference SOC s, the curve walks the SOC grid away from s using the
continuation value V_{t+1} (``optim.sdp``). A segment from SOC a to b is one micro-tranche:

  discharge a > b:  MW = (a - b) / discharge_loss(dt)
                    price = (V(a) - V(b)) / (MW * dt) + kappa      (= slope / eta_d + kappa)
  charge    a < b:  MW = (b - a) / charge_gain(dt)
                    price = (V(b) - V(a)) / (MW * dt) - kappa      (= slope * eta_c - kappa)

With V concave, discharge prices rise and charge prices fall along the curve, and every charge
bid is below every discharge offer (efficiency losses and degradation in between).
"""

from __future__ import annotations

import numpy as np

from battery_optimisation.battery import TOL, Battery
from battery_optimisation.optim.sdp import ValueFunction
from battery_optimisation.sim.clearing import Side, Tranche


def _points(grid: np.ndarray, start: float, stop: float) -> list[float]:
    """start, grid points strictly between start and stop, stop (in walking order)."""
    if stop < start:
        inner = grid[(grid < start - TOL) & (grid > stop + TOL)][::-1]
    else:
        inner = grid[(grid > start + TOL) & (grid < stop - TOL)]
    pts = [start, *inner.tolist()]
    if abs(stop - pts[-1]) > TOL:
        pts.append(stop)
    return pts


def offer_curve(
    vf: ValueFunction, t: int, soc: float, battery: Battery, dt: float
) -> tuple[list[Tranche], list[Tranche]]:
    """(discharge tranches, charge tranches) for interval ``t`` from reference ``soc``,
    each in cumulative order (first tranche = nearest to ``soc``)."""
    here = Battery(battery.config, soc)
    kappa = battery.config.kappa_sgd_per_mwh
    value = lambda s: float(np.interp(s, vf.soc, vf.values[t + 1]))  # noqa: E731

    lowest = soc - here.max_discharge_mw(dt) * here.discharge_loss(dt)
    discharge = []
    pts = _points(vf.soc, soc, lowest) if lowest < soc - TOL else []
    for a, b in zip(pts, pts[1:], strict=False):
        mw = (a - b) / here.discharge_loss(dt)
        price = (value(a) - value(b)) / (mw * dt) + kappa
        discharge.append(Tranche(side=Side.DISCHARGE, price=price, quantity_mw=mw))

    highest = soc + here.max_charge_mw(dt) * here.charge_gain(dt)
    charge = []
    pts = _points(vf.soc, soc, highest) if highest > soc + TOL else []
    for a, b in zip(pts, pts[1:], strict=False):
        mw = (b - a) / here.charge_gain(dt)
        price = (value(b) - value(a)) / (mw * dt) - kappa
        charge.append(Tranche(side=Side.CHARGE, price=price, quantity_mw=mw))
    return discharge, charge


def trim_to_band(
    discharge: list[Tranche], charge: list[Tranche], low: float, high: float
) -> tuple[list[Tranche], list[Tranche]]:
    """Keep only tranches that can clear for a price in [low, high]: discharge priced <= high,
    charge priced >= low. Curves are monotone, so this cuts their far ends."""
    return [t for t in discharge if t.price <= high], [t for t in charge if t.price >= low]


def price_take_beyond_band(
    discharge: list[Tranche],
    charge: list[Tranche],
    low: float,
    high: float,
    floor: float,
    cap: float,
) -> tuple[list[Tranche], list[Tranche]]:
    """Tranches that clear at every plausible price become price-taking (certain): discharge
    priced <= ``low`` is offered at the floor, charge priced >= ``high`` is bid at the cap.
    Feasibility can then count them as cleared for later intervals (gate closure)."""
    return (
        [t.model_copy(update={"price": floor}) if t.price <= low else t for t in discharge],
        [t.model_copy(update={"price": cap}) if t.price >= high else t for t in charge],
    )
