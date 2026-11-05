"""Offer curves from the water value (CLAUDE.md).

For interval t and reference SOC s, the curve walks the SOC grid away from s using the
continuation value V_{t+1} (``optim.sdp``). A segment from SOC a to b is one micro-tranche:

  discharge a > b:  MW = (a - b) / discharge_loss(dt)
                    price = (V(a) - V(b)) / (MW * dt) + kappa      (= slope / eta_d + kappa)
  charge    a < b:  MW = (b - a) / charge_gain(dt)
                    price = (V(b) - V(a)) / (MW * dt) - kappa      (= slope * eta_c - kappa)

With V concave, discharge prices rise and charge prices fall along the curve, and every charge
bid is below every discharge offer (efficiency losses and degradation in between).

With Markov prices (``optim.sdp``) the value after t depends on the price at t itself (a high
price announces high prices next), so slopes of one V_{t+1} do not give the curve. Instead the
curve is the optimal response itself: for each price on a fine grid across the stage's price
states, the best target SOC (``optimal_net_mw``); each step of that supply curve is a tranche.
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


GRID_PER_STATE = 50  # response-curve prices between two consecutive price states
_BEYOND = np.array([0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])  # x price-state span


def optimal_net_mw(
    vf: ValueFunction, t: int, soc: float, battery: Battery, dt: float, prices: np.ndarray
) -> np.ndarray:
    """Optimal net MW (discharge - charge) in interval ``t`` from ``soc`` for each price:
    max over reachable s' of price * sold(s -> s') - degradation + value after t (which, for
    Markov prices, depends on the price). Piecewise linear values: the optimum is at a grid
    point or a power limit."""
    here = Battery(battery.config, soc)
    lo = soc - here.max_discharge_mw(dt) * here.discharge_loss(dt)
    hi = soc + here.max_charge_mw(dt) * here.charge_gain(dt)
    grid = vf.soc[(vf.soc > lo + TOL) & (vf.soc < hi - TOL)]
    target = np.unique(np.concatenate([[lo, soc, hi], grid]))
    discharge = np.where(target < soc, (soc - target) / here.discharge_loss(dt), 0.0)
    charge = np.where(target > soc, (target - soc) / here.charge_gain(dt), 0.0)
    net = discharge - charge
    prices = np.asarray(prices, float)
    after = vf.continuation_at(t, prices)  # [P, N]
    # linear interpolation of every row at the targets: [N] -> [targets]
    position = np.interp(target, vf.soc, np.arange(vf.soc.size, dtype=float))
    left = np.minimum(np.floor(position).astype(int), vf.soc.size - 2)
    f = position - left
    base = (1 - f) * after[:, left] + f * after[:, left + 1]
    value = prices[:, None] * net[None, :] * dt + base - here.degradation_sgd(charge, discharge, dt)
    return net[value.argmax(axis=1)]


def _response_curve(
    vf: ValueFunction, t: int, soc: float, battery: Battery, dt: float
) -> tuple[list[Tranche], list[Tranche]]:
    """Offer curve = the optimal response on a price grid (Markov prices)."""
    states = vf.state_prices[t]
    span = max(states[-1] - states[0], 1.0)
    inside = np.interp(
        np.linspace(0, states.size - 1, GRID_PER_STATE * (states.size - 1) + 1),
        np.arange(states.size),
        states,
    )
    grid = np.unique(
        np.concatenate([states[0] - span * _BEYOND, inside, states[-1] + span * _BEYOND])
    )
    net = optimal_net_mw(vf, t, soc, battery, dt, grid)
    sold = np.maximum.accumulate(np.maximum(net, 0.0))  # discharge MW, rising with price
    bought = np.maximum.accumulate(np.maximum(-net, 0.0)[::-1])  # charge MW, falling price
    discharge = [
        Tranche(side=Side.DISCHARGE, price=float(p), quantity_mw=float(q))
        for p, q in zip(grid, np.diff(sold, prepend=0.0), strict=True)
        if q > TOL
    ]
    charge = [
        Tranche(side=Side.CHARGE, price=float(p), quantity_mw=float(q))
        for p, q in zip(grid[::-1], np.diff(bought, prepend=0.0), strict=True)
        if q > TOL
    ]
    return discharge, charge


def offer_curve(
    vf: ValueFunction, t: int, soc: float, battery: Battery, dt: float
) -> tuple[list[Tranche], list[Tranche]]:
    """(discharge tranches, charge tranches) for interval ``t`` from reference ``soc``,
    each in cumulative order (first tranche = nearest to ``soc``)."""
    if vf.continuation is not None:
        return _response_curve(vf, t, soc, battery, dt)
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
