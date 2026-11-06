"""Storage limits of the market engine (``MarketConfig.storage_rules``; NEMS RC383 / RC386).

For one interval with charge c, discharge d (MW) and ancillary awards a_p (MW), from the SoC at
the start of the interval:
  power, per raise group g:  d - c + sum_{p in g} a_p <= P        (one group per reserve class)
         per lower group g:  c - d + sum_{p in g} a_p <= P
  SoC, per discharge check k (h_pk = hours product p is sustained, p = energy for the schedule):
         draw_k = (d h_Ek + sum_p a_p h_pk) / eta_d - c h_Ek eta_c  <=  SoC - soc_min
       per charge check k:
         fill_k = (c h_Ek + sum_p a_p h_pk) eta_c - d h_Ek / eta_d  <=  soc_max - SoC

With ``enforce_soc`` the engine (``enforce_soc``) cuts what was cleared until it fits, keeping
what the market values most: the surplus of each cleared tranche (realised price minus offer
price, per MW), as a welfare-maximising engine would for a price taker.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import highspy

from battery_optimisation.battery import TOL, LimitBreachError
from battery_optimisation.config import BatteryConfig, Product, StorageRules
from battery_optimisation.sim.clearing import Cleared, IntervalOffers, Side

_KEEP = 1e-3  # SGD/MW tie-break: keep zero-surplus tranches rather than cut them


def _group_mw(groups: list[list[Product]], awards: Mapping[Product, float]) -> float:
    return max((sum(awards.get(p, 0.0) for p in g) for g in groups), default=0.0)


def raise_mw(rules: StorageRules, awards: Mapping[Product, float]) -> float:
    """Upward power the awards need (largest raise group)."""
    return _group_mw(rules.raise_power_groups, awards)


def lower_mw(rules: StorageRules, awards: Mapping[Product, float]) -> float:
    """Downward power the awards need (largest lower group)."""
    return _group_mw(rules.lower_power_groups, awards)


def _draw(check, battery: BatteryConfig, charge, discharge, awards) -> float:
    held = sum(mw * check.get(p, 0.0) for p, mw in awards.items())
    h = check.get(Product.ENERGY, 0.0)
    return (discharge * h + held) / battery.eta_d - charge * h * battery.eta_c


def _fill(check, battery: BatteryConfig, charge, discharge, awards) -> float:
    held = sum(mw * check.get(p, 0.0) for p, mw in awards.items())
    h = check.get(Product.ENERGY, 0.0)
    return (charge * h + held) * battery.eta_c - discharge * h / battery.eta_d


def draw_mwh(
    rules: StorageRules,
    battery: BatteryConfig,
    charge_mw: float,
    discharge_mw: float,
    awards: Mapping[Product, float],
) -> float:
    """SoC (MWh) the schedule may draw (largest discharge check); <= SoC - soc_min."""
    return max(
        (_draw(k, battery, charge_mw, discharge_mw, awards) for k in rules.discharge_checks),
        default=0.0,
    )


def fill_mwh(
    rules: StorageRules,
    battery: BatteryConfig,
    charge_mw: float,
    discharge_mw: float,
    awards: Mapping[Product, float],
) -> float:
    """SoC (MWh) the schedule may add (largest charge check); <= soc_max - SoC."""
    return max(
        (_fill(k, battery, charge_mw, discharge_mw, awards) for k in rules.charge_checks),
        default=0.0,
    )


def reserve_draw_mwh(
    rules: StorageRules, battery: BatteryConfig, awards: Mapping[Product, float]
) -> float:
    """SoC to keep above soc_min for the awards alone (no energy schedule)."""
    return max(draw_mwh(rules, battery, 0.0, 0.0, awards), 0.0)


def reserve_fill_mwh(
    rules: StorageRules, battery: BatteryConfig, awards: Mapping[Product, float]
) -> float:
    """SoC to keep below soc_max for the awards alone (no energy schedule)."""
    return max(fill_mwh(rules, battery, 0.0, 0.0, awards), 0.0)


def check_storage_rules(
    rules: StorageRules,
    battery: BatteryConfig,
    soc_mwh: float,
    charge_mw: float,
    discharge_mw: float,
    awards: Mapping[Product, float],
) -> None:
    """Raise ``LimitBreachError`` if the schedule breaks a power or SoC rule."""
    p = battery.p_max_mw
    up, down = raise_mw(rules, awards), lower_mw(rules, awards)
    if discharge_mw - charge_mw + up > p + TOL:
        raise LimitBreachError(f"power: {discharge_mw - charge_mw:.1f} MW net + {up:.1f} MW raise")
    if charge_mw - discharge_mw + down > p + TOL:
        raise LimitBreachError(
            f"power: {charge_mw - discharge_mw:.1f} MW charging + {down:.1f} MW lower"
        )
    draw = draw_mwh(rules, battery, charge_mw, discharge_mw, awards)
    if draw > soc_mwh - battery.soc_min + TOL:
        raise LimitBreachError(f"sustain: SOC {soc_mwh:.2f} MWh, schedule draws {draw:.2f} MWh")
    fill = fill_mwh(rules, battery, charge_mw, discharge_mw, awards)
    if fill > battery.soc_max - soc_mwh + TOL:
        raise LimitBreachError(f"sustain: SOC {soc_mwh:.2f} MWh, schedule adds {fill:.2f} MWh")


def _fits(rules, battery, soc, charge, discharge, awards) -> bool:
    try:
        check_storage_rules(rules, battery, soc, charge, discharge, awards)
    except LimitBreachError:
        return False
    return True


def _cleared_blocks(
    quantity: float, tranches: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """(MW, surplus per MW) of the best tranches making up ``quantity`` MW."""
    blocks, left = [], quantity
    for surplus, mw in sorted(tranches, reverse=True):
        take = min(mw, left)
        if take > TOL:
            blocks.append((take, surplus))
        left -= take
    return blocks


def enforce_soc(
    cleared: Cleared,
    offers: IntervalOffers,
    rules: StorageRules,
    battery: BatteryConfig,
    soc_mwh: float,
) -> Cleared:
    """What the engine schedules from ``cleared`` given the SoC at the start of the interval:
    unchanged if feasible, else the highest-surplus part that fits the rules (an LP over the
    cleared tranches)."""
    if _fits(rules, battery, soc_mwh, cleared.charge_mw, cleared.discharge_mw, cleared.awards):
        return cleared
    price = cleared.price
    if cleared.discharge_mw > 0:
        side, quantity = Side.DISCHARGE, cleared.discharge_mw
        energy = [(price - t.price, t.quantity_mw) for t in offers.on(side) if t.price <= price]
    else:
        side, quantity = Side.CHARGE, cleared.charge_mw
        energy = [(t.price - price, t.quantity_mw) for t in offers.on(side) if t.price >= price]
    energy_blocks = _cleared_blocks(quantity, energy)
    award_blocks = {
        p: _cleared_blocks(
            mw,
            [
                (cleared.ancillary_prices[p] - t.price, t.quantity_mw)
                for t in offers.ancillary
                if t.product is p and t.price <= cleared.ancillary_prices[p]
            ],
        )
        for p, mw in cleared.awards.items()
        if mw > TOL
    }

    h = highspy.Highs()
    h.silent()
    zero = 1.0 * h.addVariable(lb=0, ub=0)  # empty linear expression
    e = [h.addVariable(lb=0, ub=mw) for mw, _ in energy_blocks]
    a = {p: [h.addVariable(lb=0, ub=mw) for mw, _ in blocks] for p, blocks in award_blocks.items()}
    flow = sum(e, start=zero)
    d, c = (flow, zero) if side is Side.DISCHARGE else (zero, flow)
    mw = {p: sum(v, start=zero) for p, v in a.items()}
    p_max = battery.p_max_mw
    for group in rules.raise_power_groups:
        h.addConstr(d - c + sum((mw[p] for p in group if p in mw), start=zero) <= p_max)
    for group in rules.lower_power_groups:
        h.addConstr(c - d + sum((mw[p] for p in group if p in mw), start=zero) <= p_max)
    for check in rules.discharge_checks:
        h.addConstr(_draw(check, battery, c, d, mw) <= soc_mwh - battery.soc_min)
    for check in rules.charge_checks:
        h.addConstr(_fill(check, battery, c, d, mw) <= battery.soc_max - soc_mwh)
    objective = sum(
        ((s + _KEEP) * v for v, (_, s) in zip(e, energy_blocks, strict=True)), start=zero
    )
    for p, blocks in award_blocks.items():
        objective = objective + sum(
            ((s + _KEEP) * v for v, (_, s) in zip(a[p], blocks, strict=True)), start=zero
        )
    h.maximize(objective)
    if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"SoC enforcement LP: {h.modelStatusToString(h.getModelStatus())}")
    kept = float(sum(h.vals(e))) if e else 0.0
    awards = dict(cleared.awards)
    for p, v in a.items():
        awards[p] = float(sum(h.vals(v)))
    charge, discharge = (0.0, kept) if side is Side.DISCHARGE else (kept, 0.0)
    return replace(cleared, charge_mw=charge, discharge_mw=discharge, awards=awards)


def cut_mw(before: Cleared, after: Cleared) -> float:
    """MW (energy + ancillary) the engine cut for the SoC."""
    energy = (before.charge_mw - after.charge_mw) + (before.discharge_mw - after.discharge_mw)
    return energy + sum(before.awards[p] - after.awards.get(p, 0.0) for p in before.awards)
