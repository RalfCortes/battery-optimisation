"""Best reserve mix for a stage of the SDP, given the SOC at its start and its energy action.

For SOC s and energy flow (charge c, discharge d) the awards x (one per ancillary product) must
satisfy the market's storage rules (``MarketConfig.storage_rules``, see ``sim.storage_rules``)
and 0 <= x_p <= max_award_p. All rules are linear in x with a matrix that does not depend on
(s, c, d): only the right-hand side does. So the feasible set's corners are found once per
triple of active rows (``ReserveSet``): x = A_rows^-1 b_rows(s, c, d), kept where every row
holds. The stage reward is then max over corners of value . x (an LP optimum is at a corner).

``firm``: reserves get no credit from the opposite energy flow (power or SoC), as the MILP's
``firm_reserves``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from battery_optimisation.battery import TOL
from battery_optimisation.config import BatteryConfig, MarketConfig, Product


@dataclass(frozen=True)
class ReserveSet:
    """The rules as rows A x <= b0 + B @ (soc, charge, discharge, 1)."""

    products: tuple[Product, ...]
    a: np.ndarray  # [M, P]
    rhs: np.ndarray  # [M, 4]: b = rhs @ (soc, charge, discharge, 1)
    inverses: np.ndarray  # [J, P, P] inverses of the non-singular P-row subsets
    rows: np.ndarray  # [J, P] row indices of each subset


def reserve_set(market: MarketConfig, battery: BatteryConfig, firm: bool = False) -> ReserveSet:
    products = tuple(market.ancillary_products)
    rules, b = market.storage_rules, battery
    a, rhs = [], []

    def row(coef: dict[Product, float], soc: float, charge: float, discharge: float, const):
        a.append([coef.get(p, 0.0) for p in products])
        rhs.append([soc, charge, discharge, const])

    for i, p in enumerate(products):
        cap = min(market.ancillary.max_award_mw.get(p, b.p_max_mw), 2 * b.p_max_mw)
        row({p: 1.0}, 0, 0, 0, cap)
        row({p: -1.0}, 0, 0, 0, 0.0)
        del i
    for group in rules.raise_power_groups:  # d - c + sum <= P
        if any(p in products for p in group):
            row({p: 1.0 for p in group}, 0, 0.0 if firm else 1.0, -1.0, b.p_max_mw)
    for group in rules.lower_power_groups:  # c - d + sum <= P
        if any(p in products for p in group):
            row({p: 1.0 for p in group}, 0, -1.0, 0.0 if firm else 1.0, b.p_max_mw)
    for check in rules.discharge_checks:  # sum x h / eta_d <= s - min - d h/eta_d + c h eta_c
        h = check.get(Product.ENERGY, 0.0)
        coef = {p: check.get(p, 0.0) / b.eta_d for p in products if check.get(p, 0.0)}
        if coef:
            row(coef, 1.0, 0.0 if firm else h * b.eta_c, -h / b.eta_d, -b.soc_min)
    for check in rules.charge_checks:  # sum x h eta_c <= max - s - c h eta_c + d h / eta_d
        h = check.get(Product.ENERGY, 0.0)
        coef = {p: check.get(p, 0.0) * b.eta_c for p in products if check.get(p, 0.0)}
        if coef:
            row(coef, -1.0, -h * b.eta_c, 0.0 if firm else h / b.eta_d, b.soc_max)
    a_arr, rhs_arr = np.asarray(a, float), np.asarray(rhs, float)
    inverses, rows = [], []
    for subset in itertools.combinations(range(len(a_arr)), len(products)):
        sub = a_arr[list(subset)]
        if abs(np.linalg.det(sub)) > 1e-9:
            inverses.append(np.linalg.inv(sub))
            rows.append(subset)
    return ReserveSet(
        products=products,
        a=a_arr,
        rhs=rhs_arr,
        inverses=np.asarray(inverses),
        rows=np.asarray(rows, dtype=int),
    )


def corners(
    rs: ReserveSet, soc: np.ndarray, charge: np.ndarray, discharge: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Every candidate corner at each point (broadcast shape S): (x [*S, J, P], valid [*S, J]).
    The zero award is always a valid corner when the energy action itself is feasible."""
    soc, charge, discharge = np.broadcast_arrays(*map(np.asarray, (soc, charge, discharge)))
    shape = soc.shape
    z = np.stack([soc.ravel(), charge.ravel(), discharge.ravel(), np.ones(soc.size)], axis=1)
    xs, oks = [], []
    for part in np.array_split(z, max(1, z.shape[0] // 2048)):
        b = part @ rs.rhs.T  # [Z, M]
        x = np.einsum("jpq,zjq->zjp", rs.inverses, b[:, rs.rows])  # [Z, J, P]
        ok = (np.einsum("mp,zjp->zjm", rs.a, x) <= b[:, None, :] + 1e-6).all(axis=2)
        xs.append(np.where(ok[..., None], np.maximum(x, 0.0), 0.0))
        oks.append(ok)
    x, ok = np.concatenate(xs), np.concatenate(oks)
    return x.reshape(*shape, *x.shape[1:]), ok.reshape(*shape, ok.shape[-1])


def compact(x: np.ndarray, ok: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The distinct valid corners only, padded to the largest count (padding: invalid)."""
    shape, n_products = ok.shape[:-1], x.shape[-1]
    flat_x, flat_ok = x.reshape(-1, *x.shape[-2:]), ok.reshape(-1, ok.shape[-1])
    distinct = [
        np.unique(np.round(flat_x[i][flat_ok[i]], 6), axis=0) for i in range(flat_ok.shape[0])
    ]
    size = max((d.shape[0] for d in distinct), default=0) or 1
    out_x = np.zeros((len(distinct), size, n_products))
    out_ok = np.zeros((len(distinct), size), dtype=bool)
    for i, d in enumerate(distinct):
        out_x[i, : d.shape[0]], out_ok[i, : d.shape[0]] = d, True
    return out_x.reshape(*shape, size, n_products), out_ok.reshape(*shape, size)


def best_reserves(
    x: np.ndarray, ok: np.ndarray, value: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(reward [*S], awards [*S, P]) maximising ``value`` . x (SGD per MW, [P]) over the
    corners; reward 0 where nothing is worth holding."""
    v = np.where(ok, x @ value, -np.inf)
    i = v.argmax(axis=-1)
    reward = np.maximum(np.take_along_axis(v, i[..., None], axis=-1)[..., 0], 0.0)
    awards = np.take_along_axis(x, i[..., None, None], axis=-2)[..., 0, :]
    awards = np.where((reward > TOL)[..., None], awards, 0.0)
    return reward, awards


@dataclass(frozen=True)
class PairCorners:
    """Distinct valid corners for every SDP move s_i -> s_j (from the SOC grid)."""

    products: tuple[Product, ...]
    x: np.ndarray  # [N, N, V, P]
    ok: np.ndarray  # [N, N, V]

    def rewards(self, value: np.ndarray) -> np.ndarray:
        """Best reserve reward per stage and move [T, N, N] for ``value`` [T, P] (SGD/MW):
        one matrix product over every stage (BLAS), max over the corners (no awards needed).
        ``value`` [T, R, P] (reserve price states): [T, R, N, N]."""
        value = np.asarray(value, float)
        if value.ndim == 3:  # compiled loop: no [N*N*V, T*R] intermediate
            from battery_optimisation.optim.kernels import pair_rewards

            t, r, p = value.shape
            flat = pair_rewards(self.x, self.ok, np.ascontiguousarray(value.reshape(t * r, p)))
            return flat.reshape(t, r, *self.x.shape[:2])
        n, _, corners_, products = self.x.shape
        flat = self.x.reshape(-1, products) @ value.T  # [N*N*V, T]
        flat += np.where(self.ok, 0.0, -np.inf).reshape(-1, 1)
        best = flat.reshape(n, n, corners_, -1).max(axis=2)  # [N, N, T]
        return np.maximum(best, 0.0).transpose(2, 0, 1)

    def mixes(self, value: np.ndarray, policy: np.ndarray) -> np.ndarray:
        """Reserves held [T, S, N, P] on the moves of an SDP ``policy`` [T, S, N] (grid index
        of the SOC after stage t), for ``value`` [T, P] (SGD/MW): the mix behind ``rewards``.
        ``value`` [T, R, P]: joint states s = k R + r use reserve state r's values."""
        from battery_optimisation.optim.kernels import policy_mixes

        value = np.asarray(value, float)
        if value.ndim == 2:
            value = value[:, None, :]
        return policy_mixes(
            self.x, self.ok, np.ascontiguousarray(value), np.ascontiguousarray(policy, np.int16)
        )


_PAIR_CACHE: dict[tuple, PairCorners] = {}


def pair_corners(
    market: MarketConfig, battery: BatteryConfig, levels: int, dt: float, firm: bool
) -> PairCorners:
    """Corners for the moves of an SDP with ``levels`` SOC points (cached per configuration)."""
    key = (market.model_dump_json(), battery.model_dump_json(), levels, dt, firm)
    if key not in _PAIR_CACHE:
        rs = reserve_set(market, battery, firm)
        soc = np.linspace(battery.soc_min, battery.soc_max, levels)
        move = soc[None, :] - soc[:, None]
        charge = np.where(move > 0, move / (battery.eta_c * dt), 0.0)
        discharge = np.where(move < 0, -move * battery.eta_d / dt, 0.0)
        x, ok = compact(*corners(rs, np.broadcast_to(soc[:, None], move.shape), charge, discharge))
        _PAIR_CACHE[key] = PairCorners(products=rs.products, x=x, ok=ok)
    return _PAIR_CACHE[key]


def stage_values(
    market: MarketConfig,
    battery: BatteryConfig,
    products: tuple[Product, ...],
    ancillary_prices: np.ndarray,
    energy_prices: np.ndarray,
    dt: float,
) -> np.ndarray:
    """SGD per MW held for one stage [T, P]: availability price x dt; regulation also pays its
    wear and the energy AGC moves (settled at, and SoC valued at, the stage's mean energy
    price ``energy_prices`` [T])."""
    value = np.asarray(ancillary_prices, float) * dt
    if Product.REGULATION in products:
        e = market.ancillary.regulation_energy
        i = products.index(Product.REGULATION)
        settled = energy_prices * dt * (e.up - e.down)
        soc_change = dt * (e.down * battery.eta_c - e.up / battery.eta_d)
        wear = battery.kappa_sgd_per_mwh * e.throughput * dt
        value[:, i] += settled + energy_prices * soc_change - wear
    return value
