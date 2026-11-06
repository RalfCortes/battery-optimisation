"""Best reserve mix per (SOC, energy action): corners of the storage-rule polytope."""

import numpy as np
import pytest

from battery_optimisation.config import Product
from battery_optimisation.optim.reserves import best_reserves, compact, corners, reserve_set
from battery_optimisation.sim.storage_rules import check_storage_rules

REG, PRI, CON = Product.REGULATION, Product.PRIMARY_RESERVE, Product.CONTINGENCY_RESERVE


def test_corner_optimum_matches_the_rc386_example(market_cfg, battery_cfg) -> None:
    """Idle at 20 MWh, contingency 30 and primary 8 SGD/MW: contingency fills C3, primary the
    rest of C1 (same answer as the engine's LP in ``enforce_soc``)."""
    rs = reserve_set(market_cfg, battery_cfg)
    assert rs.products == (REG, PRI, CON)
    x, ok = corners(rs, np.array([20.0]), np.array([0.0]), np.array([0.0]))
    reward, awards = best_reserves(x, ok, np.array([0.0, 8.0, 30.0]))
    con = 20.0 * battery_cfg.eta_d / 0.5
    pri = (20.0 * battery_cfg.eta_d - con / 3) * 6
    assert awards[0] == pytest.approx([0.0, pri, con], abs=1e-6)
    assert reward[0] == pytest.approx(8 * pri + 30 * con)


@pytest.mark.parametrize("firm", [False, True])
def test_best_mix_is_feasible_and_beats_random_feasible_mixes(market_cfg, battery_cfg, firm):
    rng = np.random.default_rng(0)
    rs = reserve_set(market_cfg, battery_cfg, firm=firm)
    rules = market_cfg.storage_rules
    soc = rng.uniform(0, 100, 40)
    flow = rng.uniform(-100, 100, 40)
    charge, discharge = np.maximum(-flow, 0), np.maximum(flow, 0)
    x, ok = corners(rs, soc, charge, discharge)

    def fits(i: int, awards: dict) -> bool:
        """Firm: feasible without credit from the opposite flow (check each flow alone)."""
        flows = [(0.0, discharge[i]), (charge[i], 0.0)] if firm else [(charge[i], discharge[i])]
        try:
            for c, d in flows:
                check_storage_rules(rules, battery_cfg, soc[i], c, d, awards)
        except Exception:
            return False
        return True

    for _ in range(3):
        value = rng.uniform(0, 40, 3)
        reward, awards = best_reserves(x, ok, value)
        for i in range(soc.size):
            if not fits(i, {}):  # the energy action alone breaks a rule: skip
                continue
            assert fits(i, dict(zip(rs.products, awards[i], strict=True)))
            assert reward[i] == pytest.approx(float(awards[i] @ value), abs=1e-6)
            for t in rng.uniform(0, 100, (300, 3)):
                if fits(i, dict(zip(rs.products, t, strict=True))):
                    assert float(t @ value) <= reward[i] + 1e-6


def test_compact_keeps_the_same_optimum(market_cfg, battery_cfg) -> None:
    rng = np.random.default_rng(1)
    rs = reserve_set(market_cfg, battery_cfg)
    soc, flow = rng.uniform(0, 100, 30), rng.uniform(-100, 100, 30)
    x, ok = corners(rs, soc, np.maximum(-flow, 0), np.maximum(flow, 0))
    cx, cok = compact(x, ok)
    assert cx.shape[-2] < x.shape[-2]
    value = rng.uniform(0, 40, 3)
    assert best_reserves(cx, cok, value)[0] == pytest.approx(best_reserves(x, ok, value)[0])


def test_pair_rewards_match_the_per_stage_optimum(market_cfg, battery_cfg) -> None:
    from battery_optimisation.optim.reserves import pair_corners

    pc = pair_corners(market_cfg, battery_cfg, 21, market_cfg.interval_h, True)
    value = np.random.default_rng(3).uniform(0, 20, (4, 3))
    fast = pc.rewards(value)
    for t in range(4):
        assert fast[t] == pytest.approx(best_reserves(pc.x, pc.ok, value[t])[0])


def test_mixes_along_a_policy_earn_its_rewards(market_cfg, battery_cfg) -> None:
    from battery_optimisation.optim.reserves import pair_corners

    pc = pair_corners(market_cfg, battery_cfg, 21, market_cfg.interval_h, True)
    rng = np.random.default_rng(1)
    value = rng.uniform(-2, 20, (4, len(pc.products)))
    policy = rng.integers(0, 21, (4, 3, 21)).astype(np.int16)
    mixes = pc.mixes(value, policy)
    assert mixes.shape == (4, 3, 21, len(pc.products))
    rewards = pc.rewards(value)
    n = np.arange(21)[None, None, :]
    expected = rewards[np.arange(4)[:, None, None], n, policy]
    assert np.einsum("tknp,tp->tkn", mixes, value) == pytest.approx(expected, abs=1e-6)


def test_rewards_and_mixes_per_reserve_state(market_cfg, battery_cfg) -> None:
    """Reserve price states: values [T, R, P] give rewards [T, R, N, N]; on a joint policy
    (s = k R + r) the mixes use the state's own reserve values."""
    from battery_optimisation.optim.reserves import pair_corners

    pc = pair_corners(market_cfg, battery_cfg, 21, market_cfg.interval_h, True)
    rng = np.random.default_rng(2)
    value = rng.uniform(-2, 20, (4, 3, len(pc.products)))  # T=4, R=3
    rewards = pc.rewards(value)
    assert rewards.shape == (4, 3, 21, 21)
    for r in range(3):
        assert rewards[:, r] == pytest.approx(pc.rewards(value[:, r]))
    policy = rng.integers(0, 21, (4, 2 * 3, 21)).astype(np.int16)  # K=2 energy states
    mixes = pc.mixes(value, policy)
    n = np.arange(21)[None, :]
    for s in range(6):
        r = s % 3
        got = np.einsum("tnp,tp->tn", mixes[:, s], value[:, r])
        expected = rewards[np.arange(4)[:, None], r, n, policy[:, s]]
        assert got == pytest.approx(expected, abs=1e-6)
