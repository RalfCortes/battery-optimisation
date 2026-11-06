import pytest
from synthetic import FixedStrategy, make_prices

from battery_optimisation.battery import Battery
from battery_optimisation.columns import Col
from battery_optimisation.sim.clearing import Market
from battery_optimisation.sim.metrics import Period
from battery_optimisation.sim.simulator import Simulator


def test_cycles_metric(market_cfg, battery_cfg) -> None:
    """Fill from soc_min to soc_max and empty again: exactly one equivalent full cycle."""
    cfg = battery_cfg.model_copy(update={"soc_init": battery_cfg.soc_min})
    dt, n = market_cfg.interval_h, 4
    charge = cfg.usable_mwh / (cfg.eta_c * dt) / n
    discharge = cfg.usable_mwh * cfg.eta_d / dt / n
    strategy = FixedStrategy([charge] * n + [0.0] * n, [0.0] * n + [discharge] * n, market_cfg)
    m = Market(market_cfg, make_prices([50.0] * n + [200.0] * n))
    dispatch = Simulator(m, Battery(cfg)).run(strategy)

    # offers are rounded down to 0.1 MW (market format): one cycle up to that step
    assert dispatch.total().cycles == pytest.approx(1.0, rel=5e-3)
    assert len(dispatch.stats(Period.DAY)) == 1
    assert dispatch.frame[Col.SOC_MWH][-1] == pytest.approx(cfg.soc_min, abs=0.5)
