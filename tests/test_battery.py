import pytest

from battery_optimisation.battery import Battery, LimitBreachError

DT = 0.5


def test_step_applies_efficiencies_and_degradation(battery_cfg) -> None:
    b = Battery(battery_cfg)  # starts at soc_init
    step = b.step(charge_mw=50.0, discharge_mw=0.0, dt=DT)
    expected = battery_cfg.soc_init + battery_cfg.eta_c * 50.0 * DT
    assert step.soc_mwh == pytest.approx(expected)
    assert b.soc_mwh == pytest.approx(expected)
    assert step.degradation_sgd == pytest.approx(battery_cfg.kappa_sgd_per_mwh * 50.0 * DT)

    step = b.step(charge_mw=0.0, discharge_mw=40.0, dt=DT)
    assert step.soc_mwh == pytest.approx(expected - 40.0 * DT / battery_cfg.eta_d)


def test_coefficients_match_the_soc_equation(battery_cfg) -> None:
    b = Battery(battery_cfg)
    assert b.charge_gain(DT) == pytest.approx(battery_cfg.eta_c * DT)
    assert b.discharge_loss(DT) == pytest.approx(DT / battery_cfg.eta_d)


def test_headroom_limited_by_power_and_soc(battery_cfg) -> None:
    # Power-limited: plenty of room / energy for a full-power half hour.
    assert Battery(battery_cfg, soc_mwh=battery_cfg.soc_min).max_charge_mw(DT) == 100.0
    assert Battery(battery_cfg, soc_mwh=battery_cfg.soc_max).max_discharge_mw(DT) == 100.0
    # Energy-limited: 50 MWh supports only 50 * eta_d / 0.5 = 92.2 MW for half an hour.
    half = Battery(battery_cfg, soc_mwh=50.0)
    assert half.max_discharge_mw(DT) == pytest.approx(50.0 * battery_cfg.eta_d / DT)

    nearly_full = Battery(battery_cfg, soc_mwh=battery_cfg.soc_max - 1.0)
    assert nearly_full.max_charge_mw(DT) == pytest.approx(1.0 / (battery_cfg.eta_c * DT))
    nearly_empty = Battery(battery_cfg, soc_mwh=battery_cfg.soc_min + 1.0)
    assert nearly_empty.max_discharge_mw(DT) == pytest.approx(1.0 * battery_cfg.eta_d / DT)


@pytest.mark.parametrize(
    ("charge_mw", "discharge_mw", "match"),
    [
        (0.0, 200.0, "power"),  # above p_max
        (0.0, 100.0, "soc"),  # needs 100 * 0.5 / eta_d = 54 MWh, only 50 available
        (10.0, 10.0, "both"),  # simultaneous
    ],
)
def test_step_rejects_breaches_and_keeps_state(battery_cfg, charge_mw, discharge_mw, match):
    b = Battery(battery_cfg, soc_mwh=50.0)
    with pytest.raises(LimitBreachError, match=match):
        b.step(charge_mw=charge_mw, discharge_mw=discharge_mw, dt=DT)
    assert b.soc_mwh == 50.0


def test_initial_soc_must_be_within_bounds(battery_cfg) -> None:
    with pytest.raises(ValueError):
        Battery(battery_cfg, soc_mwh=battery_cfg.soc_max + 1)
