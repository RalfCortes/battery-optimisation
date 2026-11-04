from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text())


def test_battery_config_has_required_keys() -> None:
    cfg = _load("battery.yaml")
    required = {
        "p_max_mw",
        "e_max_mwh",
        "soc_min",
        "soc_max",
        "soc_init",
        "eta_c",
        "eta_d",
        "kappa_sgd_per_mwh",
    }
    assert required <= cfg.keys()
    assert 0 <= cfg["soc_min"] <= cfg["soc_init"] <= cfg["soc_max"] <= cfg["e_max_mwh"]
    assert 0 < cfg["eta_c"] <= 1 and 0 < cfg["eta_d"] <= 1


def test_market_config_has_required_keys() -> None:
    cfg = _load("market.yaml")
    required = {"interval_h", "n_tranches", "price_floor", "price_cap", "products"}
    assert required <= cfg.keys()
    assert cfg["interval_h"] * cfg["intervals_per_day"] == 24
    assert cfg["price_floor"] < cfg["price_cap"]
