from battery_optimisation.cli.bidding import bidding_tag
from battery_optimisation.config import FeasibilityMode, PriceModel, ReservePlanner, load_bidding


def classic():
    """Worst-case feasibility, MILP reserve plan (the tags below name them)."""
    b = load_bidding()
    return b.model_copy(
        update={
            "feasibility": b.feasibility.model_copy(update={"mode": FeasibilityMode.WORST_CASE}),
            "reserve_planner": ReservePlanner.MILP,
        }
    )


def test_bidding_runs_get_distinct_folders() -> None:
    b = classic().model_copy(update={"price_model": PriceModel.INDEPENDENT})
    firm = b.model_copy(update={"firm_reserves": True})
    hybrid = b.model_copy(update={"firm_reserves": False})
    prob = b.model_copy(
        update={
            "feasibility": b.feasibility.model_copy(update={"mode": FeasibilityMode.PROBABILISTIC})
        }
    )
    tags = [
        bidding_tag(firm, False),
        bidding_tag(firm, True),
        bidding_tag(hybrid, True),
        bidding_tag(prob, False),
    ]
    assert tags[:3] == ["worst_case", "worst_case_as", "worst_case_as_hybrid"]
    assert len(set(tags)) == 4


def test_markov_runs_get_their_own_folders() -> None:
    b = classic()
    markov = b.model_copy(update={"price_model": PriceModel.MARKOV, "firm_reserves": False})
    fixed = markov.model_copy(update={"markov": markov.markov.model_copy(update={"rho": 0.83})})
    copula = markov.model_copy(update={"markov": markov.markov.model_copy(update={"rho": None})})
    assert bidding_tag(copula, False) == "worst_case_markov"
    assert bidding_tag(fixed, False) == "worst_case_markov_r0.83"
    assert bidding_tag(copula, True) == "worst_case_markov_as_hybrid"


def test_default_runs_are_tagged_market_and_sdp_reserves() -> None:
    from battery_optimisation.config import AncillaryPrice, ReservePrices

    b = load_bidding()
    assert bidding_tag(b, False) == "market_markov"
    assert bidding_tag(b, True) == "market_markov_as_sdpres"
    stoch = b.model_copy(
        update={"reserve_planner": ReservePlanner.STOCHASTIC_MILP, "stochastic_relax": True}
    )
    assert bidding_tag(stoch, True) == "market_markov_as_stoch20lp"
    mean = b.model_copy(update={"ancillary_price": AncillaryPrice.MEAN})
    assert bidding_tag(mean, True) == "market_markov_as_sdpres_ancmean"
    assert bidding_tag(mean, False) == "market_markov"  # no reserves: no effect
    states = b.model_copy(update={"reserve_prices": ReservePrices.MARKOV})
    assert bidding_tag(states, True) == "market_markov_as_sdpres_resmkv5"


def test_each_bidding_run_gets_its_own_folder(tmp_path) -> None:
    """Runs are never overwritten: bidding/<tag>/<timestamp>/, with ``latest`` pointing at the
    newest one."""
    from datetime import datetime

    from battery_optimisation.bidding.report import list_bidding_runs, new_bidding_dir

    first = new_bidding_dir(tmp_path, "market_markov_as_sdpres", datetime(2026, 11, 5, 14, 0))
    second = new_bidding_dir(tmp_path, "market_markov_as_sdpres", datetime(2026, 11, 5, 15, 30))
    assert first != second and first.is_dir() and second.is_dir()
    assert first.relative_to(tmp_path).parts == (
        "bidding",
        "market_markov_as_sdpres",
        "20261105-1400",
    )
    latest = tmp_path / "bidding" / "market_markov_as_sdpres" / "latest"
    assert latest.resolve() == second.resolve()
    (first / "report.json").write_text("{}")
    (second / "report.json").write_text("{}")
    assert list_bidding_runs(tmp_path) == [second, first]  # newest first, no "latest" twice


def test_continuous_price_state_runs_get_their_own_folders() -> None:
    b = load_bidding().model_copy(update={"price_model": PriceModel.CONTINUOUS})
    assert bidding_tag(b, False) == f"market_continuous{b.continuous.nodes}"
    fixed = b.model_copy(update={"markov": b.markov.model_copy(update={"rho": 0.83})})
    assert bidding_tag(fixed, True) == f"market_continuous{b.continuous.nodes}_r0.83_as_sdpres"
