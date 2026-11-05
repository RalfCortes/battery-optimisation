from battery_optimisation.cli.bidding import bidding_tag
from battery_optimisation.config import FeasibilityMode, PriceModel, load_bidding


def test_bidding_runs_get_distinct_folders() -> None:
    b = load_bidding().model_copy(update={"price_model": PriceModel.INDEPENDENT})
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
    b = load_bidding()
    markov = b.model_copy(update={"price_model": PriceModel.MARKOV, "firm_reserves": False})
    fixed = markov.model_copy(update={"markov": markov.markov.model_copy(update={"rho": 0.83})})
    copula = markov.model_copy(update={"markov": markov.markov.model_copy(update={"rho": None})})
    assert bidding_tag(copula, False) == "worst_case_markov"
    assert bidding_tag(fixed, False) == "worst_case_markov_r0.83"
    assert bidding_tag(copula, True) == "worst_case_markov_as_hybrid"
