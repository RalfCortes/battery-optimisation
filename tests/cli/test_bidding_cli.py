from battery_optimisation.cli.bidding import bidding_tag
from battery_optimisation.config import FeasibilityMode, load_bidding


def test_bidding_runs_get_distinct_folders() -> None:
    b = load_bidding()
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
