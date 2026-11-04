from statistics import NormalDist

import numpy as np
import pytest
from battery_optimisation.forecast.quantiles import inverse_cdf, pit, rearrange

LEVELS = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
NORMAL = np.array([NormalDist(100, 20).inv_cdf(q) for q in LEVELS])


def test_pit_matches_levels_at_the_quantiles() -> None:
    assert pit(LEVELS, NORMAL, NORMAL) == pytest.approx(LEVELS)


def test_pit_inverse_roundtrip_including_tails() -> None:
    y = np.array([40.0, 70.0, 100.0, 130.0, 160.0, 200.0])  # 40 and 200 are beyond q01/q99
    u = pit(LEVELS, NORMAL, y)
    assert np.all((u > 0) & (u < 1))
    assert np.all(np.diff(u) > 0)
    assert u[0] < LEVELS[0] and u[-1] > LEVELS[-1]
    assert inverse_cdf(LEVELS, NORMAL, u) == pytest.approx(y)


def test_rows_are_independent() -> None:
    values = np.stack([NORMAL, NORMAL + 50])
    u = pit(LEVELS, values, np.array([100.0, 150.0]))
    assert u == pytest.approx([0.5, 0.5])


def test_rearrange_fixes_crossing_quantiles() -> None:
    crossed = np.array([[1.0, 3.0, 2.0, 4.0]])
    assert rearrange(crossed).tolist() == [[1.0, 2.0, 3.0, 4.0]]
