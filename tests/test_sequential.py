import pytest
import numpy as np
from src.sequential_testing import SequentialTester


@pytest.fixture
def tester():
    return SequentialTester(alpha=0.05, max_n=10000, n_looks=5, spending_function="obrien_fleming")


def test_alpha_spending_at_1(tester):
    spent = tester.alpha_spending(1.0)
    assert abs(spent - tester.alpha) < 1e-4


def test_alpha_spending_monotone(tester):
    fracs = [0.2, 0.4, 0.6, 0.8, 1.0]
    spends = [tester.alpha_spending(f) for f in fracs]
    assert all(spends[i] <= spends[i + 1] for i in range(len(spends) - 1))


def test_pocock_spending_at_1():
    tester = SequentialTester(spending_function="pocock", alpha=0.05)
    assert abs(tester.alpha_spending(1.0) - 0.05) < 0.01


def test_run_sequence_returns_result(tester):
    rng = np.random.default_rng(0)
    ctrl = rng.normal(0, 1, 10000)
    trt = rng.normal(0.5, 1, 10000)
    result = tester.run_full_sequence(ctrl, trt)
    assert result.final_decision in ("reject_null", "fail_to_reject")
    assert len(result.looks) > 0


def test_no_false_positives_under_null():
    """Family-wise error rate should be ≤ α under the null (stochastic check)."""
    rejections = 0
    n_sim = 200
    rng = np.random.default_rng(999)
    for _ in range(n_sim):
        ctrl = rng.normal(0, 1, 2000)
        trt = rng.normal(0, 1, 2000)
        t = SequentialTester(alpha=0.05, max_n=4000, n_looks=5)
        r = t.run_full_sequence(ctrl, trt)
        if r.final_decision == "reject_null":
            rejections += 1
    fwer = rejections / n_sim
    assert fwer <= 0.07  # Tightened bound; properly spending alpha achieves ~0.04-0.05


def test_total_alpha_spent_le_alpha(tester):
    rng = np.random.default_rng(42)
    ctrl = rng.normal(0, 1, 5000)
    trt = rng.normal(0.01, 1, 5000)
    result = tester.run_full_sequence(ctrl, trt)
    assert result.total_alpha_spent <= tester.alpha + 1e-6
    # Total incremental alpha across evaluated looks must sum to total_alpha_spent
    assert abs(sum(s.alpha_spent for s in result.looks) - result.total_alpha_spent) < 1e-6


def test_obrien_fleming_boundaries_decreasing(tester):
    """O'Brien-Fleming boundaries should decrease with increasing information fraction."""
    fracs = [0.2, 0.4, 0.6, 0.8, 1.0]
    boundaries = []
    t = SequentialTester(alpha=0.05, max_n=10000, n_looks=5, spending_function="obrien_fleming")
    for i, frac in enumerate(fracs):
        b = t.boundary_at_look(i, frac)
        boundaries.append(b)
        # Update alpha spent so subsequent boundaries represent marginal looks
        t._alpha_spent_so_far = t.alpha_spending(frac)
    for i in range(len(boundaries) - 1):
        assert boundaries[i] > boundaries[i + 1]
