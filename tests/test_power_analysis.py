import pytest
import numpy as np
from src.power_analysis import PowerAnalysis


@pytest.fixture
def pa():
    return PowerAnalysis(alpha=0.05, power=0.80)


def test_sample_size_proportions_basic(pa):
    n = pa.sample_size_proportions(baseline_rate=0.10, min_detectable_effect=0.02)
    assert n > 0
    assert isinstance(n, int)


def test_sample_size_decreases_with_larger_effect(pa):
    n_small = pa.sample_size_proportions(0.10, 0.01)
    n_large = pa.sample_size_proportions(0.10, 0.05)
    assert n_small > n_large


def test_sample_size_means_basic(pa):
    n = pa.sample_size_means(mean=50.0, std=20.0, min_detectable_effect=5.0)
    assert n > 0


def test_achieved_power_increases_with_n(pa):
    p1 = pa.achieved_power(n=500, effect_size=0.2)
    p2 = pa.achieved_power(n=5000, effect_size=0.2)
    assert p2 > p1


def test_achieved_power_at_design_n(pa):
    n = pa.sample_size_means(50, 20, 5.0)
    power = pa.achieved_power(n=n, effect_size=5.0 / 20.0)
    assert power >= 0.75


def test_power_curve_length(pa):
    effect_sizes = np.linspace(0.01, 0.5, 50)
    powers = pa.power_curve(effect_sizes, n=1000)
    assert len(powers) == 50
    assert all(0 <= p <= 1 for p in powers)


def test_mde_positive(pa):
    mde = pa.minimum_detectable_effect(n=1000, std=20.0)
    assert mde > 0
