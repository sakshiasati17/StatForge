import pytest
import numpy as np
from src.cuped import CUPED


@pytest.fixture
def correlated_data():
    rng = np.random.default_rng(42)
    n = 1000  # per group
    pre_c = rng.normal(50, 20, n)
    pre_t = rng.normal(50, 20, n)
    post_ctrl = 0.6 * pre_c + rng.normal(0, 16, n)
    post_trt = 0.6 * pre_t + rng.normal(5, 16, n)
    return pre_c, pre_t, post_ctrl, post_trt


def test_fit_returns_theta(correlated_data):
    pre_c, pre_t, post_c, post_t = correlated_data
    pre = np.concatenate([pre_c, pre_t])
    post = np.concatenate([post_c, post_t])
    cuped = CUPED()
    theta = cuped.fit(pre, post)
    assert isinstance(theta, float)
    assert theta != 0


def test_variance_reduction_positive(correlated_data):
    pre_c, pre_t, post_c, post_t = correlated_data
    cuped = CUPED()
    result = cuped.fit_transform(post_c, post_t, pre_c, pre_t)
    assert result.variance_reduction_pct > 0


def test_variance_reduction_reasonable(correlated_data):
    pre_c, pre_t, post_c, post_t = correlated_data
    cuped = CUPED()
    result = cuped.fit_transform(post_c, post_t, pre_c, pre_t)
    # With ρ=0.6, expect roughly 36% variance reduction (ρ²)
    assert 10 < result.variance_reduction_pct < 80


def test_adjusted_arrays_same_length(correlated_data):
    pre_c, pre_t, post_c, post_t = correlated_data
    cuped = CUPED()
    result = cuped.fit_transform(post_c, post_t, pre_c, pre_t)
    assert len(result.adjusted_control) == len(post_c)
    assert len(result.adjusted_treatment) == len(post_t)


def test_transform_without_fit_raises(correlated_data):
    pre_c, pre_t, post_c, post_t = correlated_data
    cuped = CUPED()
    with pytest.raises(ValueError):
        cuped.transform(post_c, pre_c)
