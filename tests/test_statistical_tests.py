import pytest
import numpy as np
from src.statistical_tests import StatisticalTestEngine


@pytest.fixture
def engine():
    return StatisticalTestEngine(alpha=0.05)


@pytest.fixture
def null_data():
    rng = np.random.default_rng(0)
    return rng.normal(0, 1, 1000), rng.normal(0, 1, 1000)


@pytest.fixture
def effect_data():
    rng = np.random.default_rng(1)
    return rng.normal(0, 1, 5000), rng.normal(0.3, 1, 5000)


def test_t_test_null_not_significant(engine, null_data):
    ctrl, trt = null_data
    result = engine.t_test(ctrl, trt)
    assert result.p_value > 0.01  # With small n, should mostly not reject


def test_t_test_effect_significant(engine, effect_data):
    ctrl, trt = effect_data
    result = engine.t_test(ctrl, trt)
    assert result.significant
    assert result.p_value < 0.05


def test_mann_whitney_returns_result(engine, null_data):
    ctrl, trt = null_data
    result = engine.mann_whitney(ctrl, trt)
    assert 0 <= result.p_value <= 1
    assert result.test_name == "Mann-Whitney U"


def test_chi_square_detects_difference(engine):
    result = engine.chi_square(
        conversions_control=100, n_control=1000,
        conversions_treatment=150, n_treatment=1000,
    )
    assert result.significant
    assert result.effect_size > 0


def test_chi_square_null_not_significant(engine):
    result = engine.chi_square(
        conversions_control=100, n_control=1000,
        conversions_treatment=102, n_treatment=1000,
    )
    assert not result.significant


def test_result_ci_contains_zero_under_null(engine, null_data):
    ctrl, trt = null_data
    result = engine.t_test(ctrl, trt)
    assert result.confidence_interval[0] < 0 < result.confidence_interval[1]


def test_to_dict_keys(engine, null_data):
    ctrl, trt = null_data
    d = engine.t_test(ctrl, trt).to_dict()
    assert all(k in d for k in ["test", "p_value", "effect_size", "ci_lower", "ci_upper", "significant"])


def test_one_tailed_treatment_greater_significant(effect_data):
    ctrl, trt = effect_data
    engine_one_tailed = StatisticalTestEngine(alpha=0.05, two_tailed=False)
    result = engine_one_tailed.t_test(ctrl, trt)
    assert result.significant
    assert result.statistic > 0
    assert result.p_value < 0.05
    assert result.confidence_interval[1] == float("inf")
    assert result.confidence_interval[0] > 0

    # If treatment is worse, one-tailed test in 'greater' direction should not reject
    result_reversed = engine_one_tailed.t_test(trt, ctrl)
    assert not result_reversed.significant
    assert result_reversed.p_value > 0.50

    mw_result = engine_one_tailed.mann_whitney(ctrl, trt)
    assert mw_result.significant
    assert mw_result.p_value < 0.05
    assert mw_result.confidence_interval[1] == float("inf")


def test_treatment_worse_direction(engine):
    """When treatment is worse than control, effect_size and statistic are negative."""
    ctrl = np.array([10.0, 11.0, 12.0, 10.5, 11.5])
    trt = np.array([5.0, 6.0, 5.5, 4.5, 6.5])
    res = engine.t_test(ctrl, trt)
    assert res.significant
    assert res.statistic < 0
    assert res.effect_size < 0
    assert res.confidence_interval[1] < 0

    engine_one_tailed = StatisticalTestEngine(alpha=0.05, two_tailed=False)
    one_res = engine_one_tailed.t_test(ctrl, trt)
    assert not one_res.significant
    assert one_res.p_value > 0.90


def test_constant_outcomes_identical(engine):
    """Identical constant groups should yield p=1.0 and not crash."""
    ctrl = np.full(50, 5.0)
    trt = np.full(50, 5.0)
    res = engine.t_test(ctrl, trt)
    assert not res.significant
    assert res.p_value == 1.0
    assert res.statistic == 0.0

    mw_res = engine.mann_whitney(ctrl, trt)
    assert not mw_res.significant
    assert mw_res.p_value == 1.0


def test_constant_outcomes_different_levels(engine):
    """Different constant groups should detect difference without zero-division crash."""
    ctrl = np.full(20, 2.0)
    trt = np.full(20, 4.0)
    res = engine.t_test(ctrl, trt)
    assert res.significant
    assert res.p_value == 0.0
    assert res.confidence_interval[0] == 2.0


def test_small_sample_sizes(engine):
    """Minimal sample sizes (n=2) run correctly without degrees of freedom crashing."""
    ctrl = np.array([1.0, 2.0])
    trt = np.array([3.0, 4.0])
    res = engine.t_test(ctrl, trt)
    assert res.sample_size_control == 2
    assert res.sample_size_treatment == 2
    assert res.statistic > 0


def test_unequal_sample_sizes_and_variances(engine):
    """Welch's t-test handles heavily unbalanced groups with unequal variances."""
    rng = np.random.default_rng(42)
    ctrl = rng.normal(10.0, 1.0, size=25)      # small group, small variance
    trt = rng.normal(10.0, 10.0, size=2500)    # huge group, huge variance
    res = engine.t_test(ctrl, trt)
    # Under identical means, should not reject at alpha=0.05
    assert not res.significant
    assert 0.01 < res.p_value < 0.99
    assert res.confidence_interval[0] < 0 < res.confidence_interval[1]


def test_chi_square_zero_and_full_conversions(engine):
    """Chi-square handles boundary cases where no one or everyone converts."""
    # 0 conversions in both
    res_zero = engine.chi_square(conversions_control=0, n_control=200,
                                 conversions_treatment=0, n_treatment=200)
    assert not res_zero.significant
    assert res_zero.p_value == 1.0
    assert res_zero.statistic == 0.0

    # 100% conversions in both
    res_full = engine.chi_square(conversions_control=150, n_control=150,
                                 conversions_treatment=150, n_treatment=150)
    assert not res_full.significant
    assert res_full.p_value == 1.0
    assert res_full.statistic == 0.0


def test_insufficient_sample_raises(engine):
    """t-test requires at least 2 observations per group."""
    with pytest.raises(ValueError, match="at least 2 observations"):
        engine.t_test(np.array([1.0]), np.array([2.0, 3.0]))

