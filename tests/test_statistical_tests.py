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
