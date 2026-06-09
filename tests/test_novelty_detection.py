import pytest
import numpy as np
from src.novelty_detection import NoveltyDetector


@pytest.fixture
def detector():
    return NoveltyDetector(alpha=0.05)


@pytest.fixture
def novelty_data(detector):
    return detector.simulate_novelty_data(n_days=21, n_per_day=1000,
                                          initial_lift=0.20, decay_rate=0.018, seed=42)


@pytest.fixture
def stable_data(detector):
    return detector.simulate_novelty_data(n_days=21, n_per_day=1000,
                                          initial_lift=0.10, decay_rate=0.0, seed=42)


def test_detects_novelty_with_high_decay(detector, novelty_data):
    ctrl, trt = novelty_data
    result = detector.detect(ctrl, trt)
    assert result.novelty_detected


def test_no_novelty_with_stable_effect(detector, stable_data):
    ctrl, trt = stable_data
    result = detector.detect(ctrl, trt)
    assert not result.novelty_detected


def test_slope_negative_on_novelty(detector, novelty_data):
    ctrl, trt = novelty_data
    result = detector.detect(ctrl, trt)
    assert result.slope < 0


def test_daily_lifts_length(detector, novelty_data):
    ctrl, trt = novelty_data
    result = detector.detect(ctrl, trt)
    assert len(result.daily_lifts) == len(ctrl)


def test_r_squared_in_range(detector, novelty_data):
    ctrl, trt = novelty_data
    result = detector.detect(ctrl, trt)
    assert 0 <= result.r_squared <= 1


def test_too_few_days_raises(detector):
    ctrl = {0: np.random.normal(0, 1, 100), 1: np.random.normal(0, 1, 100)}
    trt = {0: np.random.normal(1, 1, 100), 1: np.random.normal(1, 1, 100)}
    with pytest.raises(ValueError):
        detector.detect(ctrl, trt)


def test_recommendation_is_string(detector, novelty_data):
    ctrl, trt = novelty_data
    result = detector.detect(ctrl, trt)
    assert isinstance(result.recommendation, str)
    assert len(result.recommendation) > 0
