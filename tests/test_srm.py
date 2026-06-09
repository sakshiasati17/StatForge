import pytest
from src.srm_detection import SRMDetector


@pytest.fixture
def detector():
    return SRMDetector(alpha=0.01)


def test_no_srm_balanced(detector):
    result = detector.check({"control": 5000, "treatment": 5000})
    assert not result.srm_detected


def test_srm_detected_severe(detector):
    result = detector.check({"control": 3000, "treatment": 7000})
    assert result.srm_detected


def test_no_srm_small_deviation(detector):
    # 51/49 split when expecting 50/50 — should not trigger SRM
    result = detector.check({"control": 5100, "treatment": 4900})
    assert not result.srm_detected


def test_custom_expected_ratio(detector):
    result = detector.check(
        {"control": 3300, "treatment": 6700},
        expected_ratios={"control": 0.33, "treatment": 0.67},
    )
    assert not result.srm_detected


def test_result_has_message(detector):
    result = detector.check({"control": 5000, "treatment": 5000})
    assert isinstance(result.message, str)
    assert len(result.message) > 0


def test_to_dict_keys(detector):
    result = detector.check({"control": 5000, "treatment": 5000})
    d = result.to_dict()
    assert "srm_detected" in d
    assert "p_value" in d
    assert "chi2_statistic" in d
