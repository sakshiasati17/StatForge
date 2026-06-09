import pytest
from src.multiple_testing import MultipleTestingCorrection


@pytest.fixture
def corrector():
    return MultipleTestingCorrection(alpha=0.05)


def test_bonferroni_adjusts_upward(corrector):
    p_values = [0.01, 0.02, 0.03]
    result = corrector.bonferroni(p_values)
    assert all(adj >= orig for adj, orig in zip(result.adjusted_p_values, p_values))


def test_bh_less_conservative_than_bonferroni(corrector):
    p_values = [0.001, 0.01, 0.02, 0.03, 0.04, 0.06, 0.1, 0.2, 0.4, 0.8]
    bonf = corrector.bonferroni(p_values)
    bh = corrector.benjamini_hochberg(p_values)
    assert bh.n_rejected >= bonf.n_rejected


def test_bonferroni_significant_small_p(corrector):
    p_values = [0.001, 0.5, 0.5, 0.5, 0.5]
    result = corrector.bonferroni(p_values)
    assert result.rejected[0] is True


def test_adjusted_p_capped_at_1(corrector):
    p_values = [0.5, 0.6, 0.7, 0.8, 0.9]
    result = corrector.bonferroni(p_values)
    assert all(p <= 1.0 for p in result.adjusted_p_values)


def test_compare_methods_dataframe(corrector):
    p_values = [0.01, 0.03, 0.05, 0.10, 0.20]
    df = corrector.compare_methods(p_values)
    assert "bonferroni_adj_p" in df.columns
    assert "bh_adj_p" in df.columns
    assert len(df) == 5


def test_metric_names_passed_through(corrector):
    names = ["revenue", "conversion", "session_time"]
    result = corrector.bonferroni([0.01, 0.05, 0.10], metric_names=names)
    assert result.metric_names == names
