"""
Simulation-based statistical validation.
Runs thousands of experiments under known conditions to verify:
  - Type-I error rate (false positive rate under null)
  - Type-II error rate (false negative rate under alternative)
  - Confidence interval coverage
  - CUPED variance reduction is real
"""
import pytest
import numpy as np
from src.statistical_tests import StatisticalTestEngine
from src.cuped import CUPED
from src.power_analysis import PowerAnalysis


N_SIM = 5000
ALPHA = 0.05
TOLERANCE = 0.025  # allow ±2.5pp Monte Carlo variance around nominal rate


# ---------------------------------------------------------------------------
# Type-I error (false positive rate under the null)
# ---------------------------------------------------------------------------

class TestTypeIError:

    def test_t_test_type_i_error(self):
        """t-test false positive rate should be ≈ alpha under H0."""
        rng = np.random.default_rng(42)
        engine = StatisticalTestEngine(alpha=ALPHA)
        rejections = sum(
            engine.t_test(rng.normal(0, 1, 200), rng.normal(0, 1, 200)).significant
            for _ in range(N_SIM)
        )
        fpr = rejections / N_SIM
        assert fpr <= ALPHA + TOLERANCE, f"Type-I error {fpr:.3f} exceeds alpha+tolerance"

    def test_chi_square_type_i_error(self):
        """Chi-square false positive rate should be ≈ alpha under H0."""
        rng = np.random.default_rng(43)
        engine = StatisticalTestEngine(alpha=ALPHA)
        rejections = 0
        for _ in range(N_SIM):
            # same conversion rate in both groups
            n = 500
            p = 0.10
            conv_c = int(rng.binomial(n, p))
            conv_t = int(rng.binomial(n, p))
            if engine.chi_square(conv_c, n, conv_t, n).significant:
                rejections += 1
        fpr = rejections / N_SIM
        assert fpr <= ALPHA + TOLERANCE, f"Type-I error {fpr:.3f} exceeds alpha+tolerance"

    def test_mann_whitney_type_i_error(self):
        """Mann-Whitney U false positive rate should be ≈ alpha under H0."""
        rng = np.random.default_rng(44)
        engine = StatisticalTestEngine(alpha=ALPHA)
        n_sim = 2000  # MW is slower; 2K gives stable estimate
        rejections = sum(
            engine.mann_whitney(rng.normal(0, 1, 200), rng.normal(0, 1, 200)).significant
            for _ in range(n_sim)
        )
        fpr = rejections / n_sim
        assert fpr <= ALPHA + TOLERANCE, f"Type-I error {fpr:.3f} exceeds alpha+tolerance"


# ---------------------------------------------------------------------------
# Type-II error / power (false negative rate under alternative)
# ---------------------------------------------------------------------------

class TestPower:

    def test_t_test_power_at_designed_n(self):
        """t-test should achieve ≥ 75% power at the sample size designed for 80%."""
        rng = np.random.default_rng(50)
        pa = PowerAnalysis(alpha=ALPHA, power=0.80)
        effect = 0.3  # Cohen's d
        std = 1.0
        mde = effect * std
        n = pa.sample_size_means(mean=0.0, std=std, min_detectable_effect=mde)

        engine = StatisticalTestEngine(alpha=ALPHA)
        detections = sum(
            engine.t_test(rng.normal(0, std, n), rng.normal(effect, std, n)).significant
            for _ in range(N_SIM)
        )
        power_achieved = detections / N_SIM
        assert power_achieved >= 0.75, f"Power {power_achieved:.3f} below threshold"

    def test_power_increases_with_effect_size(self):
        """Larger effects should be detected more often."""
        rng = np.random.default_rng(51)
        engine = StatisticalTestEngine(alpha=ALPHA)
        n = 300

        def detection_rate(effect):
            return sum(
                engine.t_test(rng.normal(0, 1, n), rng.normal(effect, 1, n)).significant
                for _ in range(1000)
            ) / 1000

        power_small = detection_rate(0.1)
        power_large = detection_rate(0.5)
        assert power_large > power_small


# ---------------------------------------------------------------------------
# Confidence interval coverage
# ---------------------------------------------------------------------------

class TestCICoverage:

    def test_t_test_ci_coverage(self):
        """95% CI should contain the true mean difference ≈ 95% of the time."""
        rng = np.random.default_rng(60)
        engine = StatisticalTestEngine(alpha=ALPHA)
        true_diff = 0.5
        covered = 0
        for _ in range(N_SIM):
            ctrl = rng.normal(0, 1, 200)
            trt = rng.normal(true_diff, 1, 200)
            result = engine.t_test(ctrl, trt)
            lo, hi = result.confidence_interval
            if lo <= true_diff <= hi:
                covered += 1
        coverage = covered / N_SIM
        assert abs(coverage - 0.95) <= TOLERANCE, f"CI coverage {coverage:.3f} far from 0.95"

    def test_t_test_ci_contains_zero_under_null(self):
        """Under H0, CI should contain zero most of the time (≥ 1 - alpha)."""
        rng = np.random.default_rng(61)
        engine = StatisticalTestEngine(alpha=ALPHA)
        contains_zero = sum(
            (lambda r: r.confidence_interval[0] <= 0 <= r.confidence_interval[1])(
                engine.t_test(rng.normal(0, 1, 200), rng.normal(0, 1, 200))
            )
            for _ in range(N_SIM)
        )
        rate = contains_zero / N_SIM
        assert rate >= 1 - ALPHA - TOLERANCE, f"Zero coverage rate {rate:.3f} too low"


# ---------------------------------------------------------------------------
# CUPED variance reduction is real
# ---------------------------------------------------------------------------

class TestCUPEDSimulation:

    def test_cuped_reduces_variance(self):
        """Adjusted metric variance should be lower than raw variance on average."""
        rng = np.random.default_rng(70)
        cuped = CUPED()
        reductions = []
        for _ in range(500):
            n = 500
            pre_c = rng.normal(50, 20, n)
            pre_t = rng.normal(50, 20, n)
            post_c = 0.6 * pre_c + rng.normal(0, 10, n)
            post_t = 0.6 * pre_t + rng.normal(5, 10, n)
            result = cuped.fit_transform(post_c, post_t, pre_c, pre_t)
            reductions.append(result.variance_reduction_pct)
        assert np.mean(reductions) > 5.0, "CUPED should reduce variance by >5% on average with ρ≈0.6"

    def test_cuped_zero_correlation_no_reduction(self):
        """With zero correlation covariate, CUPED should give near-zero reduction."""
        rng = np.random.default_rng(71)
        cuped = CUPED()
        reductions = []
        for _ in range(200):
            n = 500
            post_c = rng.normal(50, 20, n)
            post_t = rng.normal(55, 20, n)
            pre_c = rng.normal(50, 20, n)   # uncorrelated
            pre_t = rng.normal(50, 20, n)
            result = cuped.fit_transform(post_c, post_t, pre_c, pre_t)
            reductions.append(result.variance_reduction_pct)
        assert np.mean(reductions) < 10.0, "Uncorrelated covariate should give minimal reduction"
