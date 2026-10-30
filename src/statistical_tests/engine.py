import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class TestResult:
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: tuple
    sample_size_control: int
    sample_size_treatment: int
    significant: bool
    alpha: float
    power: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "test": self.test_name,
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 4),
            "effect_size": round(self.effect_size, 4),
            "ci_lower": round(self.confidence_interval[0], 4),
            "ci_upper": round(self.confidence_interval[1], 4),
            "n_control": self.sample_size_control,
            "n_treatment": self.sample_size_treatment,
            "significant": self.significant,
            "alpha": self.alpha,
        }


class StatisticalTestEngine:
    """Runs t-test, Mann-Whitney U, and chi-square tests for A/B experiments."""

    def __init__(self, alpha: float = 0.05, two_tailed: bool = True):
        self.alpha = alpha
        self.two_tailed = two_tailed

    def t_test(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        alpha: Optional[float] = None,
        equal_var: bool = False,
    ) -> TestResult:
        """Welch's t-test for comparing means between two groups."""
        alpha = alpha if alpha is not None else self.alpha
        alternative = "two-sided" if self.two_tailed else "greater"

        # Compare treatment vs control so positive statistic corresponds to treatment > control
        stat, p = stats.ttest_ind(treatment, control, equal_var=equal_var, alternative=alternative)

        mean_diff = float(np.mean(treatment) - np.mean(control))
        n_c, n_t = len(control), len(treatment)
        var_c = float(np.var(control, ddof=1)) if n_c > 1 else 0.0
        var_t = float(np.var(treatment, ddof=1)) if n_t > 1 else 0.0

        pooled_std = float(np.sqrt((var_c + var_t) / 2))
        effect_size = mean_diff / pooled_std if pooled_std > 0 else 0.0

        if equal_var:
            df = float(n_c + n_t - 2)
            s_pooled = ((n_c - 1) * var_c + (n_t - 1) * var_t) / df if df > 0 else 1.0
            se = float(np.sqrt(s_pooled * (1 / n_c + 1 / n_t)))
        else:
            # Welch-Satterthwaite degrees of freedom
            v_c = var_c / n_c if n_c > 0 else 0.0
            v_t = var_t / n_t if n_t > 0 else 0.0
            se = float(np.sqrt(v_c + v_t))
            num = (v_c + v_t) ** 2
            denom = ((v_c ** 2) / (n_c - 1) + (v_t ** 2) / (n_t - 1)) if (n_c > 1 and n_t > 1) else 1.0
            df = float(num / denom) if denom > 0 else 1.0

        if self.two_tailed:
            t_crit = float(stats.t.ppf(1 - alpha / 2, df=df))
            ci = (mean_diff - t_crit * se, mean_diff + t_crit * se)
        else:
            t_crit = float(stats.t.ppf(1 - alpha, df=df))
            ci = (mean_diff - t_crit * se, float("inf"))

        return TestResult(
            test_name="Welch's t-test",
            statistic=float(stat),
            p_value=float(p),
            effect_size=effect_size,
            confidence_interval=ci,
            sample_size_control=n_c,
            sample_size_treatment=n_t,
            significant=float(p) < alpha,
            alpha=alpha,
        )

    def mann_whitney(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        alpha: Optional[float] = None,
    ) -> TestResult:
        """Non-parametric Mann-Whitney U test for comparing distributions."""
        alpha = alpha if alpha is not None else self.alpha
        alternative = "two-sided" if self.two_tailed else "greater"

        # Compare treatment vs control so alternative="greater" tests treatment > control
        stat, p = stats.mannwhitneyu(treatment, control, alternative=alternative)

        n_c, n_t = len(control), len(treatment)
        r = 1 - (2 * float(stat)) / (n_c * n_t) if (n_c > 0 and n_t > 0) else 0.0

        ci = self._bootstrap_ci(control, treatment, np.median, alpha)

        return TestResult(
            test_name="Mann-Whitney U",
            statistic=float(stat),
            p_value=float(p),
            effect_size=r,
            confidence_interval=ci,
            sample_size_control=n_c,
            sample_size_treatment=n_t,
            significant=float(p) < alpha,
            alpha=alpha,
        )

    def chi_square(
        self,
        conversions_control: int,
        n_control: int,
        conversions_treatment: int,
        n_treatment: int,
        alpha: Optional[float] = None,
    ) -> TestResult:
        """Chi-square test for comparing conversion rates."""
        alpha = alpha if alpha is not None else self.alpha

        observed = np.array([
            [conversions_control, n_control - conversions_control],
            [conversions_treatment, n_treatment - conversions_treatment],
        ])
        stat, p, dof, expected = stats.chi2_contingency(observed, correction=False)

        rate_c = conversions_control / n_control
        rate_t = conversions_treatment / n_treatment
        relative_lift = (rate_t - rate_c) / rate_c if rate_c > 0 else 0.0

        ci = self._proportion_ci(rate_c, rate_t, n_control, n_treatment, alpha)

        return TestResult(
            test_name="Chi-square",
            statistic=float(stat),
            p_value=float(p),
            effect_size=relative_lift,
            confidence_interval=ci,
            sample_size_control=n_control,
            sample_size_treatment=n_treatment,
            significant=float(p) < alpha,
            alpha=alpha,
        )

    def _bootstrap_ci(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        stat_fn,
        alpha: float,
        n_boot: int = 1000,
        seed: int = 42,
    ) -> tuple:
        rng = np.random.default_rng(seed)
        diffs = []
        for _ in range(n_boot):
            c_boot = rng.choice(control, size=len(control), replace=True)
            t_boot = rng.choice(treatment, size=len(treatment), replace=True)
            diffs.append(stat_fn(t_boot) - stat_fn(c_boot))
        if self.two_tailed:
            lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            return float(lo), float(hi)
        else:
            lo = np.percentile(diffs, 100 * alpha)
            return float(lo), float("inf")

    def _proportion_ci(
        self,
        rate_c: float,
        rate_t: float,
        n_c: int,
        n_t: int,
        alpha: float,
    ) -> tuple:
        diff = rate_t - rate_c
        se = np.sqrt(rate_c * (1 - rate_c) / n_c + rate_t * (1 - rate_t) / n_t)
        if self.two_tailed:
            z = stats.norm.ppf(1 - alpha / 2)
            return float(diff - z * se), float(diff + z * se)
        else:
            z = stats.norm.ppf(1 - alpha)
            return float(diff - z * se), float("inf")
