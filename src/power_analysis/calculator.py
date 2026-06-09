import numpy as np
from scipy import stats
from typing import Optional


class PowerAnalysis:
    """Sample size and power calculations for A/B tests."""

    def __init__(self, alpha: float = 0.05, power: float = 0.80, two_tailed: bool = True):
        self.alpha = alpha
        self.power = power
        self.two_tailed = two_tailed

    def sample_size_proportions(
        self,
        baseline_rate: float,
        min_detectable_effect: float,
        alpha: Optional[float] = None,
        power: Optional[float] = None,
    ) -> int:
        """Calculate required sample size per variant for a proportion test."""
        alpha = alpha or self.alpha
        power = power or self.power

        treatment_rate = baseline_rate + min_detectable_effect
        effect_size = self._cohens_h(baseline_rate, treatment_rate)
        return self._sample_size_from_effect(effect_size, alpha, power)

    def sample_size_means(
        self,
        mean: float,
        std: float,
        min_detectable_effect: float,
        alpha: Optional[float] = None,
        power: Optional[float] = None,
    ) -> int:
        """Calculate required sample size per variant for a means test."""
        alpha = alpha or self.alpha
        power = power or self.power

        effect_size = min_detectable_effect / std
        return self._sample_size_from_effect(effect_size, alpha, power)

    def achieved_power(
        self,
        n: int,
        effect_size: float,
        alpha: Optional[float] = None,
    ) -> float:
        """Calculate statistical power for a given sample size and effect size."""
        alpha = alpha or self.alpha
        z_alpha = stats.norm.ppf(1 - alpha / (2 if self.two_tailed else 1))
        z = effect_size * np.sqrt(n / 2) - z_alpha
        return float(stats.norm.cdf(z))

    def minimum_detectable_effect(
        self,
        n: int,
        std: float,
        alpha: Optional[float] = None,
        power: Optional[float] = None,
    ) -> float:
        """Calculate MDE given a fixed sample size."""
        alpha = alpha or self.alpha
        power = power or self.power

        z_alpha = stats.norm.ppf(1 - alpha / (2 if self.two_tailed else 1))
        z_beta = stats.norm.ppf(power)
        return float((z_alpha + z_beta) * std * np.sqrt(2 / n))

    def power_curve(
        self,
        effect_sizes: np.ndarray,
        n: int,
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """Return power values across a range of effect sizes for a fixed n."""
        alpha = alpha or self.alpha
        return np.array([self.achieved_power(n, es, alpha) for es in effect_sizes])

    def sample_size_curve(
        self,
        effect_sizes: np.ndarray,
        alpha: Optional[float] = None,
        power: Optional[float] = None,
    ) -> np.ndarray:
        """Return required sample sizes across a range of effect sizes."""
        alpha = alpha or self.alpha
        power = power or self.power
        return np.array([self._sample_size_from_effect(es, alpha, power) for es in effect_sizes])

    def _cohens_h(self, p1: float, p2: float) -> float:
        return float(2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1)))

    def _sample_size_from_effect(self, effect_size: float, alpha: float, power: float) -> int:
        z_alpha = stats.norm.ppf(1 - alpha / (2 if self.two_tailed else 1))
        z_beta = stats.norm.ppf(power)
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        return int(np.ceil(n))
