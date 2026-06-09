import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SRMResult:
    chi2_statistic: float
    p_value: float
    srm_detected: bool
    alpha: float
    observed_counts: Dict[str, int]
    expected_counts: Dict[str, float]
    observed_ratios: Dict[str, float]
    expected_ratios: Dict[str, float]
    message: str

    def to_dict(self) -> dict:
        return {
            "srm_detected": self.srm_detected,
            "chi2_statistic": round(self.chi2_statistic, 4),
            "p_value": round(self.p_value, 6),
            "alpha": self.alpha,
            "message": self.message,
            "observed_counts": self.observed_counts,
            "expected_counts": {k: round(v, 1) for k, v in self.expected_counts.items()},
        }


class SRMDetector:
    """
    Sample Ratio Mismatch (SRM) detector.

    SRM occurs when the actual assignment ratio differs from the intended ratio,
    indicating broken randomization. An experiment with SRM should not be trusted —
    any metric difference may be due to selection bias rather than treatment effect.

    Detection: chi-square goodness-of-fit test on group assignment counts.
    """

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def check(
        self,
        observed_counts: Dict[str, int],
        expected_ratios: Optional[Dict[str, float]] = None,
        alpha: Optional[float] = None,
    ) -> SRMResult:
        """
        Check for SRM given observed assignment counts.

        Args:
            observed_counts: e.g. {"control": 4823, "treatment": 5177}
            expected_ratios: e.g. {"control": 0.5, "treatment": 0.5}
                             Defaults to equal allocation.
            alpha: significance level (default 0.01 — conservative for SRM)
        """
        alpha = alpha or self.alpha
        groups = list(observed_counts.keys())
        observed = np.array([observed_counts[g] for g in groups], dtype=float)
        total = observed.sum()

        if expected_ratios is None:
            expected_ratios = {g: 1.0 / len(groups) for g in groups}

        ratios = np.array([expected_ratios[g] for g in groups])
        ratios = ratios / ratios.sum()
        expected = total * ratios

        chi2, p = stats.chisquare(observed, f_exp=expected)

        srm_detected = float(p) < alpha
        obs_ratios = {g: float(observed_counts[g] / total) for g in groups}
        exp_ratios = {g: float(expected_ratios[g]) for g in groups}

        if srm_detected:
            max_deviation = max(abs(obs_ratios[g] - exp_ratios[g]) for g in groups)
            message = (
                f"SRM DETECTED (p={p:.6f}). "
                f"Max ratio deviation: {max_deviation:.3f}. "
                "Experiment results are unreliable. Investigate randomization pipeline."
            )
        else:
            message = f"No SRM detected (p={p:.6f}). Assignment ratios look healthy."

        return SRMResult(
            chi2_statistic=float(chi2),
            p_value=float(p),
            srm_detected=srm_detected,
            alpha=alpha,
            observed_counts=dict(observed_counts),
            expected_counts={g: float(expected[i]) for i, g in enumerate(groups)},
            observed_ratios=obs_ratios,
            expected_ratios=exp_ratios,
            message=message,
        )

    def check_from_dataframe(
        self,
        df,
        group_column: str = "variant",
        expected_ratios: Optional[Dict[str, float]] = None,
        alpha: Optional[float] = None,
    ) -> SRMResult:
        """Convenience wrapper that takes a DataFrame and counts from group_column."""
        counts = df[group_column].value_counts().to_dict()
        return self.check(counts, expected_ratios, alpha)
