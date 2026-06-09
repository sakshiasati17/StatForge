import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Literal, Optional


@dataclass
class CorrectionResult:
    method: str
    original_p_values: List[float]
    adjusted_p_values: List[float]
    rejected: List[bool]
    alpha: float
    n_rejected: int
    metric_names: List[str]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "metric": self.metric_names,
            "p_value": self.original_p_values,
            "adjusted_p_value": [round(p, 6) for p in self.adjusted_p_values],
            "rejected": self.rejected,
        })


class MultipleTestingCorrection:
    """
    Multiple testing correction to control family-wise error rate (FWER)
    or false discovery rate (FDR) when testing several metrics simultaneously.

    Methods:
    - Bonferroni: controls FWER, conservative
    - Benjamini-Hochberg: controls FDR, less conservative
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def bonferroni(
        self,
        p_values: List[float],
        metric_names: Optional[List[str]] = None,
        alpha: Optional[float] = None,
    ) -> CorrectionResult:
        """
        Bonferroni correction: adjusted_alpha = alpha / n_tests.
        Controls FWER at alpha level. Very conservative when tests are correlated.
        """
        alpha = alpha or self.alpha
        n = len(p_values)
        adjusted = [min(p * n, 1.0) for p in p_values]
        rejected = [p < alpha for p in adjusted]

        return CorrectionResult(
            method="Bonferroni",
            original_p_values=list(p_values),
            adjusted_p_values=adjusted,
            rejected=rejected,
            alpha=alpha,
            n_rejected=sum(rejected),
            metric_names=metric_names or [f"metric_{i}" for i in range(n)],
        )

    def benjamini_hochberg(
        self,
        p_values: List[float],
        metric_names: Optional[List[str]] = None,
        alpha: Optional[float] = None,
    ) -> CorrectionResult:
        """
        Benjamini-Hochberg procedure: controls FDR at alpha level.
        Less conservative than Bonferroni; preferred when many metrics are tested.
        """
        alpha = alpha or self.alpha
        n = len(p_values)
        order = np.argsort(p_values)
        sorted_p = np.array(p_values)[order]

        adjusted = np.zeros(n)
        adjusted[order] = sorted_p * n / (np.arange(1, n + 1))

        # Enforce monotonicity: cummin from the right
        for i in range(n - 2, -1, -1):
            adjusted[order[i]] = min(adjusted[order[i]], adjusted[order[i + 1]])

        adjusted = np.minimum(adjusted, 1.0)
        rejected = [float(p) < alpha for p in adjusted]

        return CorrectionResult(
            method="Benjamini-Hochberg",
            original_p_values=list(p_values),
            adjusted_p_values=list(adjusted),
            rejected=rejected,
            alpha=alpha,
            n_rejected=sum(rejected),
            metric_names=metric_names or [f"metric_{i}" for i in range(n)],
        )

    def compare_methods(
        self,
        p_values: List[float],
        metric_names: Optional[List[str]] = None,
        alpha: Optional[float] = None,
    ) -> pd.DataFrame:
        """Run both corrections and return a comparison DataFrame."""
        bonf = self.bonferroni(p_values, metric_names, alpha)
        bh = self.benjamini_hochberg(p_values, metric_names, alpha)

        df = bonf.to_dataframe().rename(columns={
            "adjusted_p_value": "bonferroni_adj_p",
            "rejected": "bonferroni_rejected",
        })
        bh_df = bh.to_dataframe()[["adjusted_p_value", "rejected"]].rename(columns={
            "adjusted_p_value": "bh_adj_p",
            "rejected": "bh_rejected",
        })
        return pd.concat([df, bh_df], axis=1)


