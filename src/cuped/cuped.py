import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class CUPEDResult:
    theta: float
    variance_reduction_pct: float
    original_variance: float
    adjusted_variance: float
    adjusted_control: np.ndarray
    adjusted_treatment: np.ndarray
    original_control: np.ndarray
    original_treatment: np.ndarray

    def summary(self) -> dict:
        return {
            "theta": round(self.theta, 4),
            "variance_reduction_pct": round(self.variance_reduction_pct, 2),
            "original_variance": round(self.original_variance, 6),
            "adjusted_variance": round(self.adjusted_variance, 6),
            "sample_size_reduction_pct": round(self.variance_reduction_pct, 2),
        }


class CUPED:
    """
    CUPED (Controlled-experiment Using Pre-Experiment Data) for variance reduction.

    Reduces metric variance by regressing out pre-experiment covariate signal,
    enabling the same effect to be detected with fewer users or in less time.

    Reference: Deng et al. (2013) - "Improving the Sensitivity of Online
    Controlled Experiments by Utilizing Pre-Experiment Data"
    """

    def __init__(self):
        self.theta: Optional[float] = None
        self.covariate_mean: Optional[float] = None

    def fit(
        self,
        covariate: np.ndarray,
        metric: np.ndarray,
    ) -> float:
        """
        Compute theta (regression coefficient) from covariate and metric.
        theta = Cov(Y, X) / Var(X)
        """
        cov_matrix = np.cov(metric, covariate)
        self.theta = float(cov_matrix[0, 1] / cov_matrix[1, 1])
        self.covariate_mean = float(np.mean(covariate))
        return self.theta

    def transform(
        self,
        metric: np.ndarray,
        covariate: np.ndarray,
        theta: Optional[float] = None,
        covariate_mean: Optional[float] = None,
    ) -> np.ndarray:
        """
        Apply CUPED adjustment: Y_cuped = Y - theta * (X - E[X])
        """
        theta = theta if theta is not None else self.theta
        covariate_mean = covariate_mean if covariate_mean is not None else self.covariate_mean

        if theta is None or covariate_mean is None:
            raise ValueError("Call fit() before transform(), or pass theta and covariate_mean explicitly.")

        return metric - theta * (covariate - covariate_mean)

    def fit_transform(
        self,
        control_metric: np.ndarray,
        treatment_metric: np.ndarray,
        control_covariate: np.ndarray,
        treatment_covariate: np.ndarray,
    ) -> CUPEDResult:
        """
        Fit theta on pooled data, then apply CUPED to both groups.
        Returns adjusted metrics and variance reduction statistics.
        """
        pooled_metric = np.concatenate([control_metric, treatment_metric])
        pooled_covariate = np.concatenate([control_covariate, treatment_covariate])

        self.fit(pooled_covariate, pooled_metric)

        adj_control = self.transform(control_metric, control_covariate)
        adj_treatment = self.transform(treatment_metric, treatment_covariate)

        original_var = float(np.var(pooled_metric))
        adjusted_var = float(np.var(np.concatenate([adj_control, adj_treatment])))
        variance_reduction_pct = (1 - adjusted_var / original_var) * 100

        return CUPEDResult(
            theta=self.theta,
            variance_reduction_pct=variance_reduction_pct,
            original_variance=original_var,
            adjusted_variance=adjusted_var,
            adjusted_control=adj_control,
            adjusted_treatment=adj_treatment,
            original_control=control_metric,
            original_treatment=treatment_metric,
        )

    def required_sample_size_reduction(self, variance_reduction_pct: float) -> float:
        """
        Given a variance reduction %, return the % reduction in required sample size.
        Because n ∝ sigma^2, a 40% variance reduction → 40% fewer users needed.
        """
        return variance_reduction_pct
