import numpy as np
import pandas as pd
from typing import Optional, Tuple


class ExperimentDataGenerator:
    """
    Generates synthetic e-commerce A/B experiment data with realistic properties:
    - Pre-experiment covariate (e.g., 30-day revenue before experiment)
    - Treatment/control assignment
    - Binary conversion metric
    - Continuous revenue metric (zero-inflated log-normal)
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        n_control: int = 5000,
        n_treatment: int = 5000,
        baseline_conversion: float = 0.10,
        treatment_effect_conversion: float = 0.02,
        baseline_revenue_mean: float = 50.0,
        baseline_revenue_std: float = 120.0,
        treatment_effect_revenue: float = 5.0,
        covariate_correlation: float = 0.6,
        srm_inject: bool = False,
        srm_ratio: float = 0.55,
    ) -> pd.DataFrame:
        """
        Generate a full experiment dataset.

        Returns DataFrame with columns:
            user_id, variant, pre_revenue (covariate),
            converted, revenue, session_count
        """
        if srm_inject:
            n_treatment = int((n_control + n_treatment) * srm_ratio)
            n_control = int((n_control + n_treatment) * (1 - srm_ratio))

        control_df = self._generate_group(
            n=n_control,
            variant="control",
            conversion_rate=baseline_conversion,
            revenue_mean=baseline_revenue_mean,
            revenue_std=baseline_revenue_std,
            treatment_effect_revenue=0.0,
            treatment_effect_conversion=0.0,
            covariate_correlation=covariate_correlation,
            id_start=0,
        )
        treatment_df = self._generate_group(
            n=n_treatment,
            variant="treatment",
            conversion_rate=baseline_conversion,
            revenue_mean=baseline_revenue_mean,
            revenue_std=baseline_revenue_std,
            treatment_effect_revenue=treatment_effect_revenue,
            treatment_effect_conversion=treatment_effect_conversion,
            covariate_correlation=covariate_correlation,
            id_start=n_control,
        )

        df = pd.concat([control_df, treatment_df], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        return df

    def _generate_group(
        self,
        n: int,
        variant: str,
        conversion_rate: float,
        revenue_mean: float,
        revenue_std: float,
        treatment_effect_revenue: float,
        treatment_effect_conversion: float,
        covariate_correlation: float,
        id_start: int,
    ) -> pd.DataFrame:
        # Pre-experiment revenue (covariate) — zero-inflated log-normal
        pre_nonzero_mask = self.rng.random(n) < 0.40
        pre_revenue = np.where(
            pre_nonzero_mask,
            np.exp(self.rng.normal(3.5, 1.0, n)),
            0.0,
        )

        # Post-experiment revenue correlated with pre-revenue
        noise = self.rng.normal(0, revenue_std, n)
        base_revenue_latent = revenue_mean + covariate_correlation * (pre_revenue - np.mean(pre_revenue)) + np.sqrt(1 - covariate_correlation ** 2) * noise

        # Treatment lifts revenue for converters
        actual_conversion_rate = conversion_rate + treatment_effect_conversion
        converted = self.rng.random(n) < actual_conversion_rate

        revenue = np.where(
            converted,
            np.maximum(base_revenue_latent + treatment_effect_revenue, 0.01),
            0.0,
        )

        session_count = self.rng.poisson(3.5 + (0.5 if variant == "treatment" else 0), n)

        return pd.DataFrame({
            "user_id": range(id_start, id_start + n),
            "variant": variant,
            "pre_revenue": np.round(pre_revenue, 2),
            "converted": converted.astype(int),
            "revenue": np.round(np.maximum(revenue, 0), 2),
            "session_count": session_count,
        })

    def generate_multi_metric(
        self,
        n_per_group: int = 5000,
        n_metrics: int = 10,
        true_effects: int = 3,
        seed: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, list]:
        """
        Generate data with multiple metrics for multiple testing demos.
        Returns (dataframe, list of metric names with true effects).
        """
        if seed:
            rng = np.random.default_rng(seed)
        else:
            rng = self.rng

        data = {}
        true_effect_metrics = [f"metric_{i}" for i in range(true_effects)]

        for i in range(n_metrics):
            control_vals = rng.normal(100, 20, n_per_group)
            if i < true_effects:
                treatment_vals = rng.normal(103, 20, n_per_group)
            else:
                treatment_vals = rng.normal(100, 20, n_per_group)
            data[f"metric_{i}_control"] = control_vals
            data[f"metric_{i}_treatment"] = treatment_vals

        return pd.DataFrame(data), true_effect_metrics
