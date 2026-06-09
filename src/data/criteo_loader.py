import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


class CriteoLoader:
    """
    Loader for the Criteo Uplift Modeling Dataset.

    Dataset: 13.9M rows, real A/B test from Criteo (ad tech company).
    Download: https://www.kaggle.com/datasets/arashnic/uplift-modeling
              or: pip install kaggle && kaggle datasets download arashnic/uplift-modeling

    Columns:
        f0–f11  : anonymized user features (f0 correlates well with visit/conversion)
        treatment: 1 = treatment group, 0 = control group
        conversion: 1 = user converted, 0 = did not
        visit    : 1 = user visited, 0 = did not
        exposure : whether treatment ad was actually delivered
    """

    REQUIRED_COLUMNS = {"treatment", "conversion", "visit", "f0"}

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self._df: Optional[pd.DataFrame] = None

    def load(self, sample_n: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
        """
        Load dataset, optionally sampling for faster iteration.

        Args:
            sample_n: if set, load a random sample of this many rows
            seed: random seed for sampling
        """
        if not self.filepath.exists():
            raise FileNotFoundError(
                f"Criteo dataset not found at {self.filepath}.\n"
                "Download from: https://www.kaggle.com/datasets/arashnic/uplift-modeling\n"
                "Expected filename: criteo-uplift-v2.1.csv"
            )

        df = pd.read_csv(self.filepath)
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Dataset missing expected columns: {missing}")

        if sample_n and sample_n < len(df):
            df = df.sample(n=sample_n, random_state=seed).reset_index(drop=True)

        self._df = df
        return df

    def split_groups(self, df: Optional[pd.DataFrame] = None) -> tuple:
        """Return (control_df, treatment_df)."""
        df = df if df is not None else self._df
        if df is None:
            raise ValueError("Call load() first.")
        return df[df.treatment == 0].copy(), df[df.treatment == 1].copy()

    def to_experiment_format(
        self,
        df: Optional[pd.DataFrame] = None,
        covariate_col: str = "f0",
    ) -> dict:
        """
        Convert Criteo dataset to StatForge experiment format.

        Uses f0 as the CUPED pre-experiment covariate — it has the highest
        correlation with conversion/visit among the feature columns.

        Returns dict compatible with StatisticalTestEngine + CUPED inputs.
        """
        df = df if df is not None else self._df
        if df is None:
            raise ValueError("Call load() first.")

        ctrl, trt = self.split_groups(df)

        return {
            "control_conversions": ctrl["conversion"].values,
            "treatment_conversions": trt["conversion"].values,
            "control_visits": ctrl["visit"].values,
            "treatment_visits": trt["visit"].values,
            "control_covariate": ctrl[covariate_col].values,
            "treatment_covariate": trt[covariate_col].values,
            "control_revenue": ctrl["visit"].values.astype(float),
            "treatment_revenue": trt["visit"].values.astype(float),
            "n_control": len(ctrl),
            "n_treatment": len(trt),
            "covariate_col": covariate_col,
        }

    def feature_covariate_correlations(self, df: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Show correlation of each feature column with 'visit' metric.
        Helps pick the best CUPED covariate.
        """
        df = df if df is not None else self._df
        feature_cols = [c for c in df.columns if c.startswith("f")]
        return df[feature_cols].corrwith(df["visit"]).sort_values(ascending=False)

    def summary(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Print experiment summary statistics."""
        df = df if df is not None else self._df
        ctrl, trt = self.split_groups(df)
        return pd.DataFrame({
            "group": ["control", "treatment"],
            "n": [len(ctrl), len(trt)],
            "conversion_rate": [ctrl.conversion.mean(), trt.conversion.mean()],
            "visit_rate": [ctrl.visit.mean(), trt.visit.mean()],
            "split_ratio": [len(ctrl) / len(df), len(trt) / len(df)],
        })
