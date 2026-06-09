import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class NoveltyResult:
    daily_lifts: List[float]
    days: List[int]
    slope: float
    intercept: float
    p_value: float
    r_squared: float
    novelty_detected: bool
    recommendation: str
    stabilization_day: Optional[int]

    def to_dict(self) -> dict:
        return {
            "novelty_detected": self.novelty_detected,
            "slope": round(self.slope, 6),
            "p_value": round(self.p_value, 4),
            "r_squared": round(self.r_squared, 4),
            "recommendation": self.recommendation,
            "stabilization_day": self.stabilization_day,
        }


class NoveltyDetector:
    """
    Detects novelty effects in A/B experiments by tracking daily treatment lift.

    A novelty effect occurs when users react to newness rather than genuine value.
    It manifests as a significantly declining treatment effect over time.
    This is a violation of SUTVA (Stable Unit Treatment Value Assumption).

    Detection: linear regression on daily lift values. Significant negative slope
    indicates declining effect — flag as novelty.
    """

    def __init__(self, alpha: float = 0.05, min_days: int = 5):
        self.alpha = alpha
        self.min_days = min_days

    def detect(
        self,
        daily_control: Dict[int, np.ndarray],
        daily_treatment: Dict[int, np.ndarray],
        alpha: Optional[float] = None,
    ) -> NoveltyResult:
        """
        Compute per-day treatment lift and test for declining trend.

        Args:
            daily_control: {day_index: array of metric values for control}
            daily_treatment: {day_index: array of metric values for treatment}
        """
        alpha = alpha or self.alpha
        days = sorted(daily_control.keys())

        if len(days) < self.min_days:
            raise ValueError(f"Need at least {self.min_days} days of data, got {len(days)}.")

        daily_lifts = []
        for day in days:
            c_mean = float(np.mean(daily_control[day]))
            t_mean = float(np.mean(daily_treatment[day]))
            lift = (t_mean - c_mean) / c_mean if c_mean != 0 else 0.0
            daily_lifts.append(lift)

        x = np.arange(len(days), dtype=float)
        slope, intercept, r_value, p_value, _ = stats.linregress(x, daily_lifts)

        novelty_detected = bool(slope < 0 and float(p_value) < alpha)
        stabilization_day = self._estimate_stabilization(daily_lifts, slope, intercept)

        if novelty_detected:
            recommendation = (
                f"Novelty effect detected (slope={slope:.4f}, p={p_value:.4f}). "
                f"Effect is declining. "
                + (f"Estimated stabilization around day {stabilization_day}. " if stabilization_day else "")
                + "Extend experiment and re-evaluate before shipping."
            )
        else:
            recommendation = (
                f"No novelty effect detected (slope={slope:.4f}, p={p_value:.4f}). "
                "Treatment effect appears stable."
            )

        return NoveltyResult(
            daily_lifts=daily_lifts,
            days=list(days),
            slope=float(slope),
            intercept=float(intercept),
            p_value=float(p_value),
            r_squared=float(r_value ** 2),
            novelty_detected=novelty_detected,
            recommendation=recommendation,
            stabilization_day=stabilization_day,
        )

    def simulate_novelty_data(
        self,
        n_days: int = 21,
        n_per_day: int = 500,
        initial_lift: float = 0.20,
        decay_rate: float = 0.015,
        base_mean: float = 50.0,
        base_std: float = 20.0,
        seed: int = 42,
    ) -> tuple:
        """
        Simulate experiment data with a decaying novelty effect.
        Returns (daily_control, daily_treatment).
        """
        rng = np.random.default_rng(seed)
        daily_control = {}
        daily_treatment = {}

        for day in range(n_days):
            current_lift = max(initial_lift - decay_rate * day, 0.0)
            daily_control[day] = rng.normal(base_mean, base_std, n_per_day)
            daily_treatment[day] = rng.normal(
                base_mean * (1 + current_lift), base_std, n_per_day
            )

        return daily_control, daily_treatment

    def _estimate_stabilization(
        self,
        daily_lifts: List[float],
        slope: float,
        intercept: float,
    ) -> Optional[int]:
        """Estimate day where trend crosses zero or plateaus (< 1% lift)."""
        if slope >= 0:
            return None
        # Day where trend line hits 1% lift
        target = 0.01
        if intercept <= target:
            return None
        day = int((target - intercept) / slope)
        return max(day, len(daily_lifts)) if day > len(daily_lifts) else None
