import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import List, Optional, Literal


@dataclass
class SequentialTestState:
    look_number: int
    information_fraction: float
    alpha_spent: float
    alpha_boundary: float
    p_value: float
    z_statistic: float
    n_control: int
    n_treatment: int
    should_stop: bool
    reject_null: bool


@dataclass
class SequentialTestResult:
    looks: List[SequentialTestState] = field(default_factory=list)
    final_decision: str = "continue"
    total_alpha_spent: float = 0.0

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame([
            {
                "look": s.look_number,
                "info_fraction": s.information_fraction,
                "alpha_spent": s.alpha_spent,
                "alpha_boundary": s.alpha_boundary,
                "p_value": s.p_value,
                "z_stat": s.z_statistic,
                "n_control": s.n_control,
                "n_treatment": s.n_treatment,
                "stop": s.should_stop,
                "reject": s.reject_null,
            }
            for s in self.looks
        ])


class SequentialTester:
    """
    Sequential testing with alpha-spending functions for safe early stopping.

    Implements O'Brien-Fleming and Pocock alpha-spending functions.
    Prevents type-I error inflation from peeking at results mid-experiment.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        max_n: int = 10000,
        n_looks: int = 5,
        spending_function: Literal["obrien_fleming", "pocock"] = "obrien_fleming",
    ):
        self.alpha = alpha
        self.max_n = max_n
        self.n_looks = n_looks
        self.spending_function = spending_function
        self._alpha_spent_so_far = 0.0

    def alpha_spending(self, information_fraction: float) -> float:
        """
        Cumulative alpha spent by information fraction t ∈ (0, 1].
        O'Brien-Fleming: spends little early, more near end.
        Pocock: spends uniformly.
        """
        t = np.clip(information_fraction, 1e-10, 1.0)
        if self.spending_function == "obrien_fleming":
            return float(2 * (1 - stats.norm.cdf(stats.norm.ppf(1 - self.alpha / 2) / np.sqrt(t))))
        elif self.spending_function == "pocock":
            return float(self.alpha * np.log(1 + (np.e - 1) * t))
        else:
            raise ValueError(f"Unknown spending function: {self.spending_function}")

    def boundary_at_look(self, look_index: int, information_fraction: float) -> float:
        """
        Compute the incremental alpha available at this look, then return
        the corresponding z-score boundary.
        """
        cumulative_spend = self.alpha_spending(information_fraction)
        incremental_spend = max(cumulative_spend - self._alpha_spent_so_far, 1e-10)
        z_boundary = stats.norm.ppf(1 - incremental_spend / 2)
        return float(z_boundary)

    def evaluate_look(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        look_index: int,
        information_fraction: Optional[float] = None,
    ) -> SequentialTestState:
        """
        Evaluate whether to stop at this look given current data.
        information_fraction = current_n / max_n if not provided.
        """
        n_c, n_t = len(control), len(treatment)
        current_n = n_c + n_t

        if information_fraction is None:
            information_fraction = min(current_n / self.max_n, 1.0)

        z_boundary = self.boundary_at_look(look_index, information_fraction)

        stat, p = stats.ttest_ind(control, treatment, equal_var=False)
        z_stat = float(stat)

        cumulative_spend = self.alpha_spending(information_fraction)
        incremental_spend = max(cumulative_spend - self._alpha_spent_so_far, 1e-10)
        reject = abs(z_stat) >= z_boundary

        if reject:
            self._alpha_spent_so_far = cumulative_spend

        return SequentialTestState(
            look_number=look_index + 1,
            information_fraction=information_fraction,
            alpha_spent=incremental_spend,
            alpha_boundary=z_boundary,
            p_value=float(p),
            z_statistic=z_stat,
            n_control=n_c,
            n_treatment=n_t,
            should_stop=reject,
            reject_null=reject,
        )

    def run_full_sequence(
        self,
        control_data: np.ndarray,
        treatment_data: np.ndarray,
        look_fractions: Optional[List[float]] = None,
    ) -> SequentialTestResult:
        """
        Run through all pre-planned looks on the full dataset.
        Simulates sequential monitoring with data arriving over time.
        """
        self._alpha_spent_so_far = 0.0
        result = SequentialTestResult()

        if look_fractions is None:
            look_fractions = [(i + 1) / self.n_looks for i in range(self.n_looks)]

        n_total = min(len(control_data), len(treatment_data))

        for i, frac in enumerate(look_fractions):
            n_at_look = int(frac * n_total)
            c_slice = control_data[:n_at_look]
            t_slice = treatment_data[:n_at_look]

            state = self.evaluate_look(c_slice, t_slice, i, frac)
            result.looks.append(state)
            result.total_alpha_spent += state.alpha_spent

            if state.should_stop:
                result.final_decision = "reject_null"
                break
        else:
            last = result.looks[-1] if result.looks else None
            result.final_decision = "reject_null" if (last and last.reject_null) else "fail_to_reject"

        return result
