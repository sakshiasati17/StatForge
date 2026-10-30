import numpy as np
from scipy import stats
from scipy.optimize import brentq
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Sequence


from functools import lru_cache

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
    t_statistic: Optional[float] = None
    cumulative_alpha: float = 0.0
    nominal_alpha: float = 0.0


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
                "cumulative_alpha": s.cumulative_alpha,
                "alpha_boundary": s.alpha_boundary,
                "nominal_alpha": s.nominal_alpha,
                "p_value": s.p_value,
                "z_stat": s.z_statistic,
                "t_stat": s.t_statistic if s.t_statistic is not None else s.z_statistic,
                "n_control": s.n_control,
                "n_treatment": s.n_treatment,
                "stop": s.should_stop,
                "reject": s.reject_null,
            }
            for s in self.looks
        ])


# Boundary used when no alpha is available at a look (effectively "never reject").
_MAX_Z = 8.0


@lru_cache(maxsize=128)
def _solve_group_sequential_boundaries_cached(
    fracs_tuple: tuple,
    cum_tuple: tuple,
    n_grid: int = 501,
) -> tuple:
    fracs = np.asarray(fracs_tuple, dtype=float)
    cum = np.asarray(cum_tuple, dtype=float)
    boundaries: List[float] = []
    prev_alpha = 0.0
    grid = dens = None

    for k, (t, a) in enumerate(zip(fracs, cum)):
        inc = a - prev_alpha
        if k == 0:
            c = float(stats.norm.isf(inc / 2)) if inc > 0 else np.inf
            c_grid = min(c, _MAX_Z)
            sd = np.sqrt(t)
            grid = np.linspace(-c_grid * sd, c_grid * sd, n_grid)
            dens = stats.norm.pdf(grid / sd) / sd
        else:
            sd_inc = np.sqrt(t - fracs[k - 1])
            h = grid[1] - grid[0]
            w = np.full(n_grid, h)
            w[0] = w[-1] = h / 2
            wd = w * dens  # sub-density mass of continuing paths at previous look
            sqrt_t = np.sqrt(t)

            def exceed(cz: float) -> float:
                b = cz * sqrt_t
                return float(np.sum(wd * (stats.norm.sf((b - grid) / sd_inc)
                                          + stats.norm.cdf((-b - grid) / sd_inc))))

            if inc <= 1e-15 or exceed(_MAX_Z) >= inc:
                c = np.inf if inc <= 1e-15 else _MAX_Z
            else:
                c = float(brentq(lambda cz: exceed(cz) - inc, 1e-6, _MAX_Z, xtol=1e-8))
            c_grid = min(c, _MAX_Z)
            new_grid = np.linspace(-c_grid * sqrt_t, c_grid * sqrt_t, n_grid)
            kernel = stats.norm.pdf((new_grid[:, None] - grid[None, :]) / sd_inc) / sd_inc
            dens = kernel @ wd
            grid = new_grid

        boundaries.append(float(c))
        prev_alpha = a

    return tuple(boundaries)


def group_sequential_boundaries(
    information_fractions: Sequence[float],
    cumulative_alpha: Sequence[float],
    n_grid: int = 501,
) -> List[float]:
    """
    Exact two-sided group-sequential z boundaries for a given alpha-spending schedule.

    Uses the Armitage–McPherson–Rowe recursive numerical integration on the score
    process S_k = Z_k * sqrt(t_k), whose increments are independent N(0, t_k - t_{k-1})
    under H0. Boundary c_k solves

        P(|Z_1| < c_1, ..., |Z_{k-1}| < c_{k-1}, |Z_k| >= c_k) = alpha(t_k) - alpha(t_{k-1})

    so the overall type-I error equals alpha(t_K), with the correlation between looks
    (corr(Z_i, Z_j) = sqrt(t_i / t_j)) fully accounted for.
    """
    fracs = np.asarray(information_fractions, dtype=float)
    cum = np.asarray(cumulative_alpha, dtype=float)
    if len(fracs) != len(cum):
        raise ValueError("information_fractions and cumulative_alpha must have equal length")
    if np.any(fracs <= 0) or np.any(fracs > 1 + 1e-12) or np.any(np.diff(fracs) <= 0):
        raise ValueError("information fractions must be strictly increasing in (0, 1]")

    fracs_key = tuple(round(float(f), 8) for f in fracs)
    cum_key = tuple(round(float(a), 8) for a in cum)
    return list(_solve_group_sequential_boundaries_cached(fracs_key, cum_key, n_grid=n_grid))


class SequentialTester:
    """
    Group-sequential testing with Lan–DeMets alpha-spending for safe early stopping.

    Spending functions:
      - "obrien_fleming": alpha(t) = 2 * (1 - Phi(z_{1-alpha/2} / sqrt(t)))
      - "pocock":         alpha(t) = alpha * ln(1 + (e - 1) * t)

    Boundaries are exact: each look's critical value is solved by recursive numerical
    integration over the joint null distribution of all looks so far, so the overall
    type-I error equals alpha (not a conservative Bonferroni-style approximation).
    Boundaries depend only on the information fractions observed so far, so unplanned
    look timing is supported (the Lan–DeMets property).
    """

    def __init__(
        self,
        alpha: float = 0.05,
        max_n: int = 10000,
        n_looks: int = 5,
        spending_function: Literal["obrien_fleming", "pocock"] = "obrien_fleming",
    ):
        if spending_function not in ("obrien_fleming", "pocock"):
            raise ValueError(f"Unknown spending function: {spending_function}")
        self.alpha = alpha
        self.max_n = max_n
        self.n_looks = n_looks
        self.spending_function = spending_function
        self._fractions: List[float] = []
        self._alpha_spent_so_far = 0.0

    def alpha_spending(self, information_fraction: float) -> float:
        """Cumulative alpha spent by information fraction t in (0, 1]."""
        t = float(np.clip(information_fraction, 1e-10, 1.0))
        if self.spending_function == "obrien_fleming":
            return float(2 * stats.norm.sf(stats.norm.isf(self.alpha / 2) / np.sqrt(t)))
        return float(self.alpha * np.log(1 + (np.e - 1) * t))

    def compute_boundaries(self, look_fractions: Optional[Sequence[float]] = None) -> List[float]:
        """Exact |z| critical values for a full look schedule (default: equally spaced)."""
        if look_fractions is None:
            look_fractions = [(i + 1) / self.n_looks for i in range(self.n_looks)]
        cum = [self.alpha_spending(f) for f in look_fractions]
        return group_sequential_boundaries(look_fractions, cum)

    def boundary_at_look(self, look_index: int, information_fraction: float) -> float:
        """
        Exact |z| boundary at look `look_index` given the information fractions of all
        previous looks recorded on this tester (via evaluate_look / run_full_sequence).
        """
        if len(self._fractions) == look_index:
            self._fractions.append(information_fraction)
            fracs = list(self._fractions)
        elif len(self._fractions) > look_index:
            self._fractions = self._fractions[:look_index] + [information_fraction]
            fracs = list(self._fractions)
        else:
            # Earlier looks not recorded: assume equal spacing up to current fraction
            fracs = [(i + 1) * information_fraction / (look_index + 1) for i in range(look_index + 1)]
            self._fractions = fracs
        return self.compute_boundaries(fracs)[-1]

    def evaluate_look(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        look_index: int,
        information_fraction: Optional[float] = None,
        boundary: Optional[float] = None,
    ) -> SequentialTestState:
        """
        Evaluate whether to stop at this look given current data.
        information_fraction = current_n / max_n if not provided.
        """
        control = np.asarray(control, dtype=float)
        treatment = np.asarray(treatment, dtype=float)
        n_c, n_t = len(control), len(treatment)

        if information_fraction is None:
            information_fraction = min((n_c + n_t) / self.max_n, 1.0)

        if boundary is not None:
            z_boundary = float(boundary)
        else:
            z_boundary = self.boundary_at_look(look_index, information_fraction)
        self._fractions = self._fractions[:look_index] + [information_fraction]

        mean_c = float(np.mean(control)) if n_c else 0.0
        mean_t = float(np.mean(treatment)) if n_t else 0.0
        var_c = float(np.var(control, ddof=1)) if n_c > 1 else 0.0
        var_t = float(np.var(treatment, ddof=1)) if n_t > 1 else 0.0
        se = float(np.sqrt(var_c / n_c + var_t / n_t)) if (n_c > 0 and n_t > 0) else 0.0
        z_stat = float((mean_t - mean_c) / se) if se > 0 else 0.0

        if n_c > 1 and n_t > 1 and se > 0:
            t_stat, p = stats.ttest_ind(treatment, control, equal_var=False)
        else:
            t_stat, p = 0.0, 1.0

        prev_cum = self.alpha_spending(self._fractions[look_index - 1]) if look_index > 0 else 0.0
        cumulative_spend = self.alpha_spending(information_fraction)
        reject = bool(abs(z_stat) >= z_boundary)
        self._alpha_spent_so_far = cumulative_spend

        return SequentialTestState(
            look_number=look_index + 1,
            information_fraction=information_fraction,
            alpha_spent=cumulative_spend - prev_cum,
            alpha_boundary=z_boundary,
            p_value=float(p),
            z_statistic=z_stat,
            t_statistic=float(t_stat),
            n_control=n_c,
            n_treatment=n_t,
            should_stop=reject,
            reject_null=reject,
            cumulative_alpha=cumulative_spend,
            nominal_alpha=float(2 * stats.norm.sf(z_boundary)),
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
        self._fractions = []
        result = SequentialTestResult()

        if look_fractions is None:
            look_fractions = [(i + 1) / self.n_looks for i in range(self.n_looks)]

        boundaries = self.compute_boundaries(look_fractions)
        n_c_total, n_t_total = len(control_data), len(treatment_data)

        for i, frac in enumerate(look_fractions):
            c_slice = control_data[: int(round(frac * n_c_total))]
            t_slice = treatment_data[: int(round(frac * n_t_total))]

            state = self.evaluate_look(c_slice, t_slice, i, frac, boundary=boundaries[i])
            result.looks.append(state)
            result.total_alpha_spent = self._alpha_spent_so_far

            if state.should_stop:
                result.final_decision = "reject_null"
                break
        else:
            result.final_decision = "fail_to_reject"

        return result
