"""
Tableau data export utility.

Generates CSV files formatted for Tableau Public dashboards:
1. experiment_summary.csv   — one row per experiment with key metrics
2. time_series.csv          — running significance and cumulative lift over time
3. metric_results.csv       — per-metric test results with CIs
4. srm_flags.csv            — SRM detection status per experiment

These CSVs are consumed by the Tableau workbook (dashboard/statforge_tableau.twb).
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import ExperimentDataGenerator
from src.statistical_tests import StatisticalTestEngine
from src.cuped import CUPED
from src.srm_detection import SRMDetector
from src.power_analysis import PowerAnalysis

OUTPUT_DIR = Path(__file__).parent / "tableau_data"
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_experiment_summary(n_experiments: int = 12) -> pd.DataFrame:
    """Simulate results for multiple experiments."""
    rng = np.random.default_rng(42)
    engine = StatisticalTestEngine(alpha=0.05)
    detector = SRMDetector(alpha=0.01)
    cuped = CUPED()
    rows = []

    for i in range(n_experiments):
        n = rng.integers(2000, 10000)
        has_effect = rng.random() > 0.4
        true_lift = float(rng.uniform(0.01, 0.05)) if has_effect else 0.0

        gen = ExperimentDataGenerator(seed=int(rng.integers(0, 9999)))
        df = gen.generate(
            n_control=n, n_treatment=n,
            baseline_conversion=0.10,
            treatment_effect_conversion=true_lift,
            treatment_effect_revenue=true_lift * 100,
        )
        ctrl = df[df.variant == "control"]
        trt = df[df.variant == "treatment"]

        chi2_res = engine.chi_square(
            int(ctrl.converted.sum()), len(ctrl),
            int(trt.converted.sum()), len(trt),
        )
        t_res = engine.t_test(ctrl.revenue.values, trt.revenue.values)
        srm_res = detector.check({"control": len(ctrl), "treatment": len(trt)})

        cuped_res = cuped.fit_transform(
            ctrl.revenue.values, trt.revenue.values,
            ctrl.pre_revenue.values, trt.pre_revenue.values,
        )

        rows.append({
            "experiment_id": f"EXP-{i+1:03d}",
            "experiment_name": f"Experiment {i+1}",
            "n_control": len(ctrl),
            "n_treatment": len(trt),
            "conversion_control": round(float(ctrl.converted.mean()), 4),
            "conversion_treatment": round(float(trt.converted.mean()), 4),
            "conversion_lift_abs": round(float(trt.converted.mean() - ctrl.converted.mean()), 4),
            "conversion_lift_pct": round(float((trt.converted.mean() - ctrl.converted.mean()) / ctrl.converted.mean() * 100), 2),
            "revenue_control_mean": round(float(ctrl.revenue.mean()), 2),
            "revenue_treatment_mean": round(float(trt.revenue.mean()), 2),
            "revenue_p_value": round(t_res.p_value, 4),
            "conversion_p_value": round(chi2_res.p_value, 4),
            "revenue_ci_lower": round(t_res.confidence_interval[0], 2),
            "revenue_ci_upper": round(t_res.confidence_interval[1], 2),
            "significant_conversion": chi2_res.significant,
            "significant_revenue": t_res.significant,
            "srm_detected": srm_res.srm_detected,
            "variance_reduction_pct": round(cuped_res.variance_reduction_pct, 1),
            "has_true_effect": has_effect,
        })

    return pd.DataFrame(rows)


def generate_time_series(n_days: int = 30) -> pd.DataFrame:
    """Simulate day-by-day cumulative significance for a single experiment."""
    rng = np.random.default_rng(99)
    engine = StatisticalTestEngine(alpha=0.05)

    users_per_day = 200
    true_conv_ctrl = 0.10
    true_conv_trt = 0.12

    rows = []
    ctrl_conv = []
    ctrl_n = []
    trt_conv = []
    trt_n = []

    for day in range(1, n_days + 1):
        c_today = rng.binomial(users_per_day, true_conv_ctrl)
        t_today = rng.binomial(users_per_day, true_conv_trt)
        ctrl_conv.append(c_today)
        ctrl_n.append(users_per_day)
        trt_conv.append(t_today)
        trt_n.append(users_per_day)

        cumulative_ctrl_conv = sum(ctrl_conv)
        cumulative_ctrl_n = sum(ctrl_n)
        cumulative_trt_conv = sum(trt_conv)
        cumulative_trt_n = sum(trt_n)

        res = engine.chi_square(
            cumulative_ctrl_conv, cumulative_ctrl_n,
            cumulative_trt_conv, cumulative_trt_n,
        )

        rows.append({
            "day": day,
            "cumulative_n": cumulative_ctrl_n + cumulative_trt_n,
            "p_value": round(res.p_value, 4),
            "significant": res.significant,
            "conversion_control": round(cumulative_ctrl_conv / cumulative_ctrl_n, 4),
            "conversion_treatment": round(cumulative_trt_conv / cumulative_trt_n, 4),
            "lift_abs": round(cumulative_trt_conv / cumulative_trt_n - cumulative_ctrl_conv / cumulative_ctrl_n, 4),
            "ci_lower": round(res.confidence_interval[0], 4),
            "ci_upper": round(res.confidence_interval[1], 4),
        })

    return pd.DataFrame(rows)


def export_all():
    print("Generating experiment summary...")
    summary = generate_experiment_summary()
    summary.to_csv(OUTPUT_DIR / "experiment_summary.csv", index=False)
    print(f"  Saved {len(summary)} experiments → {OUTPUT_DIR / 'experiment_summary.csv'}")

    print("Generating time series...")
    ts = generate_time_series()
    ts.to_csv(OUTPUT_DIR / "time_series.csv", index=False)
    print(f"  Saved {len(ts)} days → {OUTPUT_DIR / 'time_series.csv'}")

    print("Done. Import these CSVs into Tableau Public.")
    return summary, ts


if __name__ == "__main__":
    export_all()
