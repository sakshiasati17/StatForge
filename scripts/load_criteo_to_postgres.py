"""
Criteo to PostgreSQL Loader.

Idempotently loads 500,000 rows of the Criteo Uplift dataset into PostgreSQL:
- Upserts the experiment record ("Criteo Uplift v2.1", intended split 0.85 treatment)
- Clears existing data for the experiment on rerun to prevent duplicate entries
- Bulk inserts 500,000 assignments and observations (metric = 'visit', covariate = 'f9')
- Runs statistical tests & CUPED, and persists the analysis results via db.write_result()
- Verifies exact 500,000 row counts
"""
from __future__ import annotations

import argparse
import io
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sqlalchemy import text

from src.cuped import CUPED
from src.data.criteo_loader import CriteoLoader
from src.db import connection, get_engine, upsert_experiment, write_result
from src.srm_detection import SRMDetector
from src.statistical_tests import StatisticalTestEngine


def load_criteo_to_postgres(
    data_path: str = "data/criteo-uplift-v2.1.csv",
    sample_n: int = 500000,
    seed: int = 42,
) -> int:
    print(f"Loading {sample_n:,} rows from {data_path} (seed={seed})...")
    loader = CriteoLoader(data_path)
    df = loader.load(sample_n=sample_n, seed=seed)

    experiment_name = "Criteo Uplift v2.1"
    intended_split = 0.85  # documented 85% treatment / 15% control
    alpha = 0.05
    power_target = 0.80

    print("Upserting experiment metadata...")
    exp_id = upsert_experiment(
        name=experiment_name,
        description="Criteo Uplift Modeling v2.1 benchmark (500K sample, visit metric, f9 covariate).",
        alpha=alpha,
        power_target=power_target,
        intended_split=intended_split,
        status="completed",
    )
    print(f"Experiment ID: {exp_id}")

    # Idempotent cleanup: remove previous run records for this experiment
    print("Clearing any existing data for experiment to prevent duplicates...")
    with connection() as conn:
        conn.execute(
            text("DELETE FROM experiment_observations WHERE experiment_id = :eid"),
            {"eid": exp_id},
        )
        conn.execute(
            text("DELETE FROM experiment_assignments WHERE experiment_id = :eid"),
            {"eid": exp_id},
        )
        conn.execute(
            text("DELETE FROM experiment_results WHERE experiment_id = :eid"),
            {"eid": exp_id},
        )

    # Prepare dataframes for fast COPY bulk insertion
    print("Preparing bulk data...")
    user_ids = np.arange(1, len(df) + 1, dtype=np.int64)
    variants = np.where(df["treatment"] == 1, "treatment", "control")
    base_time = datetime(2026, 6, 1, 0, 0, 0)

    # Distribute observations across 14 simulated days for realistic time-series
    day_offsets = np.random.default_rng(seed).integers(0, 14, size=len(df))
    timestamps = [
        (base_time + timedelta(days=int(d), minutes=int(i % 1440))).isoformat()
        for i, d in enumerate(day_offsets)
    ]

    assignments_df = pd.DataFrame({
        "experiment_id": exp_id,
        "user_id": user_ids,
        "variant": variants,
        "assigned_at": timestamps,
    })

    observations_df = pd.DataFrame({
        "experiment_id": exp_id,
        "user_id": user_ids,
        "metric_name": "visit",
        "metric_value": df["visit"].values.astype(float),
        "pre_value": df["f9"].values.astype(float),
        "observed_at": timestamps,
    })

    # High-speed bulk COPY using psycopg2 raw connection
    engine = get_engine()
    raw_conn = engine.raw_connection()
    try:
        cur = raw_conn.cursor()

        print("Bulk copying assignments...")
        assign_buffer = io.StringIO()
        assignments_df.to_csv(assign_buffer, index=False, header=False, sep="\t")
        assign_buffer.seek(0)
        cur.copy_from(
            assign_buffer,
            "experiment_assignments",
            columns=("experiment_id", "user_id", "variant", "assigned_at"),
        )

        print("Bulk copying observations...")
        obs_buffer = io.StringIO()
        observations_df.to_csv(obs_buffer, index=False, header=False, sep="\t")
        obs_buffer.seek(0)
        cur.copy_from(
            obs_buffer,
            "experiment_observations",
            columns=("experiment_id", "user_id", "metric_name", "metric_value", "pre_value", "observed_at"),
        )

        raw_conn.commit()
        cur.close()
    finally:
        raw_conn.close()

    # Verify counts
    with connection() as conn:
        n_assign = conn.execute(
            text("SELECT count(*) FROM experiment_assignments WHERE experiment_id = :eid"),
            {"eid": exp_id},
        ).scalar()
        n_obs = conn.execute(
            text("SELECT count(*) FROM experiment_observations WHERE experiment_id = :eid"),
            {"eid": exp_id},
        ).scalar()

    print(f"Verified row counts in PostgreSQL: {n_assign:,} assignments, {n_obs:,} observations")
    assert n_assign == len(df), f"Expected {len(df)} assignments, got {n_assign}"
    assert n_obs == len(df), f"Expected {len(df)} observations, got {n_obs}"

    # Run analysis and write result to experiment_results
    print("Computing statistical test and CUPED variance reduction...")
    ctrl_df = df[df.treatment == 0]
    trt_df = df[df.treatment == 1]
    n_ctrl = len(ctrl_df)
    n_trt = len(trt_df)

    srm = SRMDetector(alpha=0.01).check(
        {"control": n_ctrl, "treatment": n_trt},
        expected_ratios={"control": 0.15, "treatment": 0.85},
    )

    stat_engine = StatisticalTestEngine(alpha=alpha, two_tailed=True)
    visit_test = stat_engine.chi_square(
        conversions_control=int(ctrl_df.visit.sum()),
        n_control=n_ctrl,
        conversions_treatment=int(trt_df.visit.sum()),
        n_treatment=n_trt,
    )

    cuped = CUPED()
    cuped_res = cuped.fit_transform(
        control_metric=ctrl_df["visit"].values.astype(float),
        treatment_metric=trt_df["visit"].values.astype(float),
        control_covariate=ctrl_df["f9"].values.astype(float),
        treatment_covariate=trt_df["f9"].values.astype(float),
    )

    print("Writing result to experiment_results...")
    write_result(
        experiment_id=exp_id,
        metric_name="visit",
        test_type="Chi-square + CUPED",
        p_value=float(visit_test.p_value),
        adjusted_p_value=float(visit_test.p_value),
        effect_size=float(visit_test.effect_size),
        ci_lower=float(visit_test.confidence_interval[0]),
        ci_upper=float(visit_test.confidence_interval[1]),
        n_control=n_ctrl,
        n_treatment=n_trt,
        significant=bool(visit_test.significant),
        srm_detected=bool(srm.srm_detected),
        variance_reduction=float(cuped_res.variance_reduction_pct),
    )

    # Also record conversion metric result
    conv_test = stat_engine.chi_square(
        conversions_control=int(ctrl_df.conversion.sum()),
        n_control=n_ctrl,
        conversions_treatment=int(trt_df.conversion.sum()),
        n_treatment=n_trt,
    )
    write_result(
        experiment_id=exp_id,
        metric_name="conversion",
        test_type="Chi-square",
        p_value=float(conv_test.p_value),
        adjusted_p_value=float(conv_test.p_value),
        effect_size=float(conv_test.effect_size),
        ci_lower=float(conv_test.confidence_interval[0]),
        ci_upper=float(conv_test.confidence_interval[1]),
        n_control=n_ctrl,
        n_treatment=n_trt,
        significant=bool(conv_test.significant),
        srm_detected=bool(srm.srm_detected),
        variance_reduction=None,
    )

    print("Database loading and analysis complete!")
    return exp_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load Criteo dataset into PostgreSQL")
    parser.add_argument("--data-path", default="data/criteo-uplift-v2.1.csv")
    parser.add_argument("--sample-n", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    load_criteo_to_postgres(
        data_path=args.data_path,
        sample_n=args.sample_n,
        seed=args.seed,
    )
