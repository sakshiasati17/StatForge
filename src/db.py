"""
PostgreSQL connection layer for StatForge.

Uses SQLAlchemy core (not ORM) — lightweight, explicit, easy to reason about.
Configure via environment variables or .env file (see .env.example).
"""

import os
from contextlib import contextmanager
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from dotenv import load_dotenv

load_dotenv()


def _build_connection_string() -> str:
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "statforge")
    user = os.getenv("POSTGRES_USER", "statforge_user")
    password = os.getenv("POSTGRES_PASSWORD", "")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


def get_engine(pool_size: int = 5, max_overflow: int = 10):
    """
    Create a connection-pooled SQLAlchemy engine.
    Uses QueuePool — suitable for multi-threaded Streamlit.
    """
    return create_engine(
        _build_connection_string(),
        poolclass=QueuePool,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,
    )


_engine = None


def engine():
    """Singleton engine — reused across requests."""
    global _engine
    if _engine is None:
        _engine = get_engine()
    return _engine


@contextmanager
def connection():
    """Context manager for a single DB connection."""
    conn = engine().connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Experiment writes ──────────────────────────────────────────────────────────

def upsert_experiment(
    name: str,
    description: str = "",
    alpha: float = 0.05,
    power_target: float = 0.80,
    intended_split: float = 0.50,
    status: str = "draft",
) -> int:
    """Insert or update an experiment. Returns experiment_id."""
    sql = text("""
        INSERT INTO experiments (name, description, alpha, power_target, intended_split, status)
        VALUES (:name, :description, :alpha, :power_target, :intended_split, :status)
        ON CONFLICT (name) DO UPDATE SET
            description = EXCLUDED.description,
            alpha = EXCLUDED.alpha,
            power_target = EXCLUDED.power_target,
            intended_split = EXCLUDED.intended_split,
            status = EXCLUDED.status,
            updated_at = NOW()
        RETURNING id
    """)
    with connection() as conn:
        result = conn.execute(sql, {
            "name": name,
            "description": description,
            "alpha": alpha,
            "power_target": power_target,
            "intended_split": intended_split,
            "status": status,
        })
        return result.fetchone()[0]


def write_result(
    experiment_id: int,
    metric_name: str,
    test_type: str,
    p_value: float,
    adjusted_p_value: Optional[float],
    effect_size: float,
    ci_lower: float,
    ci_upper: float,
    n_control: int,
    n_treatment: int,
    significant: bool,
    srm_detected: bool = False,
    variance_reduction: Optional[float] = None,
) -> None:
    """Upsert a single metric result for an experiment."""
    sql = text("""
        INSERT INTO experiment_results (
            experiment_id, metric_name, test_type,
            p_value, adjusted_p_value, effect_size,
            ci_lower, ci_upper, n_control, n_treatment,
            significant, srm_detected, variance_reduction
        ) VALUES (
            :experiment_id, :metric_name, :test_type,
            :p_value, :adjusted_p_value, :effect_size,
            :ci_lower, :ci_upper, :n_control, :n_treatment,
            :significant, :srm_detected, :variance_reduction
        )
        ON CONFLICT (experiment_id, metric_name) DO UPDATE SET
            test_type = EXCLUDED.test_type,
            p_value = EXCLUDED.p_value,
            adjusted_p_value = EXCLUDED.adjusted_p_value,
            effect_size = EXCLUDED.effect_size,
            ci_lower = EXCLUDED.ci_lower,
            ci_upper = EXCLUDED.ci_upper,
            n_control = EXCLUDED.n_control,
            n_treatment = EXCLUDED.n_treatment,
            significant = EXCLUDED.significant,
            srm_detected = EXCLUDED.srm_detected,
            variance_reduction = EXCLUDED.variance_reduction,
            computed_at = NOW()
    """)
    with connection() as conn:
        conn.execute(sql, {
            "experiment_id": experiment_id,
            "metric_name": metric_name,
            "test_type": test_type,
            "p_value": p_value,
            "adjusted_p_value": adjusted_p_value,
            "effect_size": effect_size,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_control": n_control,
            "n_treatment": n_treatment,
            "significant": significant,
            "srm_detected": srm_detected,
            "variance_reduction": variance_reduction,
        })


# ── Experiment reads ───────────────────────────────────────────────────────────

def read_experiments(status: Optional[str] = None) -> pd.DataFrame:
    """Load all experiments (optionally filtered by status) as a DataFrame."""
    where = "WHERE status = :status" if status else ""
    sql = text(f"SELECT * FROM experiments {where} ORDER BY created_at DESC")
    with connection() as conn:
        return pd.read_sql(sql, conn, params={"status": status} if status else {})


def read_results(experiment_id: int) -> pd.DataFrame:
    """Load all metric results for an experiment."""
    sql = text("SELECT * FROM experiment_results WHERE experiment_id = :eid ORDER BY metric_name")
    with connection() as conn:
        return pd.read_sql(sql, conn, params={"eid": experiment_id})


def read_all_results() -> pd.DataFrame:
    """Load all results joined with experiment names — used by Streamlit dashboard."""
    sql = text("""
        SELECT
            e.name AS experiment_name,
            e.status,
            r.*
        FROM experiment_results r
        JOIN experiments e ON e.id = r.experiment_id
        ORDER BY r.computed_at DESC
    """)
    with connection() as conn:
        return pd.read_sql(sql, conn)


def health_check() -> bool:
    """Return True if DB is reachable."""
    try:
        with connection() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
