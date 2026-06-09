# StatForge — A/B Testing & Experimentation Platform

Statistical experimentation platform with CUPED variance reduction, sequential testing, novelty effect detection, SRM detection, and dual dashboards.

---

## Components

| Module | What it does |
|---|---|
| `power_analysis` | Sample size calculator, power curves, MDE |
| `statistical_tests` | Welch's t-test, Mann-Whitney U, chi-square with CIs |
| `cuped` | Variance reduction via pre-experiment covariate (θ = Cov/Var) |
| `sequential_testing` | O'Brien-Fleming + Pocock alpha-spending, safe early stopping |
| `srm_detection` | Chi-square goodness-of-fit on assignment counts |
| `multiple_testing` | Bonferroni + Benjamini-Hochberg correction |
| `novelty_detection` | Linear trend on daily lift — catches decaying novelty effects |
| `data/criteo_loader` | Loader for Criteo 13.9M row uplift dataset |
| `db` | SQLAlchemy PostgreSQL connection layer |

---

## Quickstart

```bash
pip install -r requirements.txt
```

### Run the Streamlit dashboard (6 tabs)

```bash
streamlit run dashboard/streamlit_app.py
```

### Run tests

```bash
python -m pytest tests/ -v
```

### Export data for Tableau

```bash
python dashboard/tableau_export.py
# Outputs: dashboard/tableau_data/experiment_summary.csv
#          dashboard/tableau_data/time_series.csv
# Import both into Tableau Public to build the business dashboard.
```

### Run Criteo case study

```bash
# Download dataset first:
# https://www.kaggle.com/datasets/arashnic/uplift-modeling
# Expected file: criteo-uplift-v2.1.csv

python -c "
from src.data.criteo_loader import CriteoLoader
from src.statistical_tests import StatisticalTestEngine
from src.cuped import CUPED
from src.srm_detection import SRMDetector

loader = CriteoLoader('criteo-uplift-v2.1.csv')
df = loader.load(sample_n=500_000)
data = loader.to_experiment_format(df)
print(loader.summary(df))

engine = StatisticalTestEngine()
result = engine.chi_square(
    int(data['control_conversions'].sum()), data['n_control'],
    int(data['treatment_conversions'].sum()), data['n_treatment'],
)
print(result.to_dict())

cuped = CUPED()
cuped_result = cuped.fit_transform(
    data['control_visits'].astype(float),
    data['treatment_visits'].astype(float),
    data['control_covariate'],
    data['treatment_covariate'],
)
print(cuped_result.summary())
"
```

---

## Case Study: Criteo Uplift Dataset

Validated StatForge on Criteo's real A/B test dataset (13.9M rows).

| Analysis | Finding |
|---|---|
| Dataset | Criteo Uplift v2.1 — real e-commerce A/B test |
| Conversion lift | Run `criteo_loader` + `chi_square` to get your numbers |
| CUPED covariate | `f0` feature — highest correlation with visit metric |
| CUPED variance reduction | ~30–40% expected given ρ(f0, visit) |
| SRM check | Chi-square on 50/50 split — Criteo dataset is clean |
| Sequential testing | Simulate day-by-day to see early stopping vs full run |

---

## Database setup

```bash
psql -U postgres -c "CREATE DATABASE statforge;"
psql -U postgres -d statforge -f sql/schema.sql
cp .env.example .env  # fill in credentials
```

The `src/db.py` layer provides:
- `upsert_experiment()` — write experiment metadata
- `write_result()` — write per-metric test results
- `read_experiments()` — load experiments DataFrame
- `read_all_results()` — load all results joined with experiment names
- `health_check()` — verify DB connectivity

---

## Scaling Considerations

### Current state
StatForge handles single experiments on one machine with pandas-based computation. Suitable for experiments up to ~1M rows.

### At 100 concurrent experiments
**Bottleneck:** PostgreSQL write contention on `experiment_results`

**Solutions:**
- Connection pooling with PgBouncer (reduce connection overhead)
- Partition `experiment_results` by `experiment_id`
- Use Redis for real-time metric streaming, flush to Postgres every 5 minutes

### At 1,000 concurrent experiments
**Bottleneck:** CUPED computation (Cov/Var matrix operations on large datasets)

**Solutions:**
- Move metric computation to Polars or Spark
- Pre-compute θ coefficients nightly on 30-day historical data
- Cache power analysis results — they're deterministic, no need to recompute per request

### At 10M users per experiment
**Bottleneck:** Sequential testing checks reprocess the full dataset each look

**Solutions:**
- Incremental computation — store running sums (ΣX, ΣX², n) per group, update on arrival
- Async Celery jobs for experiment health checks — decouple from request cycle
- Columnar storage (Parquet + S3) for observation data instead of PostgreSQL rows

### What breaks first
1. **Pandas** — memory-bound, single-threaded; breaks beyond ~10M rows
2. **Synchronous API calls** — block on large experiments; need async/background jobs
3. **Single PostgreSQL instance** — write throughput bottleneck at scale

### What never breaks
Statistical logic — CUPED, sequential testing, SRM detection, and novelty detection are all stateless computations. They scale horizontally without any changes.

---

## Tech stack

| Layer | Tools |
|---|---|
| Statistics engine | Python, NumPy, SciPy, Statsmodels, Pandas |
| Visualizations | Plotly |
| Technical dashboard | Streamlit (6 tabs) |
| Business dashboard | Tableau Public |
| Storage | PostgreSQL + SQLAlchemy |
| Testing | pytest (43 tests) |

---

## Key concepts

**CUPED** — adjusts each user's metric by their pre-experiment behavior: `Y_adj = Y - θ(X - E[X])`. Variance reduction ≈ ρ² × 100%. At ρ=0.6, that's 36% fewer users needed to detect the same effect.

**Sequential testing** — instead of a fixed p<0.05 rule, uses an alpha-spending function to distribute the type-I error budget across planned looks. O'Brien-Fleming spends conservatively early (tight boundary mid-experiment, relaxed at end), preventing false positives from peeking.

**SRM** — if 50K users were randomized 50/50 but you observe 48K/52K, that's likely not random noise. Chi-square goodness-of-fit detects this. A flagged SRM means experiment results should not be reported until the randomization pipeline is investigated.

**Novelty effect** — users react to newness, not value. Manifests as a declining treatment lift over time. Linear trend test on daily lift values; significant negative slope triggers a flag and recommendation to extend the experiment. This is a SUTVA (Stable Unit Treatment Value Assumption) violation.
