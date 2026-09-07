# StatForge — A/B Testing & Experimentation Platform

7 statistical modules built from scratch — CUPED, power analysis, SRM detection, sequential testing, novelty effect detection, multiple testing correction, and statistical tests — validated against a 500K-row real e-commerce dataset.

---

## Modules

| Module | What it does |
|---|---|
| `power_analysis` | Sample size, power curves, MDE |
| `statistical_tests` | Welch's t-test, Mann-Whitney U, chi-square with CIs |
| `cuped` | Variance reduction via pre-experiment covariate (θ = Cov/Var) |
| `sequential_testing` | O'Brien-Fleming + Pocock alpha-spending, safe early stopping |
| `srm_detection` | Chi-square goodness-of-fit on assignment counts |
| `multiple_testing` | Bonferroni + Benjamini-Hochberg correction |
| `novelty_detection` | Linear trend on daily lift — catches decaying treatment effects |

## Quickstart

```bash
pip install -r requirements.txt
streamlit run dashboard/streamlit_app.py   # 6-tab dashboard
python -m pytest tests/ -v                # 52 tests
```

## Criteo Case Study Results

Validated on 500K-row sample of Criteo's 13.9M-row uplift dataset.

| Check | Result |
|---|---|
| Dataset | Criteo Uplift v2.1 — 500K rows, 15/85 control/treatment split |
| SRM | No SRM detected — p=0.9968 |
| Conversion lift | +62.59% relative lift (p < 0.0001) |
| CUPED — wrong covariate (`f0`) | 1.84% variance reduction |
| CUPED — correct covariate (`f9`, ρ≈0.50) | **24.74% variance reduction** — ~25% lower sample requirements |

`f9` was identified via `feature_covariate_correlations()` — covariate selection is the difference between near-zero and meaningful variance reduction.

## Simulation-Based Validation

9 tests verify statistical correctness across 2K–5K simulated experiments:
- Type-I error rate ≈5% under null (t-test, chi-square, Mann-Whitney)
- t-test achieves ≥75% power at designed sample size
- 95% CI contains true value ≈95% of the time
- CUPED reduces variance with correlated covariate, near-zero with uncorrelated

## dbt Layer

5 SQL models transforming raw PostgreSQL tables into analytics-ready marts:

```
stg_experiments + stg_experiment_results + stg_observations
    → mart_experiment_summary (one row per experiment, feeds dashboards)
    → mart_daily_metrics (daily lift + 7-day rolling average)
```

```bash
cd dbt/statforge && dbt run && dbt test
```

## Database Setup

```bash
psql -U postgres -c "CREATE DATABASE statforge;"
psql -U postgres -d statforge -f sql/schema.sql
cp .env.example .env
```

## Tech Stack

| Layer | Tools |
|---|---|
| Statistics | Python, NumPy, SciPy, Statsmodels, Pandas |
| Dashboards | Streamlit (6 tabs), Tableau Public, Plotly |
| Storage | PostgreSQL, SQLAlchemy, dbt |
| Testing | pytest — 43 unit tests + 9 simulation tests |

## Key Concepts

**CUPED** — `Y_adj = Y - θ(X - E[X])`, θ = Cov(Y,X)/Var(X). Variance reduction ≈ ρ² × 100%. Wrong covariate (f0): 1.84%. Correct covariate (f9, ρ≈0.50): 24.74%.

**Sequential testing** — alpha-spending distributes type-I error budget across planned looks. O'Brien-Fleming spends conservatively early, preventing false positives from peeking.

**SRM** — chi-square goodness-of-fit on observed vs. expected assignment counts. A flagged SRM means results should not be reported until randomization is investigated.

**Novelty effect** — declining treatment lift over time from users reacting to newness. Detected via linear trend on daily lift; significant negative slope triggers a flag.

---

MIT License
