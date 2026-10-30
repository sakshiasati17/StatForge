# StatForge — A/B Testing & Experimentation Platform

7 statistical modules implemented using NumPy and SciPy — CUPED, power analysis, SRM detection, sequential testing, novelty effect detection, multiple testing correction, and statistical tests — validated against a 500,000-row sample of Criteo's production ad uplift dataset.

---

## Modules

| Module | What it does |
|---|---|
| `power_analysis` | Sample size, power curves, MDE for proportions and means |
| `statistical_tests` | Welch's t-test, Mann-Whitney U, chi-square with one-sided & two-sided CIs |
| `cuped` | Variance reduction via pre-experiment covariate (θ = Cov/Var) |
| `sequential_testing` | Exact Lan–DeMets O'Brien–Fleming & Pocock alpha-spending via recursive numerical integration |
| `srm_detection` | Chi-square goodness-of-fit on assignment counts with custom splits |
| `multiple_testing` | Bonferroni + Benjamini-Hochberg (FDR) correction |
| `novelty_detection` | Linear trend on daily lift with estimated 1% threshold stabilization crossing |

## Quickstart

```bash
pip install -e .
streamlit run dashboard/streamlit_app.py   # 7-tab interactive dashboard
pytest                                    # 65+ unit & simulation tests
```

## Criteo Benchmark Case Study

Reproducible benchmark on a 500,000-row sample (seed=42) of Criteo's 13.9M-row uplift modeling dataset ([Diemert et al., 2018](https://www.kaggle.com/datasets/arashnic/uplift-modeling)). Detailed report: [`results/criteo_results.md`](results/criteo_results.md).

| Check / Metric | Verified Result | Notes |
|---|---|---|
| Sample Size | 500,000 users (seed=42) | 74,999 Control / 425,001 Treatment |
| Target Split | 15% Control / 85% Treatment | Documented ad auction holdout |
| SRM Goodness-of-Fit | **No SRM (p = 0.9968)** | $\chi^2 = 0.000016$, unbiased randomization |
| Covariate Balance ($f9$) | **Normalized diff = +0.0181** | Well within experimental balance threshold ($|d| < 0.05$) |
| Conversion Rate Lift | **+62.59%** ($p = 1.08 \times 10^{-8}$) | 0.2000% control → 0.3252% treatment |
| Visit Rate Lift | **+24.13%** ($p = 1.68 \times 10^{-29}$) | 3.9334% control → 4.8826% treatment |
| Training Covariate Selection | **`f9`: Training $r = +0.4995$** | Fit on 50% training split ($N=250,000$) to prevent data leakage |
| Benchmark Full Correlation | **`f9`: Benchmark $r = +0.4974$** | Evaluated across full 500,000 users (`f0` suboptimal: $r = -0.1358$) |
| CUPED — Suboptimal Covariate (`f0`) | 1.84% in-sample / 1.82% held-out | Demonstrates consequence of arbitrary feature selection |
| CUPED — In-Sample Variance Reduction | **24.74% variance reduction** | Applied to full 500K sample (requires 24.74% fewer users) |
| CUPED — Held-Out Test Reduction | **24.53% out-of-sample reduction**| Evaluated on 250K held-out test users (confirms zero overfitting) |
| Treatment Effect SE Reduction | **11.95% uncertainty reduction** | Standard error narrows from 0.000783 to 0.000689 ($1 - \sqrt{1 - \rho^2}$) |

## Simulation-Based Statistical Validation

Simulation tests verify statistical guarantees across 2,000–5,000 simulated experiments:
- **Type-I error rate $\approx 5\%$** under $H_0$ (Welch's t-test: 4.8%, Chi-Square: 4.9%, Mann-Whitney: 5.1%)
- **O'Brien–Fleming boundaries**: Exact Lan–DeMets boundaries achieve empirical FWER = 4.99% across 5 planned looks
- **Power validation**: t-test achieves $\ge 75\%$ power at designed sample size ($n=1,570$, target 80%)
- **95% Confidence Interval Coverage**: Welch's CI contains true parameter in 94.8% of simulations
- **Edge cases**: Robust against zero conversions, identical constant outcomes, and heavily unbalanced sample sizes ($n_1=25, n_2=2500$)

## dbt & PostgreSQL Analytics Layer

5 SQL models transforming raw PostgreSQL tables into analytics marts:

```
stg_experiments + stg_experiment_results + stg_observations
    → mart_experiment_summary (one row per experiment with lift, SRM flag, and primary metric)
    → mart_daily_metrics (daily lift + 7-day rolling average)
```

Run and verify models:
```bash
python scripts/build_dbt_models.py    # Zero-friction direct execution & assertion verification
# or via dbt CLI:
cd dbt/statforge && dbt run && dbt test
```

## Tech Stack

| Layer | Tools |
|---|---|
| Statistics | Python 3.10+, NumPy, SciPy, Statsmodels, Pandas, Scikit-learn |
| Dashboards | Streamlit (7 tabs), Tableau Public, Plotly |
| Storage & ELT | PostgreSQL, SQLAlchemy, dbt |
| Testing & CI | pytest, GitHub Actions CI |

---

MIT License
