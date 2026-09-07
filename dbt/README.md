# StatForge dbt Layer

Transforms raw PostgreSQL tables into analytics-ready marts for dashboards and reporting.

## Models

### Staging (views — lightweight cleaning)

| Model | Source table | What it does |
|---|---|---|
| `stg_experiments` | `experiments` | Type-casts, trims whitespace, filters nulls |
| `stg_experiment_results` | `experiment_results` | Validates p-values in [0,1], type-casts |
| `stg_observations` | `experiment_observations` | Adds `observed_date`, validates group names |

### Marts (tables — ready for BI tools)

| Model | What it does |
|---|---|
| `mart_experiment_summary` | One row per experiment: lift, p-value, CI, SRM flag, power |
| `mart_daily_metrics` | Daily treatment lift + 7-day rolling average per experiment |

## Setup

```bash
pip install dbt-postgres

# Copy profiles to ~/.dbt/ or set DBT_PROFILES_DIR
cp dbt/profiles.yml ~/.dbt/profiles.yml

# Set env vars (or .env)
export DB_HOST=localhost DB_USER=postgres DB_PASSWORD=... DB_NAME=statforge

cd dbt/statforge
dbt deps
dbt run          # build all models
dbt test         # run schema + custom tests
dbt docs generate && dbt docs serve   # interactive lineage graph
```

## DAG

```
experiments (raw)          experiment_results (raw)    experiment_observations (raw)
       |                           |                              |
stg_experiments            stg_experiment_results        stg_observations
       |                           |                              |
       +---------------------------+------------------------------+
                                   |
                    mart_experiment_summary
                    mart_daily_metrics
```

## Custom tests

- `assert_p_value_in_range` — fails if any p-value falls outside [0, 1]
- `assert_no_negative_users` — fails if control or treatment N ≤ 0
