# StatForge — A/B Testing & Experimentation Platform

Statistical experimentation platform with CUPED variance reduction, sequential testing, SRM detection, and dual dashboards.

---

## Components

| Module | What it does |
|---|---|
| `power_analysis` | Sample size calculator, power curves, MDE |
| `statistical_tests` | Welch's t-test, Mann-Whitney U, chi-square |
| `cuped` | Variance reduction via pre-experiment covariate |
| `sequential_testing` | Alpha-spending functions (O'Brien-Fleming, Pocock) |
| `srm_detection` | Chi-square test for broken randomization |
| `multiple_testing` | Bonferroni + Benjamini-Hochberg correction |

---

## Quickstart

```bash
pip install -r requirements.txt
```

### Run the Streamlit dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

### Run tests

```bash
pytest tests/ -v --cov=src
```

### Export data for Tableau

```bash
python dashboard/tableau_export.py
```
Then import the CSVs from `dashboard/tableau_data/` into Tableau Public.

---

## Database setup

```bash
psql -U postgres -c "CREATE DATABASE statforge;"
psql -U postgres -d statforge -f sql/schema.sql
cp .env.example .env  # fill in credentials
```

---

## Tech stack

| Layer | Tools |
|---|---|
| Statistics engine | Python, NumPy, SciPy, Statsmodels, Pandas |
| Visualizations | Plotly |
| Technical dashboard | Streamlit |
| Business dashboard | Tableau Public |
| Storage | PostgreSQL |
| Testing | pytest + pytest-cov |

---

## Key concepts

**CUPED** — adjusts each user's metric by their pre-experiment behavior: `Y_adj = Y - θ(X - E[X])`. Variance reduction ≈ ρ² (squared correlation between pre and post metric). This means the same effect is detectable with fewer users.

**Sequential testing** — instead of a fixed p<0.05 rule, uses an alpha-spending function to distribute the type-I error budget across planned looks. The O'Brien-Fleming function spends conservatively early, allowing a tight boundary mid-experiment and a relaxed one at the end.

**SRM** — if 50K users were randomized 50/50 but you observe 48K/52K, that's likely not random noise. Chi-square goodness-of-fit detects this. A flagged SRM means the experiment results should not be reported until the pipeline is investigated.
