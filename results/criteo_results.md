# Criteo Uplift Modeling v2.1 — A/B Testing & CUPED Benchmark

Reproducible analysis on a **500,000-row sample** (seed=42) of the Criteo Uplift dataset.

---

## 1. Dataset & Randomization Summary

| Metric | Value |
|---|---|
| Dataset | Criteo Uplift v2.1 (Diemert et al.) |
| Rows Sampled | 500,000 (seed=42) |
| Control Users | 74,999 (15.00%) |
| Treatment Users | 425,001 (85.00%) |
| Target Split Ratio | 15% Control / 85% Treatment |
| SRM $\chi^2$ Statistic | 0.000016 |
| **SRM p-value** | **0.9968** (No SRM detected) |

**Pre-Treatment Covariate Validity**: Features `f0`–`f11` represent anonymized user browser and historical activity metrics captured before ad delivery. Because randomization is unbiased ($p=0.9968$), covariates satisfy $X \perp \!\! \perp T$ and do not introduce bias into treatment effect estimates.

---

## 2. Hypothesis Testing Results

| Outcome | Control Rate | Treatment Rate | Relative Lift | p-value | Significant ($\alpha=0.05$) |
|---|---|---|---|---|---|
| **Conversion** | 0.2000% (150) | 0.3252% (1,382) | **+62.59%** | 1.08e-08 | **Yes** |
| **Visit** | 3.9334% (2,950) | 4.8826% (20,751) | **+24.13%** | 1.68e-29 | **Yes** |

---

## 3. Covariate Correlations with Visit

Feature ranking based on Pearson correlation with `visit`:

| Feature | Pearson $\rho$ | Notes |
|---|---|---|
| **`f9`** | **+0.4974** | **Strongest positive predictor — optimal CUPED covariate** |
| `f4` | +0.2723 | Moderate positive correlation |
| `f10` | +0.2109 | Moderate positive correlation |
| `f1` | +0.1130 | Weak positive correlation |
| `f7` | +0.0969 | Weak positive correlation |
| `f2` | -0.0313 | Near zero |
| **`f0`** | **-0.1358** | **Suboptimal covariate (common default mistake)** |
| `f6` | -0.1400 | Negative correlation |
| `f5` | -0.1432 | Negative correlation |
| `f3` | -0.2102 | Moderate negative correlation |
| `f11` | -0.2180 | Moderate negative correlation |
| `f8` | -0.4595 | Strong negative correlation |

---

## 4. CUPED Variance Reduction & Sample Efficiency

Formula: $Y_{adj} = Y - \theta(X - E[X])$, $\theta = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}$

Theoretical maximum variance reduction: $\rho^2 \times 100\%$

| Covariate | In-Sample Variance Reduction | Held-Out Out-of-Sample Reduction | $\theta$ | Effective Sample Savings |
|---|---|---|---|---|
| **`f0` (wrong)** | **1.84%** | **1.82%** | -0.005366 | ~1.8% lower sample |
| **`f9` (optimal)** | **24.74%** | **24.53%** | 0.015060 | **~24.7% lower sample** |

**Conclusion**: Proper covariate selection via `feature_covariate_correlations()` accounts for the 13x difference in variance reduction (1.84% vs 24.74%), enabling experiments to reach equivalent statistical power with **~25% fewer users**. Out-of-sample evaluation on held-out test data confirms the reduction is robust and does not overfit (24.53% out-of-sample).
