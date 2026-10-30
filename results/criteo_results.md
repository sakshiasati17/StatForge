# Criteo Uplift Modeling v2.1 — A/B Testing & CUPED Benchmark

Reproducible analysis on a **500,000-row sample** (seed=42) of the Criteo Uplift dataset.

---

## 1. Dataset & Randomization Summary

| Metric | Value |
|---|---|
| Dataset | Criteo Uplift v2.1 (Diemert et al., 2018) |
| Citation | ACM KDD Workshop on AdKDD & TargetAd (2018) |
| Rows Sampled | 500,000 (seed=42) |
| Control Users | 74,999 (15.00%) |
| Treatment Users | 425,001 (85.00%) |
| Target Split Ratio | 15% Control / 85% Treatment (documented auction holdout) |
| SRM $\chi^2$ Statistic | 0.000016 |
| **SRM p-value** | **0.9968** (No SRM detected) |

**Covariate Balance & Pre-Treatment Integrity**:
- `f9` normalized difference between treatment and control: **+0.0181** ($|d| < 0.05$ threshold).
- Features `f0`–`f11` are anonymized pre-experiment context features. The minimal standardized difference confirms no substantial selection imbalance between variants.

---

## 2. Hypothesis Testing Results

| Outcome | Control Rate | Treatment Rate | Relative Lift | p-value | Significant ($\alpha=0.05$) |
|---|---|---|---|---|---|
| **Conversion** | 0.2000% (150) | 0.3252% (1,382) | **+62.59%** | 1.08e-08 | **Yes** |
| **Visit** | 3.9334% (2,950) | 4.8826% (20,751) | **+24.13%** | 1.68e-29 | **Yes** |

---

## 3. Covariate Correlations with Visit (Train Partition, No Leakage)

Feature selection was conducted strictly on a 50% training split (250,000 rows):
- Optimal selected feature: **`f9`** (Pearson $\rho = +0.4995$)

Full-sample correlation ranking:

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

## 4. CUPED Variance Reduction & Standard Error Uncertainty

Formula: $Y_{adj} = Y - \theta(X - E[X])$, $\theta = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}$

Theoretical maximum variance reduction: $\rho^2 \times 100\%$

| Covariate | In-Sample Var Reduction | Held-Out Test Var Reduction | $\theta$ | Treatment Effect SE Reduction |
|---|---|---|---|---|
| **`f0` (suboptimal)** | **1.84%** | **1.82%** | -0.005366 | **0.67%** |
| **`f9` (optimal)** | **24.74%** | **24.53%** | 0.015060 | **11.95%** |

**Conclusion**:
1. **Variance Reduction**: Proper covariate selection via training correlation yields **24.74%** variance reduction with `f9` vs only **1.84%** with default `f0` (a 13x improvement).
2. **Standard Error Uncertainty**: Estimator standard error narrows by **11.95%**, shrinking confidence intervals and accelerating experiment velocity.
3. **Generalization**: On held-out test data, `f9` achieves **24.53%** variance reduction, confirming zero overfitting.
