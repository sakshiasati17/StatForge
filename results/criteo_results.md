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

## 3. Covariate Correlations with Visit: Training Sample vs Full Benchmark

To prevent data leakage, covariate evaluation was conducted in two separate steps:

1. **Training Sample ($N = 250,000$, 50% random split):**
   - Used **strictly for feature ranking and covariate selection**.
   - Optimal feature: **`f9`** with **training correlation $r = +0.4995$**.
   - Suboptimal feature: **`f0`** with training correlation $r = -0.1367$.
   - Selecting the covariate on the training split guarantees zero leakage into evaluation.

2. **Full Benchmark Sample ($N = 500,000$, seed=42):**
   - Used for full-sample benchmarking and overall descriptive reporting.
   - Benchmark correlation for `f9`: **$r = +0.4974$**.
   - Benchmark correlation for `f0`: **$r = -0.1358$**.

Full-sample correlation ranking across all 12 pre-treatment features:

| Feature | Benchmark Correlation ($r$) | Training Correlation ($r$) | Role & Description |
|---|---|---|---|
| **`f9`** | **+0.4974** | **+0.4995** | **Optimal selected covariate (highest positive correlation with visit)** |
| `f4` | +0.2723 | +0.2708 | Secondary positive predictor |
| `f10` | +0.2109 | +0.2140 | Moderate positive predictor |
| `f1` | +0.1130 | +0.1126 | Weak positive predictor |
| `f7` | +0.0969 | +0.0985 | Weak positive predictor |
| `f2` | -0.0313 | -0.0319 | Near zero correlation |
| **`f0`** | **-0.1358** | **-0.1367** | **Suboptimal default covariate (demonstrates failure of guessing)** |
| `f6` | -0.1400 | -0.1426 | Weak negative predictor |
| `f5` | -0.1432 | -0.1451 | Weak negative predictor |
| `f3` | -0.2102 | -0.2137 | Moderate negative predictor |
| `f11` | -0.2180 | -0.2136 | Moderate negative predictor |
| `f8` | -0.4595 | -0.4625 | Strong negative predictor |

---

## 4. CUPED Variance Reduction: In-Sample vs Held-Out Out-of-Sample

Formula: $Y_{adj} = Y - \theta(X - E[X])$, $\theta = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}$

Theoretical maximum variance reduction: $\rho^2 \times 100\%$

| Covariate | In-Sample Variance Reduction | Held-Out Test Variance Reduction | $\theta$ Coefficient | Treatment Effect SE Reduction | Equivalent User Savings |
|---|---|---|---|---|---|
| **`f0` (suboptimal)** | **1.84%** | **1.82%** | -0.005366 | **0.67%** | ~1.8% |
| **`f9` (optimal)** | **24.74%** | **24.53%** | 0.015060 | **11.95%** | **24.74%** |

### Mathematical & Methodological Distinctions:

1. **In-Sample Variance Reduction (24.74%):**
   - $\theta$ is fitted and evaluated on the entire 500,000-user sample.
   - Raw metric variance decreases from **$0.04516$** to **$0.03398$**.
   - Directly translates to requiring **24.74% fewer users** to achieve equivalent statistical power at $\alpha=0.05$.

2. **Held-Out Test Variance Reduction (24.53%):**
   - To prove that CUPED does **not overfit**, $\theta = 0.01516$ is learned strictly on the 250,000 training users, and then applied out-of-sample to the 250,000 held-out test users.
   - Test variance decreases from **$0.04483$** to **$0.03383$** (**24.53% reduction**).
   - The close agreement (24.74% in-sample vs 24.53% out-of-sample) confirms genuine generalizability.

3. **Treatment Effect Standard Error Uncertainty Reduction (11.95%):**
   - Standard error scales with the square root of variance: $\text{SE} \propto \sqrt{\text{Var}}$.
   - Uncertainty reduction is $1 - \sqrt{1 - \rho^2} = 1 - \sqrt{1 - 0.2474} \approx \mathbf{11.95\%}$.
   - The treatment effect standard error narrows from **$0.000783$** to **$0.000689$**, shrinking confidence interval widths and enabling faster decision-making.
