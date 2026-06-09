import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from src.power_analysis import PowerAnalysis
from src.statistical_tests import StatisticalTestEngine
from src.cuped import CUPED
from src.sequential_testing import SequentialTester
from src.srm_detection import SRMDetector
from src.multiple_testing import MultipleTestingCorrection
from src.data import ExperimentDataGenerator
from src.novelty_detection import NoveltyDetector

st.set_page_config(
    page_title="StatForge — A/B Testing Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #1e2130;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #2d3147;
    }
    .srm-alert {
        background: #3d1a1a;
        border-left: 4px solid #ff4b4b;
        padding: 12px;
        border-radius: 4px;
    }
    .srm-ok {
        background: #1a3d1a;
        border-left: 4px solid #00cc44;
        padding: 12px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

st.title("StatForge — A/B Testing & Experimentation Platform")
st.caption("Statistical experimentation engine with power analysis, CUPED, sequential testing, and SRM detection")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Power Analysis",
    "Statistical Tests",
    "CUPED Variance Reduction",
    "Sequential Testing",
    "SRM + Multiple Testing",
    "Novelty Effect",
])

# ── Tab 1: Power Analysis ──────────────────────────────────────────────────────
with tab1:
    st.header("Power Analysis & Sample Size Calculator")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Parameters")
        test_type = st.selectbox("Metric type", ["Proportions (conversion rate)", "Means (revenue)"])
        alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01)
        power_target = st.slider("Target power (1-β)", 0.70, 0.99, 0.80, 0.01)

        if "Proportions" in test_type:
            baseline = st.number_input("Baseline conversion rate", 0.01, 0.50, 0.10, 0.01)
            mde = st.number_input("Minimum detectable effect (absolute)", 0.001, 0.10, 0.02, 0.001, format="%.3f")
        else:
            mean_val = st.number_input("Baseline mean", 1.0, 1000.0, 50.0)
            std_val = st.number_input("Standard deviation", 1.0, 500.0, 120.0)
            mde_abs = st.number_input("Minimum detectable effect (absolute)", 0.1, 100.0, 5.0)

    with col2:
        pa = PowerAnalysis(alpha=alpha, power=power_target)

        if "Proportions" in test_type:
            n = pa.sample_size_proportions(baseline, mde, alpha, power_target)
            effect_sizes = np.linspace(0.005, 0.10, 100)
            mde_values = [pa.sample_size_proportions(baseline, e, alpha, power_target) for e in effect_sizes]
        else:
            n = pa.sample_size_means(mean_val, std_val, mde_abs, alpha, power_target)
            effect_sizes = np.linspace(0.5, 20.0, 100)
            mde_values = [pa.sample_size_means(mean_val, std_val, e, alpha, power_target) for e in effect_sizes]

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Required n per variant", f"{n:,}")
        col_b.metric("Total sample needed", f"{n * 2:,}")
        col_c.metric("Significance level", f"{alpha:.0%}")

        # Sample size curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=effect_sizes, y=mde_values,
            mode="lines", line=dict(color="#636EFA", width=2),
            fill="tozeroy", fillcolor="rgba(99,110,250,0.1)",
        ))
        fig.update_layout(
            title="Sample Size vs Effect Size",
            xaxis_title="Minimum Detectable Effect",
            yaxis_title="Required Sample Size (per variant)",
            height=300, template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Power curve for fixed n
        effect_range = np.linspace(0.01, 0.30, 200)
        power_vals = pa.power_curve(effect_range, n, alpha)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=effect_range, y=power_vals,
            mode="lines", line=dict(color="#EF553B", width=2),
        ))
        fig2.add_hline(y=power_target, line_dash="dash", line_color="gray",
                       annotation_text=f"Target power={power_target:.0%}")
        fig2.update_layout(
            title=f"Power Curve (n={n:,} per variant)",
            xaxis_title="True Effect Size (Cohen's d)",
            yaxis_title="Statistical Power",
            height=300, template="plotly_dark",
        )
        st.plotly_chart(fig2, use_container_width=True)


# ── Tab 2: Statistical Tests ───────────────────────────────────────────────────
with tab2:
    st.header("Statistical Testing Engine")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Generate Experiment Data")
        n_users = st.slider("Users per variant", 500, 10000, 3000, 500)
        conv_baseline = st.slider("Baseline conversion rate", 0.05, 0.30, 0.10, 0.01)
        treatment_lift = st.slider("True treatment lift (absolute)", 0.0, 0.05, 0.02, 0.005)
        rev_lift = st.slider("Revenue lift per converter ($)", 0.0, 20.0, 5.0, 1.0)
        test_alpha = st.slider("α level", 0.01, 0.10, 0.05, 0.01)

        gen = ExperimentDataGenerator(seed=99)
        df = gen.generate(
            n_control=n_users, n_treatment=n_users,
            baseline_conversion=conv_baseline,
            treatment_effect_conversion=treatment_lift,
            treatment_effect_revenue=rev_lift,
        )

        control = df[df.variant == "control"]
        treatment = df[df.variant == "treatment"]

    with col2:
        engine = StatisticalTestEngine(alpha=test_alpha)

        st.subheader("Test Results")
        res_chi = engine.chi_square(
            int(control.converted.sum()), len(control),
            int(treatment.converted.sum()), len(treatment),
            alpha=test_alpha,
        )
        res_t = engine.t_test(
            control.revenue.values, treatment.revenue.values, alpha=test_alpha
        )
        res_mw = engine.mann_whitney(
            control.revenue.values, treatment.revenue.values, alpha=test_alpha
        )

        results_df = pd.DataFrame([res_chi.to_dict(), res_t.to_dict(), res_mw.to_dict()])
        st.dataframe(results_df, use_container_width=True)

        # Conversion rate comparison
        fig = go.Figure()
        rates = {
            "Control": control.converted.mean(),
            "Treatment": treatment.converted.mean(),
        }
        colors = ["#636EFA", "#EF553B"]
        for i, (name, rate) in enumerate(rates.items()):
            ci_half = 1.96 * np.sqrt(rate * (1 - rate) / n_users)
            fig.add_trace(go.Bar(
                x=[name], y=[rate], error_y=dict(type="data", array=[ci_half]),
                name=name, marker_color=colors[i],
            ))
        fig.update_layout(
            title="Conversion Rate with 95% CI",
            yaxis_title="Conversion Rate",
            yaxis_tickformat=".1%",
            template="plotly_dark", height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Revenue distribution
        fig2 = go.Figure()
        fig2.add_trace(go.Violin(
            y=control[control.revenue > 0].revenue, name="Control",
            side="negative", line_color="#636EFA", fillcolor="rgba(99,110,250,0.3)",
        ))
        fig2.add_trace(go.Violin(
            y=treatment[treatment.revenue > 0].revenue, name="Treatment",
            side="positive", line_color="#EF553B", fillcolor="rgba(239,85,59,0.3)",
        ))
        fig2.update_layout(
            title="Revenue Distribution (converters only)",
            yaxis_title="Revenue ($)", template="plotly_dark", height=300,
        )
        st.plotly_chart(fig2, use_container_width=True)


# ── Tab 3: CUPED ───────────────────────────────────────────────────────────────
with tab3:
    st.header("CUPED — Variance Reduction Using Pre-Experiment Covariate")
    st.info(
        "CUPED removes covariate signal from the post-experiment metric, reducing variance "
        "by 30–50%. This means you detect the same effect size with fewer users — or reach "
        "significance faster. Formula: **Y_adj = Y − θ(X − E[X])** where θ = Cov(Y,X)/Var(X)."
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        n_cuped = st.slider("Users per variant", 1000, 10000, 5000, 500, key="cuped_n")
        corr = st.slider("Covariate correlation (ρ)", 0.1, 0.9, 0.6, 0.05)
        cuped_lift = st.slider("Revenue lift ($)", 0.0, 20.0, 5.0, 1.0, key="cuped_lift")

        gen2 = ExperimentDataGenerator(seed=7)
        df2 = gen2.generate(
            n_control=n_cuped, n_treatment=n_cuped,
            covariate_correlation=corr,
            treatment_effect_revenue=cuped_lift,
        )
        ctrl = df2[df2.variant == "control"]
        trt = df2[df2.variant == "treatment"]

        cuped = CUPED()
        result = cuped.fit_transform(
            control_metric=ctrl.revenue.values,
            treatment_metric=trt.revenue.values,
            control_covariate=ctrl.pre_revenue.values,
            treatment_covariate=trt.pre_revenue.values,
        )

    with col2:
        summary = result.summary()
        c1, c2, c3 = st.columns(3)
        c1.metric("θ (theta)", f"{summary['theta']:.4f}",
                  help="Regression coefficient: how much pre-revenue predicts post-revenue")
        c2.metric("Variance reduction", f"{summary['variance_reduction_pct']:.1f}%",
                  help="ρ² × 100 — at ρ=0.6 expect ~36%")
        c3.metric("Sample size reduction", f"{summary['sample_size_reduction_pct']:.1f}%",
                  help="Because n ∝ σ², same % as variance reduction")

        # Density plot: original vs CUPED-adjusted
        try:
            import plotly.figure_factory as ff
            pooled_orig = np.concatenate([result.original_control, result.original_treatment])
            pooled_adj = np.concatenate([result.adjusted_control, result.adjusted_treatment])
            # Clip extreme tails for readability
            p1, p99 = np.percentile(pooled_orig, [1, 99])
            orig_clipped = pooled_orig[(pooled_orig >= p1) & (pooled_orig <= p99)]
            adj_clipped = pooled_adj[(pooled_adj >= p1) & (pooled_adj <= p99)]
            fig_density = ff.create_distplot(
                [orig_clipped.tolist(), adj_clipped.tolist()],
                ["Original", "CUPED Adjusted"],
                show_hist=False, show_rug=False,
                colors=["#636EFA", "#00CC96"],
            )
            fig_density.update_layout(
                title="Variance Narrowing: Original vs CUPED-Adjusted Distribution",
                xaxis_title="Revenue ($)", yaxis_title="Density",
                template="plotly_dark", height=280,
            )
            st.plotly_chart(fig_density, use_container_width=True)
        except Exception:
            # Fallback to histogram overlay if figure_factory fails
            fig_density = go.Figure()
            pooled_orig = np.concatenate([result.original_control, result.original_treatment])
            pooled_adj = np.concatenate([result.adjusted_control, result.adjusted_treatment])
            fig_density.add_trace(go.Histogram(x=pooled_orig, name="Original", opacity=0.6,
                                               nbinsx=80, marker_color="#636EFA"))
            fig_density.add_trace(go.Histogram(x=pooled_adj, name="CUPED Adjusted", opacity=0.6,
                                               nbinsx=80, marker_color="#00CC96"))
            fig_density.update_layout(barmode="overlay", template="plotly_dark", height=280,
                                      title="Variance Narrowing: Original vs CUPED-Adjusted")
            st.plotly_chart(fig_density, use_container_width=True)

        # Scatter: pre vs post with θ line
        rng_scatter = np.random.default_rng(0)
        idx = rng_scatter.choice(len(ctrl), min(400, len(ctrl)), replace=False)
        x_scatter = ctrl.pre_revenue.values[idx]
        y_scatter = ctrl.revenue.values[idx]
        x_line = np.linspace(x_scatter.min(), x_scatter.max(), 100)
        y_line = result.theta * (x_line - float(np.mean(ctrl.pre_revenue.values))) + float(np.mean(ctrl.revenue.values))

        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=x_scatter, y=y_scatter, mode="markers",
            marker=dict(color="#636EFA", opacity=0.35, size=4), name="Users",
        ))
        fig_scatter.add_trace(go.Scatter(
            x=x_line, y=y_line, mode="lines",
            line=dict(color="#EF553B", width=2), name=f"θ={result.theta:.3f}",
        ))
        fig_scatter.update_layout(
            title=f"Pre vs Post Revenue — Regression Slope (θ={result.theta:.3f})",
            xaxis_title="Pre-experiment revenue ($)", yaxis_title="Post-experiment revenue ($)",
            template="plotly_dark", height=260,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Sample size vs correlation interactive chart
    st.subheader("Sample Size Reduction vs Pre/Post Correlation")
    st.caption("Drag the ρ slider above — the curve shows how sample size shrinks as covariate correlation increases.")
    correlations = np.linspace(0.0, 0.95, 200)
    # Sample size scales with (1 - ρ²) relative to ρ=0 baseline
    baseline_n = 5000
    sample_sizes = [int(baseline_n * (1 - r ** 2)) for r in correlations]
    fig_ss = go.Figure()
    fig_ss.add_trace(go.Scatter(
        x=correlations, y=sample_sizes,
        mode="lines", line=dict(color="#AB63FA", width=2.5),
        fill="tozeroy", fillcolor="rgba(171,99,250,0.12)",
    ))
    fig_ss.add_vline(x=corr, line_dash="dash", line_color="#EF553B",
                     annotation_text=f"Current ρ={corr:.2f} → n={int(baseline_n*(1-corr**2)):,}",
                     annotation_position="top right")
    fig_ss.update_layout(
        title="Required Sample Size vs Covariate Correlation (baseline n=5,000)",
        xaxis_title="Correlation ρ (pre/post metric)",
        yaxis_title="Required sample size per variant",
        template="plotly_dark", height=280,
    )
    st.plotly_chart(fig_ss, use_container_width=True)


# ── Tab 4: Sequential Testing ──────────────────────────────────────────────────
with tab4:
    st.header("Sequential Testing — Safe Early Stopping")
    st.info(
        "Checking p-values repeatedly and stopping when p<0.05 inflates type-I error. "
        "Sequential testing with alpha-spending functions controls this inflation, "
        "allowing you to stop early if evidence is strong enough."
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        seq_n = st.slider("Max users per variant", 1000, 20000, 10000, 1000)
        n_looks = st.slider("Number of planned looks", 2, 10, 5)
        seq_alpha = st.slider("Overall α", 0.01, 0.10, 0.05, 0.01, key="seq_alpha")
        spending_fn = st.selectbox("Alpha-spending function", ["obrien_fleming", "pocock"])
        true_effect = st.slider("True effect size (Cohen's d)", 0.0, 0.5, 0.2, 0.05)

        rng = np.random.default_rng(55)
        control_seq = rng.normal(0, 1, seq_n)
        treatment_seq = rng.normal(true_effect, 1, seq_n)

        tester = SequentialTester(
            alpha=seq_alpha, max_n=seq_n * 2,
            n_looks=n_looks, spending_function=spending_fn,
        )
        seq_result = tester.run_full_sequence(control_seq, treatment_seq)

    with col2:
        seq_df = seq_result.to_dataframe()

        status_color = "🟢" if seq_result.final_decision == "reject_null" else "🔴"
        st.subheader(f"Decision: {status_color} {seq_result.final_decision.replace('_', ' ').title()}")

        # Sequential monitoring chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=seq_df["look"], y=seq_df["z_stat"].abs(),
            mode="lines+markers", name="|Z statistic|",
            line=dict(color="#636EFA", width=2),
            marker=dict(size=8),
        ))
        fig.add_trace(go.Scatter(
            x=seq_df["look"], y=seq_df["alpha_boundary"],
            mode="lines", name="α-boundary",
            line=dict(color="#EF553B", width=2, dash="dash"),
        ))
        fig.add_hline(
            y=1.96, line_dash="dot", line_color="gray",
            annotation_text="Naive z=1.96 (unadjusted)",
        )
        fig.update_layout(
            title="Sequential Monitoring: Z-statistic vs Alpha Boundary",
            xaxis_title="Look number",
            yaxis_title="|Z statistic|",
            template="plotly_dark", height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Alpha spending curve
        fracs = np.linspace(0.01, 1.0, 200)
        spent = [tester.alpha_spending(f) for f in fracs]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=fracs, y=spent, mode="lines",
                                   line=dict(color="#00CC96", width=2),
                                   name=spending_fn.replace("_", "-").title()))
        fig2.add_hline(y=seq_alpha, line_dash="dash", line_color="gray",
                       annotation_text=f"Total α={seq_alpha}")
        fig2.update_layout(
            title="Alpha Spending Function",
            xaxis_title="Information fraction (t)",
            yaxis_title="Cumulative α spent",
            template="plotly_dark", height=280,
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(seq_df, use_container_width=True)


# ── Tab 5: SRM + Multiple Testing ─────────────────────────────────────────────
with tab5:
    col_left, col_right = st.columns(2)

    with col_left:
        st.header("SRM Detection")
        st.info("Sample Ratio Mismatch (SRM) = actual assignment ratio ≠ intended ratio. "
                "An experiment with SRM cannot be trusted.")

        n_total_srm = st.number_input("Total users", 1000, 100000, 10000, 1000)
        intended_split = st.slider("Intended control fraction", 0.3, 0.7, 0.5, 0.05)
        observed_control_frac = st.slider(
            "Observed control fraction", 0.3, 0.7, 0.5, 0.001,
            help="Adjust this to simulate SRM"
        )
        srm_alpha = st.slider("SRM α", 0.001, 0.05, 0.01, 0.001, key="srm_alpha")

        n_ctrl_obs = int(n_total_srm * observed_control_frac)
        n_trt_obs = n_total_srm - n_ctrl_obs

        detector = SRMDetector(alpha=srm_alpha)
        srm_result = detector.check(
            {"control": n_ctrl_obs, "treatment": n_trt_obs},
            {"control": intended_split, "treatment": 1 - intended_split},
            alpha=srm_alpha,
        )

        if srm_result.srm_detected:
            st.markdown(f'<div class="srm-alert">⚠️ {srm_result.message}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="srm-ok">✅ {srm_result.message}</div>', unsafe_allow_html=True)

        srm_dict = srm_result.to_dict()
        c1, c2 = st.columns(2)
        c1.metric("χ² statistic", f"{srm_dict['chi2_statistic']:.4f}")
        c2.metric("p-value", f"{srm_dict['p_value']:.6f}")

        fig = go.Figure(data=[
            go.Bar(name="Observed", x=["Control", "Treatment"],
                   y=[n_ctrl_obs, n_trt_obs], marker_color=["#636EFA", "#EF553B"]),
            go.Bar(name="Expected", x=["Control", "Treatment"],
                   y=[n_total_srm * intended_split, n_total_srm * (1 - intended_split)],
                   marker_color=["rgba(99,110,250,0.4)", "rgba(239,85,59,0.4)"]),
        ])
        fig.update_layout(
            barmode="group", title="Observed vs Expected Assignment Counts",
            template="plotly_dark", height=280,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.header("Multiple Testing Correction")
        st.info("Testing 10 metrics simultaneously at α=0.05 → ~40% chance of ≥1 false positive. "
                "Corrections control this error rate.")

        n_metrics = st.slider("Number of metrics tested", 2, 20, 10)
        n_true_effects = st.slider("Metrics with true effects", 0, 10, 3)
        mt_alpha = st.slider("α level", 0.01, 0.10, 0.05, 0.01, key="mt_alpha")

        rng_mt = np.random.default_rng(77)
        raw_p_values = []
        metric_names = []
        for i in range(n_metrics):
            if i < n_true_effects:
                # Simulate a true effect — small p-value
                p = float(rng_mt.beta(1, 20))
            else:
                # Null — uniform p under H0
                p = float(rng_mt.uniform(0, 1))
            raw_p_values.append(p)
            metric_names.append(f"metric_{i+1}" + (" ✓" if i < n_true_effects else ""))

        corrector = MultipleTestingCorrection(alpha=mt_alpha)
        comparison_df = corrector.compare_methods(raw_p_values, metric_names, mt_alpha)
        st.dataframe(comparison_df.style.applymap(
            lambda v: "background-color: #1a3d1a" if v is True else
                      ("background-color: #3d1a1a" if v is False else ""),
            subset=["bonferroni_rejected", "bh_rejected"]
        ), use_container_width=True)

        # Rejection comparison
        bonf_result = corrector.bonferroni(raw_p_values, metric_names, mt_alpha)
        bh_result = corrector.benjamini_hochberg(raw_p_values, metric_names, mt_alpha)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, n_metrics + 1)),
            y=sorted(raw_p_values),
            mode="markers+lines", name="Raw p-values",
            marker=dict(color="#636EFA", size=8),
        ))
        fig.add_hline(y=mt_alpha, line_dash="dash", line_color="white",
                      annotation_text=f"α={mt_alpha} (uncorrected)")
        fig.add_hline(y=mt_alpha / n_metrics, line_dash="dot", line_color="#EF553B",
                      annotation_text=f"Bonferroni α/{n_metrics}")
        fig.update_layout(
            title="p-values vs Correction Thresholds",
            xaxis_title="Rank",
            yaxis_title="p-value",
            template="plotly_dark", height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Uncorrected rejections", sum(p < mt_alpha for p in raw_p_values))
        c2.metric("Bonferroni rejections", bonf_result.n_rejected)
        c3.metric("B-H rejections", bh_result.n_rejected)


# ── Tab 6: Novelty Effect ──────────────────────────────────────────────────────
with tab6:
    st.header("Novelty Effect Detection")
    st.info(
        "Users often react to novelty rather than genuine value. A new button color shows "
        "+20% clicks on day 1, +5% on day 7, +2% on day 14 — that's novelty, not improvement. "
        "Detection: fit a linear trend to daily lift values. Significant negative slope = flag."
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        n_days = st.slider("Experiment duration (days)", 7, 30, 21)
        n_per_day = st.slider("Users per day per variant", 100, 2000, 300, 100)
        initial_lift = st.slider("Initial lift (day 1)", 0.02, 0.40, 0.20, 0.01)
        decay_rate = st.slider("Daily decay rate", 0.0, 0.03, 0.015, 0.001,
                               help="0 = stable effect, >0 = novelty decay")
        novelty_alpha = st.slider("α for trend test", 0.01, 0.10, 0.05, 0.01, key="nov_alpha")

        detector = NoveltyDetector(alpha=novelty_alpha)
        daily_ctrl, daily_trt = detector.simulate_novelty_data(
            n_days=n_days,
            n_per_day=n_per_day,
            initial_lift=initial_lift,
            decay_rate=decay_rate,
        )
        nov_result = detector.detect(daily_ctrl, daily_trt, alpha=novelty_alpha)

    with col2:
        if nov_result.novelty_detected:
            st.markdown(
                f'<div class="srm-alert">⚠️ {nov_result.recommendation}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="srm-ok">✅ {nov_result.recommendation}</div>',
                unsafe_allow_html=True,
            )

        c1, c2, c3 = st.columns(3)
        c1.metric("Trend slope", f"{nov_result.slope:.4f}",
                  delta="declining" if nov_result.slope < 0 else "stable",
                  delta_color="inverse" if nov_result.slope < 0 else "normal")
        c2.metric("Trend p-value", f"{nov_result.p_value:.4f}")
        c3.metric("R²", f"{nov_result.r_squared:.3f}",
                  help="How much of daily lift variance is explained by the linear trend")

        # Daily lift + trend line
        days_list = nov_result.days
        lifts = nov_result.daily_lifts
        trend = [nov_result.slope * i + nov_result.intercept for i in range(len(days_list))]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=days_list, y=[l * 100 for l in lifts],
            mode="lines+markers", name="Daily lift (%)",
            line=dict(color="#636EFA", width=2),
            marker=dict(size=7),
        ))
        fig.add_trace(go.Scatter(
            x=days_list, y=[t * 100 for t in trend],
            mode="lines", name="Linear trend",
            line=dict(color="#EF553B", width=2, dash="dash"),
        ))
        fig.add_hline(y=0, line_color="gray", line_dash="dot")
        fig.update_layout(
            title="Daily Treatment Lift Over Time",
            xaxis_title="Day", yaxis_title="Treatment lift (%)",
            template="plotly_dark", height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Novelty vs stable comparison
        st.subheader("Novelty vs Stable Effect — Side by Side")
        fig2 = make_subplots(rows=1, cols=2,
                             subplot_titles=("High decay (novelty)", "No decay (stable)"))
        for col_idx, dr in enumerate([0.018, 0.0], start=1):
            dc, dt = detector.simulate_novelty_data(n_days=n_days, n_per_day=n_per_day,
                                                     initial_lift=initial_lift, decay_rate=dr)
            nr = detector.detect(dc, dt)
            fig2.add_trace(go.Scatter(
                x=nr.days, y=[l * 100 for l in nr.daily_lifts],
                mode="lines+markers",
                line=dict(color="#636EFA" if col_idx == 1 else "#00CC96", width=2),
                showlegend=False,
            ), row=1, col=col_idx)
            trend2 = [nr.slope * i + nr.intercept for i in range(len(nr.days))]
            fig2.add_trace(go.Scatter(
                x=nr.days, y=[t * 100 for t in trend2],
                mode="lines", line=dict(color="#EF553B", dash="dash", width=1.5),
                name="Trend", showlegend=col_idx == 1,
            ), row=1, col=col_idx)
        fig2.update_layout(template="plotly_dark", height=280)
        st.plotly_chart(fig2, use_container_width=True)
