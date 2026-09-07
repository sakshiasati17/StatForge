"""
Reproducible experiment report generator.
Produces a self-contained HTML report for a single experiment run,
including power analysis, test results, CUPED output, and SRM check.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from src.cuped import CUPED
from src.power_analysis import PowerAnalysis
from src.srm_detection import SRMDetector
from src.statistical_tests import StatisticalTestEngine


@dataclass
class ExperimentInput:
    name: str
    control: np.ndarray
    treatment: np.ndarray
    control_covariate: Optional[np.ndarray] = None
    treatment_covariate: Optional[np.ndarray] = None
    baseline_rate: Optional[float] = None
    min_detectable_effect: Optional[float] = None
    expected_split: Optional[dict] = None  # e.g. {"control": 0.5, "treatment": 0.5}
    description: str = ""


def _fig_to_html(fig: go.Figure) -> str:
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def _distribution_figure(control: np.ndarray, treatment: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=control, name="Control", opacity=0.6,
                               marker_color="#4C78A8", nbinsx=40))
    fig.add_trace(go.Histogram(x=treatment, name="Treatment", opacity=0.6,
                               marker_color="#F58518", nbinsx=40))
    fig.update_layout(
        title=title, barmode="overlay",
        xaxis_title="Value", yaxis_title="Count",
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=60, b=40, l=40, r=20),
        height=350,
    )
    return fig


def _power_curve_figure(pa: PowerAnalysis, n: int) -> go.Figure:
    effect_sizes = np.linspace(0.05, 1.0, 80)
    powers = pa.power_curve(effect_sizes, n=n)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=effect_sizes, y=powers, mode="lines",
                             line=dict(color="#4C78A8", width=2)))
    fig.add_hline(y=0.80, line_dash="dash", line_color="gray",
                  annotation_text="80% power", annotation_position="right")
    fig.update_layout(
        title="Power Curve", xaxis_title="Effect Size (Cohen's d)",
        yaxis_title="Power", yaxis_range=[0, 1],
        margin=dict(t=60, b=40, l=40, r=20), height=300,
    )
    return fig


def generate_report(exp: ExperimentInput, output_path: str = "reports/report.html") -> str:
    """Generate a self-contained HTML experiment report. Returns path to file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    engine = StatisticalTestEngine(alpha=0.05)
    pa = PowerAnalysis(alpha=0.05, power=0.80)
    srm = SRMDetector(alpha=0.01)

    # --- Statistical test ---
    test_result = engine.t_test(exp.control, exp.treatment)

    # --- SRM check ---
    group_counts = {"control": len(exp.control), "treatment": len(exp.treatment)}
    expected = exp.expected_split or {"control": 0.5, "treatment": 0.5}
    srm_result = srm.check(group_counts, expected_ratios=expected)

    # --- Power analysis ---
    n = len(exp.control)
    effect_size = abs(np.mean(exp.treatment) - np.mean(exp.control)) / np.std(np.concatenate([exp.control, exp.treatment]))
    achieved_power = pa.achieved_power(n=n, effect_size=effect_size)

    # --- CUPED ---
    cuped_html = ""
    cuped_summary = {}
    if exp.control_covariate is not None and exp.treatment_covariate is not None:
        cuped = CUPED()
        cuped_result = cuped.fit_transform(
            exp.control.astype(float), exp.treatment.astype(float),
            exp.control_covariate.astype(float), exp.treatment_covariate.astype(float),
        )
        cuped_summary = {
            "Variance Reduction": f"{cuped_result.variance_reduction_pct:.2f}%",
            "θ (theta)": f"{cuped_result.theta:.4f}",
            "Adjusted Control Mean": f"{np.mean(cuped_result.adjusted_control):.4f}",
            "Adjusted Treatment Mean": f"{np.mean(cuped_result.adjusted_treatment):.4f}",
        }
        # re-run t-test on adjusted values
        adj_result = engine.t_test(cuped_result.adjusted_control, cuped_result.adjusted_treatment)
        cuped_html = f"""
        <section>
          <h2>CUPED Variance Reduction</h2>
          {_table(cuped_summary)}
          <p><strong>Adjusted p-value:</strong> {adj_result.p_value:.4f}
             &nbsp; <strong>Significant:</strong> {"Yes ✓" if adj_result.significant else "No"}</p>
        </section>
        """

    # --- Figures ---
    dist_fig = _distribution_figure(exp.control, exp.treatment, "Metric Distribution")
    power_fig = _power_curve_figure(pa, n)

    # --- Build HTML ---
    status_color = "#2d8a4e" if test_result.significant else "#c0392b"
    status_text = "SIGNIFICANT" if test_result.significant else "NOT SIGNIFICANT"
    srm_color = "#c0392b" if srm_result.srm_detected else "#2d8a4e"
    srm_text = "SRM DETECTED ⚠" if srm_result.srm_detected else "No SRM ✓"

    summary_rows = {
        "Experiment": exp.name,
        "Generated": timestamp,
        "Control N": f"{len(exp.control):,}",
        "Treatment N": f"{len(exp.treatment):,}",
        "Control Mean": f"{np.mean(exp.control):.4f}",
        "Treatment Mean": f"{np.mean(exp.treatment):.4f}",
        "Absolute Lift": f"{np.mean(exp.treatment) - np.mean(exp.control):.4f}",
        "Effect Size (Cohen's d)": f"{effect_size:.4f}",
        "p-value": f"{test_result.p_value:.4f}",
        "95% CI": f"[{test_result.confidence_interval[0]:.4f}, {test_result.confidence_interval[1]:.4f}]",
        "Achieved Power": f"{achieved_power:.1%}",
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Experiment Report: {exp.name}</title>
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          max-width: 1000px; margin: 40px auto; padding: 0 24px; color: #1a1a2e; background: #f8f9fa; }}
  h1 {{ border-bottom: 3px solid #4C78A8; padding-bottom: 8px; }}
  h2 {{ color: #4C78A8; margin-top: 36px; }}
  section {{ background: white; border-radius: 8px; padding: 24px; margin-bottom: 24px;
             box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
  .badge {{ display: inline-block; padding: 6px 16px; border-radius: 20px; font-weight: 700;
            font-size: 14px; color: white; background: {status_color}; }}
  .srm-badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 13px;
                font-weight: 600; color: white; background: {srm_color}; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #e5e7eb; }}
  th {{ background: #f3f4f6; font-weight: 600; }}
  tr:last-child td {{ border-bottom: none; }}
  .meta {{ color: #6b7280; font-size: 13px; }}
</style>
</head>
<body>
<h1>Experiment Report</h1>
<p class="meta">Experiment: <strong>{exp.name}</strong> &nbsp;|&nbsp; Generated: {timestamp}</p>
{f'<p class="meta">{exp.description}</p>' if exp.description else ""}

<section>
  <h2>Result &nbsp; <span class="badge">{status_text}</span>
       &nbsp; <span class="srm-badge">{srm_text}</span></h2>
  {_table(summary_rows)}
</section>

<section>
  <h2>Metric Distribution</h2>
  {_fig_to_html(dist_fig)}
</section>

<section>
  <h2>Power Analysis</h2>
  <p>Achieved power at current sample size: <strong>{achieved_power:.1%}</strong></p>
  {_fig_to_html(power_fig)}
</section>

{cuped_html}

<section>
  <h2>SRM Check</h2>
  <p>χ² statistic: <strong>{srm_result.chi2_statistic:.4f}</strong>
     &nbsp; p-value: <strong>{srm_result.p_value:.4f}</strong></p>
  <p>{srm_result.message}</p>
</section>

<footer style="color:#9ca3af;font-size:12px;text-align:center;margin-top:40px;">
  StatForge — A/B Testing & Experimentation Platform
</footer>
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html, encoding="utf-8")
    return output_path


def _table(rows: dict) -> str:
    rows_html = "".join(
        f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows.items()
    )
    return f"<table>{rows_html}</table>"


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 1000
    pre_c = rng.normal(50, 20, n)
    pre_t = rng.normal(50, 20, n)
    ctrl = 0.6 * pre_c + rng.normal(100, 15, n)
    trt = 0.6 * pre_t + rng.normal(105, 15, n)

    exp = ExperimentInput(
        name="Homepage CTA Button Color",
        control=ctrl,
        treatment=trt,
        control_covariate=pre_c,
        treatment_covariate=pre_t,
        description="Testing orange vs blue CTA button on homepage. Primary metric: session duration.",
    )
    path = generate_report(exp, "reports/sample_report.html")
    print(f"Report saved to: {path}")
