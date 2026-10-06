"""Exploratory statistical evaluation for uploaded efficacy, safety, and exposure data."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.statistical_analysis import (
    clinical_interpretation,
    descriptive_statistics,
    endpoint_records,
    global_statistical_test,
    placebo_comparisons,
    responder_analysis,
    safety_efficacy_matrix,
    safety_profile,
    treatment_leaderboard,
)
from utils.statistical_reporting import (
    build_excel_summary,
    build_pdf_report,
    build_regulatory_review_report,
    export_dependencies_missing,
)
from utils.treatment_efficacy_analytics import derive_efficacy_records, treatment_assignments

DATASET_STATE_KEY = "sdtm_datasets"


def session_datasets() -> dict[str, pd.DataFrame]:
    """Return detected-domain dataframes retained in the active Streamlit session."""
    datasets = st.session_state.get(DATASET_STATE_KEY, {})
    if not isinstance(datasets, Mapping):
        return {}
    return {
        str(domain).upper(): dataframe
        for domain, dataframe in datasets.items()
        if isinstance(dataframe, pd.DataFrame)
    }


def dark_chart(figure: Any) -> Any:
    """Apply the established dark Plotly treatment used across TrialOps pages."""
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=12, r=12, t=42, b=12),
        legend_title_text="",
    )
    return figure


def number(value: float | int | None, digits: int = 3) -> str:
    """Format nullable numerical outputs without treating unavailable values as zero."""
    if value is None or pd.isna(value):
        return "Not available"
    return f"{float(value):,.{digits}f}" if isinstance(value, float) else f"{value:,}"


def render_confidence_plots(comparisons: pd.DataFrame, descriptives: pd.DataFrame) -> None:
    """Render placebo-difference and arm-mean confidence interval visualizations."""
    st.subheader("Confidence Interval Visualization")
    if comparisons.empty:
        st.info("Placebo comparison plots require a placebo reference and evaluable treatment-arm observations.")
        return
    plotted = comparisons.dropna(subset=["Mean Difference vs Placebo"]).copy()
    plotted["Lower Error"] = plotted["Mean Difference vs Placebo"] - plotted["95% CI Lower"]
    plotted["Upper Error"] = plotted["95% CI Upper"] - plotted["Mean Difference vs Placebo"]

    forest_column, difference_column = st.columns(2)
    with forest_column:
        forest = dark_chart(
            go.Figure(
                go.Scatter(
                    x=plotted["Mean Difference vs Placebo"],
                    y=plotted["Treatment Arm"],
                    mode="markers",
                    error_x=dict(type="data", array=plotted["Upper Error"], arrayminus=plotted["Lower Error"]),
                    marker=dict(size=10, color="#4C9AFF"),
                    hovertemplate="%{y}<br>Difference: %{x:.3f}<extra></extra>",
                )
            )
        )
        forest.add_vline(x=0, line_dash="dash", line_color="#B7C9D6")
        forest.update_layout(title="Forest Plot: Mean Difference vs Placebo", xaxis_title="Mean Improvement Difference")
        st.plotly_chart(forest, use_container_width=True)
    with difference_column:
        difference = dark_chart(
            px.bar(
                plotted,
                x="Treatment Arm",
                y="Mean Difference vs Placebo",
                color="Treatment Arm",
                title="Mean Difference vs Placebo",
            )
        )
        difference.add_hline(y=0, line_dash="dash", line_color="#B7C9D6")
        difference.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Mean Improvement Difference")
        st.plotly_chart(difference, use_container_width=True)

    error_records = descriptives.dropna(subset=["Mean Improvement", "95% CI Lower", "95% CI Upper"]).copy()
    if error_records.empty:
        st.info("Error bars require at least two evaluable observations in a treatment arm.")
        return
    error_records["Lower Error"] = error_records["Mean Improvement"] - error_records["95% CI Lower"]
    error_records["Upper Error"] = error_records["95% CI Upper"] - error_records["Mean Improvement"]
    error_chart = dark_chart(
        go.Figure(
            go.Bar(
                x=error_records["Treatment Arm"],
                y=error_records["Mean Improvement"],
                error_y=dict(type="data", array=error_records["Upper Error"], arrayminus=error_records["Lower Error"]),
                marker_color="#2FBF71",
            )
        )
    )
    error_chart.update_layout(title="Mean Improvement with 95% Confidence Intervals", xaxis_title=None, yaxis_title="Mean Improvement")
    st.plotly_chart(error_chart, use_container_width=True)


def render_exports(
    endpoint: str,
    placebo_arm: str,
    test_result: dict[str, Any],
    descriptives: pd.DataFrame,
    comparisons: pd.DataFrame,
    responders: pd.DataFrame,
    leaderboard: pd.DataFrame,
    matrix: pd.DataFrame,
    insights: list[str],
) -> None:
    """Offer local, on-demand clinical-review report downloads."""
    st.subheader("Export Clinical-review Outputs")
    missing = export_dependencies_missing()
    if missing:
        st.warning(
            "PDF and Excel exports require the optional packages: "
            + ", ".join(missing)
            + ". Install requirements.txt in the deployment environment to enable these exports."
        )
    else:
        excel_summary = build_excel_summary(
            {
                "Descriptives": descriptives,
                "Placebo Comparisons": comparisons,
                "Responder Analysis": responders,
                "Treatment Leaderboard": leaderboard,
                "Safety Efficacy Matrix": matrix,
                "Statistical Test": pd.DataFrame([test_result]).drop(columns=["Groups"], errors="ignore"),
            }
        )
        pdf_report = build_pdf_report(
            "TrialOps Intelligence - Exploratory Statistical Review",
            endpoint,
            test_result,
            descriptives,
            insights,
        )
        excel_column, pdf_column = st.columns(2)
        with excel_column:
            st.download_button(
                "Download Excel Summary",
                data=excel_summary,
                file_name="trialops_statistical_summary.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        with pdf_column:
            st.download_button(
                "Download PDF Report",
                data=pdf_report,
                file_name="trialops_statistical_review.pdf",
                mime="application/pdf",
            )

    regulatory_report = build_regulatory_review_report(endpoint, placebo_arm, test_result, insights)
    st.download_button(
        "Download Regulatory Review Briefing",
        data=regulatory_report,
        file_name="trialops_regulatory_review_briefing.md",
        mime="text/markdown",
    )


def render_statistical_analysis() -> None:
    """Render exploratory efficacy and safety evaluation with explicit limitations."""
    datasets = session_datasets()
    st.title("Statistical Analysis")
    st.caption("Exploratory treatment-arm evaluation for clinical review. This workspace does not replace a protocol-defined statistical analysis plan.")

    efficacy = datasets.get("EFF")
    if efficacy is None:
        st.info("No detected EFF dataset is available. Upload an efficacy dataset to begin treatment-arm statistical review.")
        return

    assignments = treatment_assignments(datasets.get("DM"), datasets.get("EX"))
    if assignments.empty:
        st.info("Treatment-arm analysis requires DM ARM/ARMCD or EXTRT data associated with USUBJID.")
        return

    derived = derive_efficacy_records(efficacy)
    endpoint_options = sorted(value for value in derived["Endpoint"].dropna().unique().tolist() if value)
    if not endpoint_options:
        st.info("No populated endpoint parameter is available for statistical evaluation.")
        return

    controls = st.columns(3)
    with controls[0]:
        endpoint = st.selectbox("Endpoint", endpoint_options)
    with controls[1]:
        direction = st.selectbox(
            "Improvement Direction",
            ("Lower values indicate improvement", "Higher values indicate improvement"),
            help="Determines whether negative or positive change is treated as improvement for charts and responder analysis.",
        )
    records = endpoint_records(derived, assignments, endpoint, direction)
    arm_options = sorted(records["Treatment Arm"].dropna().unique().tolist())
    if len(arm_options) < 2:
        st.warning("At least two treatment arms with evaluable subjects are required for statistical comparison.")
        return
    default_placebo = next((arm for arm in arm_options if "placebo" in arm.casefold()), arm_options[0])
    with controls[2]:
        placebo_arm = st.selectbox("Placebo Reference Arm", arm_options, index=arm_options.index(default_placebo))
    threshold = st.number_input("Responder Improvement Threshold", value=0.0, step=0.1, help="A subject is a responder when improvement is greater than or equal to this threshold.")

    descriptives = descriptive_statistics(records)
    test_result = global_statistical_test(records)
    comparisons = placebo_comparisons(records, placebo_arm)
    responders = responder_analysis(records, float(threshold))
    safety = safety_profile(assignments, datasets.get("AE"), datasets.get("LB"))
    leaderboard = treatment_leaderboard(descriptives, responders, safety)
    matrix, efficacy_threshold, risk_threshold = safety_efficacy_matrix(leaderboard)
    insights = clinical_interpretation(test_result, placebo_arm, comparisons)

    st.subheader("Treatment Comparison Analysis")
    metrics = st.columns(4)
    metrics[0].metric("Total Subjects", int(records["USUBJID"].nunique()))
    metrics[1].metric("Treatment Arms", len(arm_options))
    metrics[2].metric("Test", test_result["Test Name"])
    metrics[3].metric("P-value", number(test_result["P-value"], 4))
    st.dataframe(descriptives, hide_index=True, use_container_width=True)

    test_column, effect_column = st.columns(2)
    with test_column:
        st.subheader("Statistical Testing")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Test Name": test_result["Test Name"],
                        "Test Statistic": test_result["Test Statistic"],
                        "P-value": test_result["P-value"],
                        "Interpretation": test_result["Significance Interpretation"],
                    }
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
    with effect_column:
        st.subheader("Effect Size Analysis")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Metric": test_result["Effect Metric"],
                        "Effect Size": test_result["Effect Size"],
                        "Interpretation": test_result["Effect Interpretation"],
                    }
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("Placebo Comparisons")
    st.caption("Mean differences and confidence intervals are descriptive pairwise comparisons. They are not multiplicity-adjusted inferential results.")
    st.dataframe(comparisons, hide_index=True, use_container_width=True)
    render_confidence_plots(comparisons, descriptives)

    st.subheader("Responder Analysis")
    st.caption(f"Response threshold: improvement greater than or equal to {float(threshold):.2f}.")
    responder_column, responder_chart_column = st.columns(2)
    with responder_column:
        st.dataframe(responders, hide_index=True, use_container_width=True)
    with responder_chart_column:
        responder_chart = dark_chart(
            px.bar(responders, x="Treatment Arm", y="Response Rate (%)", color="Treatment Arm", title="Response Rate by Treatment Arm")
        )
        responder_chart.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Response Rate (%)")
        st.plotly_chart(responder_chart, use_container_width=True)

    st.subheader("Treatment Ranking")
    st.caption("Ranks are descriptive: higher observed improvement/response is better; lower descriptive safety risk score is better.")
    st.dataframe(leaderboard, hide_index=True, use_container_width=True)

    st.subheader("Safety vs Efficacy Matrix")
    if matrix.empty or risk_threshold is None:
        st.info("Safety-versus-efficacy classification requires uploaded AE or LB data with treatment-arm assignments.")
    else:
        matrix_chart = dark_chart(
            px.scatter(
                matrix,
                x="Mean Improvement",
                y="Safety Risk Score",
                color="Matrix Classification",
                text="Treatment Arm",
                size="N",
                title="Observed Efficacy and Descriptive Safety Risk by Arm",
            )
        )
        matrix_chart.add_vline(x=efficacy_threshold, line_dash="dash", line_color="#B7C9D6")
        matrix_chart.add_hline(y=risk_threshold, line_dash="dash", line_color="#B7C9D6")
        matrix_chart.update_traces(textposition="top center")
        matrix_chart.update_layout(xaxis_title="Mean Improvement", yaxis_title="Descriptive Safety Risk Score")
        st.plotly_chart(matrix_chart, use_container_width=True)
        st.caption("High/low labels are relative to the observed median treatment-arm improvement and safety-risk score in this upload.")
        st.dataframe(matrix.reindex(columns=["Treatment Arm", "Mean Improvement", "Safety Risk Score", "Matrix Classification"]), hide_index=True, use_container_width=True)

    st.subheader("Clinical Interpretation Engine")
    for insight in insights:
        st.info(insight)
    st.warning("Interpret results alongside the protocol, statistical analysis plan, endpoint definition, data quality review, and clinical context.")

    render_exports(endpoint, placebo_arm, test_result, descriptives, comparisons, responders, leaderboard, matrix, insights)


render_statistical_analysis()
