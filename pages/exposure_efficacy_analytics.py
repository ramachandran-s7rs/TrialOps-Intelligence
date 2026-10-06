"""Treatment exposure, efficacy, and treatment-arm clinical review."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.lab_analytics import classify_lab_results
from utils.treatment_efficacy_analytics import (
    ae_safety_by_arm,
    descriptive_statistics,
    derive_efficacy_records,
    dose_distribution,
    efficacy_by_arm,
    efficacy_summary,
    efficacy_with_arms,
    exposure_summary,
    lab_safety_by_arm,
    lab_trend_by_arm,
    route_distribution,
    treatment_assignments,
    two_group_comparison,
)

DATASET_STATE_KEY = "sdtm_datasets"


def session_datasets() -> dict[str, pd.DataFrame]:
    """Return detected-domain datasets available in the current session."""
    datasets = st.session_state.get(DATASET_STATE_KEY, {})
    if not isinstance(datasets, Mapping):
        return {}
    return {
        str(domain).upper(): dataframe
        for domain, dataframe in datasets.items()
        if isinstance(dataframe, pd.DataFrame)
    }


def dark_chart(figure: Any) -> Any:
    """Apply the application's dark analytical chart treatment."""
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=12, r=12, t=42, b=12),
        legend_title_text="",
    )
    return figure


def metric_value(value: float | int | None, digits: int = 1) -> str:
    """Format nullable numerical metrics without converting missing values to zero."""
    if value is None or pd.isna(value):
        return "Not available"
    return f"{value:,.{digits}f}" if isinstance(value, float) else f"{value:,}"


def selected_endpoint_records(records: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    """Return the selected endpoint's subject-level latest analysis records."""
    return records.loc[records["Endpoint"].eq(endpoint)].copy()


def render_exposure_review(ex_records: pd.DataFrame | None) -> None:
    """Render EX summary and optional dose, route, and duration distributions."""
    st.subheader("Treatment Exposure")
    if ex_records is None:
        st.info("No detected EX dataset is available. Upload SDTM exposure data to review treatment administration.")
        return

    summary = exposure_summary(ex_records)
    metrics = st.columns(3)
    metrics[0].metric("Exposed Subjects", summary["Total Exposed Subjects"])
    metrics[1].metric("Treatments Recorded", len(summary["Treatment Names"]))
    metrics[2].metric(
        "Mean Exposure Duration (Days)", metric_value(summary["Mean Exposure Duration (Days)"])
    )

    treatment_names = summary["Treatment Names"]
    if treatment_names:
        st.caption("Recorded treatments: " + ", ".join(treatment_names))
    else:
        st.info("EXTRT is not populated, so treatment names cannot be summarized from exposure records.")

    dose_column, route_column = st.columns(2)
    with dose_column:
        dose_counts = dose_distribution(ex_records)
        st.caption("Dose Distribution")
        if dose_counts.empty:
            st.info("No populated EXDOSE and EXDOSU values are available for dose distribution analysis.")
        else:
            figure = dark_chart(px.bar(dose_counts, x="Dose", y="Exposure Records", title="Dose Distribution"))
            figure.update_layout(showlegend=False, xaxis_title=None)
            st.plotly_chart(figure, use_container_width=True)
    with route_column:
        routes = route_distribution(ex_records)
        st.caption("Route Distribution")
        if routes.empty:
            st.info("No populated EXROUTE values are available for route distribution analysis.")
        else:
            figure = dark_chart(px.bar(routes, x="Route", y="Exposure Records", title="Route Distribution"))
            figure.update_layout(showlegend=False, xaxis_title=None)
            st.plotly_chart(figure, use_container_width=True)


def render_efficacy_review(
    efficacy_records: pd.DataFrame | None, assignments: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Render endpoint-level descriptive efficacy analytics and comparisons."""
    st.subheader("Efficacy Analytics")
    if efficacy_records is None:
        st.info(
            "No detected efficacy dataset is available. Upload a detected EFF dataset with USUBJID, PARAM or PARAMCD, and endpoint values to review outcomes."
        )
        return pd.DataFrame(), pd.DataFrame()

    derived = derive_efficacy_records(efficacy_records)
    endpoint_options = sorted(value for value in derived["Endpoint"].dropna().unique().tolist() if value)
    if not endpoint_options:
        st.info("No populated endpoint parameter is available for efficacy review.")
        return derived, pd.DataFrame()

    endpoint = st.selectbox("Efficacy Endpoint", endpoint_options, key="efficacy_endpoint")
    endpoint_records = efficacy_with_arms(selected_endpoint_records(derived, endpoint), assignments)
    selected_arms = st.multiselect(
        "Treatment Arms",
        sorted(endpoint_records["Treatment Arm"].dropna().unique().tolist()),
        default=sorted(endpoint_records["Treatment Arm"].dropna().unique().tolist()),
        key="efficacy_treatment_arms",
    )
    endpoint_records = endpoint_records.loc[endpoint_records["Treatment Arm"].isin(selected_arms)].copy()
    summary = efficacy_summary(endpoint_records)
    by_arm = efficacy_by_arm(endpoint_records)

    metrics = st.columns(5)
    metrics[0].metric("Subjects Evaluated", summary["Total Subjects Evaluated"])
    metrics[1].metric("Mean Baseline", metric_value(summary["Mean Baseline Value"]))
    metrics[2].metric("Mean Post-treatment", metric_value(summary["Mean Post-treatment Value"]))
    metrics[3].metric("Mean Change", metric_value(summary["Mean Change from Baseline"]))
    metrics[4].metric("Mean Percentage Change", metric_value(summary["Mean Percentage Change"]) + "%")

    st.caption("Subjects by Treatment Arm")
    if by_arm.empty:
        st.info("No subject-level efficacy observations are available for the selected endpoint and treatment arms.")
        return derived, endpoint_records
    st.dataframe(by_arm, hide_index=True, use_container_width=True)

    first_row, second_row = st.columns(2)
    with first_row:
        figure = dark_chart(
            px.bar(by_arm, x="Treatment Arm", y="Mean Change", color="Treatment Arm", title="Mean Change from Baseline by Arm")
        )
        figure.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Mean Change")
        st.plotly_chart(figure, use_container_width=True)
    with second_row:
        figure = dark_chart(
            px.bar(
                by_arm,
                x="Treatment Arm",
                y="Mean Percentage Change",
                color="Treatment Arm",
                title="Mean Percentage Change by Arm",
            )
        )
        figure.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Mean Percentage Change")
        st.plotly_chart(figure, use_container_width=True)

    third_row, fourth_row = st.columns(2)
    with third_row:
        figure = dark_chart(
            px.box(
                endpoint_records,
                x="Treatment Arm",
                y="Post-treatment Value",
                color="Treatment Arm",
                points="all",
                title="Endpoint Distribution by Arm",
            )
        )
        figure.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Post-treatment Value")
        st.plotly_chart(figure, use_container_width=True)
    with fourth_row:
        figure = dark_chart(
            px.strip(
                endpoint_records,
                x="Treatment Arm",
                y="Change",
                color="Treatment Arm",
                hover_data=["USUBJID", "Visit"],
                title="Subject-level Change from Baseline",
            )
        )
        figure.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Change")
        st.plotly_chart(figure, use_container_width=True)

    st.subheader("Descriptive Result")
    st.caption("Change-from-baseline summaries by treatment arm; confidence intervals use a normal approximation.")
    st.dataframe(descriptive_statistics(endpoint_records), hide_index=True, use_container_width=True)

    comparison = two_group_comparison(endpoint_records, selected_arms)
    st.subheader("Inferential Statistical Result")
    if comparison is None:
        st.info(
            "An optional two-group comparison is displayed only when exactly two selected groups each have at least 20 evaluable observations."
        )
    else:
        st.dataframe(pd.DataFrame([comparison]), hide_index=True, use_container_width=True)
        st.caption("This approximate comparison does not adjust for multiplicity, covariates, missing data, or trial design factors.")

    st.warning(
        "These are descriptive clinical-review outputs. Observed differences do not establish treatment efficacy; further statistical and clinical review is required."
    )
    return derived, endpoint_records


def render_safety_by_arm(
    ae_records: pd.DataFrame | None, lb_records: pd.DataFrame | None, assignments: pd.DataFrame
) -> None:
    """Render subject-denominator AE and laboratory summaries by treatment arm."""
    st.subheader("Safety by Treatment Arm")
    if assignments.empty:
        st.info("Treatment-arm safety comparisons require DM ARM/ARMCD or EXTRT assignment data.")
        return

    ae_summary = ae_safety_by_arm(ae_records, assignments)
    left_column, right_column = st.columns(2)
    with left_column:
        st.caption("Adverse Events by Treatment Arm")
        if ae_records is None:
            st.info("No detected AE dataset is available for treatment-arm safety review.")
        else:
            st.dataframe(ae_summary, hide_index=True, use_container_width=True)
            ae_chart = dark_chart(
                px.bar(
                    ae_summary,
                    x="Treatment Arm",
                    y="Subjects with AEs (%)",
                    color="Treatment Arm",
                    title="Subjects with at Least One AE by Arm",
                )
            )
            ae_chart.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Subjects with AEs (%)")
            st.plotly_chart(ae_chart, use_container_width=True)
    with right_column:
        st.caption("Laboratory Safety by Treatment Arm")
        if lb_records is None:
            st.info("No detected LB dataset is available for treatment-arm laboratory safety review.")
        else:
            lab_summary = lab_safety_by_arm(lb_records, assignments)
            st.dataframe(lab_summary, hide_index=True, use_container_width=True)

    if lb_records is not None:
        classified_labs = classify_lab_results(lb_records)
        test_options = sorted(
            value
            for value in classified_labs.loc[
                classified_labs["Collection Date"].notna() & classified_labs["Numeric Result"].notna(), "Lab Test"
            ].dropna().unique().tolist()
            if value
        )
        if test_options:
            selected_test = st.selectbox("Laboratory Trend by Treatment Arm", test_options, key="arm_lab_trend_test")
            trend = lab_trend_by_arm(lb_records, assignments, selected_test)
            if not trend.empty:
                figure = dark_chart(
                    px.line(
                        trend,
                        x="Collection Date",
                        y="Numeric Result",
                        color="Treatment Arm",
                        markers=True,
                        title=f"{selected_test} Mean Laboratory Trend by Arm",
                    )
                )
                figure.update_layout(xaxis_title=None, yaxis_title="Mean Result")
                st.plotly_chart(figure, use_container_width=True)
        else:
            st.info("No date-stamped numeric laboratory data are available for treatment-arm trend review.")


def render_clinical_insights(
    endpoint_records: pd.DataFrame,
    ae_records: pd.DataFrame | None,
    lb_records: pd.DataFrame | None,
    assignments: pd.DataFrame,
) -> None:
    """Render factual, deterministic treatment and safety observations."""
    st.subheader("Clinical Review Insights")
    insights: list[str] = []
    for _, row in efficacy_by_arm(endpoint_records).iterrows():
        change = row["Mean Change"]
        if pd.isna(change):
            continue
        direction = "reduction" if change < 0 else "increase"
        insights.append(
            f"Observed data show {row['Treatment Arm']} had a mean {direction} of {abs(change):.1f} units from baseline."
        )

    for _, row in ae_safety_by_arm(ae_records, assignments).iterrows():
        if int(row["Subjects in Arm"]) and int(row["Subjects with Serious AEs"]):
            insights.append(
                f"{row['Subjects with Serious AEs (%)']:.0f}% of subjects in {row['Treatment Arm']} experienced at least one serious adverse event."
            )

    if lb_records is not None and not assignments.empty:
        labs = classify_lab_results(lb_records)
        labs["USUBJID"] = labs["USUBJID"].astype("string").fillna("").str.strip()
        labs = labs.merge(assignments, on="USUBJID", how="left").fillna({"Treatment Arm": "Not assigned"})
        abnormal = labs.loc[labs["Result Status"].isin(["High", "Low"])]
        if not abnormal.empty:
            top_abnormal = (
                abnormal.groupby(["Treatment Arm", "Lab Test"], as_index=False)
                .agg(**{"Affected Subjects": ("USUBJID", "nunique")})
                .sort_values(["Affected Subjects", "Lab Test"], ascending=[False, True])
                .iloc[0]
            )
            arm_size = int(assignments.loc[assignments["Treatment Arm"].eq(top_abnormal["Treatment Arm"]), "USUBJID"].nunique())
            if arm_size:
                insights.append(
                    f"{top_abnormal['Lab Test']} abnormalities occurred in {int(top_abnormal['Affected Subjects'])} of {arm_size} subjects in {top_abnormal['Treatment Arm']}."
                )

    if not insights:
        st.info("Clinical review insights will appear when treatment, efficacy, or AE data are available by treatment arm.")
        return
    for insight in insights:
        st.info(insight)
    st.caption("Further statistical and clinical review is required before drawing treatment conclusions.")


def render_exposure_efficacy_analytics() -> None:
    """Render the complete treatment exposure and efficacy analytical workspace."""
    datasets = session_datasets()
    st.title("Exposure and Efficacy Analytics")
    st.caption("Treatment exposure, endpoint review, and arm-based safety summaries from detected SDTM data.")

    assignments = treatment_assignments(datasets.get("DM"), datasets.get("EX"))
    render_exposure_review(datasets.get("EX"))
    st.divider()
    _, endpoint_records = render_efficacy_review(datasets.get("EFF"), assignments)
    st.divider()
    render_safety_by_arm(datasets.get("AE"), datasets.get("LB"), assignments)
    st.divider()
    render_clinical_insights(endpoint_records, datasets.get("AE"), datasets.get("LB"), assignments)


render_exposure_efficacy_analytics()
