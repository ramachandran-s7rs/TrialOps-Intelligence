"""Explainable data-quality, safety, and operational anomaly review."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.anomaly_analytics import analysis_result, anomaly_insights
from utils.anomaly_reporting import build_pdf_summary, build_summary_report
from utils.statistical_reporting import build_excel_summary, export_dependencies_missing

DATASET_STATE_KEY = "sdtm_datasets"


def session_datasets() -> dict[str, pd.DataFrame]:
    """Return the current session's detected SDTM datasets."""
    datasets = st.session_state.get(DATASET_STATE_KEY, {})
    if not isinstance(datasets, Mapping):
        return {}
    return {
        str(domain).upper(): dataframe
        for domain, dataframe in datasets.items()
        if isinstance(dataframe, pd.DataFrame)
    }


def dark_chart(figure: Any) -> Any:
    """Apply the application-standard dark Plotly configuration."""
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=12, r=12, t=42, b=12),
        legend_title_text="",
    )
    return figure


def risk_count(findings: pd.DataFrame, risk_level: str) -> int:
    """Return a safe count for one risk level."""
    return int(findings["Risk Level"].eq(risk_level).sum()) if "Risk Level" in findings.columns else 0


def render_timeline_analysis(findings: pd.DataFrame) -> None:
    """Render monthly, site, and cumulative explainable anomaly trends."""
    st.subheader("Timeline Analysis")
    dated = findings.dropna(subset=["Anomaly Date"]).copy()
    if dated.empty:
        st.info("No valid source dates were available for anomaly trend analysis. Upload date-stamped SDTM records to populate this view.")
        return
    dated["Month"] = dated["Anomaly Date"].dt.to_period("M").astype(str)
    monthly = dated.groupby("Month", as_index=False).size().rename(columns={"size": "Anomalies"})
    site_trend = (
        dated.loc[dated["Site ID"].ne("Not available")]
        .groupby("Site ID", as_index=False)
        .size()
        .rename(columns={"size": "Anomalies"})
    )
    cumulative = dated.sort_values("Anomaly Date").copy()
    cumulative["Cumulative Anomalies"] = range(1, len(cumulative.index) + 1)

    monthly_column, site_column = st.columns(2)
    with monthly_column:
        chart = dark_chart(px.line(monthly, x="Month", y="Anomalies", markers=True, title="Monthly Anomaly Trend"))
        chart.update_layout(xaxis_title=None, yaxis_title="Anomalies")
        st.plotly_chart(chart, use_container_width=True)
    with site_column:
        if site_trend.empty:
            st.info("No site identifier was available for a site-level anomaly trend.")
        else:
            chart = dark_chart(
                px.bar(site_trend.sort_values("Anomalies"), x="Site ID", y="Anomalies", color="Site ID", title="Site Anomaly Trend")
            )
            chart.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Anomalies")
            st.plotly_chart(chart, use_container_width=True)
    cumulative_chart = dark_chart(
        px.line(cumulative, x="Anomaly Date", y="Cumulative Anomalies", markers=True, title="Cumulative Anomaly Count")
    )
    cumulative_chart.update_layout(xaxis_title=None, yaxis_title="Cumulative Anomalies")
    st.plotly_chart(cumulative_chart, use_container_width=True)


def render_exports(result: Mapping[str, Any], insights: list[str]) -> None:
    """Provide PDF, Excel, and summary-report downloads from the active analysis."""
    st.subheader("Export Anomaly Review")
    findings = result["Findings"]
    missing = export_dependencies_missing()
    if missing:
        st.warning(
            "PDF and Excel exports require the optional packages: "
            + ", ".join(missing)
            + ". Install requirements.txt in the deployment environment to enable these exports."
        )
    else:
        excel = build_excel_summary(
            {
                "Priority Findings": findings,
                "Anomaly Frequency": result["Frequency"],
                "Laboratory Outliers": result["Laboratory Outliers"],
                "Safety Signals": result["Safety Signals"],
                "Safety by Arm": result["Safety by Arm"],
                "Site Risk": result["Site Risk"],
            }
        )
        pdf = build_pdf_summary(result, findings, insights)
        excel_column, pdf_column = st.columns(2)
        with excel_column:
            st.download_button(
                "Download Excel Review",
                data=excel,
                file_name="trialops_anomaly_review.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        with pdf_column:
            st.download_button(
                "Download PDF Summary",
                data=pdf,
                file_name="trialops_anomaly_detection_summary.pdf",
                mime="application/pdf",
            )
    st.download_button(
        "Download Anomaly Detection Summary Report",
        data=build_summary_report(result, findings, insights),
        file_name="trialops_anomaly_detection_summary.md",
        mime="text/markdown",
    )


def render_anomaly_detection() -> None:
    """Render the complete clinical operations anomaly-review workspace."""
    datasets = session_datasets()
    st.title("Anomaly Detection")
    st.caption("Explainable rule-based review of clinical data quality, safety, protocol, and site-performance signals.")
    if not datasets:
        st.info("No detected SDTM datasets are available. Upload clinical source data to begin anomaly review.")
        return

    result = analysis_result(datasets)
    findings = result["Findings"]
    st.subheader("Study Overview")
    metrics = st.columns(4)
    metrics[0].metric("Total Subjects", result["Total Subjects"])
    metrics[1].metric("Total Sites", result["Total Sites"])
    metrics[2].metric("Records Reviewed", result["Total Records"])
    metrics[3].metric("Anomalies Detected", len(findings.index))
    risk_metrics = st.columns(3)
    risk_metrics[0].metric("High Risk Signals", risk_count(findings, "High"))
    risk_metrics[1].metric("Medium Risk Signals", risk_count(findings, "Medium"))
    risk_metrics[2].metric("Low Risk Signals", risk_count(findings, "Low"))

    st.subheader("Data Quality Anomalies")
    frequency = result["Frequency"]
    if frequency.empty:
        st.success("No explainable data-quality, safety, laboratory, or operational anomalies were detected in the current uploaded datasets.")
    else:
        st.dataframe(frequency.sort_values("Count", ascending=False), hide_index=True, use_container_width=True)

    st.subheader("Laboratory Outlier Detection")
    lab_outliers = result["Laboratory Outliers"]
    if lab_outliers.empty:
        st.info("No laboratory outliers were identified from numeric results and supplied reference ranges.")
    else:
        st.dataframe(lab_outliers, hide_index=True, use_container_width=True)

    st.subheader("Safety Signal Detection")
    safety_signals = result["Safety Signals"]
    safety_by_arm = result["Safety by Arm"]
    safety_column, arm_column = st.columns(2)
    with safety_column:
        if safety_signals.empty:
            st.info("No multiple-serious-AE or severe/non-serious AE patterns were detected.")
        else:
            st.dataframe(safety_signals, hide_index=True, use_container_width=True)
    with arm_column:
        if safety_by_arm.empty:
            st.info("Treatment-arm safety comparison requires AE data and DM ARM/ARMCD or EXTRT assignments.")
        else:
            st.dataframe(safety_by_arm, hide_index=True, use_container_width=True)

    st.subheader("Site Risk Monitoring")
    site_risk = result["Site Risk"]
    if site_risk.empty:
        st.info("No site identifiers were detected. Upload SITEID, SITE, CENTER, CENTRE, or SITE_NUMBER to enable site-risk monitoring.")
    else:
        st.dataframe(site_risk, hide_index=True, use_container_width=True)
        chart = dark_chart(
            px.bar(site_risk, x="Site ID", y="Risk Score", color="Risk Level", title="Site Operational Risk Score")
        )
        chart.update_layout(xaxis_title=None, yaxis_title="Risk Score")
        st.plotly_chart(chart, use_container_width=True)

    render_timeline_analysis(findings)

    st.subheader("Risk Prioritization")
    if findings.empty:
        st.info("No priority review records are available from the active anomaly rules.")
    else:
        st.dataframe(
            findings.reindex(
                columns=["Subject ID", "Site ID", "Issue Type", "Severity", "Risk Score", "Recommended Review", "Dataset", "Source Row", "Message"]
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("Automated Clinical Insights")
    insights = anomaly_insights(result)
    for insight in insights:
        st.info(insight)
    st.caption("Signals are review priorities, not diagnoses, audit findings, or proof of a protocol deviation.")

    render_exports(result, insights)


render_anomaly_detection()
