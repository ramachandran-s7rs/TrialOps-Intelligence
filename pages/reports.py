"""Sponsor-facing clinical operations reporting from active study data."""

from __future__ import annotations
from collections.abc import Mapping
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.report_analytics import build_report_data, build_report_pdf
from utils.statistical_reporting import build_excel_summary, export_dependencies_missing

REPORT_TYPES = (
    "Clinical Data Quality Report", "Query Management Report", "Safety Surveillance Report", "Protocol Deviation Report",
    "Laboratory Review Report", "Exposure and Efficacy Report", "Site Performance Report", "Study Executive Summary",
)


def datasets() -> dict[str, pd.DataFrame]:
    value = st.session_state.get("sdtm_datasets", {})
    return {str(key).upper(): frame for key, frame in value.items() if isinstance(frame, pd.DataFrame)} if isinstance(value, Mapping) else {}


def chart(figure):
    figure.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=12,r=12,t=40,b=12), legend_title_text="")
    return figure


def render_exports(name: str, data: dict, selected: pd.DataFrame) -> None:
    st.subheader("Export Report")
    st.download_button("Download CSV", selected.to_csv(index=False).encode("utf-8"), file_name="trialops_selected_report.csv", mime="text/csv")
    missing = export_dependencies_missing()
    if missing:
        st.warning("PDF and Excel downloads require: " + ", ".join(missing) + ". Install requirements.txt in the deployment environment.")
        return
    excel = build_excel_summary({key: value for key, value in data.items() if isinstance(value, pd.DataFrame)})
    pdf = build_report_pdf(name, data["Summary"], selected, data["Executive Insights"])
    first, second = st.columns(2)
    with first: st.download_button("Download Excel Study Package", excel, file_name="trialops_study_package.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with second: st.download_button("Download PDF Report", pdf, file_name="trialops_selected_report.pdf", mime="application/pdf")


def render_reports() -> None:
    study = datasets(); st.title("Reports"); st.caption("Audit-ready clinical operations, safety, data-quality, and study-performance reporting.")
    if not study:
        st.info("No detected SDTM datasets are available. Upload study data to generate clinical operations reports."); return
    validation = st.session_state.get("validation_results")
    data = build_report_data(study, validation if isinstance(validation, pd.DataFrame) else pd.DataFrame())
    summary = data["Summary"]
    st.subheader("Report Dashboard")
    rows = ["Total Subjects","Total Sites","Total Queries","Open Queries","Total Validation Findings","Protocol Deviations","Safety Signals","Detected Anomalies"]
    for group in (rows[:4], rows[4:]):
        columns = st.columns(4)
        for column, label in zip(columns, group): column.metric(label, summary[label])
    report_type = st.selectbox("Report Template", REPORT_TYPES)
    selected = pd.DataFrame()
    if report_type == "Clinical Data Quality Report":
        st.subheader(report_type); st.dataframe(data["Quality"], hide_index=True, use_container_width=True); st.dataframe(data["Domains"], hide_index=True, use_container_width=True); selected = data["Quality"]
    elif report_type == "Query Management Report":
        st.subheader(report_type); left, right = st.columns(2)
        with left:
            st.plotly_chart(chart(px.pie(data["Query Status"], names="Status", values="Queries", title="Queries by Status")), use_container_width=True)
        with right:
            st.plotly_chart(chart(px.bar(data["Query Domain"], x="Dataset", y="Queries", title="Queries by Domain")), use_container_width=True)
        if not data["Query Time"].empty: st.plotly_chart(chart(px.line(data["Query Time"], x="Month", y="Queries", markers=True, title="Queries Over Time")), use_container_width=True)
        selected = data["Queries"]
    elif report_type == "Safety Surveillance Report":
        st.subheader(report_type); st.dataframe(data["Safety Summary"], hide_index=True, use_container_width=True); st.dataframe(data["Safety by Arm"], hide_index=True, use_container_width=True)
        if not data["Safety Severity"].empty: st.plotly_chart(chart(px.bar(data["Safety Severity"], x="Severity", y="Event Count", color="Severity", title="AE Severity Distribution")), use_container_width=True)
        selected = data["Safety by Arm"]
    elif report_type == "Protocol Deviation Report":
        st.subheader(report_type); deviations = data["Deviations"]; st.dataframe(deviations, hide_index=True, use_container_width=True)
        if not deviations.empty:
            categories = deviations["Issue Type"].value_counts().rename_axis("Category").reset_index(name="Deviations")
            st.plotly_chart(chart(px.bar(categories, x="Category", y="Deviations", title="Deviation Categories")), use_container_width=True)
            dated = deviations.dropna(subset=["Anomaly Date"]).copy()
            if not dated.empty:
                dated["Month"] = pd.to_datetime(dated["Anomaly Date"]).dt.to_period("M").astype(str)
                monthly = dated.groupby("Month", as_index=False).size().rename(columns={"size":"Deviations"})
                st.plotly_chart(chart(px.line(monthly, x="Month", y="Deviations", markers=True, title="Monthly Protocol Deviation Trend")), use_container_width=True)
        selected = deviations
    elif report_type == "Laboratory Review Report":
        st.subheader(report_type); st.json(data["Laboratory"]); st.dataframe(data["Lab Alerts"], hide_index=True, use_container_width=True); selected = data["Lab Alerts"]
    elif report_type == "Exposure and Efficacy Report":
        st.subheader(report_type); st.dataframe(data["Efficacy"], hide_index=True, use_container_width=True)
        efficacy_records = data["Efficacy Records"]
        if not efficacy_records.empty and efficacy_records["Change"].notna().any():
            by_endpoint = efficacy_records.groupby("Endpoint", as_index=False)["Change"].mean().rename(columns={"Change":"Mean Change"})
            st.plotly_chart(chart(px.bar(by_endpoint, x="Endpoint", y="Mean Change", title="Mean Endpoint Change")), use_container_width=True)
        selected = data["Efficacy"]
    elif report_type == "Site Performance Report":
        st.subheader(report_type); st.dataframe(data["Site Performance"], hide_index=True, use_container_width=True)
        if not data["Site Performance"].empty: st.plotly_chart(chart(px.bar(data["Site Performance"], x="Site ID", y="Performance Score", color="Performance Classification", title="Site Performance Score")), use_container_width=True)
        selected = data["Site Performance"]
    else:
        st.subheader(report_type)
        for insight in data["Executive Insights"]: st.info(insight)
        st.caption("Narrative outputs are sponsor-style study observations and require review against source data, protocol, and clinical context.")
        selected = pd.DataFrame([{"Insight": insight} for insight in data["Executive Insights"]])
    render_exports(report_type, data, selected)


render_reports()
