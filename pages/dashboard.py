"""Study-level analytics for the TrialOps Intelligence workspace."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from database import fetch_audit_history, fetch_queries
from utils.lab_analytics import classify_lab_results, laboratory_summary
from utils.safety_analytics import (
    safety_insights,
    safety_summary,
    serious_event_distribution,
    severity_distribution,
    top_adverse_events,
)
from utils.statistical_analysis import (
    descriptive_statistics as statistical_descriptives,
    endpoint_records as statistical_endpoint_records,
    global_statistical_test,
    safety_profile,
)
from utils.treatment_efficacy_analytics import (
    derive_efficacy_records,
    efficacy_summary,
    exposure_summary,
    subject_identifiers,
    treatment_assignments,
)

DATASET_STATE_KEY = "sdtm_datasets"
METADATA_STATE_KEY = "sdtm_upload_metadata"
VALIDATION_RESULTS_KEY = "validation_results"
ACTIVITY_EVENTS_KEY = "study_activity_events"

QUERY_STATUSES = ("Open", "Answered", "Closed")
SEVERITIES = ("Critical", "Major", "Minor")
STATUS_COLORS = {"Open": "#4C9AFF", "Answered": "#FFB020", "Closed": "#2FBF71"}
SEVERITY_COLORS = {"Critical": "#FF4B4B", "Major": "#FF8C00", "Minor": "#FFD43B"}


def session_datasets() -> dict[str, pd.DataFrame]:
    """Return valid uploaded dataframes from the current Streamlit session."""
    datasets = st.session_state.get(DATASET_STATE_KEY, {})
    if not isinstance(datasets, Mapping):
        return {}
    return {
        str(domain).upper(): dataframe
        for domain, dataframe in datasets.items()
        if isinstance(dataframe, pd.DataFrame)
    }


def validation_findings() -> pd.DataFrame:
    """Return the latest in-session validation results in a safe tabular form."""
    findings = st.session_state.get(VALIDATION_RESULTS_KEY)
    return findings.copy() if isinstance(findings, pd.DataFrame) else pd.DataFrame()


def count_unique_subjects(datasets: Mapping[str, pd.DataFrame]) -> int:
    """Count non-blank USUBJID values across every uploaded domain."""
    subject_ids: set[str] = set()
    for dataframe in datasets.values():
        subject_column = next(
            (str(column) for column in dataframe.columns if str(column).strip().upper() == "USUBJID"),
            None,
        )
        if subject_column is None:
            continue

        values = dataframe[subject_column].astype("string").dropna().str.strip()
        subject_ids.update(value for value in values.tolist() if value)
    return len(subject_ids)


def dataset_inventory(datasets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Build the dataset inventory from detected domains and session metadata."""
    metadata = st.session_state.get(METADATA_STATE_KEY, {})
    metadata = metadata if isinstance(metadata, Mapping) else {}
    inventory: list[dict[str, Any]] = []

    for domain, dataframe in sorted(datasets.items()):
        details = metadata.get(domain, {})
        detected_domain = details.get("detected_domain", domain) if isinstance(details, Mapping) else domain
        inventory.append(
            {
                "Dataset": domain,
                "Detected Domain": str(detected_domain).upper(),
                "Record Count": len(dataframe.index),
                "Column Count": len(dataframe.columns),
            }
        )

    return pd.DataFrame(
        inventory,
        columns=["Dataset", "Detected Domain", "Record Count", "Column Count"],
    )


def dataset_metadata(datasets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Build provenance details for each currently uploaded detected dataset."""
    metadata = st.session_state.get(METADATA_STATE_KEY, {})
    metadata = metadata if isinstance(metadata, Mapping) else {}
    rows: list[dict[str, Any]] = []

    for domain, dataframe in sorted(datasets.items()):
        details = metadata.get(domain, {})
        details = details if isinstance(details, Mapping) else {}
        uploaded_at = details.get("uploaded_at")
        upload_timestamp = (
            "Not available"
            if uploaded_at is None or pd.isna(uploaded_at) or not str(uploaded_at).strip()
            else str(uploaded_at)
        )
        rows.append(
            {
                "Dataset Name": domain,
                "Detected Domain": str(details.get("detected_domain", domain)).upper(),
                "Record Count": len(dataframe.index),
                "Column Count": len(dataframe.columns),
                "Upload Timestamp": upload_timestamp,
            }
        )

    return pd.DataFrame(
        rows,
        columns=["Dataset Name", "Detected Domain", "Record Count", "Column Count", "Upload Timestamp"],
    )


def category_counts(dataframe: pd.DataFrame, column: str, categories: tuple[str, ...]) -> pd.DataFrame:
    """Return a complete category count table, including zero-count categories."""
    if column not in dataframe.columns:
        counts = pd.Series(0, index=categories, dtype="int64")
    else:
        # Convert categoricals to ordinary strings before counting. Filling an
        # unknown value directly into a categorical series raises a TypeError.
        counts = (
            dataframe[column]
            .astype(str)
            .value_counts()
            .astype("int64")
            .reindex(categories, fill_value=0)
        )

    return pd.DataFrame(
        {
            column: categories,
            "Count": counts.to_numpy(dtype="int64"),
        }
    )


def dark_chart(figure: Any) -> Any:
    """Apply a consistent, compact dark presentation to a Plotly figure."""
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=12, r=12, t=32, b=12),
        legend_title_text="",
    )
    return figure


def validation_dataset_counts(findings: pd.DataFrame) -> pd.DataFrame:
    """Aggregate current validation findings by detected dataset domain."""
    if findings.empty or "Dataset" not in findings.columns:
        return pd.DataFrame(columns=["Dataset", "Findings"])

    return (
        findings.assign(Dataset=findings["Dataset"].astype("string").fillna("Unknown"))
        .groupby("Dataset", as_index=False)
        .size()
        .rename(columns={"size": "Findings"})
        .sort_values(["Findings", "Dataset"], ascending=[False, True])
    )


def display_audit_status(value: Any) -> str:
    """Return a safe status label for nullable SQLite audit values."""
    return "—" if value is None or pd.isna(value) else str(value)


def recent_activity(audit_history: pd.DataFrame) -> pd.DataFrame:
    """Combine persisted query audit events with current-session study activity."""
    session_events = st.session_state.get(ACTIVITY_EVENTS_KEY, [])
    session_activity = pd.DataFrame(session_events) if isinstance(session_events, list) else pd.DataFrame()
    if not session_activity.empty:
        session_activity = session_activity.reindex(columns=["Timestamp", "Event", "Detail"])

    audit_activity = pd.DataFrame(columns=["Timestamp", "Event", "Detail"])
    if not audit_history.empty:
        audit_activity = pd.DataFrame(
            {
                "Timestamp": audit_history.get("Timestamp", pd.Series(dtype="string")),
                "Event": "Query " + audit_history.get("Action", pd.Series(dtype="string")).astype("string").fillna("Updated"),
                "Detail": audit_history.apply(
                    lambda row: (
                        f"{row.get('Query ID', 'Query')}: "
                        f"{display_audit_status(row.get('Old Status'))} → "
                        f"{display_audit_status(row.get('New Status'))}"
                    ),
                    axis=1,
                ),
            }
        )

    combined = pd.concat([session_activity, audit_activity], ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=["Timestamp", "Event", "Detail"])

    combined["_sort_timestamp"] = pd.to_datetime(combined["Timestamp"], errors="coerce", utc=True)
    combined = combined.sort_values("_sort_timestamp", ascending=False, na_position="last")
    combined["Timestamp"] = combined["_sort_timestamp"].dt.strftime("%Y-%m-%d %H:%M UTC").fillna("Unknown")
    return combined[["Timestamp", "Event", "Detail"]].head(12).reset_index(drop=True)


def render_severity_card(label: str, count: int, color: str) -> None:
    """Render a compact color-coded validation severity summary card."""
    st.markdown(
        f"""
        <div class="severity-card" style="border-left-color: {color};">
            <div class="severity-label">{label}</div>
            <div class="severity-value">{count:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safety_metric_value(value: int | None) -> str:
    """Format optional AESER-derived counts without implying missing data is zero."""
    return "Not available" if value is None else f"{value:,}"


def render_drug_safety_overview(datasets: Mapping[str, pd.DataFrame]) -> None:
    """Render AE-driven drug safety review analytics when an AE dataset is available."""
    st.subheader("Drug Safety Overview")
    ae_records = datasets.get("AE")
    if ae_records is None:
        st.info("No detected AE dataset is available. Upload adverse-event data to populate the drug safety review.")
        return

    summary = safety_summary(ae_records)
    metrics = st.columns(4)
    metrics[0].metric("Total Adverse Events", safety_metric_value(summary["Total Adverse Events"]))
    metrics[1].metric("Subjects with Adverse Events", safety_metric_value(summary["Subjects with Adverse Events"]))
    metrics[2].metric("Serious Adverse Events", safety_metric_value(summary["Serious Adverse Events"]))
    metrics[3].metric("Non-Serious Adverse Events", safety_metric_value(summary["Non-Serious Adverse Events"]))

    top_events = top_adverse_events(ae_records)
    st.subheader("Top 10 Adverse Events")
    if top_events.empty:
        st.info("No populated AEDECOD or AETERM values are available to summarize adverse-event terms.")
    else:
        top_events_figure = dark_chart(
            px.bar(
                top_events.sort_values("Event Count"),
                x="Event Count",
                y="Adverse Event",
                orientation="h",
                title="Top 10 Adverse Events",
            )
        )
        top_events_figure.update_layout(showlegend=False, xaxis_title="Events", yaxis_title=None)
        st.plotly_chart(top_events_figure, use_container_width=True)

    severity_column, serious_column = st.columns(2)
    with severity_column:
        st.subheader("AE Severity Distribution")
        severity_counts = severity_distribution(ae_records)
        if severity_counts is None:
            st.info("AESEV is not available in the uploaded AE dataset, so severity distribution cannot be displayed.")
        elif not int(severity_counts["Event Count"].sum()):
            st.info("No recognized Mild, Moderate, or Severe AESEV values are available for severity analysis.")
        else:
            severity_figure = dark_chart(
                px.bar(
                    severity_counts,
                    x="Severity",
                    y="Event Count",
                    color="Severity",
                    title="AE Severity Distribution",
                )
            )
            severity_figure.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Events")
            st.plotly_chart(severity_figure, use_container_width=True)
    with serious_column:
        st.subheader("Serious Event Analysis")
        serious_counts = serious_event_distribution(ae_records)
        if serious_counts is None:
            st.info("AESER is not available in the uploaded AE dataset, so serious-event analysis cannot be displayed.")
        elif not int(serious_counts["Event Count"].sum()):
            st.info("No recognized AESER Yes or No values are available for serious-event analysis.")
        else:
            serious_figure = dark_chart(
                px.pie(
                    serious_counts,
                    names="Seriousness",
                    values="Event Count",
                    title="Serious and Non-Serious Events",
                )
            )
            st.plotly_chart(serious_figure, use_container_width=True)

    st.subheader("Safety Insights")
    insights = safety_insights(ae_records)
    if not insights:
        st.info("Safety insights will appear when populated AE term or seriousness data is available.")
    else:
        for insight in insights:
            st.info(insight)


def render_laboratory_overview(datasets: Mapping[str, pd.DataFrame]) -> None:
    """Render study-level LB KPIs from results and supplied reference limits."""
    st.subheader("Laboratory Overview")
    lb_records = datasets.get("LB")
    if lb_records is None:
        st.info("No detected LB dataset is available. Upload laboratory data to populate study-level laboratory metrics.")
        return

    summary = laboratory_summary(classify_lab_results(lb_records))
    metrics = st.columns(4)
    for container, (label, value) in zip(metrics, summary.items()):
        container.metric(label, value)


def render_treatment_efficacy_overview(datasets: Mapping[str, pd.DataFrame]) -> None:
    """Render live treatment-arm, exposure, safety, and efficacy headline metrics."""
    st.subheader("Treatment and Efficacy Overview")
    assignments = treatment_assignments(datasets.get("DM"), datasets.get("EX"))
    ex_records = datasets.get("EX")
    ae_records = datasets.get("AE")
    efficacy_records = datasets.get("EFF")

    exposed_subjects = exposure_summary(ex_records)["Total Exposed Subjects"] if ex_records is not None else 0
    treatment_arms = (
        int(assignments.loc[assignments["Treatment Arm"].ne("Not assigned"), "Treatment Arm"].nunique())
        if not assignments.empty
        else 0
    )
    ae_subjects = 0
    serious_ae_subjects = 0
    if ae_records is not None:
        ae_subjects = int(subject_identifiers(ae_records).replace("", pd.NA).nunique(dropna=True))
        serious_column = next(
            (str(column) for column in ae_records.columns if str(column).strip().upper() == "AESER"), None
        )
        if serious_column is not None:
            serious_values = ae_records[serious_column].astype("string").fillna("").str.strip().str.upper()
            serious_subjects = subject_identifiers(ae_records).loc[serious_values.isin({"Y", "YES", "TRUE", "1"})]
            serious_ae_subjects = int(serious_subjects.replace("", pd.NA).nunique(dropna=True))

    evaluable_subjects = 0
    mean_change: float | None = None
    if efficacy_records is not None:
        derived = derive_efficacy_records(efficacy_records)
        summary = efficacy_summary(derived)
        evaluable_subjects = int(summary["Total Subjects Evaluated"])
        mean_change = summary["Mean Change from Baseline"]

    metrics = st.columns(6)
    metrics[0].metric("Treatment Arms", treatment_arms)
    metrics[1].metric("Exposed Subjects", exposed_subjects)
    metrics[2].metric("Subjects with AEs", ae_subjects)
    metrics[3].metric("Serious AE Subjects", serious_ae_subjects)
    metrics[4].metric("Efficacy-Evaluable Subjects", evaluable_subjects)
    metrics[5].metric("Mean Change from Baseline", "Not available" if mean_change is None else f"{mean_change:,.1f}")


def render_statistical_overview(datasets: Mapping[str, pd.DataFrame]) -> None:
    """Render endpoint-aware statistical-review headline metrics without hardcoded values."""
    st.subheader("Statistical Review Overview")
    assignments = treatment_assignments(datasets.get("DM"), datasets.get("EX"))
    efficacy = datasets.get("EFF")
    if efficacy is None or assignments.empty:
        st.info("Upload detected efficacy and treatment-arm data to populate the statistical-review overview.")
        return

    derived = derive_efficacy_records(efficacy)
    endpoints = sorted(value for value in derived["Endpoint"].dropna().unique().tolist() if value)
    significant = 0
    non_significant = 0
    for endpoint in endpoints:
        test = global_statistical_test(
            statistical_endpoint_records(derived, assignments, endpoint, "Lower values indicate improvement")
        )
        p_value = test.get("P-value")
        if p_value is None:
            continue
        if p_value < 0.05:
            significant += 1
        else:
            non_significant += 1

    safety = safety_profile(assignments, datasets.get("AE"), datasets.get("LB"))
    highest_risk = "Not available"
    if not safety.empty and safety["Safety Risk Score"].notna().any():
        highest_risk = str(safety.sort_values("Safety Risk Score", ascending=False).iloc[0]["Treatment Arm"])
    best_arm = "Select endpoint"
    if len(endpoints) == 1:
        metrics = statistical_descriptives(
            statistical_endpoint_records(derived, assignments, endpoints[0], "Lower values indicate improvement")
        )
        if not metrics.empty and metrics["Mean Improvement"].notna().any():
            best_arm = str(metrics.sort_values("Mean Improvement", ascending=False).iloc[0]["Treatment Arm"])

    metrics = st.columns(6)
    metrics[0].metric("Total Subjects", count_unique_subjects(datasets))
    metrics[1].metric("Treatment Arms", int(assignments["Treatment Arm"].nunique()))
    metrics[2].metric("Significant Endpoints", significant)
    metrics[3].metric("Non-Significant Endpoints", non_significant)
    metrics[4].metric("Best Performing Arm", best_arm)
    metrics[5].metric("Highest Risk Arm", highest_risk)
    if len(endpoints) > 1:
        st.caption("Best-performing-arm selection is available in Statistical Analysis because multiple endpoint units should not be combined.")


def render_dashboard() -> None:
    """Render a live study overview from session datasets and SQLite query data."""
    datasets = session_datasets()
    findings = validation_findings()
    queries = fetch_queries()
    audit_history = fetch_audit_history()

    status_counts = category_counts(queries, "Status", QUERY_STATUSES)
    severity_counts = category_counts(findings, "Severity", SEVERITIES)
    query_severity_counts = category_counts(queries, "Severity", SEVERITIES)
    inventory = dataset_inventory(datasets)
    metadata = dataset_metadata(datasets)

    st.title("Clinical Data Review Dashboard")
    st.caption("Live study overview from uploaded SDTM data, validation results, and persisted query records.")

    st.markdown(
        """
        <style>
        .severity-card {
            background: rgba(255, 255, 255, 0.04);
            border-left: 5px solid;
            border-radius: 0.45rem;
            min-height: 5.4rem;
            padding: 0.85rem 1rem;
        }
        .severity-label { color: #D0D7DE; font-size: 0.92rem; }
        .severity-value { color: #FFFFFF; font-size: 1.7rem; font-weight: 650; margin-top: 0.25rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Study Overview")
    first_row = st.columns(4)
    first_row[0].metric("Datasets Uploaded", len(datasets))
    first_row[1].metric("Records Loaded", sum(len(dataframe.index) for dataframe in datasets.values()))
    first_row[2].metric("Total Subjects", count_unique_subjects(datasets))
    first_row[3].metric("Validation Findings", len(findings.index))

    second_row = st.columns(3)
    for metric_column, status in zip(second_row, QUERY_STATUSES):
        metric_column.metric(f"{status} Queries", int(status_counts.loc[status_counts["Status"] == status, "Count"].iloc[0]))

    st.divider()
    render_drug_safety_overview(datasets)

    st.divider()
    render_laboratory_overview(datasets)

    st.divider()
    render_treatment_efficacy_overview(datasets)

    st.divider()
    render_statistical_overview(datasets)

    st.subheader("Validation Findings Summary")
    severity_columns = st.columns(3)
    for card_column, severity in zip(severity_columns, SEVERITIES):
        count = int(severity_counts.loc[severity_counts["Severity"] == severity, "Count"].iloc[0])
        with card_column:
            render_severity_card(severity, count, SEVERITY_COLORS[severity])

    st.subheader("Dataset Inventory")
    if inventory.empty:
        st.info("No uploaded SDTM datasets are available for the study inventory. Upload source data to begin clinical review.")
    else:
        st.dataframe(inventory, hide_index=True, use_container_width=True)

    st.subheader("Dataset Metadata")
    if metadata.empty:
        st.info("Dataset metadata will appear after detected SDTM source data is uploaded for this study session.")
    else:
        st.dataframe(metadata, hide_index=True, use_container_width=True)

    st.subheader("Query Analytics")
    status_chart_column, severity_chart_column = st.columns(2)
    with status_chart_column:
        status_figure = dark_chart(
            px.bar(
                status_counts,
                x="Status",
                y="Count",
                color="Status",
                color_discrete_map=STATUS_COLORS,
                title="Query Status Distribution",
            )
        )
        status_figure.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Queries")
        st.plotly_chart(status_figure, use_container_width=True)
    with severity_chart_column:
        severity_figure = dark_chart(
            px.bar(
                query_severity_counts,
                x="Severity",
                y="Count",
                color="Severity",
                color_discrete_map=SEVERITY_COLORS,
                title="Query Severity Distribution",
            )
        )
        severity_figure.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Queries")
        st.plotly_chart(severity_figure, use_container_width=True)

    st.subheader("Validation Analytics")
    findings_by_dataset = validation_dataset_counts(findings)
    if findings_by_dataset.empty:
        st.info("No validation findings are available for analysis. Run the Validation Engine to populate this clinical data-quality view.")
    else:
        findings_figure = dark_chart(
            px.bar(
                findings_by_dataset,
                x="Dataset",
                y="Findings",
                color="Dataset",
                title="Validation Findings by Dataset",
            )
        )
        findings_figure.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Findings")
        st.plotly_chart(findings_figure, use_container_width=True)

    st.subheader("Recent Activity")
    activity = recent_activity(audit_history)
    if activity.empty:
        st.info("No recent clinical-review activity is available. Dataset uploads, validation runs, and query updates will be recorded here.")
    else:
        st.dataframe(activity, hide_index=True, use_container_width=True)


render_dashboard()
