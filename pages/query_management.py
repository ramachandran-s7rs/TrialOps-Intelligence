"""Manage persistent clinical data queries and their lifecycle history."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from database import (
    ACTIVE_QUERY_STATUSES,
    QUERY_EXPORT_COLUMNS,
    QUERY_STATUSES,
    clear_lab_alerts,
    clear_queries,
    create_queries_from_findings,
    fetch_audit_history,
    fetch_queries,
    initialise_database,
    update_query,
)


QUERY_COLUMNS = [
    "Query ID",
    "Subject ID",
    "Dataset",
    "Source Row",
    "Field",
    "Rule",
    "Issue Type",
    "Description",
    "Severity",
    "Status",
    "Response",
    "Resolution Note",
    "Created At",
    "Updated At",
    "Closed At",
]
AUDIT_COLUMNS = ["Query ID", "Action", "Old Status", "New Status", "Timestamp"]
STATUSES = QUERY_STATUSES
SEVERITIES = ("Critical", "Major", "Minor")
DATASET_STATE_KEYS = (
    "sdtm_datasets",
    "sdtm_upload_metadata",
    "sdtm_upload_fingerprints",
    "sdtm_pending_uploads",
    "study_activity_events",
)
VALIDATION_STATE_KEYS = (
    "validation_results",
    "validation_source_fingerprints",
    "auto_closed_queries",
)


def validation_finding_records() -> list[dict[str, object]]:
    """Return current session-scoped validation findings as query candidates."""
    findings = st.session_state.get("validation_results")
    if not isinstance(findings, pd.DataFrame) or findings.empty:
        return []
    return findings.to_dict(orient="records")


def filter_queries(
    queries: pd.DataFrame,
    statuses: list[str],
    severities: list[str],
    datasets: list[str],
    subject_id: str,
    search_text: str,
) -> pd.DataFrame:
    """Apply UI filters without modifying persistent query records."""
    filtered = queries.copy()
    if statuses:
        filtered = filtered[filtered["Status"].isin(statuses)]
    if severities:
        filtered = filtered[filtered["Severity"].isin(severities)]
    if datasets:
        filtered = filtered[filtered["Dataset"].isin(datasets)]
    if subject_id.strip():
        filtered = filtered[filtered["Subject ID"].str.contains(subject_id.strip(), case=False, na=False)]
    if search_text.strip():
        search_columns = ("Query ID", "Subject ID", "Rule", "Description")
        matched = pd.Series(False, index=filtered.index)
        for column_name in search_columns:
            matched |= filtered[column_name].str.contains(search_text.strip(), case=False, na=False)
        filtered = filtered[matched]
    return filtered


def render_metrics(queries: pd.DataFrame) -> None:
    """Display lifecycle and severity counts from the persisted query register."""
    metrics = st.columns(7)
    metrics[0].metric("Total Queries", len(queries.index))
    for container, status in zip(metrics[1:4], STATUSES):
        container.metric(status, int((queries["Status"] == status).sum()))
    for container, severity in zip(metrics[4:], SEVERITIES):
        container.metric(severity, int((queries["Severity"] == severity).sum()))


def clear_uploaded_datasets() -> None:
    """Remove only session-scoped uploaded datasets and reset the uploader widget."""
    for key in DATASET_STATE_KEYS:
        st.session_state.pop(key, None)
    st.session_state["sdtm_uploader_version"] = st.session_state.get("sdtm_uploader_version", 0) + 1


def clear_validation_findings() -> None:
    """Remove session-scoped findings without modifying persisted queries."""
    for key in VALIDATION_STATE_KEYS:
        st.session_state.pop(key, None)


@st.dialog("Clear uploaded datasets")
def confirm_clear_uploaded_datasets() -> None:
    """Require explicit confirmation before clearing uploaded datasets."""
    st.warning("This removes uploaded datasets from the active browser session only.")
    cancel, confirm = st.columns(2)
    with cancel:
        if st.button("Cancel", key="cancel_clear_uploaded"):
            st.rerun()
    with confirm:
        if st.button("Clear uploaded datasets", type="primary", key="confirm_clear_uploaded"):
            clear_uploaded_datasets()
            st.rerun()


@st.dialog("Clear validation findings")
def confirm_clear_validation_findings() -> None:
    """Require explicit confirmation before clearing validation findings."""
    st.warning("This removes validation findings from the active browser session. Persisted queries remain unchanged.")
    cancel, confirm = st.columns(2)
    with cancel:
        if st.button("Cancel", key="cancel_clear_findings"):
            st.rerun()
    with confirm:
        if st.button("Clear validation findings", type="primary", key="confirm_clear_findings"):
            clear_validation_findings()
            st.rerun()


@st.dialog("Clear all queries")
def confirm_clear_queries() -> None:
    """Require explicit confirmation before deleting persistent queries and audit records."""
    st.warning("This permanently deletes all persisted queries and their audit history.")
    cancel, confirm = st.columns(2)
    with cancel:
        if st.button("Cancel", key="cancel_clear_queries"):
            st.rerun()
    with confirm:
        if st.button("Clear queries", type="primary", key="confirm_clear_queries"):
            clear_queries()
            st.rerun()


@st.dialog("Full environment reset")
def confirm_full_environment_reset() -> None:
    """Require explicit confirmation before clearing session and persistent state."""
    st.error("This permanently deletes all queries and audit history and clears session uploads and findings.")
    cancel, confirm = st.columns(2)
    with cancel:
        if st.button("Cancel", key="cancel_full_reset"):
            st.rerun()
    with confirm:
        if st.button("Full environment reset", type="primary", key="confirm_full_reset"):
            clear_uploaded_datasets()
            clear_validation_findings()
            clear_queries()
            clear_lab_alerts()
            st.rerun()


initialise_database()

st.title("Query Management")
st.caption("Create, review, resolve, and audit clinical data queries generated from validation findings.")

findings = validation_finding_records()
sync_column, source_column = st.columns([1, 3])
with sync_column:
    sync_requested = st.button("Sync Validation Findings", type="primary", disabled=not findings)
with source_column:
    st.caption(
        f"{len(findings):,} current finding(s) available for synchronization."
        if findings
        else "Run the Validation Engine before synchronizing findings."
    )

if sync_requested:
    created, skipped = create_queries_from_findings(findings)
    st.success(f"{created} new queries created. {skipped} existing active queries reused.")

queries = fetch_queries()
if queries.empty:
    queries = pd.DataFrame(columns=QUERY_COLUMNS)

st.divider()
st.subheader("Query Overview")
render_metrics(queries)

st.subheader("Filters")
filter_columns = st.columns(4)
with filter_columns[0]:
    selected_statuses = st.multiselect("Status", STATUSES, default=list(STATUSES))
with filter_columns[1]:
    selected_severities = st.multiselect("Severity", SEVERITIES, default=list(SEVERITIES))
with filter_columns[2]:
    dataset_options = sorted(queries["Dataset"].dropna().unique().tolist())
    selected_datasets = st.multiselect("Dataset", dataset_options, default=dataset_options)
with filter_columns[3]:
    subject_filter = st.text_input("Subject ID")

search_text = st.text_input("Search Query ID, Subject ID, Rule, or Description")
filtered_queries = filter_queries(
    queries,
    selected_statuses,
    selected_severities,
    selected_datasets,
    subject_filter,
    search_text,
)

st.divider()
st.subheader("Active Queries")
active_queries = filtered_queries[filtered_queries["Status"].isin(ACTIVE_QUERY_STATUSES)]
st.caption(f"Showing {len(active_queries.index):,} active query record(s).")
st.dataframe(active_queries, hide_index=True, use_container_width=True)

st.divider()
st.subheader("Query Update")
if filtered_queries.empty:
    st.info("No clinical data queries match the selected review filters. Adjust the filters or synchronize current validation findings.")
else:
    selected_query_id = st.selectbox("Query ID", filtered_queries["Query ID"].tolist())
    selected_query = queries.loc[queries["Query ID"] == selected_query_id].iloc[0]
    selected_status = str(selected_query["Status"])
    selected_status = selected_status if selected_status in STATUSES else STATUSES[0]

    with st.form(f"update_query_{selected_query_id}"):
        status = st.selectbox("Status", STATUSES, index=STATUSES.index(selected_status))
        response = st.text_area("Response", value=selected_query["Response"], height=100)
        resolution_note = st.text_area("Resolution Note", value=selected_query["Resolution Note"], height=100)
        save_requested = st.form_submit_button("Save query update")

    if save_requested:
        try:
            update_query(selected_query_id, status, response, resolution_note)
        except ValueError as error:
            st.error(str(error))
        else:
            st.success(f"{selected_query_id} updated.")
            st.rerun()

st.divider()
st.subheader("Audit History")
audit_history = fetch_audit_history()
if audit_history.empty:
    st.info("No query lifecycle events are available for this study. Audit history will populate when queries are synchronized or updated.")
else:
    st.dataframe(audit_history, hide_index=True, use_container_width=True)

st.divider()
st.subheader("Export Options")
query_export = filtered_queries.reindex(columns=QUERY_EXPORT_COLUMNS)
st.download_button(
    "Export Queries CSV",
    data=query_export.to_csv(index=False).encode("utf-8"),
    file_name="trialops_queries.csv",
    mime="text/csv",
)
st.download_button(
    "Export Audit Trail CSV",
    data=audit_history.reindex(columns=AUDIT_COLUMNS).to_csv(index=False).encode("utf-8"),
    file_name="trialops_query_audit_trail.csv",
    mime="text/csv",
)

st.divider()
st.subheader("Admin Utilities")
st.caption("Administrative actions require confirmation. Query resets permanently remove SQLite records.")
admin_columns = st.columns(4)
with admin_columns[0]:
    if st.button("Clear Uploaded Datasets"):
        confirm_clear_uploaded_datasets()
with admin_columns[1]:
    if st.button("Clear Validation Findings"):
        confirm_clear_validation_findings()
with admin_columns[2]:
    if st.button("Clear Queries"):
        confirm_clear_queries()
with admin_columns[3]:
    if st.button("Full Environment Reset", type="primary"):
        confirm_full_environment_reset()
