"""Participant-level clinical review across uploaded SDTM domains."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from database import fetch_queries
from utils.lab_analytics import abnormal_results, classify_lab_results, laboratory_summary, laboratory_trend_data
from utils.safety_analytics import serious_event_distribution, severity_distribution, subject_safety_summary
from utils.treatment_efficacy_analytics import (
    derive_efficacy_records,
    efficacy_with_arms,
    exposure_records,
    latest_efficacy_records,
    treatment_assignments,
)

DATASET_STATE_KEY = "sdtm_datasets"
VALIDATION_RESULTS_KEY = "validation_results"
QUERY_STATUSES = ("Open", "Answered", "Closed")


def session_datasets() -> dict[str, pd.DataFrame]:
    """Return uploaded detected-domain dataframes from the active session."""
    datasets = st.session_state.get(DATASET_STATE_KEY, {})
    if not isinstance(datasets, Mapping):
        return {}
    return {
        str(domain).upper(): dataframe
        for domain, dataframe in datasets.items()
        if isinstance(dataframe, pd.DataFrame)
    }


def find_column(dataframe: pd.DataFrame, variable: str) -> str | None:
    """Find a variable by its normalized SDTM name."""
    normalized_variable = variable.upper()
    return next(
        (str(column) for column in dataframe.columns if str(column).strip().upper() == normalized_variable),
        None,
    )


def subject_ids(datasets: Mapping[str, pd.DataFrame]) -> list[str]:
    """Return sorted, non-blank USUBJID values across every uploaded domain."""
    identifiers: set[str] = set()
    for dataframe in datasets.values():
        subject_column = find_column(dataframe, "USUBJID")
        if subject_column is None:
            continue
        values = dataframe[subject_column].astype("string").dropna().str.strip()
        identifiers.update(value for value in values.tolist() if value)
    return sorted(identifiers)


def subject_records(dataframe: pd.DataFrame, subject_id: str) -> pd.DataFrame:
    """Return records matching a subject in one domain without changing source data."""
    subject_column = find_column(dataframe, "USUBJID")
    if subject_column is None:
        return dataframe.iloc[0:0].copy()
    values = dataframe[subject_column].astype("string").fillna("").str.strip()
    return dataframe.loc[values.eq(subject_id)].copy()


def records_by_domain(datasets: Mapping[str, pd.DataFrame], subject_id: str) -> dict[str, pd.DataFrame]:
    """Collect subject records from every uploaded domain that contains USUBJID."""
    return {
        domain: records
        for domain, dataframe in datasets.items()
        if not (records := subject_records(dataframe, subject_id)).empty
    }


def display_value(value: Any) -> str:
    """Format nullable source values for profile metrics."""
    return "Not reported" if value is None or pd.isna(value) or str(value).strip() == "" else str(value)


def profile_value(profile: pd.Series, column: str | None) -> str:
    """Distinguish an unavailable profile variable from an unreported value."""
    return "Not available" if column is None else display_value(profile[column])


def subject_profile(dm_records: pd.DataFrame, subject_id: str) -> None:
    """Render a subject profile using DM as the preferred demographic source."""
    st.subheader("Subject Profile")
    if dm_records.empty:
        st.info("No DM demographic record is available for this subject. The clinical profile is limited to the selected subject ID.")
        st.metric("Subject ID", subject_id)
        return

    profile = dm_records.iloc[0]
    subject_column = find_column(dm_records, "USUBJID")
    sex_column = find_column(dm_records, "SEX")
    age_column = find_column(dm_records, "AGE")
    race_column = find_column(dm_records, "RACE")
    country_column = find_column(dm_records, "COUNTRY")

    metrics = st.columns(5)
    metrics[0].metric("Subject ID", profile_value(profile, subject_column) if subject_column else subject_id)
    metrics[1].metric("Sex", profile_value(profile, sex_column))
    metrics[2].metric("Age", profile_value(profile, age_column))
    metrics[3].metric("Race", profile_value(profile, race_column))
    metrics[4].metric("Country", profile_value(profile, country_column))

    with st.expander("DM source record"):
        st.dataframe(dm_records, hide_index=True, use_container_width=True)


def render_subject_treatment_summary(
    records: Mapping[str, pd.DataFrame],
    subject_id: str,
    dm_dataset: pd.DataFrame | None,
    ex_dataset: pd.DataFrame | None,
    efficacy_dataset: pd.DataFrame | None,
) -> None:
    """Render compact treatment, exposure, efficacy, and safety context for one subject."""
    st.subheader("Treatment and Outcome Summary")
    assignments = treatment_assignments(dm_dataset, ex_dataset)
    treatment_arm = "Not available"
    subject_assignment = assignments.loc[assignments["USUBJID"].eq(subject_id)]
    if not subject_assignment.empty:
        treatment_arm = str(subject_assignment.iloc[0]["Treatment Arm"])

    subject_exposure = records.get("EX", pd.DataFrame())
    exposure = exposure_records(subject_exposure) if not subject_exposure.empty else pd.DataFrame()
    treatment = treatment_arm
    dose = "Not reported"
    exposure_dates = "Not reported"
    if not exposure.empty:
        recorded_treatments = [value for value in exposure["Treatment"].tolist() if value]
        if recorded_treatments:
            treatment = recorded_treatments[0]
        recorded_doses = [
            " ".join(part for part in (str(row["Dose"]), str(row["Dose Unit"])) if part).strip()
            for _, row in exposure.iterrows()
            if row["Dose"] or row["Dose Unit"]
        ]
        if recorded_doses:
            dose = recorded_doses[0]
        start_dates = exposure["Exposure Start"].dropna()
        end_dates = exposure["Exposure End"].dropna()
        if not start_dates.empty or not end_dates.empty:
            start = start_dates.min().date().isoformat() if not start_dates.empty else "Not reported"
            end = end_dates.max().date().isoformat() if not end_dates.empty else "Not reported"
            exposure_dates = f"{start} to {end}"

    first_row = st.columns(3)
    first_row[0].metric("Treatment", treatment)
    first_row[1].metric("Dose", dose)
    first_row[2].metric("Exposure Dates", exposure_dates)

    efficacy_rows = pd.DataFrame()
    if efficacy_dataset is not None:
        derived = derive_efficacy_records(efficacy_dataset)
        subject_efficacy = derived.loc[subject_ids_from_frame(derived).eq(subject_id)].copy()
        efficacy_rows = efficacy_with_arms(latest_efficacy_records(subject_efficacy), assignments)

    ae_records = records.get("AE", pd.DataFrame())
    safety = subject_safety_summary(ae_records)
    lb_records = records.get("LB", pd.DataFrame())
    abnormal_lab_count = len(abnormal_results(classify_lab_results(lb_records)).index) if not lb_records.empty else 0

    if efficacy_rows.empty:
        st.info("No endpoint results are available for this subject in the uploaded efficacy dataset.")
        second_row = st.columns(3)
        second_row[0].metric("Adverse Events", safety["Adverse Events"])
        second_row[1].metric(
            "Serious AEs",
            "Not available" if safety["Serious Adverse Events"] is None else safety["Serious Adverse Events"],
        )
        second_row[2].metric("Abnormal Lab Results", abnormal_lab_count)
        return

    endpoint = efficacy_rows.iloc[0]
    second_row = st.columns(7)
    second_row[0].metric("Baseline", display_value(endpoint["Baseline"]))
    second_row[1].metric("Latest Value", display_value(endpoint["Post-treatment Value"]))
    second_row[2].metric("Change", display_value(endpoint["Change"]))
    percentage_change = endpoint["Percentage Change"]
    second_row[3].metric("Percentage Change", "Not reported" if pd.isna(percentage_change) else f"{percentage_change:.1f}%")
    second_row[4].metric("Adverse Events", safety["Adverse Events"])
    second_row[5].metric(
        "Serious AEs",
        "Not available" if safety["Serious Adverse Events"] is None else safety["Serious Adverse Events"],
    )
    second_row[6].metric("Abnormal Lab Results", abnormal_lab_count)

    if len(efficacy_rows.index) > 1:
        st.caption("Additional subject-level efficacy endpoints")
        st.dataframe(
            efficacy_rows.reindex(
                columns=["Endpoint", "Visit", "Baseline", "Post-treatment Value", "Change", "Percentage Change"]
            ),
            hide_index=True,
            use_container_width=True,
        )


def subject_ids_from_frame(dataframe: pd.DataFrame) -> pd.Series:
    """Return cleaned subject IDs for a dataframe without relying on filename metadata."""
    subject_column = find_column(dataframe, "USUBJID")
    if subject_column is None:
        return pd.Series("", index=dataframe.index, dtype="string")
    return dataframe[subject_column].astype("string").fillna("").str.strip()


def source_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Present records with their original CSV row number for review traceability."""
    review_frame = dataframe.copy()
    review_frame.insert(0, "Source Row", [int(index) + 2 for index in review_frame.index])
    return review_frame


def render_domain_records(records: Mapping[str, pd.DataFrame]) -> None:
    """Render a domain inventory and complete subject-level record extracts."""
    st.subheader("Available Domain Records")
    if not records:
        st.info("No uploaded SDTM records are available for this subject across the current clinical review datasets.")
        return

    inventory = pd.DataFrame(
        [{"Dataset": domain, "Record Count": len(dataframe.index)} for domain, dataframe in sorted(records.items())]
    )
    st.dataframe(inventory, hide_index=True, use_container_width=True)

    for domain, dataframe in sorted(records.items()):
        with st.expander(f"{domain} records ({len(dataframe.index)})", expanded=domain == "DM"):
            st.dataframe(source_rows(dataframe), hide_index=True, use_container_width=True)


def render_subject_safety_review(
    records: Mapping[str, pd.DataFrame], subject_id: str, ae_dataset: pd.DataFrame | None
) -> None:
    """Render the selected subject's AE burden, risk flag, and event records."""
    st.subheader("Subject Safety Review")
    if ae_dataset is None:
        st.info("No detected AE dataset is available, so subject-level drug safety review cannot be displayed.")
        return

    ae_records = records.get("AE", ae_dataset.iloc[0:0].copy())

    summary = subject_safety_summary(ae_records)
    metrics = st.columns(3)
    metrics[0].metric("Adverse Events", summary["Adverse Events"])
    metrics[1].metric(
        "Serious Adverse Events",
        "Not available" if summary["Serious Adverse Events"] is None else summary["Serious Adverse Events"],
    )
    metrics[2].metric("Safety Risk", f"{summary['Risk Level']} Risk")

    st.info(
        f"Subject {subject_id} is classified as {summary['Risk Level']} Risk based on the configured adverse-event review rules."
    )

    if ae_records.empty:
        st.info("No adverse events were reported for this subject in the uploaded AE dataset.")
        return

    severity_counts = severity_distribution(ae_records)
    st.caption("AE severity breakdown")
    if severity_counts is None:
        st.info("AESEV is not available in this subject's AE records, so severity breakdown cannot be displayed.")
    else:
        st.dataframe(severity_counts, hide_index=True, use_container_width=True)

    serious_counts = serious_event_distribution(ae_records)
    if serious_counts is not None:
        st.caption("AE seriousness breakdown")
        st.dataframe(serious_counts, hide_index=True, use_container_width=True)

    st.caption("Adverse event records")
    st.dataframe(source_rows(ae_records), hide_index=True, use_container_width=True)


def render_subject_laboratory_review(
    records: Mapping[str, pd.DataFrame], subject_id: str, lb_dataset: pd.DataFrame | None
) -> None:
    """Render selected-subject laboratory classifications, alerts, and trends."""
    st.subheader("Subject Laboratory Review")
    if lb_dataset is None:
        st.info("No detected LB dataset is available, so subject-level laboratory review cannot be displayed.")
        return

    lb_records = records.get("LB", lb_dataset.iloc[0:0].copy())
    classified_results = classify_lab_results(lb_records)
    summary = laboratory_summary(classified_results)
    metrics = st.columns(4)
    for container, (label, value) in zip(metrics, summary.items()):
        container.metric(label, value)

    if classified_results.empty:
        st.info("No laboratory records were reported for this subject in the uploaded LB dataset.")
        return

    alerts = abnormal_results(classified_results)
    if alerts.empty:
        st.success("No subject-level laboratory results are outside the supplied reference ranges.")
    else:
        st.caption("Subject laboratory alerts")
        st.dataframe(
            alerts.reindex(
                columns=[
                    "Lab Test",
                    "Numeric Result",
                    "Lab Unit",
                    "Reference Low",
                    "Reference High",
                    "Result Status",
                    "Collection Date",
                    "Source Row",
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )

    trend_tests = classified_results.loc[
        classified_results["Collection Date"].notna()
        & classified_results["Numeric Result"].notna()
        & classified_results["Lab Test"].ne(""),
        "Lab Test",
    ].drop_duplicates().sort_values().tolist()
    if trend_tests:
        selected_test = st.selectbox(
            "Laboratory Trend Test",
            trend_tests,
            key=f"patient_lab_trend_{subject_id}",
        )
        trend = laboratory_trend_data(classified_results, selected_test)
        trend_figure = px.line(
            trend,
            x="Collection Date",
            y="Numeric Result",
            color="Result Status",
            markers=True,
            hover_data={"Lab Unit": True, "Reference Low": True, "Reference High": True},
            title=f"{selected_test} Results Over Time",
            template="plotly_dark",
        )
        trend_figure.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=12, r=12, t=40, b=12),
            legend_title_text="",
            xaxis_title=None,
            yaxis_title="Result",
        )
        st.plotly_chart(trend_figure, use_container_width=True)
    else:
        st.info("No date-stamped numeric laboratory results are available for a subject-level trend chart.")

    st.caption("Subject laboratory records")
    st.dataframe(classified_results, hide_index=True, use_container_width=True)


def current_findings(subject_id: str) -> pd.DataFrame:
    """Return current validation findings associated with the selected subject."""
    findings = st.session_state.get(VALIDATION_RESULTS_KEY)
    if not isinstance(findings, pd.DataFrame) or findings.empty or "USUBJID" not in findings.columns:
        return pd.DataFrame()

    finding_subjects = findings["USUBJID"].astype("string").fillna("").str.strip()
    return findings.loc[finding_subjects.eq(subject_id)].copy()


def related_queries(subject_id: str) -> pd.DataFrame:
    """Return persisted query records associated with the selected subject."""
    queries = fetch_queries()
    if queries.empty or "Subject ID" not in queries.columns:
        return queries.iloc[0:0].copy()

    query_subjects = queries["Subject ID"].astype("string").fillna("").str.strip()
    return queries.loc[query_subjects.eq(subject_id)].copy()


def render_query_summary(queries: pd.DataFrame) -> None:
    """Display status counts for the selected subject's persistent queries."""
    st.subheader("Query Status Summary")
    metrics = st.columns(4)
    metrics[0].metric("Total Queries", len(queries.index))
    for container, status in zip(metrics[1:], QUERY_STATUSES):
        count = int((queries["Status"].astype("string") == status).sum()) if "Status" in queries.columns else 0
        container.metric(status, count)


def date_columns(dataframe: pd.DataFrame) -> list[str]:
    """Identify SDTM-like date variables usable for a subject event timeline."""
    return [
        str(column)
        for column in dataframe.columns
        if str(column).strip().upper().endswith("DTC") or "DATE" in str(column).strip().upper()
    ]


def event_description(domain: str, record: pd.Series, source_row: int) -> str:
    """Provide a concise timeline label using common SDTM clinical variables."""
    preferred_variables = {
        "AE": ("AETERM", "AEDECOD"),
        "LB": ("LBTEST", "LBTESTCD"),
        "VS": ("VSTEST", "VSTESTCD"),
    }
    for variable in preferred_variables.get(domain, ()):
        column = find_column(pd.DataFrame([record]), variable)
        if column is not None and display_value(record[column]) != "Not reported":
            return f"{domain}: {record[column]}"
    return f"{domain} record at source row {source_row}"


def subject_timeline(records: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Build date-stamped clinical events from records available for the subject."""
    events: list[dict[str, Any]] = []
    for domain, dataframe in records.items():
        for column in date_columns(dataframe):
            parsed_dates = pd.to_datetime(dataframe[column].astype("string"), errors="coerce")
            for position, event_date in enumerate(parsed_dates):
                if pd.isna(event_date):
                    continue
                source_index = dataframe.index[position]
                record = dataframe.iloc[position]
                events.append(
                    {
                        "Event Date": event_date,
                        "Dataset": domain,
                        "Date Field": column,
                        "Source Row": int(source_index) + 2,
                        "Event": event_description(domain, record, int(source_index) + 2),
                    }
                )

    if not events:
        return pd.DataFrame(columns=["Event Date", "Dataset", "Date Field", "Source Row", "Event"])
    return pd.DataFrame(events).sort_values("Event Date").reset_index(drop=True)


def render_timeline(records: Mapping[str, pd.DataFrame]) -> None:
    """Render a dark Plotly timeline when date-bearing subject records exist."""
    st.subheader("Subject Timeline")
    timeline = subject_timeline(records)
    if timeline.empty:
        st.info("No date fields were detected in the uploaded datasets. Timeline visualization requires at least one recognized date variable.")
        return

    figure = px.scatter(
        timeline,
        x="Event Date",
        y="Dataset",
        color="Dataset",
        hover_name="Event",
        hover_data={"Date Field": True, "Source Row": True, "Event Date": "|%Y-%m-%d"},
        title="Clinical Event Timeline",
    )
    figure.update_traces(marker={"size": 11})
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=12, r=12, t=40, b=12),
        xaxis_title=None,
        yaxis_title=None,
        legend_title_text="",
    )
    st.plotly_chart(figure, use_container_width=True)
    with st.expander("Timeline event details"):
        st.dataframe(timeline, hide_index=True, use_container_width=True)


def render_patient_360() -> None:
    """Render the subject review workspace from detected uploaded datasets."""
    datasets = session_datasets()
    st.title("Patient 360")
    st.caption("Subject-level clinical review across uploaded SDTM datasets and persisted data queries.")

    if not datasets:
        st.info("No detected SDTM datasets are available for subject review. Upload clinical source data to begin Patient 360 review.")
        return

    available_subjects = subject_ids(datasets)
    if not available_subjects:
        st.warning("The uploaded datasets do not contain a usable USUBJID value. Subject-level clinical review cannot be started.")
        return

    search_text = st.text_input("Search Subject ID", placeholder="Type part of a USUBJID")
    matching_subjects = [
        subject_id for subject_id in available_subjects if search_text.strip().upper() in subject_id.upper()
    ]
    if not matching_subjects:
        st.warning("No subject IDs match the current search criteria. Refine the search or review the uploaded SDTM data.")
        return

    selected_subject = st.selectbox(
        "Select Subject ID",
        matching_subjects,
        help="The list is compiled from USUBJID values across all uploaded detected domains.",
    )
    records = records_by_domain(datasets, selected_subject)
    dm_records = records.get("DM", pd.DataFrame())

    subject_profile(dm_records, selected_subject)
    render_subject_treatment_summary(
        records,
        selected_subject,
        datasets.get("DM"),
        datasets.get("EX"),
        datasets.get("EFF"),
    )
    render_domain_records(records)
    render_subject_safety_review(records, selected_subject, datasets.get("AE"))
    render_subject_laboratory_review(records, selected_subject, datasets.get("LB"))

    findings = current_findings(selected_subject)
    queries = related_queries(selected_subject)

    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader("Validation Findings")
        if findings.empty:
            st.success("No current clinical validation findings are associated with this subject.")
        else:
            st.dataframe(findings, hide_index=True, use_container_width=True)
    with right_column:
        render_query_summary(queries)
        st.subheader("Related Queries")
        if queries.empty:
            st.info("No persisted clinical data queries are associated with this subject.")
        else:
            st.dataframe(queries, hide_index=True, use_container_width=True)

    render_timeline(records)


render_patient_360()
