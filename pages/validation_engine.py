"""Clinical data validation rules for session-scoped CDISC SDTM datasets."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Iterable

import pandas as pd
import streamlit as st

from database import auto_close_resolved_queries

ISSUE_COLUMNS = [
    "Dataset",
    "Source row",
    "USUBJID",
    "Severity",
    "Rule",
    "Field",
    "Observed value",
    "Message",
]
SEVERITIES = ("Critical", "Major", "Minor")
VALIDATION_RESULTS_KEY = "validation_results"
VALIDATION_SOURCE_KEY = "validation_source_fingerprints"
AUTO_CLOSED_QUERIES_KEY = "auto_closed_queries"
ACTIVITY_EVENTS_KEY = "study_activity_events"
FINDING_EXPORT_COLUMNS = [
    "Dataset",
    "USUBJID",
    "Rule",
    "Field",
    "Severity",
    "Message",
    "Source row",
]


def column_lookup(dataframe: pd.DataFrame) -> dict[str, str]:
    """Map normalised SDTM variable names to their original column names."""
    lookup: dict[str, str] = {}
    for column_name in dataframe.columns:
        lookup.setdefault(str(column_name).strip().upper(), str(column_name))
    return lookup


def find_column(dataframe: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Return the first available column matching one of the SDTM candidates."""
    lookup = column_lookup(dataframe)
    for candidate in candidates:
        resolved_name = lookup.get(candidate.upper())
        if resolved_name is not None:
            return resolved_name
    return None


def missing_value_mask(values: pd.Series) -> pd.Series:
    """Return a mask for null, empty, and whitespace-only values."""
    cleaned_values = values.astype("string").str.strip()
    return (values.isna() | cleaned_values.isna() | cleaned_values.eq("")).fillna(True).astype(bool)


def dataframe_fingerprint(dataframe: pd.DataFrame) -> str:
    """Create a stable signature for a dataset's columns, values, and row order."""
    fingerprint = sha256()
    fingerprint.update("\x1f".join(map(str, dataframe.columns)).encode("utf-8"))
    fingerprint.update(pd.util.hash_pandas_object(dataframe, index=True).to_numpy().tobytes())
    return fingerprint.hexdigest()


def validation_source_signature(datasets: dict[str, pd.DataFrame]) -> dict[str, str]:
    """Return the current in-memory signature for every uploaded dataset."""
    return {dataset_name.upper(): dataframe_fingerprint(dataframe) for dataset_name, dataframe in datasets.items()}


def record_validation_activity(results: pd.DataFrame, dataset_count: int, auto_closed_count: int) -> None:
    """Add a concise validation-run event for the session dashboard."""
    detail = f"Validated {dataset_count} dataset(s) and identified {len(results.index)} finding(s)."
    if auto_closed_count:
        detail = f"{detail} Automatically closed {auto_closed_count} resolved query record(s)."

    activity_events = st.session_state.setdefault(ACTIVITY_EVENTS_KEY, [])
    activity_events.append(
        {
            "Event": "Validation Run Completed",
            "Detail": detail,
            "Timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }
    )
    st.session_state[ACTIVITY_EVENTS_KEY] = activity_events[-100:]


def display_value(value: Any) -> str:
    """Format a source value for a human-readable validation finding."""
    return "" if pd.isna(value) else str(value)


def subject_identifiers(dataframe: pd.DataFrame) -> pd.Series:
    """Return USUBJID values, or blank values when the variable is unavailable."""
    subject_column = find_column(dataframe, ("USUBJID",))
    if subject_column is None:
        return pd.Series("", index=dataframe.index, dtype="string")
    return dataframe[subject_column].astype("string").fillna("").str.strip()


def dataset_issue(
    dataset_name: str,
    severity: str,
    rule: str,
    field: str,
    message: str,
) -> dict[str, object]:
    """Create a dataset-level finding when a required variable is absent."""
    return {
        "Dataset": dataset_name,
        "Source row": None,
        "USUBJID": "",
        "Severity": severity,
        "Rule": rule,
        "Field": field,
        "Observed value": "",
        "Message": message,
    }


def row_issues(
    dataset_name: str,
    dataframe: pd.DataFrame,
    failed_mask: pd.Series,
    severity: str,
    rule: str,
    field: str,
    message: str,
) -> list[dict[str, object]]:
    """Create row-level findings for the records selected by a boolean mask."""
    subjects = subject_identifiers(dataframe)
    issues: list[dict[str, object]] = []

    for position, failed in enumerate(failed_mask.tolist()):
        if not failed:
            continue

        observed_value = display_value(dataframe.iloc[position][field]) if field in dataframe.columns else ""
        issues.append(
            {
                "Dataset": dataset_name,
                "Source row": position + 2,
                "USUBJID": str(subjects.iloc[position]),
                "Severity": severity,
                "Rule": rule,
                "Field": field,
                "Observed value": observed_value,
                "Message": message,
            }
        )

    return issues


def validate_required_dm_fields(dataframe: pd.DataFrame) -> list[dict[str, object]]:
    """Validate required subject-level DM variables."""
    issues: list[dict[str, object]] = []
    required_fields = {
        "USUBJID": ("Critical", "DM - Missing USUBJID"),
        "SEX": ("Major", "DM - Missing SEX"),
        "AGE": ("Major", "DM - Missing AGE"),
    }

    for variable, (severity, rule) in required_fields.items():
        column_name = find_column(dataframe, (variable,))
        if column_name is None:
            issues.append(
                dataset_issue(
                    "DM",
                    severity,
                    rule,
                    variable,
                    f"Required DM variable {variable} is not present in the dataset.",
                )
            )
            continue

        issues.extend(
            row_issues(
                "DM",
                dataframe,
                missing_value_mask(dataframe[column_name]),
                severity,
                rule,
                column_name,
                f"{variable} is missing.",
            )
        )

    return issues


def validate_required_ae_fields(dataframe: pd.DataFrame) -> list[dict[str, object]]:
    """Validate required adverse-event variables in the AE domain."""
    issues: list[dict[str, object]] = []
    required_fields = {
        "USUBJID": ("Critical", "AE - Missing USUBJID"),
        "AETERM": ("Major", "AE - Missing AETERM"),
    }

    for variable, (severity, rule) in required_fields.items():
        column_name = find_column(dataframe, (variable,))
        if column_name is None:
            issues.append(
                dataset_issue(
                    "AE",
                    severity,
                    rule,
                    variable,
                    f"Required AE variable {variable} is not present in the dataset.",
                )
            )
            continue

        issues.extend(
            row_issues(
                "AE",
                dataframe,
                missing_value_mask(dataframe[column_name]),
                severity,
                rule,
                column_name,
                f"{variable} is missing.",
            )
        )

    return issues


def validate_laboratory_results(dataframe: pd.DataFrame) -> list[dict[str, object]]:
    """Validate missing results and high ALT/AST values in the LB domain."""
    issues: list[dict[str, object]] = []
    value_columns = [
        column_name
        for column_name in (
            find_column(dataframe, ("LBSTRESN",)),
            find_column(dataframe, ("LBSTRESC",)),
            find_column(dataframe, ("LBORRES",)),
        )
        if column_name is not None
    ]

    if not value_columns:
        issues.append(
            dataset_issue(
                "LB",
                "Major",
                "LB - Missing laboratory values",
                "LBSTRESN / LBSTRESC / LBORRES",
                "No laboratory result variable was found in the dataset.",
            )
        )
    else:
        result_is_missing = pd.Series(True, index=dataframe.index)
        for column_name in value_columns:
            result_is_missing &= missing_value_mask(dataframe[column_name])

        issues.extend(
            row_issues(
                "LB",
                dataframe,
                result_is_missing,
                "Major",
                "LB - Missing laboratory values",
                value_columns[0],
                "No laboratory result value is present.",
            )
        )

    test_column = find_column(dataframe, ("LBTESTCD",))
    numeric_result_column = find_column(dataframe, ("LBSTRESN", "LBORRES"))
    if test_column is None or numeric_result_column is None:
        issues.append(
            dataset_issue(
                "LB",
                "Major",
                "LB - ALT/AST threshold check unavailable",
                "LBTESTCD / LBSTRESN",
                "LBTESTCD and a numeric laboratory result field are required to evaluate ALT and AST thresholds.",
            )
        )
        return issues

    test_codes = dataframe[test_column].astype("string").fillna("").str.strip().str.upper()
    numeric_results = pd.to_numeric(dataframe[numeric_result_column], errors="coerce")
    threshold_rules = (
        ("ALT", 120, "LB - ALT > 120"),
        ("AST", 100, "LB - AST > 100"),
    )
    for test_code, threshold, rule in threshold_rules:
        issues.extend(
            row_issues(
                "LB",
                dataframe,
                test_codes.eq(test_code) & numeric_results.gt(threshold),
                "Major",
                rule,
                numeric_result_column,
                f"{test_code} result exceeds {threshold}.",
            )
        )

    return issues


def validate_vital_signs(dataframe: pd.DataFrame) -> list[dict[str, object]]:
    """Validate implausibly low and high heart-rate values in the VS domain."""
    issues: list[dict[str, object]] = []
    test_column = find_column(dataframe, ("VSTESTCD",))
    result_column = find_column(dataframe, ("VSSTRESN", "VSORRES"))
    if test_column is None or result_column is None:
        issues.append(
            dataset_issue(
                "VS",
                "Major",
                "VS - Heart rate check unavailable",
                "VSTESTCD / VSSTRESN",
                "VSTESTCD and a numeric vital-sign result field are required to evaluate heart-rate thresholds.",
            )
        )
        return issues

    test_codes = dataframe[test_column].astype("string").fillna("").str.strip().str.upper()
    heart_rate_tests = test_codes.isin({"HR", "PULSE", "HEARTRATE", "HEART RATE"})
    numeric_results = pd.to_numeric(dataframe[result_column], errors="coerce")

    issues.extend(
        row_issues(
            "VS",
            dataframe,
            heart_rate_tests & numeric_results.lt(30),
            "Major",
            "VS - Heart Rate < 30",
            result_column,
            "Heart Rate is below 30.",
        )
    )
    issues.extend(
        row_issues(
            "VS",
            dataframe,
            heart_rate_tests & numeric_results.gt(180),
            "Major",
            "VS - Heart Rate > 180",
            result_column,
            "Heart Rate exceeds 180.",
        )
    )

    return issues


def is_date_column(column_name: str) -> bool:
    """Identify common SDTM date/date-time variables without altering source data."""
    normalised_name = column_name.strip().upper()
    return normalised_name.endswith("DTC") or normalised_name.endswith("DATE") or normalised_name in {
        "VISITDT",
        "VISIT_DATE",
    }


def validate_dates(dataset_name: str, dataframe: pd.DataFrame) -> list[dict[str, object]]:
    """Flag invalid and future values in available SDTM date/date-time variables."""
    issues: list[dict[str, object]] = []
    today_utc = pd.Timestamp.now(tz="UTC").normalize()

    for column_name in (str(column) for column in dataframe.columns if is_date_column(str(column))):
        source_values = dataframe[column_name]
        values_present = ~missing_value_mask(source_values)
        parsed_dates = pd.to_datetime(source_values, errors="coerce", utc=True)
        invalid_dates = values_present & parsed_dates.isna()
        visit_rule = "Date - Invalid visit date" if "VISIT" in column_name.upper() else "Date - Invalid date"

        issues.extend(
            row_issues(
                dataset_name,
                dataframe,
                invalid_dates,
                "Major",
                visit_rule,
                column_name,
                "Date value cannot be interpreted as a valid date.",
            )
        )
        issues.extend(
            row_issues(
                dataset_name,
                dataframe,
                values_present & parsed_dates.notna() & parsed_dates.dt.normalize().gt(today_utc),
                "Major",
                "Date - Future date",
                column_name,
                "Date value is later than the current date.",
            )
        )

    return issues


def validate_sdtm_datasets(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Run all configured validations against the uploaded SDTM datasets."""
    issues: list[dict[str, object]] = []

    for dataset_name, dataframe in datasets.items():
        domain = dataset_name.upper()
        if domain == "DM":
            issues.extend(validate_required_dm_fields(dataframe))
        elif domain == "AE":
            issues.extend(validate_required_ae_fields(dataframe))
        elif domain == "LB":
            issues.extend(validate_laboratory_results(dataframe))
        elif domain == "VS":
            issues.extend(validate_vital_signs(dataframe))

        issues.extend(validate_dates(domain, dataframe))

    results = pd.DataFrame(issues, columns=ISSUE_COLUMNS)
    if results.empty:
        return results

    severity_order = pd.CategoricalDtype(categories=SEVERITIES, ordered=True)
    results["Severity"] = results["Severity"].astype(severity_order)
    return results.sort_values(["Severity", "Dataset", "Source row", "Rule"], kind="stable").reset_index(drop=True)


def validation_sources_changed(datasets: dict[str, pd.DataFrame]) -> bool:
    """Return whether the in-memory datasets changed since the latest validation."""
    return st.session_state.get(VALIDATION_SOURCE_KEY) != validation_source_signature(datasets)


def run_validation(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Execute validation and retain the latest result in the active session."""
    results = validate_sdtm_datasets(datasets)
    auto_closed_count = auto_close_resolved_queries(
        results.to_dict(orient="records"),
        datasets.keys(),
    )
    st.session_state[VALIDATION_RESULTS_KEY] = results
    st.session_state[VALIDATION_SOURCE_KEY] = validation_source_signature(datasets)
    st.session_state[AUTO_CLOSED_QUERIES_KEY] = auto_closed_count
    record_validation_activity(results, len(datasets), auto_closed_count)
    return results


st.title("Validation Engine")
st.caption("Review clinical data-quality findings for the SDTM datasets uploaded in this session.")

uploaded_datasets: dict[str, pd.DataFrame] = st.session_state.get("sdtm_datasets", {})
if not uploaded_datasets:
    st.info("No detected SDTM datasets are available for clinical validation. Upload DM, AE, LB, or VS data to begin review.")
else:
    run_requested = st.button("Run validation", type="primary")
    if run_requested or validation_sources_changed(uploaded_datasets) or VALIDATION_RESULTS_KEY not in st.session_state:
        validation_results = run_validation(uploaded_datasets)
    else:
        validation_results = st.session_state[VALIDATION_RESULTS_KEY]

    auto_closed_count = int(st.session_state.get(AUTO_CLOSED_QUERIES_KEY, 0))
    if auto_closed_count:
        st.info(f"{auto_closed_count} query record(s) were automatically closed because the issue is no longer detected.")

    st.subheader("Validation summary")
    dataset_metric, finding_metric = st.columns(2)
    dataset_metric.metric("Datasets validated", len(uploaded_datasets))
    finding_metric.metric("Total findings", len(validation_results.index))

    severity_metrics = st.columns(len(SEVERITIES))
    for container, severity in zip(severity_metrics, SEVERITIES):
        issue_count = int((validation_results["Severity"] == severity).sum()) if not validation_results.empty else 0
        container.metric(severity, issue_count)

    st.subheader("Validation results")
    if validation_results.empty:
        st.success("Validation completed with no findings under the configured clinical data-quality rules.")
    else:
        st.dataframe(validation_results, hide_index=True, use_container_width=True)

    findings_export = validation_results.reindex(columns=FINDING_EXPORT_COLUMNS).rename(
        columns={"USUBJID": "Subject ID", "Source row": "Source Row"}
    )
    st.download_button(
        "Export Findings",
        data=findings_export.to_csv(index=False).encode("utf-8"),
        file_name="trialops_validation_findings.csv",
        mime="text/csv",
    )

    export_content = validation_results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Export validation results as CSV",
        data=export_content,
        file_name="trialops_validation_results.csv",
        mime="text/csv",
    )
