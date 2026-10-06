"""Explainable clinical operations anomaly detection for uploaded SDTM domains."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from database import fetch_queries
from utils.lab_analytics import abnormal_results, classify_lab_results
from utils.safety_analytics import serious_masks
from utils.treatment_efficacy_analytics import ae_safety_by_arm, find_column, subject_identifiers, treatment_assignments


RISK_SCORES = {"High": 90, "Medium": 60, "Low": 30}
CRITICAL_FIELDS = {
    "DM": (("USUBJID",), ("SEX",), ("AGE",)),
    "AE": (("USUBJID",), ("AETERM",)),
    "LB": (("USUBJID",), ("LBSTRESN", "LBORRES")),
    "VS": (("USUBJID",), ("VSSTRESN", "VSORRES")),
    "EX": (("USUBJID",),),
    "EFF": (("USUBJID",), ("AVAL", "CHG")),
}


def cleaned_text(values: pd.Series) -> pd.Series:
    """Return trimmed nullable values without altering uploaded source datasets."""
    return values.astype("string").fillna("").str.strip()


def source_row(index: Any) -> int:
    """Map a dataframe index back to the source CSV row when possible."""
    try:
        return int(index) + 2
    except (TypeError, ValueError):
        return 0


def site_column(dataframe: pd.DataFrame) -> str | None:
    """Find a common clinical site identifier without relying on filenames."""
    for variable in ("SITEID", "SITE", "CENTER", "CENTRE", "SITE_NUMBER"):
        column = find_column(dataframe, variable)
        if column is not None:
            return column
    return None


def site_values(dataframe: pd.DataFrame) -> pd.Series:
    """Return normalized site values or empty values when site data is absent."""
    column = site_column(dataframe)
    return cleaned_text(dataframe[column]) if column is not None else pd.Series("", index=dataframe.index, dtype="string")


def subject_site_map(datasets: Mapping[str, pd.DataFrame]) -> dict[str, str]:
    """Map subjects to their first populated site, preferring DM records."""
    mapping: dict[str, str] = {}
    ordered_domains = ["DM"] + sorted(domain for domain in datasets if domain != "DM")
    for domain in ordered_domains:
        dataframe = datasets.get(domain)
        if not isinstance(dataframe, pd.DataFrame):
            continue
        subjects = subject_identifiers(dataframe)
        sites = site_values(dataframe)
        for subject, site in zip(subjects, sites):
            if subject and site and str(subject) not in mapping:
                mapping[str(subject)] = str(site)
    return mapping


def _risk(severity: str) -> tuple[str, int]:
    """Assign a transparent risk band and score from a fixed severity mapping."""
    risk_level = {"Critical": "High", "Major": "Medium", "Minor": "Low"}.get(severity, "Low")
    return risk_level, RISK_SCORES[risk_level]


def _finding(
    domain: str,
    index: Any,
    subject_id: str,
    site_id: str,
    issue_type: str,
    severity: str,
    message: str,
    recommended_review: str,
    anomaly_date: Any = None,
) -> dict[str, Any]:
    """Create a standardized, explainable anomaly finding."""
    risk_level, risk_score = _risk(severity)
    return {
        "Subject ID": subject_id or "Not reported",
        "Site ID": site_id or "Not available",
        "Dataset": domain,
        "Source Row": source_row(index),
        "Issue Type": issue_type,
        "Severity": severity,
        "Risk Level": risk_level,
        "Risk Score": risk_score,
        "Anomaly Date": anomaly_date,
        "Message": message,
        "Recommended Review": recommended_review,
    }


def _missing_mask(dataframe: pd.DataFrame, variables: tuple[str, ...]) -> pd.Series:
    """Return rows where every listed alternative is unavailable or blank."""
    values: list[pd.Series] = []
    for variable in variables:
        column = find_column(dataframe, variable)
        values.append(
            cleaned_text(dataframe[column]).eq("")
            if column is not None
            else pd.Series(True, index=dataframe.index)
        )
    return pd.concat(values, axis=1).all(axis=1)


def _record_date(dataframe: pd.DataFrame) -> pd.Series:
    """Return the first parseable SDTM-like date for each record, if one exists."""
    date_columns = [
        str(column)
        for column in dataframe.columns
        if str(column).strip().upper().endswith("DTC") or str(column).strip().upper() == "ADT"
    ]
    dates = pd.Series(pd.NaT, index=dataframe.index, dtype="datetime64[ns]")
    for column in date_columns:
        parsed = pd.to_datetime(dataframe[column].astype("string"), errors="coerce")
        dates = dates.fillna(parsed)
    return dates


def data_quality_findings(datasets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Detect explainable missing, duplicate, date, and visit-sequence anomalies."""
    rows: list[dict[str, Any]] = []
    site_map = subject_site_map(datasets)
    for domain, dataframe in datasets.items():
        if dataframe.empty:
            continue
        subjects = subject_identifiers(dataframe)
        sites = site_values(dataframe)
        record_dates = _record_date(dataframe)
        for alternatives in CRITICAL_FIELDS.get(domain, ()): 
            missing = _missing_mask(dataframe, alternatives)
            field_label = " or ".join(alternatives)
            severity = "Critical" if alternatives == ("USUBJID",) else "Major"
            for index in dataframe.index[missing]:
                subject = str(subjects.loc[index])
                site = str(sites.loc[index]) or site_map.get(subject, "")
                rows.append(
                    _finding(
                        domain,
                        index,
                        subject,
                        site,
                        "Missing critical value",
                        severity,
                        f"{field_label} is missing or blank.",
                        "Verify the source record and complete the required value.",
                        record_dates.loc[index],
                    )
                )

        if domain == "AE":
            start_date = find_column(dataframe, "AESTDTC")
            if start_date is None:
                missing_dates = pd.Series(True, index=dataframe.index)
            else:
                missing_dates = cleaned_text(dataframe[start_date]).eq("")
            for index in dataframe.index[missing_dates]:
                subject = str(subjects.loc[index])
                site = str(sites.loc[index]) or site_map.get(subject, "")
                rows.append(
                    _finding(
                        domain,
                        index,
                        subject,
                        site,
                        "Missing adverse event date",
                        "Major",
                        "AESTDTC is missing or blank.",
                        "Review adverse-event onset documentation and complete the start date if available.",
                        record_dates.loc[index],
                    )
                )

        duplicate_rows = dataframe.duplicated(keep="first")
        for index in dataframe.index[duplicate_rows]:
            subject = str(subjects.loc[index])
            site = str(sites.loc[index]) or site_map.get(subject, "")
            rows.append(
                _finding(
                    domain,
                    index,
                    subject,
                    site,
                    "Duplicate record",
                    "Major",
                    "This record duplicates an earlier record in the same detected dataset.",
                    "Confirm whether the duplicate should be removed or retained with a documented distinction.",
                    record_dates.loc[index],
                )
            )

        for column in dataframe.columns:
            variable = str(column).strip().upper()
            if not (variable.endswith("DTC") or variable == "ADT"):
                continue
            populated = cleaned_text(dataframe[column]).ne("")
            parsed = pd.to_datetime(dataframe[column].astype("string"), errors="coerce")
            for index in dataframe.index[populated & parsed.isna()]:
                subject = str(subjects.loc[index])
                site = str(sites.loc[index]) or site_map.get(subject, "")
                rows.append(
                    _finding(
                        domain,
                        index,
                        subject,
                        site,
                        "Invalid date",
                        "Major",
                        f"{variable} contains a non-parseable date value.",
                        "Verify the source date format and correct the date value.",
                    )
                )

        visit_column = find_column(dataframe, "VISITNUM") or find_column(dataframe, "AVISITN")
        if visit_column is not None:
            visit_numbers = pd.to_numeric(dataframe[visit_column], errors="coerce")
            for subject, group in pd.DataFrame({"Subject": subjects, "Visit": visit_numbers}, index=dataframe.index).groupby("Subject"):
                previous: float | None = None
                for index, visit in group["Visit"].items():
                    if not subject or pd.isna(visit):
                        continue
                    if previous is not None and visit < previous:
                        site = str(sites.loc[index]) or site_map.get(str(subject), "")
                        rows.append(
                            _finding(
                                domain,
                                index,
                                str(subject),
                                site,
                                "Visit sequence issue",
                                "Minor",
                                f"{visit_column} is {visit:g}, following {previous:g} in source record order.",
                                "Review visit ordering and protocol visit assignment.",
                                record_dates.loc[index],
                            )
                        )
                    previous = float(visit)

    return pd.DataFrame(rows)


def laboratory_outliers(lb_records: pd.DataFrame | None, sites_by_subject: Mapping[str, str]) -> pd.DataFrame:
    """Detect reference-range, extreme-value, and sudden-change laboratory signals."""
    columns = ["Subject ID", "Site ID", "Lab Test", "Value", "Reference Range", "Risk Level", "Signal", "Source Row", "Collection Date"]
    if lb_records is None or lb_records.empty:
        return pd.DataFrame(columns=columns)
    labs = classify_lab_results(lb_records)
    records: list[dict[str, Any]] = []
    for _, row in abnormal_results(labs).iterrows():
        subject = str(row.get("USUBJID", ""))
        records.append(
            {
                "Subject ID": subject or "Not reported",
                "Site ID": sites_by_subject.get(subject, "Not available"),
                "Lab Test": row.get("Lab Test", "Not reported"),
                "Value": row.get("Numeric Result"),
                "Reference Range": f"{row.get('Reference Low', 'Not reported')} to {row.get('Reference High', 'Not reported')}",
                "Risk Level": "Medium",
                "Signal": f"{row.get('Result Status')} result outside supplied reference range",
                "Source Row": row.get("Source Row", 0),
                "Collection Date": row.get("Collection Date"),
            }
        )

    numeric = labs.loc[labs["Numeric Result"].notna() & labs["Lab Test"].ne("")].copy()
    for test_name, group in numeric.groupby("Lab Test"):
        if len(group.index) < 4 or float(group["Numeric Result"].std(ddof=1)) == 0:
            continue
        z_scores = (group["Numeric Result"] - group["Numeric Result"].mean()) / group["Numeric Result"].std(ddof=1)
        for index, z_score in z_scores.loc[z_scores.abs().ge(3)].items():
            row = group.loc[index]
            subject = str(row.get("USUBJID", ""))
            records.append(
                {
                    "Subject ID": subject or "Not reported",
                    "Site ID": sites_by_subject.get(subject, "Not available"),
                    "Lab Test": test_name,
                    "Value": row["Numeric Result"],
                    "Reference Range": f"{row.get('Reference Low', 'Not reported')} to {row.get('Reference High', 'Not reported')}",
                    "Risk Level": "High",
                    "Signal": f"Extreme value: observed z-score {z_score:.2f} within {test_name} results",
                    "Source Row": row.get("Source Row", 0),
                    "Collection Date": row.get("Collection Date"),
                }
            )

    dated = numeric.loc[numeric["Collection Date"].notna()].sort_values("Collection Date")
    for (subject, test_name), group in dated.groupby(["USUBJID", "Lab Test"]):
        prior_value: float | None = None
        for _, row in group.iterrows():
            value = float(row["Numeric Result"])
            if prior_value not in (None, 0) and abs((value - prior_value) / prior_value) >= 0.5:
                records.append(
                    {
                        "Subject ID": str(subject) or "Not reported",
                        "Site ID": sites_by_subject.get(str(subject), "Not available"),
                        "Lab Test": test_name,
                        "Value": value,
                        "Reference Range": f"{row.get('Reference Low', 'Not reported')} to {row.get('Reference High', 'Not reported')}",
                        "Risk Level": "High",
                        "Signal": "Sudden subject-level change of at least 50% from the prior result",
                        "Source Row": row.get("Source Row", 0),
                        "Collection Date": row.get("Collection Date"),
                    }
                )
            prior_value = value
    output = pd.DataFrame(records, columns=columns)
    if output.empty:
        return output
    order = {"High": 0, "Medium": 1, "Low": 2}
    output["_order"] = output["Risk Level"].map(order)
    return output.sort_values(["_order", "Lab Test"]).drop(columns="_order").drop_duplicates(
        subset=["Source Row", "Signal"], keep="first"
    )


def safety_signals(ae_records: pd.DataFrame | None, assignments: pd.DataFrame, sites_by_subject: Mapping[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect subject-level serious-event patterns and descriptive arm-level AE risk."""
    signal_columns = ["Subject ID", "Site ID", "Dataset", "Source Row", "Issue Type", "Severity", "Risk Level", "Risk Score", "Anomaly Date", "Message", "Recommended Review"]
    if ae_records is None or ae_records.empty:
        return pd.DataFrame(columns=signal_columns), pd.DataFrame()
    subjects = subject_identifiers(ae_records)
    masks = serious_masks(ae_records)
    serious = masks[0] if masks is not None else pd.Series(False, index=ae_records.index)
    rows: list[dict[str, Any]] = []
    for subject, indexes in pd.Series(subjects[serious].index, index=subjects[serious]).groupby(level=0):
        source_indexes = indexes.tolist()
        if subject and len(source_indexes) >= 2:
            rows.append(
                _finding(
                    "AE",
                    source_indexes[0],
                    str(subject),
                    sites_by_subject.get(str(subject), ""),
                    "Multiple serious adverse events",
                    "Critical",
                    f"Subject has {len(source_indexes)} AE records marked serious (AESER).",
                    "Perform prompt medical and data-review assessment of serious adverse-event records.",
                    _record_date(ae_records).loc[source_indexes[0]],
                )
            )
    severity_column = find_column(ae_records, "AESEV")
    if severity_column is not None and masks is not None:
        severe_nonserious = cleaned_text(ae_records[severity_column]).str.upper().eq("SEVERE") & ~serious
        for index in ae_records.index[severe_nonserious]:
            subject = str(subjects.loc[index])
            rows.append(
                _finding(
                    "AE",
                    index,
                    subject,
                    sites_by_subject.get(subject, ""),
                    "Unexpected severity pattern",
                    "Major",
                    "AESEV is Severe while AESER is not marked serious.",
                    "Confirm the severity and seriousness assessment against source documentation.",
                    _record_date(ae_records).loc[index],
                )
            )
    arm_table = ae_safety_by_arm(ae_records, assignments) if not assignments.empty else pd.DataFrame()
    if not arm_table.empty:
        serious_rate = arm_table["Subjects with Serious AEs (%)"].fillna(0)
        arm_table["Risk Level"] = np.select(
            [serious_rate.ge(10), serious_rate.ge(5)], ["High", "Medium"], default="Low"
        )
    return pd.DataFrame(rows, columns=signal_columns), arm_table


def site_risk_monitoring(
    datasets: Mapping[str, pd.DataFrame],
    findings: pd.DataFrame,
    ae_records: pd.DataFrame | None,
    sites_by_subject: Mapping[str, str],
) -> pd.DataFrame:
    """Calculate explainable per-site operational rates and risk classifications."""
    site_records: list[dict[str, str]] = []
    for dataframe in datasets.values():
        subjects = subject_identifiers(dataframe)
        sites = site_values(dataframe)
        site_records.extend(
            {"Subject ID": str(subject), "Site ID": str(site)}
            for subject, site in zip(subjects, sites)
            if subject and site
        )
    roster = pd.DataFrame(site_records).drop_duplicates()
    if roster.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    queries = fetch_queries()
    for site, group in roster.groupby("Site ID"):
        subjects = set(group["Subject ID"])
        denominator = len(subjects)
        site_findings = findings.loc[findings["Site ID"].eq(site)] if not findings.empty else pd.DataFrame()
        missing_count = int(site_findings["Issue Type"].eq("Missing critical value").sum()) if not site_findings.empty else 0
        visit_count = int(site_findings["Issue Type"].eq("Visit sequence issue").sum()) if not site_findings.empty else 0
        query_count = 0
        if not queries.empty and "Subject ID" in queries:
            query_count = int(queries["Subject ID"].astype("string").isin(subjects).sum())
        ae_subjects = 0
        if ae_records is not None:
            ae_subjects = int(subject_identifiers(ae_records).loc[lambda values: values.isin(subjects)].nunique())
        total_records = sum(
            int(subject_identifiers(dataframe).isin(subjects).sum()) for dataframe in datasets.values()
        )
        query_rate = (query_count / denominator) * 100 if denominator else np.nan
        missing_rate = (missing_count / total_records) * 100 if total_records else np.nan
        protocol_rate = (visit_count / denominator) * 100 if denominator else np.nan
        safety_rate = (ae_subjects / denominator) * 100 if denominator else np.nan
        risk_score = float(np.nanmean([query_rate, missing_rate, protocol_rate, safety_rate]))
        risk_level = "High" if risk_score >= 30 else "Medium" if risk_score >= 10 else "Low"
        rows.append(
            {
                "Site ID": site,
                "Subjects": denominator,
                "Query Rate (%)": query_rate,
                "Missing Data Rate (%)": missing_rate,
                "Protocol Deviation Rate (%)": protocol_rate,
                "Safety Event Rate (%)": safety_rate,
                "Risk Score": risk_score,
                "Risk Level": risk_level,
            }
        )
    return pd.DataFrame(rows).sort_values("Risk Score", ascending=False).reset_index(drop=True)


def add_operational_signals(findings: pd.DataFrame, site_risk: pd.DataFrame) -> pd.DataFrame:
    """Append high/medium site-risk signals to the subject/site priority dataset."""
    if site_risk.empty:
        return findings
    additional: list[dict[str, Any]] = []
    for _, row in site_risk.loc[site_risk["Risk Level"].isin(["High", "Medium"])].iterrows():
        severity = "Critical" if row["Risk Level"] == "High" else "Major"
        signal = _finding(
                "Operational",
                0,
                "Not applicable",
                str(row["Site ID"]),
                "Site performance risk",
                severity,
                f"Site risk score is {row['Risk Score']:.1f}, based on observed query, missing-data, visit-sequence, and AE rates.",
                "Review site-level data quality, protocol adherence, and safety follow-up activity.",
            )
        signal["Source Row"] = 0
        additional.append(signal)
    if not additional:
        return findings
    return pd.DataFrame([*findings.to_dict("records"), *additional])


def analysis_result(datasets: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame | int]:
    """Create the complete explainable anomaly-review dataset for the active session."""
    base = data_quality_findings(datasets)
    sites = subject_site_map(datasets)
    lab = laboratory_outliers(datasets.get("LB"), sites)
    assignments = treatment_assignments(datasets.get("DM"), datasets.get("EX"))
    safety, arm_comparison = safety_signals(datasets.get("AE"), assignments, sites)
    lab_findings: list[dict[str, Any]] = []
    for _, row in lab.iterrows():
        severity = "Critical" if row["Risk Level"] == "High" else "Major"
        lab_findings.append(
            _finding(
                "LB",
                int(row["Source Row"]) - 2 if pd.notna(row["Source Row"]) else 0,
                str(row["Subject ID"]),
                str(row["Site ID"]),
                str(row["Signal"]),
                severity,
                f"{row['Lab Test']}: {row['Value']} (reference range {row['Reference Range']}).",
                "Review laboratory source data, reference range, and medical relevance.",
                row["Collection Date"],
            )
        )
    finding_frames = [frame for frame in (base, safety, pd.DataFrame(lab_findings)) if not frame.empty]
    findings = pd.concat(finding_frames, ignore_index=True) if finding_frames else pd.DataFrame()
    if findings.empty:
        findings = pd.DataFrame(columns=["Subject ID", "Site ID", "Dataset", "Source Row", "Issue Type", "Severity", "Risk Level", "Risk Score", "Anomaly Date", "Message", "Recommended Review"])
    site_risk = site_risk_monitoring(datasets, findings, datasets.get("AE"), sites)
    findings = add_operational_signals(findings, site_risk)
    findings["Anomaly Date"] = pd.to_datetime(findings["Anomaly Date"], errors="coerce")
    findings = findings.sort_values(["Risk Score", "Anomaly Date"], ascending=[False, False], na_position="last").reset_index(drop=True)
    frequency = findings.groupby(["Issue Type", "Risk Level"], as_index=False).size().rename(columns={"size": "Count"})
    return {
        "Findings": findings,
        "Laboratory Outliers": lab,
        "Safety Signals": safety,
        "Safety by Arm": arm_comparison,
        "Site Risk": site_risk,
        "Frequency": frequency,
        "Total Subjects": len({subject for dataframe in datasets.values() for subject in subject_identifiers(dataframe) if subject}),
        "Total Sites": len(set(sites.values())),
        "Total Records": sum(len(dataframe.index) for dataframe in datasets.values()),
    }


def anomaly_insights(result: Mapping[str, pd.DataFrame | int]) -> list[str]:
    """Create factual, deterministic observations from the detected anomaly result."""
    insights: list[str] = []
    site_risk = result["Site Risk"]
    if isinstance(site_risk, pd.DataFrame) and not site_risk.empty:
        top_site = site_risk.iloc[0]
        insights.append(f"Site {top_site['Site ID']} has the highest observed anomaly risk score ({top_site['Risk Score']:.1f}).")
    lab = result["Laboratory Outliers"]
    if isinstance(lab, pd.DataFrame) and not lab.empty:
        alt = lab.loc[lab["Lab Test"].astype("string").str.upper().str.contains("ALT|ALANINE", regex=True, na=False)]
        if not alt.empty:
            insights.append(f"ALT abnormalities were observed in {alt['Subject ID'].nunique()} subject(s).")
    safety = result["Safety Signals"]
    if isinstance(safety, pd.DataFrame) and not safety.empty:
        multiple_serious = safety.loc[safety["Issue Type"].eq("Multiple serious adverse events")]
        if not multiple_serious.empty:
            insights.append(f"{multiple_serious['Subject ID'].nunique()} subject(s) reported multiple serious adverse events.")
    if not insights:
        insights.append("No explainable anomaly patterns were detected from the currently uploaded datasets and available site variables.")
    return insights
