"""Composed, audit-ready reporting metrics from active TrialOps datasets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from database import fetch_queries
from utils.anomaly_analytics import analysis_result
from utils.lab_analytics import abnormal_results, classify_lab_results, laboratory_summary
from utils.safety_analytics import safety_summary, severity_distribution
from utils.treatment_efficacy_analytics import derive_efficacy_records, exposure_summary, treatment_assignments


def build_report_pdf(title: str, summary: Mapping[str, Any], table: pd.DataFrame, insights: list[str]) -> bytes:
    """Create a concise sponsor-facing PDF from the selected report's live data."""
    from io import BytesIO
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    output = BytesIO(); doc = SimpleDocTemplate(output, pagesize=letter, leftMargin=.5*inch, rightMargin=.5*inch, topMargin=.5*inch, bottomMargin=.5*inch)
    styles = getSampleStyleSheet(); story = [Paragraph(title, styles['Title']), Paragraph('Clinical operations review output', styles['Normal']), Spacer(1, 8)]
    metrics = [['Metric', 'Value']] + [[key, str(value)] for key, value in summary.items()]
    metric_table = Table(metrics, colWidths=[3*inch, 3*inch]); metric_table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1F4E78')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),.25,colors.HexColor('#B7C9D6')),('PADDING',(0,0),(-1,-1),4)])); story.extend([metric_table, Spacer(1, 10)])
    if not table.empty:
        display = table.head(12).iloc[:, :6].fillna('Not available').astype(str)
        report_table = Table([display.columns.tolist()] + display.values.tolist(), repeatRows=1)
        report_table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1F4E78')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTSIZE',(0,0),(-1,-1),7),('GRID',(0,0),(-1,-1),.25,colors.HexColor('#B7C9D6')),('PADDING',(0,0),(-1,-1),3)])); story.extend([report_table, Spacer(1, 10)])
    story.extend(Paragraph(f'- {insight}', styles['BodyText']) for insight in insights); story.append(Paragraph('This report is a clinical-review aid and requires qualified operational, clinical, and data-management review.', styles['Italic']))
    doc.build(story); return output.getvalue()


def _subjects(datasets: Mapping[str, pd.DataFrame]) -> int:
    values: set[str] = set()
    for frame in datasets.values():
        column = next((str(c) for c in frame.columns if str(c).strip().upper() == "USUBJID"), None)
        if column is not None:
            values.update(value for value in frame[column].astype("string").fillna("").str.strip() if value)
    return len(values)


def _quality_score(records: int, missing: int, duplicates: int, invalid_dates: int) -> tuple[float, str]:
    if not records:
        return 0.0, "Needs Review"
    score = max(0.0, 100 - ((missing + duplicates + invalid_dates) / records) * 100)
    label = "Excellent" if score >= 98 else "Good" if score >= 95 else "Needs Review" if score >= 85 else "Critical"
    return score, label


def build_report_data(datasets: Mapping[str, pd.DataFrame], validation_findings: pd.DataFrame) -> dict[str, Any]:
    """Build all report metrics once from session data and persisted query state."""
    anomalies = analysis_result(datasets)
    findings = anomalies["Findings"]
    queries = fetch_queries()
    total_records = int(anomalies["Total Records"])
    missing = int(findings["Issue Type"].eq("Missing critical value").sum()) if not findings.empty else 0
    duplicates = int(findings["Issue Type"].eq("Duplicate record").sum()) if not findings.empty else 0
    invalid_dates = int(findings["Issue Type"].eq("Invalid date").sum()) if not findings.empty else 0
    quality_score, quality_label = _quality_score(total_records, missing, duplicates, invalid_dates)
    deviations = findings.loc[findings["Issue Type"].isin(["Visit sequence issue", "Site performance risk"])] if not findings.empty else pd.DataFrame()
    ae = datasets.get("AE")
    lb = datasets.get("LB")
    ex = datasets.get("EX")
    eff = datasets.get("EFF")
    assignments = treatment_assignments(datasets.get("DM"), ex)
    lab_classified = classify_lab_results(lb) if lb is not None else pd.DataFrame()
    efficacy = derive_efficacy_records(eff) if eff is not None else pd.DataFrame()
    mean_change = float(efficacy["Change"].mean()) if not efficacy.empty and efficacy["Change"].notna().any() else None
    response_rate = None
    if not efficacy.empty and efficacy["Change"].notna().any():
        response_rate = float((efficacy["Change"] < 0).mean() * 100)
    query_status = (
        queries["Status"].astype("string").value_counts().rename_axis("Status").reset_index(name="Queries")
        if not queries.empty else pd.DataFrame(columns=["Status", "Queries"])
    )
    query_domain = (
        queries["Dataset"].astype("string").value_counts().rename_axis("Dataset").reset_index(name="Queries")
        if not queries.empty else pd.DataFrame(columns=["Dataset", "Queries"])
    )
    query_time = pd.DataFrame(columns=["Month", "Queries"])
    if not queries.empty and "Created At" in queries:
        created = pd.to_datetime(queries["Created At"], errors="coerce")
        query_time = created.dropna().dt.to_period("M").astype(str).value_counts().sort_index().rename_axis("Month").reset_index(name="Queries")
    site = anomalies["Site Risk"].copy()
    if not site.empty:
        site["Anomalies"] = site["Site ID"].map(findings["Site ID"].value_counts()).fillna(0).astype(int)
        site["Performance Score"] = (100 - site["Risk Score"]).clip(lower=0)
        site["Performance Classification"] = site["Performance Score"].map(lambda value: "Excellent" if value >= 90 else "Good" if value >= 75 else "Needs Review" if value >= 50 else "High Risk")
    executive = [f"Study includes {_subjects(datasets)} subject(s) across {int(anomalies['Total Sites'])} detected site(s).", f"Data completeness is estimated at {quality_score:.1f}% ({quality_label})."]
    if not deviations.empty and _subjects(datasets): executive.append(f"Protocol-related signals occurred in {deviations['Subject ID'].replace('Not applicable', pd.NA).nunique(dropna=True) / _subjects(datasets) * 100:.1f}% of subjects.")
    if not site.empty: executive.append(f"Site {site.iloc[0]['Site ID']} has the highest observed operational risk score.")
    if ae is None: executive.append("Safety assessment is unavailable because no AE dataset is uploaded.")
    else: executive.append("Safety signals require clinical review; no automated safety conclusion is made.")
    return {
        "Summary": {"Total Subjects": _subjects(datasets), "Total Sites": int(anomalies["Total Sites"]), "Total Queries": len(queries), "Open Queries": int((queries.get("Status", pd.Series(dtype="string")) == "Open").sum()), "Total Validation Findings": len(validation_findings), "Protocol Deviations": len(deviations), "Safety Signals": len(anomalies["Safety Signals"]), "Detected Anomalies": len(findings)},
        "Domains": pd.DataFrame([{"Dataset": domain, "Records": len(frame), "Columns": len(frame.columns)} for domain, frame in sorted(datasets.items())]),
        "Quality": pd.DataFrame([{"Records Uploaded": total_records, "Missing Critical Values": missing, "Duplicate Records": duplicates, "Invalid Dates": invalid_dates, "Data Completeness (%)": quality_score, "Quality Score": quality_label}]),
        "Queries": queries, "Query Status": query_status, "Query Domain": query_domain, "Query Time": query_time,
        "Safety Summary": pd.DataFrame([safety_summary(ae)]) if ae is not None else pd.DataFrame(),
        "Safety Severity": severity_distribution(ae) if ae is not None else pd.DataFrame(), "Safety by Arm": anomalies["Safety by Arm"],
        "Deviations": deviations, "Laboratory": laboratory_summary(lab_classified) if not lab_classified.empty else {}, "Lab Alerts": abnormal_results(lab_classified) if not lab_classified.empty else pd.DataFrame(),
        "Exposure": exposure_summary(ex) if ex is not None else {}, "Efficacy": pd.DataFrame([{"Subjects Exposed": exposure_summary(ex)["Total Exposed Subjects"] if ex is not None else 0, "Treatment Arms": int(assignments["Treatment Arm"].nunique()) if not assignments.empty else 0, "Mean Endpoint Change": mean_change, "Response Rate (%)": response_rate}]), "Efficacy Records": efficacy,
        "Site Performance": site, "Anomalies": findings, "Frequency": anomalies["Frequency"], "Executive Insights": executive,
    }
