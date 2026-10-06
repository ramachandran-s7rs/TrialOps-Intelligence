"""Report builders for the explainable anomaly-detection workspace."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd


def build_pdf_summary(summary: dict[str, Any], findings: pd.DataFrame, insights: list[str]) -> bytes:
    """Create a concise, professional anomaly-review PDF from live session results."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=letter,
        leftMargin=0.55 * inch,
        rightMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    )
    styles = getSampleStyleSheet()
    story = [
        Paragraph("TrialOps Intelligence - Anomaly Detection Summary", styles["Title"]),
        Paragraph("Explainable rule-based clinical operations review", styles["Normal"]),
        Spacer(1, 10),
    ]
    metrics = [
        ["Measure", "Observed Value"],
        ["Total subjects", str(summary["Total Subjects"])],
        ["Total sites", str(summary["Total Sites"])],
        ["Records reviewed", str(summary["Total Records"])],
        ["Anomalies detected", str(len(findings.index))],
        ["High-risk signals", str(int((findings["Risk Level"] == "High").sum()))],
    ]
    metric_table = Table(metrics, colWidths=[2.4 * inch, 2.4 * inch])
    metric_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7F1D1D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D9B4B4")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FFF7F7")]),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend([metric_table, Spacer(1, 12), Paragraph("Priority Findings", styles["Heading2"])])
    priority = findings.head(10).reindex(columns=["Subject ID", "Site ID", "Issue Type", "Severity", "Risk Score"])
    if priority.empty:
        story.append(Paragraph("No anomalies were detected in the currently uploaded datasets.", styles["Normal"]))
    else:
        rows = [priority.columns.tolist()] + priority.fillna("Not available").astype(str).values.tolist()
        priority_table = Table(rows, colWidths=[1.05 * inch, 0.8 * inch, 2.1 * inch, 0.8 * inch, 0.8 * inch], repeatRows=1)
        priority_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7F1D1D")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D9B4B4")),
                    ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                    ("PADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story.append(priority_table)
    story.extend([Spacer(1, 12), Paragraph("Automated Clinical-review Insights", styles["Heading2"])])
    story.extend(Paragraph(f"- {insight}", styles["BodyText"]) for insight in insights)
    story.extend(
        [
            Spacer(1, 8),
            Paragraph(
                "This report identifies explainable data-quality and safety signals for review. It is not a diagnosis, audit conclusion, or regulatory submission.",
                styles["Italic"],
            ),
        ]
    )
    document.build(story)
    return output.getvalue()


def build_summary_report(summary: dict[str, Any], findings: pd.DataFrame, insights: list[str]) -> bytes:
    """Create a portable text clinical-operations anomaly summary."""
    lines = [
        "TrialOps Intelligence - Anomaly Detection Summary Report",
        "",
        f"Total subjects: {summary['Total Subjects']}",
        f"Total sites: {summary['Total Sites']}",
        f"Total records reviewed: {summary['Total Records']}",
        f"Total anomalies detected: {len(findings.index)}",
        f"High-risk signals: {int((findings['Risk Level'] == 'High').sum())}",
        "",
        "Automated clinical-review insights",
    ]
    lines.extend(f"- {insight}" for insight in insights)
    lines.extend(
        [
            "",
            "Limitations",
            "- Signals are deterministic rules and observed statistical thresholds, not a black-box prediction model.",
            "- Review findings against source records, protocol requirements, and clinical context before taking action.",
        ]
    )
    return "\n".join(lines).encode("utf-8")
