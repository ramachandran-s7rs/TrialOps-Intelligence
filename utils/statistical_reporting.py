"""On-demand PDF, Excel, and text reports for exploratory statistical review."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd


def export_dependencies_missing() -> list[str]:
    """Report optional export packages unavailable in the active runtime."""
    missing: list[str] = []
    try:
        import openpyxl  # noqa: F401
    except ModuleNotFoundError:
        missing.append("openpyxl")
    try:
        import reportlab  # noqa: F401
    except ModuleNotFoundError:
        missing.append("reportlab")
    return missing


def _display_value(value: Any, decimals: int = 3) -> str:
    """Provide a compact value for a human-readable clinical-review report."""
    if value is None or pd.isna(value):
        return "Not available"
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def build_excel_summary(sections: dict[str, pd.DataFrame]) -> bytes:
    """Create a styled workbook from analysis output tables."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter

    workbook = Workbook()
    workbook.remove(workbook.active)
    header_fill = PatternFill("solid", fgColor="1F4E78")
    for sheet_name, dataframe in sections.items():
        worksheet = workbook.create_sheet(title=sheet_name[:31])
        if dataframe.empty:
            worksheet.append(["No data available for this analysis section."])
            continue
        worksheet.append(list(dataframe.columns))
        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = Font(color="FFFFFF", bold=True)
        for row in dataframe.itertuples(index=False, name=None):
            worksheet.append([None if pd.isna(value) else value for value in row])
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for index, column in enumerate(worksheet.columns, start=1):
            width = min(max(len(str(cell.value or "")) for cell in column) + 2, 34)
            worksheet.column_dimensions[get_column_letter(index)].width = width

    output = BytesIO()
    workbook.save(output)
    return output.getvalue()


def build_pdf_report(
    title: str,
    endpoint: str,
    test_result: dict[str, Any],
    descriptives: pd.DataFrame,
    insights: list[str],
) -> bytes:
    """Create a concise landscape PDF statistical-review report."""
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
        Paragraph(title, styles["Title"]),
        Paragraph(f"Endpoint: {endpoint}", styles["Normal"]),
        Spacer(1, 10),
        Paragraph("Exploratory Statistical Test", styles["Heading2"]),
    ]
    test_rows = [["Measure", "Result"]] + [
        [key, _display_value(value)]
        for key, value in test_result.items()
        if key in {"Test Name", "Test Statistic", "P-value", "Significance Interpretation", "Effect Metric", "Effect Size", "Effect Interpretation"}
    ]
    test_table = Table(test_rows, colWidths=[2.3 * inch, 4.6 * inch])
    test_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#B7C9D6")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F8FB")]),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend([test_table, Spacer(1, 12), Paragraph("Treatment-arm Descriptives", styles["Heading2"])])

    display = descriptives.copy().rename(
        columns={
            "Treatment Arm": "Arm",
            "Mean Improvement": "Mean",
            "Median Improvement": "Median",
            "Standard Deviation": "SD",
            "Standard Error": "SE",
            "95% CI Lower": "CI Low",
            "95% CI Upper": "CI High",
        }
    )
    for column in display.columns:
        if pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].map(lambda value: _display_value(value, 3))
    rows = [display.columns.tolist()] + display.astype(str).values.tolist()
    widths = [1.2 * inch] + [0.78 * inch] * (len(display.columns) - 1)
    descriptive_table = Table(rows, colWidths=widths, repeatRows=1)
    descriptive_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#B7C9D6")),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("PADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.extend([descriptive_table, Spacer(1, 12), Paragraph("Clinical-review Interpretation", styles["Heading2"])])
    story.extend(Paragraph(f"- {insight}", styles["BodyText"]) for insight in insights)
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "This report is an exploratory clinical-review aid. It is not an FDA submission, statistical analysis plan, or confirmatory efficacy conclusion.",
            styles["Italic"],
        )
    )
    document.build(story)
    return output.getvalue()


def build_regulatory_review_report(
    endpoint: str,
    placebo_arm: str,
    test_result: dict[str, Any],
    insights: list[str],
) -> bytes:
    """Create a plain-text regulatory-review briefing with explicit limitations."""
    lines = [
        "TrialOps Intelligence - Exploratory Regulatory Review Briefing",
        "",
        f"Endpoint: {endpoint}",
        f"Placebo reference: {placebo_arm}",
        "",
        "Statistical method",
        f"- Test: {_display_value(test_result.get('Test Name'))}",
        f"- Test statistic: {_display_value(test_result.get('Test Statistic'))}",
        f"- P-value: {_display_value(test_result.get('P-value'), 4)}",
        f"- Interpretation: {_display_value(test_result.get('Significance Interpretation'))}",
        f"- Effect size: {_display_value(test_result.get('Effect Metric'))} = {_display_value(test_result.get('Effect Size'))}",
        "",
        "Clinical-review observations",
    ]
    lines.extend(f"- {insight}" for insight in insights)
    lines.extend(
        [
            "",
            "Limitations",
            "- Results are exploratory and rely on the uploaded datasets.",
            "- No multiplicity adjustment, covariate model, missing-data strategy, estimand definition, or protocol-specific analysis was applied.",
            "- This briefing is not an FDA submission and must be reviewed by qualified biostatistical and clinical personnel.",
        ]
    )
    return "\n".join(lines).encode("utf-8")
