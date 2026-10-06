"""Reusable laboratory result classification and analytics for uploaded LB data."""

from __future__ import annotations

from typing import Any

import pandas as pd


RESULT_STATUSES = ("High", "Low", "Normal", "Not Classified")


def find_column(dataframe: pd.DataFrame, variable: str) -> str | None:
    """Return a case-insensitive SDTM variable match, if available."""
    normalized_variable = variable.upper()
    return next(
        (str(column) for column in dataframe.columns if str(column).strip().upper() == normalized_variable),
        None,
    )


def cleaned_text(values: pd.Series) -> pd.Series:
    """Return nullable, trimmed text values without modifying the source dataset."""
    return values.astype("string").fillna("").str.strip()


def numeric_result_column(lb_records: pd.DataFrame) -> str | None:
    """Prefer standard numeric results, with original result as a fallback."""
    return find_column(lb_records, "LBSTRESN") or find_column(lb_records, "LBORRES")


def test_column(lb_records: pd.DataFrame) -> str | None:
    """Prefer the decoded lab test name, with the test code as a fallback."""
    return find_column(lb_records, "LBTEST") or find_column(lb_records, "LBTESTCD")


def classify_lab_results(lb_records: pd.DataFrame) -> pd.DataFrame:
    """Classify LB results against supplied reference limits without mutating source data."""
    classified = lb_records.copy()
    numeric_index = pd.to_numeric(pd.Series(classified.index), errors="coerce")
    source_rows = (
        numeric_index.astype(int).add(2).tolist()
        if numeric_index.notna().all()
        else list(range(2, len(classified.index) + 2))
    )
    classified.insert(0, "Source Row", source_rows)

    result_column = numeric_result_column(classified)
    low_column = find_column(classified, "LBSTNRLO")
    high_column = find_column(classified, "LBSTNRHI")
    selected_test_column = test_column(classified)
    unit_column = find_column(classified, "LBORRESU")
    date_column = find_column(classified, "LBDTC")

    classified["Numeric Result"] = (
        pd.to_numeric(classified[result_column], errors="coerce")
        if result_column is not None
        else pd.Series(pd.NA, index=classified.index, dtype="Float64")
    )
    classified["Reference Low"] = (
        pd.to_numeric(classified[low_column], errors="coerce")
        if low_column is not None
        else pd.Series(pd.NA, index=classified.index, dtype="Float64")
    )
    classified["Reference High"] = (
        pd.to_numeric(classified[high_column], errors="coerce")
        if high_column is not None
        else pd.Series(pd.NA, index=classified.index, dtype="Float64")
    )
    classified["Lab Test"] = (
        cleaned_text(classified[selected_test_column])
        if selected_test_column is not None
        else pd.Series("", index=classified.index, dtype="string")
    )
    classified["Lab Unit"] = (
        cleaned_text(classified[unit_column])
        if unit_column is not None
        else pd.Series("", index=classified.index, dtype="string")
    )
    classified["Collection Date"] = (
        pd.to_datetime(classified[date_column].astype("string"), errors="coerce")
        if date_column is not None
        else pd.Series(pd.NaT, index=classified.index)
    )

    status = pd.Series("Not Classified", index=classified.index, dtype="string")
    comparable = (
        classified["Numeric Result"].notna()
        & classified["Reference Low"].notna()
        & classified["Reference High"].notna()
    )
    status.loc[comparable & (classified["Numeric Result"] < classified["Reference Low"])] = "Low"
    status.loc[comparable & (classified["Numeric Result"] > classified["Reference High"])] = "High"
    status.loc[
        comparable
        & classified["Numeric Result"].ge(classified["Reference Low"])
        & classified["Numeric Result"].le(classified["Reference High"])
    ] = "Normal"
    classified["Result Status"] = status
    return classified


def laboratory_summary(classified_results: pd.DataFrame) -> dict[str, int]:
    """Return total and abnormal laboratory result counts."""
    statuses = classified_results.get("Result Status", pd.Series(dtype="string"))
    return {
        "Total Lab Records": len(classified_results.index),
        "Abnormal Results": int(statuses.isin(("High", "Low")).sum()),
        "High Results": int((statuses == "High").sum()),
        "Low Results": int((statuses == "Low").sum()),
    }


def abnormal_results(classified_results: pd.DataFrame) -> pd.DataFrame:
    """Return results outside their supplied reference range."""
    if "Result Status" not in classified_results.columns:
        return classified_results.iloc[0:0].copy()
    return classified_results.loc[classified_results["Result Status"].isin(("High", "Low"))].copy()


def top_abnormal_tests(classified_results: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """Return laboratory tests with the highest abnormal-result counts."""
    abnormal = abnormal_results(classified_results)
    if abnormal.empty or "Lab Test" not in abnormal.columns:
        return pd.DataFrame(columns=["Laboratory Test", "Abnormal Results"])
    counts = abnormal["Lab Test"].replace("", pd.NA).dropna().value_counts().head(limit)
    return counts.rename_axis("Laboratory Test").reset_index(name="Abnormal Results")


def laboratory_trend_data(classified_results: pd.DataFrame, test_name: str) -> pd.DataFrame:
    """Return date-stamped numeric results for one laboratory test."""
    if not {"Collection Date", "Lab Test", "Numeric Result"}.issubset(classified_results.columns):
        return pd.DataFrame()
    trend = classified_results.loc[
        classified_results["Lab Test"].eq(test_name)
        & classified_results["Collection Date"].notna()
        & classified_results["Numeric Result"].notna()
    ].copy()
    return trend.sort_values("Collection Date")


def laboratory_insights(classified_results: pd.DataFrame) -> list[str]:
    """Generate concise observations from supplied laboratory data and ranges."""
    insights: list[str] = []
    summary = laboratory_summary(classified_results)
    top_tests = top_abnormal_tests(classified_results, limit=1)
    if not top_tests.empty:
        test_name = str(top_tests.iloc[0]["Laboratory Test"])
        insights.append(f"{test_name} has the highest number of abnormal laboratory results.")
    if summary["Total Lab Records"]:
        abnormal_percentage = (summary["Abnormal Results"] / summary["Total Lab Records"]) * 100
        insights.append(f"{abnormal_percentage:.0f}% of laboratory records are outside the supplied reference ranges.")
    return insights


def lab_alert_records(classified_results: pd.DataFrame) -> list[dict[str, Any]]:
    """Return persistence-ready laboratory alerts for abnormal results only."""
    alerts = abnormal_results(classified_results)
    if alerts.empty:
        return []

    subject_column = find_column(alerts, "USUBJID")
    test_code_column = find_column(alerts, "LBTESTCD")
    source_result_column = numeric_result_column(alerts)
    records: list[dict[str, Any]] = []
    for _, row in alerts.iterrows():
        records.append(
            {
                "Subject ID": "" if subject_column is None else str(row[subject_column]).strip(),
                "Lab Test": str(row["Lab Test"]),
                "Lab Test Code": "" if test_code_column is None else str(row[test_code_column]).strip(),
                "Result Value": "" if source_result_column is None else str(row[source_result_column]).strip(),
                "Result Unit": str(row["Lab Unit"]),
                "Reference Low": row["Reference Low"],
                "Reference High": row["Reference High"],
                "Result Status": str(row["Result Status"]),
                "Collection Date": "" if pd.isna(row["Collection Date"]) else str(row["Collection Date"].date()),
                "Source Row": int(row["Source Row"]),
            }
        )
    return records
