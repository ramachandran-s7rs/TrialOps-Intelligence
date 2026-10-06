"""Reusable rule-based adverse-event safety analytics for uploaded AE data."""

from __future__ import annotations

from typing import Any

import pandas as pd


SEVERITY_CATEGORIES = ("Mild", "Moderate", "Severe")
SERIOUS_VALUES = frozenset({"Y", "YES", "TRUE", "1"})
NON_SERIOUS_VALUES = frozenset({"N", "NO", "FALSE", "0"})


def find_column(dataframe: pd.DataFrame, variable: str) -> str | None:
    """Return a case-insensitive SDTM variable match, if the variable exists."""
    normalized_variable = variable.upper()
    return next(
        (str(column) for column in dataframe.columns if str(column).strip().upper() == normalized_variable),
        None,
    )


def cleaned_text(values: pd.Series) -> pd.Series:
    """Return trimmed, nullable text values without mutating source data."""
    return values.astype("string").fillna("").str.strip()


def serious_masks(ae_records: pd.DataFrame) -> tuple[pd.Series, pd.Series] | None:
    """Return serious and non-serious masks from AESER, or None when unavailable."""
    serious_column = find_column(ae_records, "AESER")
    if serious_column is None:
        return None

    values = cleaned_text(ae_records[serious_column]).str.upper()
    return values.isin(SERIOUS_VALUES), values.isin(NON_SERIOUS_VALUES)


def adverse_event_terms(ae_records: pd.DataFrame) -> pd.Series:
    """Return preferred AEDECOD terms, falling back to AETERM where needed."""
    term_column = find_column(ae_records, "AEDECOD")
    source_term_column = find_column(ae_records, "AETERM")
    if term_column is None and source_term_column is None:
        return pd.Series(dtype="string")

    terms = (
        cleaned_text(ae_records[term_column])
        if term_column is not None
        else pd.Series("", index=ae_records.index, dtype="string")
    )
    if source_term_column is not None:
        source_terms = cleaned_text(ae_records[source_term_column])
        terms = terms.mask(terms.eq(""), source_terms)
    return terms[terms.ne("")]


def top_adverse_events(ae_records: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """Return the most frequently reported event terms for the safety dashboard."""
    counts = adverse_event_terms(ae_records).value_counts().head(limit)
    return counts.rename_axis("Adverse Event").reset_index(name="Event Count")


def severity_distribution(ae_records: pd.DataFrame) -> pd.DataFrame | None:
    """Return standard AE severity counts, or None when AESEV is unavailable."""
    severity_column = find_column(ae_records, "AESEV")
    if severity_column is None:
        return None

    labels = cleaned_text(ae_records[severity_column]).str.casefold().map(
        {"mild": "Mild", "moderate": "Moderate", "severe": "Severe"}
    )
    counts = labels.value_counts().reindex(SEVERITY_CATEGORIES, fill_value=0)
    return pd.DataFrame({"Severity": SEVERITY_CATEGORIES, "Event Count": counts.to_numpy(dtype="int64")})


def serious_event_distribution(ae_records: pd.DataFrame) -> pd.DataFrame | None:
    """Return AESER-based serious and non-serious counts, or None when unavailable."""
    masks = serious_masks(ae_records)
    if masks is None:
        return None

    serious_mask, non_serious_mask = masks
    return pd.DataFrame(
        {
            "Seriousness": ("Serious", "Non-Serious"),
            "Event Count": (int(serious_mask.sum()), int(non_serious_mask.sum())),
        }
    )


def safety_summary(ae_records: pd.DataFrame) -> dict[str, int | None]:
    """Summarize AE volume, affected subjects, and known AESER classifications."""
    subject_column = find_column(ae_records, "USUBJID")
    subject_count = 0
    if subject_column is not None:
        subject_count = int(cleaned_text(ae_records[subject_column]).replace("", pd.NA).nunique(dropna=True))

    masks = serious_masks(ae_records)
    if masks is None:
        serious_count = None
        non_serious_count = None
    else:
        serious_mask, non_serious_mask = masks
        serious_count = int(serious_mask.sum())
        non_serious_count = int(non_serious_mask.sum())

    return {
        "Total Adverse Events": len(ae_records.index),
        "Subjects with Adverse Events": subject_count,
        "Serious Adverse Events": serious_count,
        "Non-Serious Adverse Events": non_serious_count,
    }


def subject_safety_summary(ae_records: pd.DataFrame) -> dict[str, Any]:
    """Classify an individual subject's AE burden using the configured safety rules."""
    event_count = len(ae_records.index)
    masks = serious_masks(ae_records)
    serious_count = None if masks is None else int(masks[0].sum())

    if event_count >= 3 or (serious_count is not None and serious_count > 0):
        risk_level = "High"
    elif event_count == 2:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "Adverse Events": event_count,
        "Serious Adverse Events": serious_count,
        "Risk Level": risk_level,
    }


def safety_insights(ae_records: pd.DataFrame) -> list[str]:
    """Generate concise, data-derived observations for the safety dashboard."""
    insights: list[str] = []
    total_events = len(ae_records.index)
    top_events = top_adverse_events(ae_records, limit=1)
    if not top_events.empty:
        event_name = str(top_events.iloc[0]["Adverse Event"])
        insights.append(f"{event_name} is the most frequently reported adverse event.")

    summary = safety_summary(ae_records)
    serious_count = summary["Serious Adverse Events"]
    if total_events and serious_count is not None:
        serious_percentage = (serious_count / total_events) * 100
        insights.append(f"{serious_percentage:.0f}% of adverse events are classified as serious.")

    return insights
