"""Column-signature detection for supported CDISC SDTM domains."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


SUPPORTED_DOMAINS = ("DM", "AE", "LB", "VS", "EX", "EFF")


def standardize_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with uppercase, whitespace-trimmed column names.

    Duplicate names introduced by standardisation are rejected because they make
    SDTM variable selection ambiguous and could invalidate downstream checks.
    """
    standardized = dataframe.copy()
    standardized.columns = [str(column).strip().upper() for column in standardized.columns]
    duplicate_columns = standardized.columns[standardized.columns.duplicated()].unique().tolist()
    if duplicate_columns:
        raise ValueError(f"Duplicate column names after standardisation: {', '.join(duplicate_columns)}")
    return standardized


def _contains_all(columns: set[str], required: Iterable[str]) -> bool:
    return set(required).issubset(columns)


def _contains_any(columns: set[str], alternatives: Iterable[str]) -> bool:
    return bool(columns.intersection(alternatives))


def detect_candidate_domains(dataframe: pd.DataFrame) -> list[str]:
    """Return all supported domains whose required SDTM column signature matches.

    Detection deliberately considers only standardized column names. Filenames,
    paths, extensions, and row values are not used for business classification.
    """
    columns = {str(column).strip().upper() for column in dataframe.columns}
    candidates: list[str] = []

    if _contains_all(columns, ("USUBJID", "SEX", "AGE")):
        candidates.append("DM")
    if _contains_all(columns, ("USUBJID", "AETERM")):
        candidates.append("AE")
    if _contains_all(columns, ("USUBJID",)) and _contains_any(columns, ("EXTRT", "EXDOSE", "EXSTDTC")):
        candidates.append("EX")
    if _contains_all(columns, ("USUBJID",)) and _contains_any(columns, ("LBTEST", "LBTESTCD")) and _contains_any(
        columns, ("LBSTRESN", "LBORRES")
    ):
        candidates.append("LB")
    if _contains_all(columns, ("USUBJID",)) and _contains_any(columns, ("VSTEST", "VSTESTCD")) and _contains_any(
        columns, ("VSSTRESN", "VSORRES")
    ):
        candidates.append("VS")
    if _contains_all(columns, ("USUBJID",)) and _contains_any(columns, ("PARAM", "PARAMCD")) and _contains_any(
        columns, ("BASE", "AVAL", "CHG", "PCHG")
    ):
        candidates.append("EFF")

    return candidates


def detect_domain(dataframe: pd.DataFrame) -> str | None:
    """Return a single detected domain, or ``None`` when detection is not unique."""
    candidates = detect_candidate_domains(dataframe)
    return candidates[0] if len(candidates) == 1 else None
