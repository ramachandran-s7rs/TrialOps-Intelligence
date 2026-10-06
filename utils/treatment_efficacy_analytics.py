"""Shared treatment exposure, efficacy, and treatment-arm analytics."""

from __future__ import annotations

from math import erfc, sqrt
from typing import Any, Mapping

import pandas as pd

from utils.lab_analytics import abnormal_results, classify_lab_results
from utils.safety_analytics import serious_masks


def find_column(dataframe: pd.DataFrame, variable: str) -> str | None:
    """Return a case-insensitive SDTM variable match, if it is available."""
    normalized_variable = variable.upper()
    return next(
        (str(column) for column in dataframe.columns if str(column).strip().upper() == normalized_variable),
        None,
    )


def cleaned_text(values: pd.Series) -> pd.Series:
    """Return trimmed nullable text values without modifying source records."""
    return values.astype("string").fillna("").str.strip()


def subject_identifiers(dataframe: pd.DataFrame) -> pd.Series:
    """Return cleaned subject identifiers, or blank identifiers when unavailable."""
    subject_column = find_column(dataframe, "USUBJID")
    if subject_column is None:
        return pd.Series("", index=dataframe.index, dtype="string")
    return cleaned_text(dataframe[subject_column])


def treatment_assignments(
    dm_records: pd.DataFrame | None, ex_records: pd.DataFrame | None
) -> pd.DataFrame:
    """Assign subjects to treatment arms using DM ARM/ARMCD, then EXTRT as fallback."""
    assignments: dict[str, str] = {}
    if isinstance(dm_records, pd.DataFrame):
        dm_subjects = subject_identifiers(dm_records)
        arm_column = find_column(dm_records, "ARM")
        arm_code_column = find_column(dm_records, "ARMCD")
        arm_values = (
            cleaned_text(dm_records[arm_column])
            if arm_column is not None
            else pd.Series("", index=dm_records.index, dtype="string")
        )
        if arm_code_column is not None:
            arm_codes = cleaned_text(dm_records[arm_code_column])
            arm_values = arm_values.mask(arm_values.eq(""), arm_codes)
        for subject_id, arm in zip(dm_subjects, arm_values):
            if subject_id:
                assignments[str(subject_id)] = str(arm) if arm else "Not assigned"

    if isinstance(ex_records, pd.DataFrame):
        ex_subjects = subject_identifiers(ex_records)
        treatment_column = find_column(ex_records, "EXTRT")
        treatments = (
            cleaned_text(ex_records[treatment_column])
            if treatment_column is not None
            else pd.Series("", index=ex_records.index, dtype="string")
        )
        for subject_id, treatment in zip(ex_subjects, treatments):
            if subject_id and (subject_id not in assignments or assignments[subject_id] == "Not assigned"):
                assignments[str(subject_id)] = str(treatment) if treatment else "Not assigned"

    return pd.DataFrame(
        [{"USUBJID": subject_id, "Treatment Arm": arm} for subject_id, arm in sorted(assignments.items())],
        columns=["USUBJID", "Treatment Arm"],
    )


def exposure_records(ex_records: pd.DataFrame) -> pd.DataFrame:
    """Return EX records with standardized display fields and optional duration."""
    exposure = ex_records.copy()
    for variable, display_name in (
        ("EXTRT", "Treatment"),
        ("EXDOSE", "Dose"),
        ("EXDOSU", "Dose Unit"),
        ("EXROUTE", "Route"),
        ("EXDOSFRQ", "Dose Frequency"),
    ):
        column = find_column(exposure, variable)
        exposure[display_name] = cleaned_text(exposure[column]) if column is not None else ""

    start_column = find_column(exposure, "EXSTDTC")
    end_column = find_column(exposure, "EXENDTC")
    exposure["Exposure Start"] = (
        pd.to_datetime(exposure[start_column].astype("string"), errors="coerce")
        if start_column is not None
        else pd.Series(pd.NaT, index=exposure.index)
    )
    exposure["Exposure End"] = (
        pd.to_datetime(exposure[end_column].astype("string"), errors="coerce")
        if end_column is not None
        else pd.Series(pd.NaT, index=exposure.index)
    )
    exposure["Exposure Duration (Days)"] = (exposure["Exposure End"] - exposure["Exposure Start"]).dt.days
    return exposure


def exposure_summary(ex_records: pd.DataFrame) -> dict[str, Any]:
    """Return exposure-level totals and optional duration metrics."""
    exposure = exposure_records(ex_records)
    subjects = subject_identifiers(exposure).replace("", pd.NA).nunique(dropna=True)
    treatments = sorted(value for value in exposure["Treatment"].dropna().unique().tolist() if value)
    duration = exposure["Exposure Duration (Days)"].dropna()
    return {
        "Total Exposed Subjects": int(subjects),
        "Treatment Names": treatments,
        "Mean Exposure Duration (Days)": None if duration.empty else float(duration.mean()),
    }


def dose_distribution(ex_records: pd.DataFrame) -> pd.DataFrame:
    """Return counts for recorded dose plus unit combinations."""
    exposure = exposure_records(ex_records)
    labels = (exposure["Dose"] + " " + exposure["Dose Unit"]).str.strip()
    labels = labels[labels.ne("")]
    return labels.value_counts().rename_axis("Dose").reset_index(name="Exposure Records")


def route_distribution(ex_records: pd.DataFrame) -> pd.DataFrame:
    """Return counts for recorded administration routes."""
    exposure = exposure_records(ex_records)
    routes = exposure["Route"][exposure["Route"].ne("")]
    return routes.value_counts().rename_axis("Route").reset_index(name="Exposure Records")


def derive_efficacy_records(efficacy_records: pd.DataFrame) -> pd.DataFrame:
    """Derive missing change measures from BASE and AVAL without changing source data."""
    derived = efficacy_records.copy()
    parameter_column = find_column(derived, "PARAM") or find_column(derived, "PARAMCD")
    base_column = find_column(derived, "BASE")
    aval_column = find_column(derived, "AVAL")
    change_column = find_column(derived, "CHG")
    percentage_change_column = find_column(derived, "PCHG")
    visit_column = find_column(derived, "AVISIT") or find_column(derived, "AVISITN")
    date_column = find_column(derived, "ADT")

    derived["Endpoint"] = (
        cleaned_text(derived[parameter_column])
        if parameter_column is not None
        else pd.Series("Unspecified Endpoint", index=derived.index, dtype="string")
    )
    derived["Baseline"] = (
        pd.to_numeric(derived[base_column], errors="coerce")
        if base_column is not None
        else pd.Series(pd.NA, index=derived.index, dtype="Float64")
    )
    derived["Post-treatment Value"] = (
        pd.to_numeric(derived[aval_column], errors="coerce")
        if aval_column is not None
        else pd.Series(pd.NA, index=derived.index, dtype="Float64")
    )
    calculated_change = derived["Post-treatment Value"] - derived["Baseline"]
    recorded_change = pd.to_numeric(derived[change_column], errors="coerce") if change_column is not None else calculated_change
    derived["Change"] = recorded_change.where(recorded_change.notna(), calculated_change)
    calculated_percentage_change = (derived["Change"] / derived["Baseline"]) * 100
    calculated_percentage_change = calculated_percentage_change.where(derived["Baseline"].ne(0))
    recorded_percentage_change = (
        pd.to_numeric(derived[percentage_change_column], errors="coerce")
        if percentage_change_column is not None
        else calculated_percentage_change
    )
    derived["Percentage Change"] = recorded_percentage_change.where(
        recorded_percentage_change.notna(), calculated_percentage_change
    )
    derived["Visit"] = cleaned_text(derived[visit_column]) if visit_column is not None else ""
    derived["Analysis Date"] = (
        pd.to_datetime(derived[date_column].astype("string"), errors="coerce")
        if date_column is not None
        else pd.Series(pd.NaT, index=derived.index)
    )
    return derived


def latest_efficacy_records(derived_records: pd.DataFrame) -> pd.DataFrame:
    """Select the latest available observation per subject and endpoint."""
    if derived_records.empty:
        return derived_records.copy()
    records = derived_records.copy()
    records["USUBJID"] = subject_identifiers(records)
    records["_record_order"] = range(len(records.index))
    records["_date_order"] = records["Analysis Date"].fillna(pd.Timestamp.min)
    latest = records.sort_values(["USUBJID", "Endpoint", "_date_order", "_record_order"]).groupby(
        ["USUBJID", "Endpoint"], as_index=False, sort=False
    ).tail(1)
    return latest.drop(columns=["_record_order", "_date_order"])


def efficacy_with_arms(derived_records: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    """Attach treatment arms to the latest subject-level efficacy records."""
    latest = latest_efficacy_records(derived_records)
    if assignments.empty:
        latest["Treatment Arm"] = "Not assigned"
        return latest
    return latest.merge(assignments, on="USUBJID", how="left").assign(
        **{"Treatment Arm": lambda frame: frame["Treatment Arm"].fillna("Not assigned")}
    )


def efficacy_summary(records: pd.DataFrame) -> dict[str, int | float | None]:
    """Return descriptive efficacy metrics for a selected endpoint population."""
    evaluable = records.loc[records["Post-treatment Value"].notna()] if not records.empty else records
    return {
        "Total Subjects Evaluated": int(evaluable["USUBJID"].nunique()) if not evaluable.empty else 0,
        "Mean Baseline Value": None if evaluable.empty else float(evaluable["Baseline"].mean()),
        "Mean Post-treatment Value": None if evaluable.empty else float(evaluable["Post-treatment Value"].mean()),
        "Mean Change from Baseline": None if evaluable.empty else float(evaluable["Change"].mean()),
        "Mean Percentage Change": None if evaluable.empty else float(evaluable["Percentage Change"].mean()),
    }


def efficacy_by_arm(records: pd.DataFrame) -> pd.DataFrame:
    """Summarize efficacy changes by treatment arm using subject-level observations."""
    if records.empty:
        return pd.DataFrame(columns=["Treatment Arm", "Subjects", "Mean Change", "Mean Percentage Change"])
    return (
        records.groupby("Treatment Arm", as_index=False)
        .agg(
            Subjects=("USUBJID", "nunique"),
            **{"Mean Change": ("Change", "mean"), "Mean Percentage Change": ("Percentage Change", "mean")},
        )
        .sort_values("Treatment Arm")
    )


def descriptive_statistics(records: pd.DataFrame) -> pd.DataFrame:
    """Return per-arm descriptive statistics for change from baseline."""
    rows: list[dict[str, Any]] = []
    for arm, group in records.groupby("Treatment Arm", dropna=False):
        values = group["Change"].dropna()
        count = len(values.index)
        standard_deviation = values.std(ddof=1) if count > 1 else None
        standard_error = (standard_deviation / sqrt(count)) if standard_deviation is not None and count else None
        rows.append(
            {
                "Treatment Arm": arm,
                "N": count,
                "Mean": None if not count else values.mean(),
                "Median": None if not count else values.median(),
                "Standard Deviation": standard_deviation,
                "Min": None if not count else values.min(),
                "Max": None if not count else values.max(),
                "95% CI Lower (normal approximation)": None if standard_error is None else values.mean() - 1.96 * standard_error,
                "95% CI Upper (normal approximation)": None if standard_error is None else values.mean() + 1.96 * standard_error,
            }
        )
    return pd.DataFrame(rows)


def two_group_comparison(records: pd.DataFrame, arms: list[str]) -> dict[str, Any] | None:
    """Return a labeled, approximate independent two-group comparison when suitable."""
    if len(arms) != 2:
        return None
    first = records.loc[records["Treatment Arm"] == arms[0], "Change"].dropna()
    second = records.loc[records["Treatment Arm"] == arms[1], "Change"].dropna()
    if len(first.index) < 20 or len(second.index) < 20:
        return None
    first_variance = first.var(ddof=1)
    second_variance = second.var(ddof=1)
    standard_error = sqrt((first_variance / len(first.index)) + (second_variance / len(second.index)))
    if not standard_error:
        return None
    difference = float(first.mean() - second.mean())
    z_score = difference / standard_error
    return {
        "Comparison": f"{arms[0]} minus {arms[1]}",
        "Mean Difference": difference,
        "Approximate z Statistic": z_score,
        "Approximate Two-sided p-value": erfc(abs(z_score) / sqrt(2)),
        "Method": "Independent-group normal approximation; descriptive support only",
    }


def ae_safety_by_arm(ae_records: pd.DataFrame | None, assignments: pd.DataFrame) -> pd.DataFrame:
    """Calculate AE rates using treatment-arm subject counts as denominators."""
    denominator = assignments.groupby("Treatment Arm", as_index=False).agg(**{"Subjects in Arm": ("USUBJID", "nunique")})
    if ae_records is None or ae_records.empty:
        return denominator.assign(
            **{
                "Subjects with at Least One AE": 0,
                "Subjects with AEs (%)": 0.0,
                "Subjects with Serious AEs": 0,
                "Subjects with Serious AEs (%)": 0.0,
                "Subjects with Severe AEs": 0,
                "Total AEs": 0,
                "Serious AEs": 0,
                "Severe AEs": 0,
            }
        )

    safety = ae_records.copy()
    safety["USUBJID"] = subject_identifiers(safety)
    safety = safety.merge(assignments, on="USUBJID", how="left").fillna({"Treatment Arm": "Not assigned"})
    seriousness = serious_masks(safety)
    safety["_serious"] = seriousness[0] if seriousness is not None else False
    severity_column = find_column(safety, "AESEV")
    safety["_severe"] = cleaned_text(safety[severity_column]).str.upper().eq("SEVERE") if severity_column else False
    event_counts = safety.groupby("Treatment Arm", as_index=False).agg(
        **{
            "Total AEs": ("USUBJID", "size"),
            "Serious AEs": ("_serious", "sum"),
            "Severe AEs": ("_severe", "sum"),
        }
    )
    subject_counts = safety.groupby(["Treatment Arm", "USUBJID"], as_index=False).agg(
        **{"Has Serious AE": ("_serious", "any"), "Has Severe AE": ("_severe", "any")}
    )
    subject_counts = subject_counts.groupby("Treatment Arm", as_index=False).agg(
        **{
            "Subjects with at Least One AE": ("USUBJID", "nunique"),
            "Subjects with Serious AEs": ("Has Serious AE", "sum"),
            "Subjects with Severe AEs": ("Has Severe AE", "sum"),
        }
    )
    result = denominator.merge(subject_counts, on="Treatment Arm", how="left").merge(
        event_counts, on="Treatment Arm", how="left"
    ).fillna(0)
    result["Subjects with AEs (%)"] = (result["Subjects with at Least One AE"] / result["Subjects in Arm"]) * 100
    result["Subjects with Serious AEs (%)"] = (
        result["Subjects with Serious AEs"] / result["Subjects in Arm"]
    ) * 100
    return result


def lab_safety_by_arm(lb_records: pd.DataFrame | None, assignments: pd.DataFrame) -> pd.DataFrame:
    """Compare abnormal laboratory outcomes by treatment arm."""
    denominator = assignments.groupby("Treatment Arm", as_index=False).agg(**{"Subjects in Arm": ("USUBJID", "nunique")})
    if lb_records is None or lb_records.empty:
        return denominator.assign(
            **{"Subjects with Abnormal Labs": 0, "Subjects with Abnormal Labs (%)": 0.0, "High Lab Results": 0, "Low Lab Results": 0}
        )

    labs = classify_lab_results(lb_records)
    labs["USUBJID"] = subject_identifiers(labs)
    labs = labs.merge(assignments, on="USUBJID", how="left").fillna({"Treatment Arm": "Not assigned"})
    abnormal = abnormal_results(labs)
    if abnormal.empty:
        return denominator.assign(
            **{"Subjects with Abnormal Labs": 0, "Subjects with Abnormal Labs (%)": 0.0, "High Lab Results": 0, "Low Lab Results": 0}
        )
    counts = abnormal.groupby("Treatment Arm", as_index=False).agg(
        **{
            "Subjects with Abnormal Labs": ("USUBJID", "nunique"),
            "High Lab Results": ("Result Status", lambda values: (values == "High").sum()),
            "Low Lab Results": ("Result Status", lambda values: (values == "Low").sum()),
        }
    )
    result = denominator.merge(counts, on="Treatment Arm", how="left").fillna(0)
    result["Subjects with Abnormal Labs (%)"] = (
        result["Subjects with Abnormal Labs"] / result["Subjects in Arm"]
    ) * 100
    return result


def lab_trend_by_arm(lb_records: pd.DataFrame, assignments: pd.DataFrame, test_name: str) -> pd.DataFrame:
    """Return mean date-stamped laboratory values by treatment arm for one test."""
    labs = classify_lab_results(lb_records)
    labs["USUBJID"] = subject_identifiers(labs)
    labs = labs.merge(assignments, on="USUBJID", how="left").fillna({"Treatment Arm": "Not assigned"})
    filtered = labs.loc[
        labs["Lab Test"].eq(test_name) & labs["Collection Date"].notna() & labs["Numeric Result"].notna()
    ]
    return (
        filtered.groupby(["Treatment Arm", "Collection Date"], as_index=False)["Numeric Result"]
        .mean()
        .sort_values("Collection Date")
    )
