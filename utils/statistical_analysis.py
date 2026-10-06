"""Reusable exploratory statistical analysis for uploaded clinical datasets."""

from __future__ import annotations

from math import sqrt
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from utils.treatment_efficacy_analytics import (
    ae_safety_by_arm,
    efficacy_with_arms,
    lab_safety_by_arm,
)


def endpoint_records(
    derived_efficacy: pd.DataFrame,
    assignments: pd.DataFrame,
    endpoint: str,
    improvement_direction: str,
) -> pd.DataFrame:
    """Return analysis-ready subject outcomes for an endpoint and direction of benefit."""
    records = efficacy_with_arms(derived_efficacy, assignments)
    records = records.loc[records["Endpoint"].eq(endpoint)].copy()
    records["Improvement"] = pd.to_numeric(records["Change"], errors="coerce")
    if improvement_direction == "Lower values indicate improvement":
        records["Improvement"] = -records["Improvement"]
    return records.loc[records["Improvement"].notna() & records["Treatment Arm"].notna()].copy()


def _confidence_interval(values: pd.Series) -> tuple[float | None, float | None, float | None]:
    """Return mean, standard error, and t-based 95% CI for a numeric sample."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    sample_size = len(numeric.index)
    if not sample_size:
        return None, None, None
    mean = float(numeric.mean())
    if sample_size == 1:
        return mean, None, None
    standard_error = float(numeric.std(ddof=1) / sqrt(sample_size))
    margin = float(stats.t.ppf(0.975, sample_size - 1) * standard_error)
    return mean, standard_error, margin


def descriptive_statistics(records: pd.DataFrame) -> pd.DataFrame:
    """Return transparent, per-arm descriptive results for mean improvement."""
    rows: list[dict[str, Any]] = []
    for arm, group in records.groupby("Treatment Arm", dropna=False, sort=True):
        values = pd.to_numeric(group["Improvement"], errors="coerce").dropna()
        mean, standard_error, margin = _confidence_interval(values)
        rows.append(
            {
                "Treatment Arm": str(arm),
                "N": int(len(values.index)),
                "Mean Improvement": mean,
                "Median Improvement": None if values.empty else float(values.median()),
                "Standard Deviation": None if len(values.index) < 2 else float(values.std(ddof=1)),
                "Standard Error": standard_error,
                "95% CI Lower": None if margin is None or mean is None else mean - margin,
                "95% CI Upper": None if margin is None or mean is None else mean + margin,
            }
        )
    return pd.DataFrame(rows)


def cohen_d(first: pd.Series, second: pd.Series) -> float | None:
    """Calculate Cohen's d using a pooled standard deviation for two groups."""
    first_values = pd.to_numeric(first, errors="coerce").dropna()
    second_values = pd.to_numeric(second, errors="coerce").dropna()
    first_count, second_count = len(first_values.index), len(second_values.index)
    if first_count < 2 or second_count < 2:
        return None
    pooled_variance = (
        ((first_count - 1) * first_values.var(ddof=1)) + ((second_count - 1) * second_values.var(ddof=1))
    ) / (first_count + second_count - 2)
    if not pooled_variance or pd.isna(pooled_variance):
        return None
    return float((first_values.mean() - second_values.mean()) / sqrt(pooled_variance))


def effect_interpretation(value: float | None, metric: str) -> str:
    """Classify an absolute standardized effect using conventional thresholds."""
    if value is None or pd.isna(value):
        return "Not available"
    magnitude = abs(float(value))
    thresholds = (0.01, 0.06, 0.14) if metric == "Eta Squared" else (0.2, 0.5, 0.8)
    if magnitude < thresholds[0]:
        return "Negligible"
    if magnitude < thresholds[1]:
        return "Small"
    if magnitude < thresholds[2]:
        return "Medium"
    return "Large"


def global_statistical_test(records: pd.DataFrame) -> dict[str, Any]:
    """Choose a two-group t-test or multi-group ANOVA from available arms."""
    groups = {
        str(arm): pd.to_numeric(group["Improvement"], errors="coerce").dropna()
        for arm, group in records.groupby("Treatment Arm", sort=True)
    }
    groups = {arm: values for arm, values in groups.items() if len(values.index) >= 2}
    if len(groups) < 2:
        return {
            "Test Name": "Not available",
            "Test Statistic": None,
            "P-value": None,
            "Significance Interpretation": "At least two treatment arms with two evaluable subjects each are required.",
            "Effect Metric": "Not available",
            "Effect Size": None,
            "Effect Interpretation": "Not available",
        }

    group_names = list(groups)
    group_values = list(groups.values())
    if len(groups) == 2:
        result = stats.ttest_ind(group_values[0], group_values[1], equal_var=False, nan_policy="omit")
        effect = cohen_d(group_values[0], group_values[1])
        test_name = "Independent two-sample t-test (Welch)"
        effect_metric = "Cohen's d"
    else:
        result = stats.f_oneway(*group_values)
        all_values = pd.concat(group_values, ignore_index=True)
        grand_mean = float(all_values.mean())
        between_sum_squares = sum(len(values.index) * (float(values.mean()) - grand_mean) ** 2 for values in group_values)
        total_sum_squares = float(((all_values - grand_mean) ** 2).sum())
        effect = None if total_sum_squares == 0 else float(between_sum_squares / total_sum_squares)
        test_name = "One-way ANOVA"
        effect_metric = "Eta Squared"

    p_value = float(result.pvalue) if not pd.isna(result.pvalue) else None
    return {
        "Test Name": test_name,
        "Test Statistic": float(result.statistic),
        "P-value": p_value,
        "Significance Interpretation": (
            "Statistically Significant (p < 0.05)" if p_value is not None and p_value < 0.05 else "Not Statistically Significant (p >= 0.05)"
        ),
        "Effect Metric": effect_metric,
        "Effect Size": effect,
        "Effect Interpretation": effect_interpretation(effect, effect_metric),
        "Groups": group_names,
    }


def placebo_comparisons(records: pd.DataFrame, placebo_arm: str) -> pd.DataFrame:
    """Return descriptive mean-improvement differences and 95% CIs versus placebo."""
    placebo = records.loc[records["Treatment Arm"].eq(placebo_arm), "Improvement"]
    placebo = pd.to_numeric(placebo, errors="coerce").dropna()
    rows: list[dict[str, Any]] = []
    for arm, group in records.groupby("Treatment Arm", sort=True):
        values = pd.to_numeric(group["Improvement"], errors="coerce").dropna()
        if str(arm) == placebo_arm:
            rows.append(
                {
                    "Treatment Arm": str(arm),
                    "Mean Difference vs Placebo": 0.0,
                    "95% CI Lower": 0.0,
                    "95% CI Upper": 0.0,
                    "Comparison Status": "Placebo reference",
                }
            )
            continue
        if len(values.index) < 2 or len(placebo.index) < 2:
            rows.append(
                {
                    "Treatment Arm": str(arm),
                    "Mean Difference vs Placebo": None,
                    "95% CI Lower": None,
                    "95% CI Upper": None,
                    "Comparison Status": "Insufficient observations",
                }
            )
            continue
        variance_treatment = float(values.var(ddof=1))
        variance_placebo = float(placebo.var(ddof=1))
        standard_error = sqrt((variance_treatment / len(values.index)) + (variance_placebo / len(placebo.index)))
        numerator = ((variance_treatment / len(values.index)) + (variance_placebo / len(placebo.index))) ** 2
        denominator = ((variance_treatment / len(values.index)) ** 2 / (len(values.index) - 1)) + (
            (variance_placebo / len(placebo.index)) ** 2 / (len(placebo.index) - 1)
        )
        degrees_freedom = numerator / denominator if denominator else min(len(values.index), len(placebo.index)) - 1
        difference = float(values.mean() - placebo.mean())
        margin = float(stats.t.ppf(0.975, degrees_freedom) * standard_error)
        rows.append(
            {
                "Treatment Arm": str(arm),
                "Mean Difference vs Placebo": difference,
                "95% CI Lower": difference - margin,
                "95% CI Upper": difference + margin,
                "Comparison Status": "Descriptive comparison",
            }
        )
    return pd.DataFrame(rows)


def responder_analysis(records: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Summarize responders with improvement at or above the user-defined threshold."""
    rows: list[dict[str, Any]] = []
    for arm, group in records.groupby("Treatment Arm", sort=True):
        values = pd.to_numeric(group["Improvement"], errors="coerce").dropna()
        responders = int(values.ge(threshold).sum())
        total = int(len(values.index))
        rows.append(
            {
                "Treatment Arm": str(arm),
                "Responders": responders,
                "Non-Responders": total - responders,
                "Evaluable Subjects": total,
                "Response Rate (%)": None if not total else (responders / total) * 100,
            }
        )
    return pd.DataFrame(rows)


def safety_profile(
    assignments: pd.DataFrame, ae_records: pd.DataFrame | None, lb_records: pd.DataFrame | None
) -> pd.DataFrame:
    """Build an explicitly descriptive treatment-arm safety profile from AE and LB data."""
    if assignments.empty:
        return pd.DataFrame()
    profiles = assignments.groupby("Treatment Arm", as_index=False).agg(**{"Subjects in Arm": ("USUBJID", "nunique")})
    if ae_records is None:
        profiles["Subjects with AEs (%)"] = np.nan
        profiles["Subjects with Serious AEs (%)"] = np.nan
    else:
        ae_profile = ae_safety_by_arm(ae_records, assignments).reindex(
            columns=["Treatment Arm", "Subjects with AEs (%)", "Subjects with Serious AEs (%)"]
        )
        profiles = profiles.merge(ae_profile, on="Treatment Arm", how="left")
    if lb_records is None:
        profiles["Subjects with Abnormal Labs (%)"] = np.nan
    else:
        lab_profile = lab_safety_by_arm(lb_records, assignments).reindex(
            columns=["Treatment Arm", "Subjects with Abnormal Labs (%)"]
        )
        profiles = profiles.merge(lab_profile, on="Treatment Arm", how="left")

    components = profiles[["Subjects with AEs (%)", "Subjects with Serious AEs (%)", "Subjects with Abnormal Labs (%)"]].copy()
    components["Subjects with Serious AEs (%)"] = components["Subjects with Serious AEs (%)"] * 2
    profiles["Safety Risk Score"] = components.mean(axis=1, skipna=True)
    profiles.loc[components.isna().all(axis=1), "Safety Risk Score"] = np.nan
    profiles["Safety Profile"] = profiles["Safety Risk Score"].map(
        lambda value: "Not available" if pd.isna(value) else "Low" if value < 25 else "Moderate" if value < 50 else "High"
    )
    return profiles


def treatment_leaderboard(
    descriptives: pd.DataFrame, responders: pd.DataFrame, safety: pd.DataFrame
) -> pd.DataFrame:
    """Rank arms by observed improvement, response rate, and transparent safety profile."""
    leaderboard = descriptives.reindex(columns=["Treatment Arm", "Mean Improvement", "N"]).merge(
        responders.reindex(columns=["Treatment Arm", "Response Rate (%)"]), on="Treatment Arm", how="left"
    )
    if not safety.empty:
        leaderboard = leaderboard.merge(
            safety.reindex(columns=["Treatment Arm", "Safety Risk Score", "Safety Profile"]),
            on="Treatment Arm",
            how="left",
        )
    else:
        leaderboard["Safety Risk Score"] = np.nan
        leaderboard["Safety Profile"] = "Not available"
    leaderboard["Improvement Rank"] = leaderboard["Mean Improvement"].rank(method="min", ascending=False).astype("Int64")
    leaderboard["Response Rate Rank"] = leaderboard["Response Rate (%)"].rank(method="min", ascending=False).astype("Int64")
    leaderboard["Safety Rank"] = leaderboard["Safety Risk Score"].rank(method="min", ascending=True).astype("Int64")
    return leaderboard.sort_values(["Improvement Rank", "Safety Rank"], na_position="last").reset_index(drop=True)


def safety_efficacy_matrix(leaderboard: pd.DataFrame) -> tuple[pd.DataFrame, float | None, float | None]:
    """Classify arms relative to observed median efficacy and safety risk scores."""
    matrix = leaderboard.copy()
    if matrix.empty:
        return matrix, None, None
    efficacy_threshold = float(matrix["Mean Improvement"].median())
    available_risk = matrix["Safety Risk Score"].dropna()
    risk_threshold = None if available_risk.empty else float(available_risk.median())
    high_efficacy = matrix["Mean Improvement"].ge(efficacy_threshold)
    high_risk = pd.Series(False, index=matrix.index) if risk_threshold is None else matrix["Safety Risk Score"].ge(risk_threshold)
    matrix["Matrix Classification"] = np.select(
        [high_efficacy & ~high_risk, high_efficacy & high_risk, ~high_efficacy & ~high_risk, ~high_efficacy & high_risk],
        ["High Efficacy / Low Risk", "High Efficacy / High Risk", "Low Efficacy / Low Risk", "Low Efficacy / High Risk"],
        default="Safety data unavailable",
    )
    return matrix, efficacy_threshold, risk_threshold


def clinical_interpretation(
    test_result: dict[str, Any],
    placebo_arm: str,
    comparisons: pd.DataFrame,
) -> list[str]:
    """Generate conservative, deterministic observations for clinical-review use."""
    messages: list[str] = []
    p_value = test_result.get("P-value")
    if p_value is None:
        return ["Statistical interpretation is unavailable because the selected endpoint does not have sufficient evaluable observations across treatment arms."]
    outcome = "statistically significant" if p_value < 0.05 else "not statistically significant"
    messages.append(
        f"The exploratory {test_result['Test Name']} was {outcome} (p={p_value:.4f})."
    )
    if test_result.get("Effect Size") is not None:
        messages.append(
            f"The observed {test_result['Effect Metric']} was {test_result['Effect Size']:.3f}, conventionally interpreted as {test_result['Effect Interpretation'].lower()}."
        )
    active_comparisons = comparisons.loc[comparisons["Treatment Arm"].ne(placebo_arm)].dropna(
        subset=["Mean Difference vs Placebo"]
    )
    if not active_comparisons.empty:
        best = active_comparisons.sort_values("Mean Difference vs Placebo", ascending=False).iloc[0]
        messages.append(
            f"{best['Treatment Arm']} had an observed mean-improvement difference of {best['Mean Difference vs Placebo']:.2f} versus {placebo_arm}."
        )
    messages.append(
        "These exploratory results do not establish clinical benefit, safety, or regulatory approval. Confirmatory analysis and clinical review are required."
    )
    return messages
