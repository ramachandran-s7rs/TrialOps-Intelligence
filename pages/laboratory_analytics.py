"""Laboratory review workspace for uploaded CDISC SDTM LB datasets."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from database import fetch_lab_alerts, sync_lab_alerts
from utils.lab_analytics import (
    abnormal_results,
    classify_lab_results,
    lab_alert_records,
    laboratory_insights,
    laboratory_summary,
    laboratory_trend_data,
    top_abnormal_tests,
)


def session_datasets() -> dict[str, pd.DataFrame]:
    """Return uploaded detected-domain dataframes from the active session."""
    datasets = st.session_state.get("sdtm_datasets", {})
    return datasets if isinstance(datasets, dict) else {}


def dataframe_signature(dataframe: pd.DataFrame) -> str:
    """Create a stable signature for one uploaded LB dataset revision."""
    signature = sha256()
    signature.update("\x1f".join(map(str, dataframe.columns)).encode("utf-8"))
    signature.update(pd.util.hash_pandas_object(dataframe, index=True).to_numpy().tobytes())
    return signature.hexdigest()


def dark_chart(figure: Any) -> Any:
    """Apply the application's dark Plotly presentation."""
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=12, r=12, t=40, b=12),
        legend_title_text="",
    )
    return figure


def render_laboratory_trends(classified_results: pd.DataFrame) -> None:
    """Render an over-time result plot for a selectable laboratory test."""
    st.subheader("Laboratory Trends")
    trend_candidates = classified_results.loc[
        classified_results["Collection Date"].notna()
        & classified_results["Numeric Result"].notna()
        & classified_results["Lab Test"].ne(""),
        "Lab Test",
    ].drop_duplicates().sort_values().tolist()
    if not trend_candidates:
        st.info("No date-stamped numeric laboratory results are available. Trends require LBDTC and a numeric result value.")
        return

    selected_test = st.selectbox("Laboratory Test", trend_candidates, key="laboratory_trend_test")
    trend = laboratory_trend_data(classified_results, selected_test)
    trend_figure = dark_chart(
        px.line(
            trend,
            x="Collection Date",
            y="Numeric Result",
            color="Result Status",
            markers=True,
            hover_data={
                "Lab Unit": True,
                "Reference Low": True,
                "Reference High": True,
                "Source Row": True,
            },
            title=f"{selected_test} Results Over Time",
        )
    )
    if trend["Reference Low"].notna().any():
        trend_figure.add_scatter(
            x=trend["Collection Date"],
            y=trend["Reference Low"],
            mode="lines",
            name="Reference Low",
            line={"dash": "dash", "color": "#FFB020"},
        )
    if trend["Reference High"].notna().any():
        trend_figure.add_scatter(
            x=trend["Collection Date"],
            y=trend["Reference High"],
            mode="lines",
            name="Reference High",
            line={"dash": "dash", "color": "#FF4B4B"},
        )
    trend_figure.update_layout(xaxis_title=None, yaxis_title="Result")
    st.plotly_chart(trend_figure, use_container_width=True)


def render_laboratory_analytics() -> None:
    """Render classification, review, persistence, and insights for LB data."""
    st.title("Laboratory Analytics")
    st.caption("Review laboratory results against the reference ranges supplied in the uploaded LB dataset.")

    lb_records = session_datasets().get("LB")
    if not isinstance(lb_records, pd.DataFrame):
        st.info("No detected LB dataset is available. Upload laboratory data to begin result classification and review.")
        return

    classified_results = classify_lab_results(lb_records)
    summary = laboratory_summary(classified_results)
    metrics = st.columns(4)
    for container, (label, value) in zip(metrics, summary.items()):
        container.metric(label, value)

    st.subheader("Top Abnormal Laboratory Tests")
    top_tests = top_abnormal_tests(classified_results)
    if top_tests.empty:
        st.info("No laboratory results are outside the supplied reference ranges.")
    else:
        top_tests_figure = dark_chart(
            px.bar(
                top_tests.sort_values("Abnormal Results"),
                x="Abnormal Results",
                y="Laboratory Test",
                orientation="h",
                title="Top Abnormal Laboratory Tests",
            )
        )
        top_tests_figure.update_layout(showlegend=False, xaxis_title="Abnormal Results", yaxis_title=None)
        st.plotly_chart(top_tests_figure, use_container_width=True)

    render_laboratory_trends(classified_results)

    st.subheader("Critical Laboratory Alerts")
    st.caption("Alerts identify results outside the reference ranges supplied in the uploaded LB dataset.")
    alerts = abnormal_results(classified_results)
    alert_columns = [
        "USUBJID",
        "Lab Test",
        "Numeric Result",
        "Lab Unit",
        "Reference Low",
        "Reference High",
        "Result Status",
        "Collection Date",
        "Source Row",
    ]
    if alerts.empty:
        st.success("No laboratory alerts were identified from the supplied reference ranges.")
    else:
        st.dataframe(alerts.reindex(columns=alert_columns), hide_index=True, use_container_width=True)

    sync_column, status_column = st.columns([1, 3])
    with sync_column:
        sync_requested = st.button("Persist Laboratory Alerts", disabled=alerts.empty)
    with status_column:
        persisted_alerts = fetch_lab_alerts()
        st.caption(f"{len(persisted_alerts.index):,} laboratory alert record(s) are retained in SQLite.")
    if sync_requested:
        created, skipped = sync_lab_alerts(lab_alert_records(classified_results), dataframe_signature(lb_records))
        st.success(f"{created} laboratory alert record(s) persisted. {skipped} existing record(s) retained.")

    st.subheader("Laboratory Insights")
    insights = laboratory_insights(classified_results)
    if not insights:
        st.info("Laboratory insights will appear when numeric results and supplied reference ranges are available.")
    else:
        for insight in insights:
            st.info(insight)


render_laboratory_analytics()
