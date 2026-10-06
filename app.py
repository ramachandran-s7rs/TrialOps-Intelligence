"""TrialOps Intelligence application entry point."""

from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).parent

st.set_page_config(
    page_title="TrialOps Intelligence",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = {
    "Trial Operations": [
        st.Page(PROJECT_ROOT / "pages" / "dashboard.py", title="Dashboard", icon="📊", default=True),
        st.Page(PROJECT_ROOT / "pages" / "data_upload.py", title="Data Upload", icon="📤"),
        st.Page(PROJECT_ROOT / "pages" / "validation_engine.py", title="Validation Engine", icon="✅"),
        st.Page(PROJECT_ROOT / "pages" / "query_management.py", title="Query Management", icon="📝"),
    ],
    "Clinical Analytics": [
        st.Page(PROJECT_ROOT / "pages" / "patient_360.py", title="Patient 360", icon="👤"),
        st.Page(PROJECT_ROOT / "pages" / "exposure_efficacy_analytics.py", title="Efficacy Analytics", icon="📈"),
        st.Page(PROJECT_ROOT / "pages" / "statistical_analysis.py", title="Statistical Analysis", icon="📐"),
        st.Page(PROJECT_ROOT / "pages" / "laboratory_analytics.py", title="Laboratory Analytics", icon="🧫"),
        st.Page(PROJECT_ROOT / "pages" / "anomaly_detection.py", title="Anomaly Detection", icon="🚨"),
        st.Page(PROJECT_ROOT / "pages" / "reports.py", title="Reports", icon="📄"),
    ],
}

st.sidebar.title("TrialOps Intelligence")
st.sidebar.caption("Clinical Data Quality and Analytics Platform")

navigation = st.navigation(pages, position="sidebar")
navigation.run()
