"""Upload and inspect CDISC SDTM source datasets for the active session."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError, ParserError
import streamlit as st

from utils.domain_detection import detect_candidate_domains, standardize_column_names

DATASET_STATE_KEY = "sdtm_datasets"
METADATA_STATE_KEY = "sdtm_upload_metadata"
FINGERPRINT_STATE_KEY = "sdtm_upload_fingerprints"
PENDING_UPLOADS_KEY = "sdtm_pending_uploads"
UPLOADER_VERSION_KEY = "sdtm_uploader_version"
ACTIVITY_EVENTS_KEY = "study_activity_events"


def initialise_session_state() -> None:
    """Initialise the session-scoped storage used by the upload workspace."""
    st.session_state.setdefault(DATASET_STATE_KEY, {})
    st.session_state.setdefault(METADATA_STATE_KEY, {})
    st.session_state.setdefault(FINGERPRINT_STATE_KEY, {})
    st.session_state.setdefault(PENDING_UPLOADS_KEY, {})
    st.session_state.setdefault(UPLOADER_VERSION_KEY, 0)
    st.session_state.setdefault(ACTIVITY_EVENTS_KEY, [])


def utc_timestamp() -> str:
    """Return a consistently formatted UTC timestamp for study activity."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def record_activity(event: str, detail: str, timestamp: str | None = None) -> None:
    """Keep a compact in-session study activity feed for the dashboard."""
    activity_events = st.session_state.setdefault(ACTIVITY_EVENTS_KEY, [])
    activity_events.append(
        {
            "Event": event,
            "Detail": detail,
            "Timestamp": timestamp or utc_timestamp(),
        }
    )
    st.session_state[ACTIVITY_EVENTS_KEY] = activity_events[-100:]


def read_csv_dataset(file_content: bytes) -> pd.DataFrame:
    """Read a CSV payload with common clinical-data encodings.

    A parsing issue is surfaced to the user rather than silently skipping rows,
    which is essential when an uploaded file will later drive data-quality work.
    """
    try:
        return pd.read_csv(BytesIO(file_content), encoding="utf-8-sig")
    except UnicodeDecodeError:
        # Latin-1 is a safe fallback for legacy CSV exports and decodes every
        # byte sequence without dropping or transforming source rows.
        return pd.read_csv(BytesIO(file_content), encoding="latin-1")


def reset_upload_workspace() -> None:
    """Clear session-scoped datasets and create a fresh uploader widget."""
    st.session_state[DATASET_STATE_KEY] = {}
    st.session_state[METADATA_STATE_KEY] = {}
    st.session_state[FINGERPRINT_STATE_KEY] = {}
    st.session_state[PENDING_UPLOADS_KEY] = {}
    st.session_state[UPLOADER_VERSION_KEY] += 1


def store_dataset(
    domain: str,
    dataframe: pd.DataFrame,
    filename: str,
    file_fingerprint: str,
) -> str | None:
    """Store a detected domain dataset; filenames remain display metadata only."""
    datasets: dict[str, pd.DataFrame] = st.session_state[DATASET_STATE_KEY]
    metadata: dict[str, dict[str, str | int]] = st.session_state[METADATA_STATE_KEY]
    fingerprints: dict[str, str] = st.session_state[FINGERPRINT_STATE_KEY]

    if fingerprints.get(domain) == file_fingerprint:
        return None

    action = "Updated" if domain in datasets else "Loaded"
    uploaded_at = utc_timestamp()
    datasets[domain] = dataframe
    metadata[domain] = {
        "original_filename": filename,
        "detected_domain": domain,
        "records": len(dataframe.index),
        "columns": len(dataframe.columns),
        "uploaded_at": uploaded_at,
    }
    fingerprints[domain] = file_fingerprint
    record_activity(
        "Dataset Uploaded",
        f"{action} {domain} dataset with {len(dataframe.index)} record(s).",
        uploaded_at,
    )
    return f"{action} {domain} from {filename}"


def process_uploaded_files(uploaded_files: list[Any]) -> tuple[list[str], list[str]]:
    """Parse, standardise, and automatically store uniquely detected datasets."""
    imported_datasets: list[str] = []
    messages: list[str] = []
    pending_uploads: dict[str, dict[str, Any]] = st.session_state[PENDING_UPLOADS_KEY]

    for uploaded_file in uploaded_files:
        file_content = uploaded_file.getvalue()
        file_fingerprint = sha256(file_content).hexdigest()
        upload_record = pending_uploads.get(file_fingerprint)

        if upload_record is None:
            try:
                dataframe = standardize_column_names(read_csv_dataset(file_content))
            except EmptyDataError:
                messages.append(f"{uploaded_file.name}: skipped because the CSV contains no header or records.")
                continue
            except ParserError:
                messages.append(f"{uploaded_file.name}: skipped because the CSV structure could not be parsed.")
                continue
            except UnicodeDecodeError:
                messages.append(f"{uploaded_file.name}: skipped because its text encoding is not supported.")
                continue
            except ValueError as error:
                messages.append(f"{uploaded_file.name}: skipped. {error}")
                continue

            upload_record = {
                "filename": uploaded_file.name,
                "dataframe": dataframe,
                "candidates": detect_candidate_domains(dataframe),
            }
            pending_uploads[file_fingerprint] = upload_record

        candidates = upload_record["candidates"]
        if not candidates:
            messages.append(
                f"{upload_record['filename']}: domain could not be detected from the required SDTM column signature."
            )
            continue
        if len(candidates) > 1:
            continue

        result = store_dataset(candidates[0], upload_record["dataframe"], upload_record["filename"], file_fingerprint)
        if result is not None:
            imported_datasets.append(result)

    return imported_datasets, messages


def render_ambiguous_uploads() -> list[str]:
    """Offer explicit domain selection for datasets with more than one match."""
    imported_datasets: list[str] = []
    pending_uploads: dict[str, dict[str, Any]] = st.session_state[PENDING_UPLOADS_KEY]

    for file_fingerprint, upload_record in pending_uploads.items():
        candidates: list[str] = upload_record["candidates"]
        if len(candidates) <= 1:
            continue

        st.warning(
            f"{upload_record['filename']}: multiple SDTM domains match the uploaded columns ({', '.join(candidates)}). "
            "Choose the intended domain before import."
        )
        selected_domain = st.selectbox(
            f"Detected domain for {upload_record['filename']}",
            candidates,
            key=f"domain_selection_{file_fingerprint}",
        )
        if st.button(f"Import as {selected_domain}", key=f"domain_import_{file_fingerprint}"):
            result = store_dataset(
                selected_domain,
                upload_record["dataframe"],
                upload_record["filename"],
                file_fingerprint,
            )
            if result is not None:
                imported_datasets.append(result)

    return imported_datasets


def render_dataset_summary(dataset_name: str, dataframe: pd.DataFrame, metadata: dict[str, str | int]) -> None:
    """Render an accessible inspection view for one uploaded SDTM dataset."""
    filename = str(metadata["original_filename"])
    detected_domain = str(metadata["detected_domain"])
    records = int(metadata["records"])
    columns = int(metadata["columns"])

    with st.expander(f"{dataset_name} — {filename}", expanded=True):
        filename_metric, domain_metric, record_metric, column_metric = st.columns(4)
        filename_metric.metric("Original filename", filename)
        domain_metric.metric("Detected domain", detected_domain)
        record_metric.metric("Records", f"{records:,}")
        column_metric.metric("Columns", columns)

        st.subheader("Column names")
        st.dataframe(
            pd.DataFrame({"Column name": dataframe.columns.astype(str)}),
            hide_index=True,
            use_container_width=True,
        )

        st.subheader("First 10 rows")
        st.dataframe(dataframe.head(10), hide_index=True, use_container_width=True)


initialise_session_state()

st.title("Data Upload")
st.caption("Upload CSV files. SDTM and efficacy domains are detected from standardized dataset columns, not filenames.")

reset_column, _ = st.columns([1, 4])
with reset_column:
    st.button("Clear uploaded data", on_click=reset_upload_workspace)

uploader_key = f"sdtm_csv_uploader_{st.session_state[UPLOADER_VERSION_KEY]}"
uploaded_files = st.file_uploader(
    "Select SDTM CSV files",
    type=["csv"],
    accept_multiple_files=True,
    key=uploader_key,
    help="Supported domains are DM, AE, LB, VS, EX, and EFF. Filenames are retained only as metadata.",
)

if uploaded_files:
    imported_datasets, upload_messages = process_uploaded_files(uploaded_files)
    if imported_datasets:
        st.success("; ".join(imported_datasets))
    for message in upload_messages:
        st.warning(message)

manual_imports = render_ambiguous_uploads()
if manual_imports:
    st.success("; ".join(manual_imports))

datasets = st.session_state[DATASET_STATE_KEY]
metadata = st.session_state[METADATA_STATE_KEY]

if not datasets:
    st.info("No SDTM datasets are available in this clinical review session. Upload source CSV files to begin data-quality review.")
else:
    st.divider()
    st.subheader("Uploaded datasets")
    st.caption("Datasets remain available while this Streamlit session is active.")

    for dataset_name in sorted(datasets):
        render_dataset_summary(dataset_name, datasets[dataset_name], metadata[dataset_name])
