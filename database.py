"""SQLite persistence services for TrialOps Intelligence."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Any, Iterable

import pandas as pd


DATABASE_PATH = Path(__file__).parent / "data" / "trialops_intelligence.db"
QUERY_STATUSES = ("Open", "Answered", "Closed")
ACTIVE_QUERY_STATUSES = ("Open", "Answered")
QUERY_EXPORT_COLUMNS = [
    "Query ID",
    "Subject ID",
    "Dataset",
    "Source Row",
    "Field",
    "Rule",
    "Severity",
    "Status",
    "Response",
    "Resolution Note",
    "Created At",
    "Updated At",
    "Closed At",
]
LAB_ALERT_COLUMNS = [
    "Alert ID",
    "Subject ID",
    "Lab Test",
    "Lab Test Code",
    "Result Value",
    "Result Unit",
    "Reference Low",
    "Reference High",
    "Result Status",
    "Collection Date",
    "Source Row",
    "Created At",
]


def get_connection() -> sqlite3.Connection:
    """Return a configured SQLite connection."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def _utc_timestamp() -> str:
    """Return a timezone-aware, sortable timestamp for persisted records."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalise_text(value: Any) -> str:
    """Convert nullable source values to stable SQLite text values."""
    return "" if value is None or pd.isna(value) else str(value).strip()


def _normalise_source_row(value: Any) -> int:
    """Convert nullable source row values to a stable integer identity component."""
    if value is None or pd.isna(value):
        return 0
    return int(value)


def _issue_type(rule: str) -> str:
    """Extract an issue category from a validation rule label."""
    return rule.split(" - ", maxsplit=1)[-1]


def _create_query_table(connection: sqlite3.Connection, table_name: str = "queries") -> None:
    """Create the current query table schema."""
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id TEXT UNIQUE,
            subject_id TEXT NOT NULL DEFAULT '',
            dataset TEXT NOT NULL,
            source_row INTEGER NOT NULL DEFAULT 0,
            field TEXT NOT NULL,
            rule TEXT NOT NULL,
            issue_type TEXT NOT NULL,
            description TEXT NOT NULL,
            severity TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Open' CHECK (status IN ('Open', 'Answered', 'Closed')),
            response TEXT NOT NULL DEFAULT '',
            resolution_note TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            closed_at TEXT
        )
        """
    )


def _migrate_legacy_queries(connection: sqlite3.Connection) -> None:
    """Replace the Phase 1 table when it lacks the closure timestamp column."""
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(queries)")}
    if not columns or "closed_at" in columns:
        return

    _create_query_table(connection, "queries_migrated")
    connection.execute(
        """
        INSERT INTO queries_migrated (
            id, query_id, subject_id, dataset, source_row, field, rule,
            issue_type, description, severity, status, response,
            resolution_note, created_at, updated_at, closed_at
        )
        SELECT
            id, query_id, subject_id, dataset, source_row, field, rule,
            issue_type, description, severity, status, response,
            resolution_note, created_at, updated_at,
            CASE WHEN status = 'Closed' THEN updated_at ELSE NULL END
        FROM queries
        """
    )
    connection.execute("DROP TABLE queries")
    connection.execute("ALTER TABLE queries_migrated RENAME TO queries")


def initialise_database() -> None:
    """Create and migrate persistent query and audit stores."""
    with get_connection() as connection:
        _create_query_table(connection)
        _migrate_legacy_queries(connection)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS query_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id TEXT NOT NULL,
                action TEXT NOT NULL CHECK (action IN ('Created', 'Updated', 'Closed', 'Auto Closed')),
                old_status TEXT,
                new_status TEXT,
                timestamp TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS lab_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id TEXT NOT NULL DEFAULT '',
                lab_test TEXT NOT NULL DEFAULT '',
                lab_test_code TEXT NOT NULL DEFAULT '',
                result_value TEXT NOT NULL DEFAULT '',
                result_unit TEXT NOT NULL DEFAULT '',
                reference_low REAL,
                reference_high REAL,
                result_status TEXT NOT NULL CHECK (result_status IN ('High', 'Low')),
                collection_date TEXT NOT NULL DEFAULT '',
                source_row INTEGER NOT NULL,
                source_signature TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(source_signature, source_row, result_status)
            )
            """
        )
        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS ux_queries_active_identity
            ON queries (dataset, subject_id, source_row, field, rule)
            WHERE status IN ('Open', 'Answered')
            """
        )
        connection.execute("CREATE INDEX IF NOT EXISTS ix_query_audit_query_id ON query_audit (query_id)")
        connection.execute("CREATE INDEX IF NOT EXISTS ix_lab_alerts_subject_id ON lab_alerts (subject_id)")


def _record_audit(
    connection: sqlite3.Connection,
    query_id: str,
    action: str,
    old_status: str | None,
    new_status: str | None,
    timestamp: str | None = None,
) -> None:
    """Append an immutable lifecycle event for a persisted query."""
    connection.execute(
        """
        INSERT INTO query_audit (query_id, action, old_status, new_status, timestamp)
        VALUES (?, ?, ?, ?, ?)
        """,
        (query_id, action, old_status, new_status, timestamp or _utc_timestamp()),
    )


def _finding_identity(finding: dict[str, Any]) -> tuple[str, str, int, str, str]:
    """Return the normalized identity used to match a finding to a query."""
    return (
        _normalise_text(finding.get("Dataset")).upper(),
        _normalise_text(finding.get("USUBJID")),
        _normalise_source_row(finding.get("Source row")),
        _normalise_text(finding.get("Field")),
        _normalise_text(finding.get("Rule")),
    )


def create_queries_from_findings(findings: Iterable[dict[str, Any]]) -> tuple[int, int]:
    """Create one active query per current finding and reuse existing active queries."""
    initialise_database()
    created = 0
    skipped = 0
    timestamp = _utc_timestamp()

    with get_connection() as connection:
        for finding in findings:
            dataset, subject_id, source_row, field, rule = _finding_identity(finding)
            existing = connection.execute(
                """
                SELECT query_id FROM queries
                WHERE dataset = ? AND subject_id = ? AND source_row = ? AND field = ? AND rule = ?
                  AND status IN ('Open', 'Answered')
                """,
                (dataset, subject_id, source_row, field, rule),
            ).fetchone()
            if existing is not None:
                skipped += 1
                continue

            message = _normalise_text(finding.get("Message"))
            description = f"{message}\nPlease verify source data and update if appropriate."
            cursor = connection.execute(
                """
                INSERT INTO queries (
                    subject_id, dataset, source_row, field, rule, issue_type,
                    description, severity, status, response, resolution_note,
                    created_at, updated_at, closed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'Open', '', '', ?, ?, NULL)
                """,
                (
                    subject_id,
                    dataset,
                    source_row,
                    field,
                    rule,
                    _issue_type(rule),
                    description,
                    _normalise_text(finding.get("Severity")),
                    timestamp,
                    timestamp,
                ),
            )
            query_id = f"QRY-{cursor.lastrowid:06d}"
            connection.execute("UPDATE queries SET query_id = ? WHERE id = ?", (query_id, cursor.lastrowid))
            _record_audit(connection, query_id, "Created", None, "Open", timestamp)
            created += 1

    return created, skipped


def auto_close_resolved_queries(
    findings: Iterable[dict[str, Any]],
    validated_domains: Iterable[str],
) -> int:
    """Close active queries that no longer appear in the latest validation run.

    Only queries from domains validated in the current run are considered. This
    prevents a partial dataset upload from closing unrelated open queries.
    """
    initialise_database()
    active_finding_identities = {_finding_identity(finding) for finding in findings}
    domains = {str(domain).upper() for domain in validated_domains}
    if not domains:
        return 0

    placeholders = ", ".join("?" for _ in domains)
    closed_count = 0
    timestamp = _utc_timestamp()
    resolution_note = "Issue no longer detected in latest validation run."

    with get_connection() as connection:
        active_queries = connection.execute(
            f"""
            SELECT query_id, dataset, subject_id, source_row, field, rule, status
            FROM queries
            WHERE status IN ('Open', 'Answered') AND dataset IN ({placeholders})
            """,
            tuple(sorted(domains)),
        ).fetchall()

        for query in active_queries:
            identity = (
                query["dataset"],
                query["subject_id"],
                query["source_row"],
                query["field"],
                query["rule"],
            )
            if identity in active_finding_identities:
                continue

            old_status = query["status"]
            connection.execute(
                """
                UPDATE queries
                SET status = 'Closed', resolution_note = ?, closed_at = ?, updated_at = ?
                WHERE query_id = ?
                """,
                (resolution_note, timestamp, timestamp, query["query_id"]),
            )
            _record_audit(connection, query["query_id"], "Auto Closed", old_status, "Closed", timestamp)
            closed_count += 1

    return closed_count


def fetch_queries() -> pd.DataFrame:
    """Return persistent queries in newest-first order."""
    initialise_database()
    with get_connection() as connection:
        return pd.read_sql_query(
            """
            SELECT
                query_id AS 'Query ID',
                subject_id AS 'Subject ID',
                dataset AS 'Dataset',
                source_row AS 'Source Row',
                field AS 'Field',
                rule AS 'Rule',
                issue_type AS 'Issue Type',
                description AS 'Description',
                severity AS 'Severity',
                status AS 'Status',
                response AS 'Response',
                resolution_note AS 'Resolution Note',
                created_at AS 'Created At',
                updated_at AS 'Updated At',
                closed_at AS 'Closed At'
            FROM queries
            ORDER BY id DESC
            """,
            connection,
        )


def fetch_audit_history() -> pd.DataFrame:
    """Return persistent query lifecycle events in newest-first order."""
    initialise_database()
    with get_connection() as connection:
        return pd.read_sql_query(
            """
            SELECT
                query_id AS 'Query ID',
                action AS 'Action',
                old_status AS 'Old Status',
                new_status AS 'New Status',
                timestamp AS 'Timestamp'
            FROM query_audit
            ORDER BY id DESC
            """,
            connection,
        )


def _normalise_optional_number(value: Any) -> float | None:
    """Return a nullable SQLite number from a pandas-compatible scalar."""
    if value is None or pd.isna(value):
        return None
    return float(value)


def sync_lab_alerts(alerts: Iterable[dict[str, Any]], source_signature: str) -> tuple[int, int]:
    """Persist abnormal laboratory results without duplicating the same LB source row."""
    initialise_database()
    created = 0
    skipped = 0
    timestamp = _utc_timestamp()

    with get_connection() as connection:
        for alert in alerts:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO lab_alerts (
                    subject_id, lab_test, lab_test_code, result_value, result_unit,
                    reference_low, reference_high, result_status, collection_date,
                    source_row, source_signature, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _normalise_text(alert.get("Subject ID")),
                    _normalise_text(alert.get("Lab Test")),
                    _normalise_text(alert.get("Lab Test Code")),
                    _normalise_text(alert.get("Result Value")),
                    _normalise_text(alert.get("Result Unit")),
                    _normalise_optional_number(alert.get("Reference Low")),
                    _normalise_optional_number(alert.get("Reference High")),
                    _normalise_text(alert.get("Result Status")),
                    _normalise_text(alert.get("Collection Date")),
                    _normalise_source_row(alert.get("Source Row")),
                    source_signature,
                    timestamp,
                ),
            )
            if cursor.rowcount:
                created += 1
            else:
                skipped += 1
    return created, skipped


def fetch_lab_alerts() -> pd.DataFrame:
    """Return persisted laboratory alerts in newest-first order."""
    initialise_database()
    with get_connection() as connection:
        return pd.read_sql_query(
            """
            SELECT
                id AS 'Alert ID',
                subject_id AS 'Subject ID',
                lab_test AS 'Lab Test',
                lab_test_code AS 'Lab Test Code',
                result_value AS 'Result Value',
                result_unit AS 'Result Unit',
                reference_low AS 'Reference Low',
                reference_high AS 'Reference High',
                result_status AS 'Result Status',
                collection_date AS 'Collection Date',
                source_row AS 'Source Row',
                created_at AS 'Created At'
            FROM lab_alerts
            ORDER BY id DESC
            """,
            connection,
        )


def update_query(query_id: str, status: str, response: str, resolution_note: str) -> None:
    """Persist a query response, lifecycle status, and resolution note."""
    if status not in QUERY_STATUSES:
        raise ValueError("Status must be Open, Answered, or Closed.")

    initialise_database()
    timestamp = _utc_timestamp()
    with get_connection() as connection:
        existing = connection.execute(
            """
            SELECT status, dataset, subject_id, source_row, field, rule
            FROM queries WHERE query_id = ?
            """,
            (query_id,),
        ).fetchone()
        if existing is None:
            raise ValueError(f"Query {query_id} does not exist.")

        old_status = existing["status"]
        if status in ACTIVE_QUERY_STATUSES and old_status == "Closed":
            active_match = connection.execute(
                """
                SELECT query_id FROM queries
                WHERE dataset = ? AND subject_id = ? AND source_row = ? AND field = ? AND rule = ?
                  AND status IN ('Open', 'Answered') AND query_id != ?
                """,
                (
                    existing["dataset"],
                    existing["subject_id"],
                    existing["source_row"],
                    existing["field"],
                    existing["rule"],
                    query_id,
                ),
            ).fetchone()
            if active_match is not None:
                raise ValueError("This issue already has an active reissued query and cannot be reopened.")

        closed_at = timestamp if status == "Closed" and old_status != "Closed" else None
        connection.execute(
            """
            UPDATE queries
            SET status = ?, response = ?, resolution_note = ?,
                closed_at = CASE WHEN ? = 'Closed' THEN COALESCE(closed_at, ?) ELSE NULL END,
                updated_at = ?
            WHERE query_id = ?
            """,
            (status, response.strip(), resolution_note.strip(), status, closed_at, timestamp, query_id),
        )
        action = "Closed" if status == "Closed" and old_status != "Closed" else "Updated"
        _record_audit(connection, query_id, action, old_status, status, timestamp)


def clear_queries() -> None:
    """Remove all query and audit records for an explicit reset operation."""
    initialise_database()
    with get_connection() as connection:
        connection.execute("DELETE FROM query_audit")
        connection.execute("DELETE FROM queries")
        try:
            connection.execute("DELETE FROM sqlite_sequence WHERE name IN ('queries', 'query_audit')")
        except sqlite3.OperationalError:
            pass


def clear_lab_alerts() -> None:
    """Remove persisted laboratory alerts during an explicit environment reset."""
    initialise_database()
    with get_connection() as connection:
        connection.execute("DELETE FROM lab_alerts")
        try:
            connection.execute("DELETE FROM sqlite_sequence WHERE name = 'lab_alerts'")
        except sqlite3.OperationalError:
            pass
