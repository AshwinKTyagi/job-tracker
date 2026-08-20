"""Tests for connection handling and migrations (store/db.py)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from jobtrack.errors import StoreError
from jobtrack.store import SCHEMA_VERSION, Store

EXPECTED_TABLES = frozenset(
    {
        "messages",
        "classifications",
        "applications",
        "events",
        "overrides",
        "sync_state",
        "schema_version",
    }
)

EXPECTED_INDEXES = frozenset({"idx_apps_company_key", "idx_events_app", "idx_cls_review"})


def test_open_creates_parent_dirs(tmp_path: Path) -> None:
    """Store.open creates JOBTRACK_HOME-style nested directories on demand."""
    nested = tmp_path / "a" / "b" / "jobtrack.db"
    with Store.open(nested) as store:
        store.migrate()
    assert nested.is_file()


def test_migrate_creates_expected_schema(db_path: Path) -> None:
    """Running migrations against an empty file produces every table and index (I5/I4)."""
    with Store.open(db_path) as store:
        store.migrate()

    conn = sqlite3.connect(db_path)
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }
    finally:
        conn.close()

    assert tables >= EXPECTED_TABLES
    assert indexes >= EXPECTED_INDEXES


def test_applications_table_has_no_status_column(db_path: Path) -> None:
    """I4: ApplicationStatus must never be a stored column."""
    with Store.open(db_path) as store:
        store.migrate()
        columns = {
            row["name"]
            for row in store._connection.execute("PRAGMA table_info(applications)").fetchall()
        }
    assert "status" not in columns


def test_events_message_id_is_unique(db_path: Path) -> None:
    """I5/I1: at most one event row per message_id, enforced at the schema level."""
    with Store.open(db_path) as store:
        store.migrate()
        store._connection.execute(
            "INSERT INTO messages (message_id, thread_id, received_at, from_email, ingested_at) "
            "VALUES ('m1', 't1', '2026-01-01T00:00:00+00:00', 'a@b.com', '2026-01-01T00:00:00+00:00')"
        )
        store._connection.execute(
            "INSERT INTO events (message_id, event_type, occurred_at, created_at) "
            "VALUES ('m1', 'unknown', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')"
        )
        store._connection.commit()
        with pytest.raises(sqlite3.IntegrityError):
            store._connection.execute(
                "INSERT INTO events (message_id, event_type, occurred_at, created_at) "
                "VALUES ('m1', 'unknown', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')"
            )


def test_migrate_records_schema_version(db_path: Path) -> None:
    """migrate() stamps schema_version with SCHEMA_VERSION after applying migrations."""
    with Store.open(db_path) as store:
        store.migrate()
        row = store._connection.execute(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        ).fetchone()
    assert row["version"] == SCHEMA_VERSION


def test_migrate_is_idempotent(db_path: Path) -> None:
    """Calling migrate() twice on an already-migrated DB does nothing and does not raise."""
    with Store.open(db_path) as store:
        store.migrate()
        store.migrate()
        count = store._connection.execute("SELECT COUNT(*) AS n FROM schema_version").fetchone()
    assert count["n"] == 1


def test_foreign_keys_are_enforced(db_path: Path) -> None:
    """PRAGMA foreign_keys=ON per CONTRACTS.md — a dangling reference must be rejected."""
    with Store.open(db_path) as store:
        store.migrate()
        with pytest.raises(sqlite3.IntegrityError):
            store._connection.execute(
                "INSERT INTO classifications (message_id, event_type, confidence, needs_review, "
                "classifier_name, classifier_version, classified_at) "
                "VALUES ('does-not-exist', 'unknown', 0.0, 0, 'rules', '1.0.0', "
                "'2026-01-01T00:00:00+00:00')"
            )


def test_wal_mode_enabled(db_path: Path) -> None:
    """PRAGMA journal_mode=WAL per CONTRACTS.md."""
    with Store.open(db_path) as store:
        mode = store._connection.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_open_rejects_non_jobtrack_file(tmp_path: Path) -> None:
    """A file that exists but isn't a jobtrack database raises StoreError, not a crash."""
    path = tmp_path / "not_a_db.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE some_other_app_table (id INTEGER)")
    conn.commit()
    conn.close()

    with pytest.raises(StoreError):
        Store.open(path)


def test_context_manager_closes_connection(db_path: Path) -> None:
    """__exit__ closes the connection; further use raises."""
    with Store.open(db_path) as store:
        store.migrate()
        connection = store._connection
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")
