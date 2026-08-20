"""Tests for connection handling, migrations, and the write/read paths.

Every test runs against a real SQLite file in ``tmp_path`` — never a mock.
"""

from __future__ import annotations

import shutil
import sqlite3
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from jobtrack.config import Config
from jobtrack.errors import MigrationError, StoreError
from jobtrack.models import (
    ApplicationStatus,
    Classification,
    EventType,
    RawMessage,
)
from jobtrack.store import db as db_module
from jobtrack.store import repo
from jobtrack.store.db import SCHEMA_VERSION, Store, migration_files
from jobtrack.store.linker import LINK_WINDOW_DAYS

NOW = datetime(2026, 8, 18, 12, 0, 0, tzinfo=UTC)
SCHEMA_SQL = Path(db_module.__file__).resolve().parent / "schema.sql"
EXPECTED_TABLES = {
    "applications",
    "classifications",
    "events",
    "messages",
    "overrides",
    "schema_version",
    "sync_state",
}

MessageFactory = Callable[..., RawMessage]


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    """A migrated store backed by a real SQLite file in tmp_path."""
    with Store.open(tmp_path / "jobtrack.db") as opened:
        opened.migrate()
        yield opened


def classify(
    message: RawMessage,
    *,
    event_type: EventType = EventType.APPLICATION_RECEIVED,
    company: str | None = "Acme Robotics, Inc.",
    company_key: str | None = "acme robotics",
    role: str | None = "Software Engineer",
    confidence: float = 0.9,
    needs_review: bool = False,
) -> Classification:
    """Build a Classification for a message, standing in for M2."""
    return Classification(
        message_id=message.message_id,
        event_type=event_type,
        company=company,
        company_key=company_key,
        role=role,
        location="Remote",
        ats="greenhouse",
        confidence=confidence,
        needs_review=needs_review,
        evidence=["ack.subject.thanks_for_applying"],
        classifier_name="rules",
        classifier_version="1.0.0",
    )


def user_objects(connection: sqlite3.Connection) -> list[tuple[str, str, str]]:
    """Every non-internal schema object as (type, name, sql), sorted."""
    rows = connection.execute(
        "SELECT type AS type, name AS name, sql AS sql FROM sqlite_master "
        "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
    ).fetchall()
    return [(row["type"], row["name"], row["sql"]) for row in rows]


# --- open / migrate ---------------------------------------------------------


def test_open_creates_missing_parent_directories(tmp_path: Path) -> None:
    """JOBTRACK_HOME may not exist yet on a first run."""
    target = tmp_path / "nested" / "deeper" / "jobtrack.db"
    with Store.open(target):
        pass
    assert target.exists()


def test_open_rejects_a_file_that_is_not_a_database(tmp_path: Path) -> None:
    """A stray non-SQLite file is reported as a StoreError, not a sqlite3 error."""
    bogus = tmp_path / "jobtrack.db"
    bogus.write_bytes(b"this is definitely not a sqlite database" * 64)
    with pytest.raises(StoreError, match="not a readable jobtrack database"):
        Store.open(bogus)


def test_schema_version_is_zero_before_migrating(tmp_path: Path) -> None:
    """An empty file has no schema yet."""
    with Store.open(tmp_path / "jobtrack.db") as opened:
        assert opened.schema_version() == 0


def test_migrate_creates_every_table(store: Store) -> None:
    """Running the migrations against an empty file yields the whole schema."""
    connection = sqlite3.connect(store_path(store))
    connection.row_factory = sqlite3.Row
    names = {name for kind, name, _ in user_objects(connection) if kind == "table"}
    connection.close()
    assert EXPECTED_TABLES <= names
    assert store.schema_version() == SCHEMA_VERSION


def test_applications_table_has_no_status_column(store: Store) -> None:
    """Status is derived on every read and must never be persisted (I4)."""
    connection = sqlite3.connect(store_path(store))
    columns = {row[1] for row in connection.execute("PRAGMA table_info(applications)")}
    connection.close()
    assert "status" not in columns


def test_migrate_is_idempotent(store: Store) -> None:
    """A second migrate applies nothing and does not raise."""
    store.migrate()
    store.migrate()
    assert store.schema_version() == SCHEMA_VERSION


def test_migrations_match_schema_sql(tmp_path: Path) -> None:
    """schema.sql and the migration chain must describe the same database."""
    migrated = tmp_path / "migrated.db"
    with Store.open(migrated) as opened:
        opened.migrate()

    direct = tmp_path / "direct.db"
    connection = sqlite3.connect(direct)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA_SQL.read_text(encoding="utf-8"))
    connection.commit()
    expected = user_objects(connection)
    connection.close()

    check = sqlite3.connect(migrated)
    check.row_factory = sqlite3.Row
    actual = user_objects(check)
    check.close()
    assert actual == expected


def test_schema_version_constant_tracks_the_migration_files() -> None:
    """Adding a migration without bumping SCHEMA_VERSION is a bug."""
    assert max(version for version, _ in migration_files()) == SCHEMA_VERSION


def test_migration_files_requires_the_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing migrations directory is a MigrationError, not an OSError."""
    monkeypatch.setattr(db_module, "MIGRATIONS_DIR", tmp_path / "absent")
    with pytest.raises(MigrationError, match="migrations directory is missing"):
        migration_files()


def test_migration_files_rejects_a_bad_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Migration filenames must be NNNN_name.sql so ordering is unambiguous."""
    directory = tmp_path / "migrations"
    directory.mkdir()
    (directory / "initial.sql").write_text("SELECT 1;", encoding="utf-8")
    monkeypatch.setattr(db_module, "MIGRATIONS_DIR", directory)
    with pytest.raises(MigrationError, match="not NNNN_name.sql"):
        migration_files()


def test_migration_files_rejects_duplicate_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two files claiming one version would apply in an undefined order."""
    directory = tmp_path / "migrations"
    directory.mkdir()
    (directory / "0001_a.sql").write_text("SELECT 1;", encoding="utf-8")
    (directory / "0001_b.sql").write_text("SELECT 1;", encoding="utf-8")
    monkeypatch.setattr(db_module, "MIGRATIONS_DIR", directory)
    with pytest.raises(MigrationError, match="duplicate migration version"):
        migration_files()


def test_migration_files_ignores_non_sql_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stray README in the migrations directory is not a migration."""
    directory = tmp_path / "migrations"
    directory.mkdir()
    shutil.copy(db_module.MIGRATIONS_DIR / "0001_initial.sql", directory / "0001_initial.sql")
    (directory / "notes.md").write_text("ignore me", encoding="utf-8")
    monkeypatch.setattr(db_module, "MIGRATIONS_DIR", directory)
    assert [version for version, _ in migration_files()] == [1]


def test_failed_migration_leaves_the_prior_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken migration rolls back; the database stays at the version before it."""
    directory = tmp_path / "migrations"
    directory.mkdir()
    shutil.copy(db_module.MIGRATIONS_DIR / "0001_initial.sql", directory / "0001_initial.sql")
    (directory / "0002_broken.sql").write_text(
        "CREATE TABLE ok_so_far (id INTEGER);\nTHIS IS NOT SQL;\n", encoding="utf-8"
    )
    monkeypatch.setattr(db_module, "MIGRATIONS_DIR", directory)

    with Store.open(tmp_path / "jobtrack.db") as opened:
        with pytest.raises(MigrationError, match="0002_broken.sql"):
            opened.migrate()
        assert opened.schema_version() == 1
        connection = sqlite3.connect(tmp_path / "jobtrack.db")
        names = {row[0] for row in connection.execute("SELECT name FROM sqlite_master")}
        connection.close()
        assert "ok_so_far" not in names


def test_open_from_config_carries_ghost_after_days(tmp_config: Config) -> None:
    """The configured patience reaches the store that derives GHOSTED (I4)."""
    config = tmp_config.model_copy(update={"store": tmp_config.store.model_copy(
        update={"ghost_after_days": 7}
    )})
    with Store.open_from_config(config) as opened:
        opened.migrate()
        assert opened.ghost_after_days == 7
        assert config.db_path.exists()


def test_store_is_a_context_manager(tmp_path: Path) -> None:
    """Leaving the with-block closes the connection."""
    opened = Store.open(tmp_path / "jobtrack.db")
    with opened as entered:
        entered.migrate()
        assert entered is opened
    with pytest.raises(StoreError):
        opened.has_message("m1")


def test_sqlite_errors_never_escape(tmp_path: Path) -> None:
    """Driver failures are wrapped at the module boundary, always."""
    opened = Store.open(tmp_path / "jobtrack.db")
    opened.migrate()
    opened.close()
    with pytest.raises(StoreError, match="query failed"):
        opened.list_events()


# --- messages and classifications -------------------------------------------


def test_record_message_is_idempotent(store: Store, make_message: MessageFactory) -> None:
    """message_id is the universal dedupe key (I1)."""
    message = make_message(message_id="m1")
    assert not store.has_message("m1")
    store.record_message(message)
    store.record_message(message)
    assert store.has_message("m1")


def test_record_message_rejects_naive_datetimes(
    store: Store, make_message: MessageFactory
) -> None:
    """A naive received_at must not reach the disk (I7)."""
    naive = make_message(received_at=datetime(2026, 8, 18, 12, 0, 0))  # noqa: DTZ001
    with pytest.raises(StoreError, match="naive datetime"):
        store.record_message(naive)


def test_record_classification_requires_its_message(
    store: Store, make_message: MessageFactory
) -> None:
    """Foreign keys are ON: a classification cannot dangle."""
    orphan = classify(make_message(message_id="ghost"))
    with pytest.raises(StoreError, match="write failed"):
        store.record_classification(orphan)


def test_record_classification_upserts(store: Store, make_message: MessageFactory) -> None:
    """Reclassifying replaces the stored row rather than duplicating it."""
    message = make_message(message_id="m1")
    store.record_message(message)
    store.record_classification(classify(message))
    store.record_classification(classify(message, event_type=EventType.REJECTION))
    store.link_and_record_event(message, classify(message, event_type=EventType.REJECTION), now=NOW)
    events = store.list_events()
    assert len(events) == 1
    assert events[0].event_type is EventType.REJECTION


# --- linking ----------------------------------------------------------------


def test_link_and_record_event_is_idempotent(store: Store, make_message: MessageFactory) -> None:
    """I1: replaying a message produces no second event and no second application."""
    message = make_message(message_id="m1", thread_id="t1")
    classification = classify(message)
    first = store.link_and_record_event(message, classification, now=NOW)
    second = store.link_and_record_event(message, classification, now=NOW)
    assert first == second
    assert len(store.list_events()) == 1
    assert len(store.list_applications(now=NOW)) == 1


def test_unknown_classifications_are_recorded_unlinked(
    store: Store, make_message: MessageFactory
) -> None:
    """An UNKNOWN message is kept but belongs to no application."""
    message = make_message(message_id="m1")
    event = store.link_and_record_event(
        message, classify(message, event_type=EventType.UNKNOWN), now=NOW
    )
    assert event.application_id is None
    assert store.list_applications(now=NOW) == []
    assert len(store.list_events()) == 1


def test_messages_without_a_company_key_are_recorded_unlinked(
    store: Store, make_message: MessageFactory
) -> None:
    """Without a company_key (I8) there is nothing to match or create against."""
    message = make_message(message_id="m1")
    event = store.link_and_record_event(
        message, classify(message, company=None, company_key=None), now=NOW
    )
    assert event.application_id is None


def test_same_thread_links_to_one_application(
    store: Store, make_message: MessageFactory
) -> None:
    """Rule 1: a reply on the same thread joins the same application."""
    first = make_message(message_id="m1", thread_id="t1")
    second = make_message(message_id="m2", thread_id="t1")
    store.link_and_record_event(first, classify(first), now=NOW)
    event = store.link_and_record_event(
        second,
        classify(second, company_key="totally different", role="Chef", event_type=EventType.OFFER),
        now=NOW,
    )
    applications = store.list_applications(now=NOW)
    assert len(applications) == 1
    assert event.application_id == applications[0].application_id
    assert applications[0].event_count == 2


def test_same_company_and_role_links_across_threads(
    store: Store, make_message: MessageFactory
) -> None:
    """Rule 2: a rejection on a new thread still lands on the original application."""
    applied = make_message(message_id="m1", thread_id="t1", received_at=NOW - timedelta(days=20))
    rejected = make_message(message_id="m2", thread_id="t2", received_at=NOW - timedelta(days=2))
    store.link_and_record_event(applied, classify(applied), now=NOW)
    store.link_and_record_event(
        rejected,
        classify(rejected, event_type=EventType.REJECTION, role="Sr. Software Engineer II"),
        now=NOW,
    )
    applications = store.list_applications(now=NOW)
    assert len(applications) == 1
    assert applications[0].status is ApplicationStatus.REJECTED
    assert applications[0].source_thread_ids == ["t1", "t2"]
    assert applications[0].days_to_first_response == 18


def test_a_different_company_creates_a_second_application(
    store: Store, make_message: MessageFactory
) -> None:
    """Different employers never merge."""
    first = make_message(message_id="m1", thread_id="t1")
    second = make_message(message_id="m2", thread_id="t2")
    store.link_and_record_event(first, classify(first), now=NOW)
    store.link_and_record_event(
        second, classify(second, company="Globex", company_key="globex"), now=NOW
    )
    assert len(store.list_applications(now=NOW)) == 2


def test_application_ids_are_deterministic(
    tmp_path: Path, make_message: MessageFactory
) -> None:
    """The same inputs produce the same application id in a fresh database."""
    message = make_message(message_id="m1", thread_id="t1")
    classification = classify(message)
    ids: list[str] = []
    for name in ("a.db", "b.db"):
        with Store.open(tmp_path / name) as opened:
            opened.migrate()
            event = opened.link_and_record_event(message, classification, now=NOW)
            assert event.application_id is not None
            ids.append(event.application_id)
    assert ids[0] == ids[1]


def test_enrichment_fills_missing_application_fields(
    store: Store, make_message: MessageFactory
) -> None:
    """A later message supplies the role the first one lacked."""
    first = make_message(message_id="m1", thread_id="t1")
    second = make_message(message_id="m2", thread_id="t1")
    store.link_and_record_event(first, classify(first, role=None, ats=None), now=NOW)
    store.link_and_record_event(
        second, classify(second, event_type=EventType.INTERVIEW), now=NOW
    )
    application = store.list_applications(now=NOW)[0]
    assert application.role == "Software Engineer"
    assert application.ats == "greenhouse"


def test_applied_at_moves_back_to_the_earliest_message(
    store: Store, make_message: MessageFactory
) -> None:
    """Events arriving out of order still yield the true application date."""
    late = make_message(message_id="m1", thread_id="t1", received_at=NOW - timedelta(days=2))
    early = make_message(message_id="m2", thread_id="t1", received_at=NOW - timedelta(days=30))
    store.link_and_record_event(late, classify(late, event_type=EventType.INTERVIEW), now=NOW)
    store.link_and_record_event(early, classify(early), now=NOW)
    application = store.list_applications(now=NOW)[0]
    assert application.applied_at == NOW - timedelta(days=30)


# --- candidates -------------------------------------------------------------


def test_match_candidates_reports_thread_ids(
    store: Store, make_message: MessageFactory
) -> None:
    """Candidates carry every thread their application has been seen on."""
    first = make_message(message_id="m1", thread_id="t1")
    second = make_message(message_id="m2", thread_id="t2")
    store.link_and_record_event(first, classify(first), now=NOW)
    store.link_and_record_event(
        second, classify(second, event_type=EventType.INTERVIEW), now=NOW
    )
    candidates = store.match_candidates("acme robotics", "t9", within_days=LINK_WINDOW_DAYS)
    assert len(candidates) == 1
    assert candidates[0].thread_ids == ["t1", "t2"]
    assert candidates[0].company_key == "acme robotics"


def test_match_candidates_applies_the_window_prefilter(
    store: Store, make_message: MessageFactory
) -> None:
    """Stale applications are dropped before the pure linker ever sees them."""
    ancient = make_message(message_id="m1", thread_id="t1", received_at=NOW - timedelta(days=400))
    recent = make_message(message_id="m2", thread_id="t2", received_at=NOW)
    store.link_and_record_event(ancient, classify(ancient), now=NOW)
    store.link_and_record_event(
        recent, classify(recent, company="Globex", company_key="globex"), now=NOW
    )
    wide = store.match_candidates("acme robotics", "t9", within_days=500)
    narrow = store.match_candidates("acme robotics", "t9", within_days=30)
    assert len(wide) == 1
    assert narrow == []


def test_match_candidates_keeps_stale_thread_matches(
    store: Store, make_message: MessageFactory
) -> None:
    """A thread match bypasses the window entirely."""
    ancient = make_message(message_id="m1", thread_id="t1", received_at=NOW - timedelta(days=400))
    recent = make_message(message_id="m2", thread_id="t2", received_at=NOW)
    store.link_and_record_event(ancient, classify(ancient), now=NOW)
    store.link_and_record_event(
        recent, classify(recent, company="Globex", company_key="globex"), now=NOW
    )
    candidates = store.match_candidates("nothing at all", "t1", within_days=30)
    assert [c.thread_ids for c in candidates] == [["t1"]]


def test_match_candidates_is_empty_on_a_fresh_database(store: Store) -> None:
    """No applications means no candidates and no queries against a NULL anchor."""
    assert store.match_candidates("acme", "t1", within_days=30) == []


# --- reads ------------------------------------------------------------------


def test_get_application_returns_none_for_an_unknown_id(store: Store) -> None:
    """Reading a nonexistent application is not an error."""
    assert store.get_application("app_nope", now=NOW) is None


def test_get_application_matches_the_listing(
    store: Store, make_message: MessageFactory
) -> None:
    """The single-row read and the listing agree."""
    message = make_message(message_id="m1", thread_id="t1")
    store.link_and_record_event(message, classify(message), now=NOW)
    listed = store.list_applications(now=NOW)[0]
    assert store.get_application(listed.application_id, now=NOW) == listed


def test_list_applications_filters_by_status(
    store: Store, make_message: MessageFactory
) -> None:
    """Status is derived, then filtered (I4)."""
    live = make_message(message_id="m1", thread_id="t1")
    dead = make_message(message_id="m2", thread_id="t2")
    store.link_and_record_event(live, classify(live), now=NOW)
    store.link_and_record_event(
        dead,
        classify(dead, company="Globex", company_key="globex", event_type=EventType.REJECTION),
        now=NOW,
    )
    rejected = store.list_applications(now=NOW, status=ApplicationStatus.REJECTED)
    assert [row.company_key for row in rejected] == ["globex"]


def test_list_applications_filters_by_company_insensitively(
    store: Store, make_message: MessageFactory
) -> None:
    """The company filter runs against company_key, so punctuation and case do not matter."""
    message = make_message(message_id="m1", thread_id="t1")
    store.link_and_record_event(message, classify(message), now=NOW)
    assert len(store.list_applications(now=NOW, company="ACME, Robotics!")) == 1
    assert len(store.list_applications(now=NOW, company="acme")) == 1
    assert store.list_applications(now=NOW, company="globex") == []


def test_list_applications_filters_by_needs_review(
    store: Store, make_message: MessageFactory
) -> None:
    """The review flag propagates from the classification to the application."""
    unsure = make_message(message_id="m1", thread_id="t1")
    store.link_and_record_event(unsure, classify(unsure, needs_review=True), now=NOW)
    assert len(store.list_applications(now=NOW, needs_review=True)) == 1
    assert store.list_applications(now=NOW, needs_review=False) == []


def test_list_events_includes_unlinked_messages(
    store: Store, make_message: MessageFactory
) -> None:
    """list_events(None) returns everything, including UNKNOWNs (I5)."""
    linked = make_message(message_id="m1", thread_id="t1")
    unlinked = make_message(message_id="m2", thread_id="t2")
    store.link_and_record_event(linked, classify(linked), now=NOW)
    store.link_and_record_event(
        unlinked, classify(unlinked, event_type=EventType.UNKNOWN), now=NOW
    )
    application_id = store.list_applications(now=NOW)[0].application_id
    assert len(store.list_events()) == 2
    assert [e.message_id for e in store.list_events(application_id)] == ["m1"]


def test_events_are_ordered_oldest_first(store: Store, make_message: MessageFactory) -> None:
    """Ordering is by occurred_at, not by insertion order."""
    late = make_message(message_id="m1", thread_id="t1", received_at=NOW)
    early = make_message(message_id="m2", thread_id="t1", received_at=NOW - timedelta(days=5))
    store.link_and_record_event(late, classify(late, event_type=EventType.INTERVIEW), now=NOW)
    store.link_and_record_event(early, classify(early), now=NOW)
    assert [e.message_id for e in store.list_events()] == ["m2", "m1"]


def test_ghosted_status_is_derived_from_silence(
    tmp_path: Path, make_message: MessageFactory
) -> None:
    """A quiet, non-terminal application ghosts once the configured patience runs out."""
    message = make_message(message_id="m1", thread_id="t1", received_at=NOW - timedelta(days=45))
    with Store.open(tmp_path / "jobtrack.db") as opened:
        opened.migrate()
        opened.ghost_after_days = 30
        opened.link_and_record_event(message, classify(message), now=NOW)
        assert opened.list_applications(now=NOW)[0].status is ApplicationStatus.GHOSTED


# --- sync state -------------------------------------------------------------


def test_cursor_round_trips(store: Store) -> None:
    """The cursor survives a write and a read (I9)."""
    assert store.get_cursor("gmail") is None
    store.set_cursor("gmail", "history-123", synced_at=NOW)
    assert store.get_cursor("gmail") == "history-123"
    store.set_cursor("gmail", "history-456", synced_at=NOW)
    assert store.get_cursor("gmail") == "history-456"


def test_cursor_can_be_cleared(store: Store) -> None:
    """Clearing forces the next sync back onto a dated query."""
    store.set_cursor("gmail", "history-123", synced_at=NOW)
    store.set_cursor("gmail", None, synced_at=NOW)
    assert store.get_cursor("gmail") is None


def test_cursors_are_per_source(store: Store) -> None:
    """Two mailboxes do not share a cursor."""
    store.set_cursor("gmail", "g1", synced_at=NOW)
    store.set_cursor("imap", "i1", synced_at=NOW)
    assert store.get_cursor("gmail") == "g1"
    assert store.get_cursor("imap") == "i1"


def test_set_cursor_rejects_naive_datetimes(store: Store) -> None:
    """The sync timestamp is a datetime like any other (I7)."""
    with pytest.raises(StoreError, match="naive datetime"):
        store.set_cursor("gmail", "c", synced_at=datetime(2026, 8, 18))  # noqa: DTZ001


def store_path(store: Store) -> Path:
    """Recover the file a store was opened on, for schema introspection in tests."""
    connection = store._conn  # noqa: SLF001 - tests may inspect the file under test
    row = connection.execute("PRAGMA database_list").fetchall()[0]
    return Path(row[2])
