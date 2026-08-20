"""Tests for timestamp handling, row mapping, and ApplicationRow assembly."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from jobtrack.errors import StoreError
from jobtrack.models import ApplicationStatus, EventRow, EventType
from jobtrack.store import repo
from jobtrack.store.db import Store

NOW = datetime(2026, 8, 18, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    """A migrated store backed by a real SQLite file in tmp_path."""
    with Store.open(tmp_path / "jobtrack.db") as opened:
        opened.migrate()
        yield opened


def make_core(**overrides: object) -> repo.ApplicationCore:
    """Build an ApplicationCore with sensible defaults."""
    values: dict[str, object] = {
        "application_id": "app_1",
        "company": "Acme Robotics, Inc.",
        "company_key": "acme robotics",
        "role": "Software Engineer",
        "location": "Remote",
        "ats": "greenhouse",
    }
    values.update(overrides)
    return repo.ApplicationCore(**values)  # type: ignore[arg-type]  # kwargs are checked above


def make_record(
    event_type: EventType,
    *,
    days_ago: int = 0,
    event_id: int = 1,
    thread_id: str = "t1",
    needs_review: bool = False,
    is_overridden: bool = False,
    override_company: str | None = None,
    override_role: str | None = None,
    corrected_at: datetime | None = None,
) -> repo.EventRecord:
    """Build an EventRecord positioned ``days_ago`` days before NOW."""
    row = EventRow(
        event_id=event_id,
        application_id="app_1",
        message_id=f"m{event_id}",
        event_type=event_type,
        occurred_at=NOW - timedelta(days=days_ago),
        confidence=0.8,
        needs_review=needs_review,
        is_overridden=is_overridden,
        subject="subject",
        from_email="careers@example.com",
    )
    return repo.EventRecord(
        row=row,
        thread_id=thread_id,
        override_company=override_company,
        override_role=override_role,
        corrected_at=corrected_at,
    )


# --- timestamps -------------------------------------------------------------


def test_to_iso_renders_utc() -> None:
    """A UTC datetime round-trips as ISO-8601 with an explicit offset."""
    assert repo.to_iso(NOW) == "2026-08-18T12:00:00+00:00"


def test_to_iso_converts_other_zones_to_utc() -> None:
    """Any aware zone is normalized to UTC before storage (I7)."""
    eastern = timezone(timedelta(hours=-4))
    assert repo.to_iso(NOW.astimezone(eastern)) == "2026-08-18T12:00:00+00:00"


def test_to_iso_rejects_naive_datetimes() -> None:
    """A naive datetime crossing the boundary is a bug, not a coercion (I7)."""
    with pytest.raises(StoreError, match="naive datetime"):
        # Deliberately naive: to_iso must refuse it rather than guess a zone.
        repo.to_iso(datetime(2026, 8, 18, 12, 0, 0))


def test_from_iso_round_trips() -> None:
    """from_iso undoes to_iso exactly."""
    assert repo.from_iso(repo.to_iso(NOW)) == NOW


def test_from_iso_assumes_utc_for_offsetless_text() -> None:
    """Timestamps written by the database clock carry no offset in some SQLite builds."""
    assert repo.from_iso("2026-08-18T12:00:00") == NOW


def test_from_iso_accepts_the_database_clock_format() -> None:
    """strftime('%Y-%m-%dT%H:%M:%fZ') output parses back to an aware datetime."""
    parsed = repo.from_iso("2026-08-18T12:00:00.000Z")
    assert parsed == NOW


def test_from_iso_rejects_garbage() -> None:
    """A corrupt timestamp surfaces as a StoreError, never a ValueError."""
    with pytest.raises(StoreError, match="unparseable timestamp"):
        repo.from_iso("not-a-date")


def test_to_event_type_rejects_unknown_values() -> None:
    """A value outside EventType is corruption, not a silent UNKNOWN."""
    with pytest.raises(StoreError, match="unknown event_type"):
        repo.to_event_type("promotion")


# --- JSON columns -----------------------------------------------------------


def test_corrupt_labels_json_raises_store_error(store: Store, tmp_path: Path) -> None:
    """A corrupt JSON array column is reported, not silently dropped."""
    connection = sqlite3.connect(tmp_path / "jobtrack.db")
    connection.row_factory = sqlite3.Row
    connection.execute(
        "INSERT INTO messages (message_id, thread_id, received_at, from_email, labels_json) "
        "VALUES (?, ?, ?, ?, ?)",
        ("m1", "t1", repo.to_iso(NOW), "a@b.co", "{not-json"),
    )
    connection.commit()
    row = connection.execute(repo.SELECT_MESSAGE, ("m1",)).fetchone()
    with pytest.raises(StoreError, match="corrupt JSON in labels_json"):
        repo.row_to_message(row)
    connection.close()


def test_wrongly_typed_json_columns_raise_store_error(store: Store, tmp_path: Path) -> None:
    """Valid JSON of the wrong shape is still corruption."""
    connection = sqlite3.connect(tmp_path / "jobtrack.db")
    connection.row_factory = sqlite3.Row
    connection.execute(
        "INSERT INTO messages "
        "(message_id, thread_id, received_at, from_email, labels_json, headers_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("m2", "t1", repo.to_iso(NOW), "a@b.co", '{"a": 1}', "[]"),
    )
    connection.commit()
    row = connection.execute(repo.SELECT_MESSAGE, ("m2",)).fetchone()
    with pytest.raises(StoreError, match="expected a JSON array"):
        repo.row_to_message(row)

    connection.execute("UPDATE messages SET labels_json = ? WHERE message_id = ?", ("[]", "m2"))
    connection.commit()
    row = connection.execute(repo.SELECT_MESSAGE, ("m2",)).fetchone()
    with pytest.raises(StoreError, match="expected a JSON object"):
        repo.row_to_message(row)
    connection.close()


def test_corrupt_evidence_json_raises_store_error(store: Store, tmp_path: Path) -> None:
    """The classification projection guards its JSON column too."""
    connection = sqlite3.connect(tmp_path / "jobtrack.db")
    connection.row_factory = sqlite3.Row
    connection.execute(
        "INSERT INTO messages (message_id, thread_id, received_at, from_email) "
        "VALUES (?, ?, ?, ?)",
        ("m3", "t1", repo.to_iso(NOW), "a@b.co"),
    )
    connection.execute(
        "INSERT INTO classifications "
        "(message_id, event_type, evidence_json, classifier_name, classifier_version) "
        "VALUES (?, ?, ?, ?, ?)",
        ("m3", "rejection", "{oops", "rules", "1.0.0"),
    )
    connection.commit()
    row = connection.execute(repo.SELECT_CLASSIFICATION, ("m3",)).fetchone()
    with pytest.raises(StoreError, match="corrupt JSON in evidence_json"):
        repo.row_to_classification(row)
    connection.close()


# --- build_application_row --------------------------------------------------


def test_application_with_no_events_is_rejected() -> None:
    """The writer never produces an eventless application, so this is corruption."""
    with pytest.raises(StoreError, match="has no events"):
        repo.build_application_row(make_core(), [], now=NOW, ghost_after_days=30)


def test_derived_fields_come_from_the_event_history() -> None:
    """applied_at, last_event_at, counts, and elapsed days all derive from events (I4)."""
    records = [
        make_record(EventType.INTERVIEW, days_ago=3, event_id=2, thread_id="t2"),
        make_record(EventType.APPLICATION_RECEIVED, days_ago=10, event_id=1, thread_id="t1"),
    ]
    row = repo.build_application_row(make_core(), records, now=NOW, ghost_after_days=30)
    assert row.applied_at == NOW - timedelta(days=10)
    assert row.last_event_at == NOW - timedelta(days=3)
    assert row.last_event_type is EventType.INTERVIEW
    assert row.event_count == 2
    assert row.days_to_first_response == 7
    assert row.days_since_last_event == 3
    assert row.status is ApplicationStatus.INTERVIEWING
    assert row.source_thread_ids == ["t1", "t2"]


def test_thread_ids_are_deduplicated_and_sorted() -> None:
    """Several messages on one thread contribute a single thread id."""
    records = [
        make_record(EventType.APPLICATION_RECEIVED, days_ago=5, event_id=1, thread_id="tb"),
        make_record(EventType.INTERVIEW, days_ago=4, event_id=2, thread_id="tb"),
        make_record(EventType.OFFER, days_ago=1, event_id=3, thread_id="ta"),
    ]
    row = repo.build_application_row(make_core(), records, now=NOW, ghost_after_days=30)
    assert row.source_thread_ids == ["ta", "tb"]


def test_no_response_yet_leaves_days_to_first_response_none() -> None:
    """An acknowledgement on its own is not a response."""
    records = [make_record(EventType.APPLICATION_RECEIVED, days_ago=5)]
    row = repo.build_application_row(make_core(), records, now=NOW, ghost_after_days=30)
    assert row.days_to_first_response is None


def test_override_company_and_role_win_at_read_time() -> None:
    """Corrections beat the stored display values (I6)."""
    records = [
        make_record(
            EventType.REJECTION,
            days_ago=1,
            is_overridden=True,
            override_company="Globex Corporation",
            override_role="Platform Engineer",
            corrected_at=NOW,
        )
    ]
    row = repo.build_application_row(make_core(), records, now=NOW, ghost_after_days=30)
    assert row.company == "Globex Corporation"
    assert row.role == "Platform Engineer"
    assert row.company_key == "acme robotics"


def test_most_recent_correction_wins() -> None:
    """Two corrections on one application: the newer one is displayed."""
    records = [
        make_record(
            EventType.APPLICATION_RECEIVED,
            days_ago=5,
            event_id=1,
            is_overridden=True,
            override_company="First Guess",
            corrected_at=NOW - timedelta(days=2),
        ),
        make_record(
            EventType.INTERVIEW,
            days_ago=2,
            event_id=2,
            is_overridden=True,
            override_company="Second Guess",
            corrected_at=NOW - timedelta(days=1),
        ),
    ]
    row = repo.build_application_row(make_core(), records, now=NOW, ghost_after_days=30)
    assert row.company == "Second Guess"


def test_uncorrected_application_keeps_its_stored_display_values() -> None:
    """With no overrides the stored company and role are used verbatim (I8)."""
    records = [make_record(EventType.APPLICATION_RECEIVED, days_ago=1)]
    row = repo.build_application_row(make_core(), records, now=NOW, ghost_after_days=30)
    assert row.company == "Acme Robotics, Inc."
    assert row.role == "Software Engineer"


def test_needs_review_ignores_overridden_events() -> None:
    """A human already looked at an overridden event, so it no longer needs review."""
    flagged = make_record(EventType.INTERVIEW, days_ago=1, event_id=1, needs_review=True)
    corrected = make_record(
        EventType.INTERVIEW,
        days_ago=1,
        event_id=2,
        needs_review=True,
        is_overridden=True,
        override_role="Backend Engineer",
        corrected_at=NOW,
    )
    assert repo.build_application_row(
        make_core(), [flagged], now=NOW, ghost_after_days=30
    ).needs_review
    assert not repo.build_application_row(
        make_core(), [corrected], now=NOW, ghost_after_days=30
    ).needs_review


def test_corrupt_headers_json_raises_store_error(store: Store, tmp_path: Path) -> None:
    """The object-valued JSON column is guarded the same way as the array one."""
    connection = sqlite3.connect(tmp_path / "jobtrack.db")
    connection.row_factory = sqlite3.Row
    connection.execute(
        "INSERT INTO messages "
        "(message_id, thread_id, received_at, from_email, labels_json, headers_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("m4", "t1", repo.to_iso(NOW), "a@b.co", "[]", "{not-json"),
    )
    connection.commit()
    row = connection.execute(repo.SELECT_MESSAGE, ("m4",)).fetchone()
    with pytest.raises(StoreError, match="corrupt JSON in headers_json"):
        repo.row_to_message(row)
    connection.close()
