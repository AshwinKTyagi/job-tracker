"""Tests for the review queue, human corrections, and reclassify safety.

These cover the three invariants the correction path exists to protect: events are
append-only (I5), overrides win at read time and survive a reclassify (I6), and the
derived status follows the corrected history rather than the classifier's (I4).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from jobtrack.errors import StoreError
from jobtrack.models import (
    ApplicationStatus,
    Classification,
    EventType,
    Override,
    RawMessage,
)
from jobtrack.store.db import Store

NOW = datetime(2026, 8, 18, 12, 0, 0, tzinfo=UTC)

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


def stored_event_type(tmp_path: Path, message_id: str) -> str:
    """Read the untouched event_type straight from disk, bypassing overrides."""
    connection = sqlite3.connect(tmp_path / "jobtrack.db")
    connection.row_factory = sqlite3.Row
    row = connection.execute(
        "SELECT event_type AS event_type FROM events WHERE message_id = ?", (message_id,)
    ).fetchone()
    connection.close()
    return str(row["event_type"])


def count_rows(tmp_path: Path, table: str) -> int:
    """Count rows in one table. The table name is a literal chosen by the test, never input."""
    connection = sqlite3.connect(tmp_path / "jobtrack.db")
    sql = {
        "events": "SELECT count(*) FROM events",
        "applications": "SELECT count(*) FROM applications",
        "classifications": "SELECT count(*) FROM classifications",
        "overrides": "SELECT count(*) FROM overrides",
        "messages": "SELECT count(*) FROM messages",
    }[table]
    total = int(connection.execute(sql).fetchone()[0])
    connection.close()
    return total


# --- corrections ------------------------------------------------------------


def test_override_changes_the_event_type_at_read_time(
    store: Store, make_message: MessageFactory
) -> None:
    """The classifier said acknowledgement; the human says rejection, and wins (I6)."""
    message = make_message(message_id="m1", thread_id="t1")
    store.link_and_record_event(message, classify(message), now=NOW)
    store.set_override(Override(message_id="m1", event_type=EventType.REJECTION, corrected_at=NOW))
    events = store.list_events()
    assert events[0].event_type is EventType.REJECTION
    assert events[0].is_overridden
    assert store.list_applications(now=NOW)[0].status is ApplicationStatus.REJECTED


def test_override_never_mutates_the_event_row(
    store: Store, make_message: MessageFactory, tmp_path: Path
) -> None:
    """Events are append-only: the correction lives in its own table (I5)."""
    message = make_message(message_id="m1", thread_id="t1")
    store.link_and_record_event(message, classify(message), now=NOW)
    store.set_override(Override(message_id="m1", event_type=EventType.REJECTION, corrected_at=NOW))
    assert stored_event_type(tmp_path, "m1") == "application_received"
    assert count_rows(tmp_path, "events") == 1


def test_override_company_and_role_win_over_the_classifier(
    store: Store, make_message: MessageFactory
) -> None:
    """A corrected employer and title show up on the application row (I6)."""
    message = make_message(message_id="m1", thread_id="t1")
    store.link_and_record_event(message, classify(message), now=NOW)
    store.set_override(
        Override(
            message_id="m1",
            company="Globex Corporation",
            role="Platform Engineer",
            corrected_at=NOW,
            note="the ATS masked the real employer",
        )
    )
    application = store.list_applications(now=NOW)[0]
    assert application.company == "Globex Corporation"
    assert application.role == "Platform Engineer"
    assert application.company_key == "acme robotics"


def test_override_upserts_on_message_id(
    store: Store, make_message: MessageFactory, tmp_path: Path
) -> None:
    """Correcting twice replaces the correction rather than stacking rows."""
    message = make_message(message_id="m1", thread_id="t1")
    store.link_and_record_event(message, classify(message), now=NOW)
    store.set_override(Override(message_id="m1", company="First", corrected_at=NOW))
    store.set_override(
        Override(message_id="m1", company="Second", corrected_at=NOW + timedelta(hours=1))
    )
    assert count_rows(tmp_path, "overrides") == 1
    assert store.list_applications(now=NOW)[0].company == "Second"


def test_override_to_unknown_unlinks_and_prunes_the_application(
    store: Store, make_message: MessageFactory, tmp_path: Path
) -> None:
    """A message that was never job mail leaves the application empty, so it goes away."""
    message = make_message(message_id="m1", thread_id="t1")
    store.link_and_record_event(message, classify(message), now=NOW)
    assert count_rows(tmp_path, "applications") == 1

    store.set_override(Override(message_id="m1", event_type=EventType.UNKNOWN, corrected_at=NOW))
    assert store.list_applications(now=NOW) == []
    assert count_rows(tmp_path, "applications") == 0
    assert count_rows(tmp_path, "events") == 1
    assert store.list_events()[0].application_id is None


def test_override_links_a_message_the_classifier_gave_up_on(
    store: Store, make_message: MessageFactory
) -> None:
    """Correcting UNKNOWN into a real event links it to an application."""
    message = make_message(message_id="m1", thread_id="t1")
    event = store.link_and_record_event(
        message, classify(message, event_type=EventType.UNKNOWN), now=NOW
    )
    assert event.application_id is None

    store.set_override(
        Override(
            message_id="m1",
            event_type=EventType.APPLICATION_RECEIVED,
            role="Backend Engineer",
            corrected_at=NOW,
        )
    )
    applications = store.list_applications(now=NOW)
    assert len(applications) == 1
    assert store.list_events()[0].application_id == applications[0].application_id
    assert applications[0].role == "Backend Engineer"


def test_corrected_message_joins_an_existing_application(
    store: Store, make_message: MessageFactory
) -> None:
    """A rescued message links to the application it belongs to, not a fresh one."""
    applied = make_message(message_id="m1", thread_id="t1")
    missed = make_message(message_id="m2", thread_id="t2")
    store.link_and_record_event(applied, classify(applied), now=NOW)
    store.link_and_record_event(missed, classify(missed, event_type=EventType.UNKNOWN), now=NOW)
    store.set_override(Override(message_id="m2", event_type=EventType.INTERVIEW, corrected_at=NOW))
    applications = store.list_applications(now=NOW)
    assert len(applications) == 1
    assert applications[0].event_count == 2
    assert applications[0].status is ApplicationStatus.INTERVIEWING


def test_override_on_a_message_with_no_event_is_recorded_anyway(
    store: Store, make_message: MessageFactory, tmp_path: Path
) -> None:
    """A correction can arrive before the message has been linked."""
    message = make_message(message_id="m1", thread_id="t1")
    store.record_message(message)
    store.set_override(Override(message_id="m1", event_type=EventType.REJECTION, corrected_at=NOW))
    assert count_rows(tmp_path, "overrides") == 1
    assert count_rows(tmp_path, "events") == 0


def test_override_of_a_linked_event_keeps_its_application(
    store: Store, make_message: MessageFactory
) -> None:
    """Correcting a type on an already-linked event does not re-home it."""
    first = make_message(message_id="m1", thread_id="t1")
    second = make_message(message_id="m2", thread_id="t1")
    store.link_and_record_event(first, classify(first), now=NOW)
    store.link_and_record_event(second, classify(second, event_type=EventType.OFFER), now=NOW)
    before = store.list_applications(now=NOW)[0].application_id
    store.set_override(Override(message_id="m2", event_type=EventType.REJECTION, corrected_at=NOW))
    after = store.list_applications(now=NOW)
    assert [row.application_id for row in after] == [before]
    assert after[0].status is ApplicationStatus.REJECTED


def test_override_rejects_a_naive_corrected_at(store: Store, make_message: MessageFactory) -> None:
    """Corrections are timestamped tz-aware UTC like everything else (I7)."""
    message = make_message(message_id="m1", thread_id="t1")
    store.link_and_record_event(message, classify(message), now=NOW)
    naive = Override(
        message_id="m1",
        event_type=EventType.REJECTION,
        # Deliberately naive: it must be refused before it can reach disk.
        corrected_at=datetime(2026, 8, 18, 12, 0, 0),
    )
    with pytest.raises(StoreError, match="naive datetime"):
        store.set_override(naive)


# --- reclassify safety ------------------------------------------------------


def test_reclassify_preserves_overrides(store: Store, make_message: MessageFactory) -> None:
    """Wiping and re-running the classifier never clobbers a human correction (I6)."""
    message = make_message(message_id="m1", thread_id="t1")
    store.link_and_record_event(message, classify(message), now=NOW)
    store.set_override(
        Override(
            message_id="m1",
            event_type=EventType.REJECTION,
            company="Globex Corporation",
            corrected_at=NOW,
        )
    )

    store.clear_classifications(only_unreviewed=True)
    store.record_classification(classify(message, event_type=EventType.INTERVIEW))

    events = store.list_events()
    assert events[0].event_type is EventType.REJECTION
    application = store.list_applications(now=NOW)[0]
    assert application.company == "Globex Corporation"
    assert application.status is ApplicationStatus.REJECTED


def test_clear_classifications_keeps_reviewed_rows(
    store: Store, make_message: MessageFactory, tmp_path: Path
) -> None:
    """only_unreviewed keeps the rows a human already labeled, as eval data."""
    reviewed = make_message(message_id="m1", thread_id="t1")
    untouched = make_message(message_id="m2", thread_id="t2")
    store.link_and_record_event(reviewed, classify(reviewed), now=NOW)
    store.link_and_record_event(
        untouched, classify(untouched, company="Globex", company_key="globex"), now=NOW
    )
    store.set_override(Override(message_id="m1", event_type=EventType.REJECTION, corrected_at=NOW))

    cleared = store.clear_classifications(only_unreviewed=True)
    assert cleared == 1
    assert count_rows(tmp_path, "classifications") == 1


def test_clear_classifications_all_drops_everything(
    store: Store, make_message: MessageFactory, tmp_path: Path
) -> None:
    """The --all reclassify starts from a clean slate."""
    message = make_message(message_id="m1", thread_id="t1")
    store.link_and_record_event(message, classify(message), now=NOW)
    store.set_override(Override(message_id="m1", event_type=EventType.REJECTION, corrected_at=NOW))
    assert store.clear_classifications(only_unreviewed=False) == 1
    assert count_rows(tmp_path, "classifications") == 0
    assert count_rows(tmp_path, "overrides") == 1


def test_clear_classifications_never_touches_events_or_messages(
    store: Store, make_message: MessageFactory, tmp_path: Path
) -> None:
    """Only the classifications table is emptied (I5)."""
    message = make_message(message_id="m1", thread_id="t1")
    store.link_and_record_event(message, classify(message), now=NOW)
    store.clear_classifications(only_unreviewed=False)
    assert count_rows(tmp_path, "events") == 1
    assert count_rows(tmp_path, "messages") == 1
    assert count_rows(tmp_path, "applications") == 1


def test_unclassified_events_read_as_needing_review(
    store: Store, make_message: MessageFactory
) -> None:
    """Between a clear and a reclassify, an event has no confidence to report."""
    message = make_message(message_id="m1", thread_id="t1")
    store.link_and_record_event(message, classify(message), now=NOW)
    store.clear_classifications(only_unreviewed=False)
    event = store.list_events()[0]
    assert event.confidence == 0.0
    assert event.needs_review


# --- review queue -----------------------------------------------------------


def test_pending_review_lists_flagged_messages_oldest_first(
    store: Store, make_message: MessageFactory
) -> None:
    """The queue is ordered by arrival so the reviewer works through history."""
    late = make_message(message_id="m1", thread_id="t1", received_at=NOW)
    early = make_message(message_id="m2", thread_id="t2", received_at=NOW - timedelta(days=3))
    confident = make_message(message_id="m3", thread_id="t3")
    store.link_and_record_event(late, classify(late, needs_review=True), now=NOW)
    store.link_and_record_event(
        early,
        classify(early, company="Globex", company_key="globex", needs_review=True),
        now=NOW,
    )
    store.link_and_record_event(
        confident, classify(confident, company="Initech", company_key="initech"), now=NOW
    )
    queue = store.pending_review()
    assert [item.message.message_id for item in queue] == ["m2", "m1"]


def test_pending_review_respects_its_limit(store: Store, make_message: MessageFactory) -> None:
    """The CLI walks the queue in pages."""
    for index in range(3):
        message = make_message(
            message_id=f"m{index}",
            thread_id=f"t{index}",
            received_at=NOW - timedelta(days=index),
        )
        store.link_and_record_event(
            message,
            classify(message, company_key=f"company {index}", needs_review=True),
            now=NOW,
        )
    assert len(store.pending_review(limit=2)) == 2
    assert len(store.pending_review()) == 3


def test_pending_review_carries_the_message_and_the_suggestion(
    store: Store, make_message: MessageFactory
) -> None:
    """A ReviewItem has everything the reviewer needs, including the linked application."""
    message = make_message(
        message_id="m1",
        thread_id="t1",
        subject="Your application to Acme",
        body_text="Thanks for applying.",
        labels=["INBOX", "CATEGORY_UPDATES"],
        headers={"list-unsubscribe": "<mailto:x@acme.test>"},
    )
    store.link_and_record_event(message, classify(message, needs_review=True), now=NOW)
    item = store.pending_review()[0]
    assert item.message == message
    assert item.classification.needs_review
    assert item.suggested_application_id == store.list_applications(now=NOW)[0].application_id


def test_pending_review_excludes_corrected_messages(
    store: Store, make_message: MessageFactory
) -> None:
    """A corrected message has left the queue (I6)."""
    message = make_message(message_id="m1", thread_id="t1")
    store.link_and_record_event(message, classify(message, needs_review=True), now=NOW)
    assert len(store.pending_review()) == 1
    store.set_override(Override(message_id="m1", event_type=EventType.REJECTION, corrected_at=NOW))
    assert store.pending_review() == []


def test_pending_review_reports_unlinked_suggestions_as_none(
    store: Store, make_message: MessageFactory
) -> None:
    """An UNKNOWN in the queue has no application to suggest."""
    message = make_message(message_id="m1", thread_id="t1")
    store.link_and_record_event(
        message,
        classify(message, event_type=EventType.UNKNOWN, needs_review=True),
        now=NOW,
    )
    assert store.pending_review()[0].suggested_application_id is None


# --- acceptance -------------------------------------------------------------


def test_accept_classification_clears_the_review_flag(
    store: Store, make_message: MessageFactory
) -> None:
    """Accepting keeps the classifier's fields but takes the message off the queue."""
    message = make_message(message_id="m1", thread_id="t1")
    store.link_and_record_event(message, classify(message, needs_review=True), now=NOW)
    store.accept_classification("m1", now=NOW)
    assert store.pending_review() == []
    event = store.list_events()[0]
    assert event.event_type is EventType.APPLICATION_RECEIVED
    assert not event.needs_review
    assert not event.is_overridden
    assert not store.list_applications(now=NOW)[0].needs_review


def test_accept_classification_is_retained_as_labeled_data(
    store: Store, make_message: MessageFactory, tmp_path: Path
) -> None:
    """The acceptance is recorded, with all correction columns NULL."""
    message = make_message(message_id="m1", thread_id="t1")
    store.link_and_record_event(message, classify(message, needs_review=True), now=NOW)
    store.accept_classification("m1", now=NOW)
    connection = sqlite3.connect(tmp_path / "jobtrack.db")
    connection.row_factory = sqlite3.Row
    row = connection.execute(
        "SELECT event_type AS event_type, company AS company, role AS role, note AS note "
        "FROM overrides WHERE message_id = ?",
        ("m1",),
    ).fetchone()
    connection.close()
    assert row["event_type"] is None
    assert row["company"] is None
    assert row["role"] is None
    assert "accepted" in row["note"]


def test_accept_classification_does_not_clobber_a_correction(
    store: Store, make_message: MessageFactory
) -> None:
    """An acceptance must never overwrite a real correction (I6)."""
    message = make_message(message_id="m1", thread_id="t1")
    store.link_and_record_event(message, classify(message, needs_review=True), now=NOW)
    store.set_override(Override(message_id="m1", event_type=EventType.REJECTION, corrected_at=NOW))
    store.accept_classification("m1", now=NOW + timedelta(days=1))
    assert store.list_events()[0].event_type is EventType.REJECTION


def test_accept_classification_requires_a_classification(store: Store) -> None:
    """Accepting something that was never classified is an error, not a silent no-op."""
    with pytest.raises(StoreError, match="no classification recorded"):
        store.accept_classification("nope", now=NOW)


def test_accept_classification_rejects_a_naive_clock(
    store: Store, make_message: MessageFactory
) -> None:
    """The acceptance timestamp obeys I7 like every other datetime."""
    message = make_message(message_id="m1", thread_id="t1")
    store.link_and_record_event(message, classify(message, needs_review=True), now=NOW)
    with pytest.raises(StoreError, match="naive datetime"):
        # Deliberately naive.
        store.accept_classification("m1", now=datetime(2026, 8, 18, 12, 0, 0))


# --- failure paths ----------------------------------------------------------


def test_mutations_on_a_closed_store_raise_store_error(
    tmp_path: Path, make_message: MessageFactory
) -> None:
    """A dead connection fails every write as a StoreError, never a sqlite3.Error."""
    opened = Store.open(tmp_path / "jobtrack.db")
    opened.migrate()
    message = make_message(message_id="m1", thread_id="t1")
    opened.link_and_record_event(message, classify(message), now=NOW)
    opened.close()

    with pytest.raises(StoreError):
        opened.record_message(message)
    with pytest.raises(StoreError):
        opened.clear_classifications()
    with pytest.raises(StoreError):
        opened.set_override(
            Override(message_id="m1", event_type=EventType.REJECTION, corrected_at=NOW)
        )
    with pytest.raises(StoreError):
        opened.accept_classification("m1", now=NOW)
    with pytest.raises(StoreError):
        opened.pending_review()


def test_relinking_needs_a_classification_to_match_on(
    store: Store, make_message: MessageFactory
) -> None:
    """After a full wipe there is no company_key to link with, so the event stays loose."""
    message = make_message(message_id="m1", thread_id="t1")
    store.link_and_record_event(message, classify(message, event_type=EventType.UNKNOWN), now=NOW)
    store.clear_classifications(only_unreviewed=False)
    store.set_override(Override(message_id="m1", event_type=EventType.INTERVIEW, corrected_at=NOW))
    assert store.list_events()[0].application_id is None
    assert store.list_applications(now=NOW) == []


def test_a_corrected_company_is_used_when_relinking(
    store: Store, make_message: MessageFactory
) -> None:
    """The corrected display name reaches the application created by the relink."""
    message = make_message(message_id="m1", thread_id="t1")
    store.link_and_record_event(message, classify(message, event_type=EventType.UNKNOWN), now=NOW)
    store.set_override(
        Override(
            message_id="m1",
            event_type=EventType.APPLICATION_RECEIVED,
            company="Globex Corporation",
            corrected_at=NOW,
        )
    )
    application = store.list_applications(now=NOW)[0]
    assert application.company == "Globex Corporation"
    assert application.company_key == "acme robotics"
