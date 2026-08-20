"""Tests for the Store repository methods (store/repo.py).

Every test runs against a real, migrated SQLite file under tmp_path (via the `store`
fixture) rather than a mock, per CLAUDE.md.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pytest

from jobtrack.errors import StoreError
from jobtrack.models import Classification, EventType, Override, RawMessage
from jobtrack.store import Store


def _message(**overrides: object) -> RawMessage:
    defaults: dict[str, object] = {
        "message_id": "msg-0001",
        "thread_id": "thread-0001",
        "received_at": datetime(2026, 8, 1, 9, 0, 0, tzinfo=UTC),
        "from_email": "no-reply@greenhouse.io",
        "from_name": "Acme Robotics",
        "subject": "Thanks for applying to Acme Robotics",
        "body_text": "We received your application.",
    }
    defaults.update(overrides)
    return RawMessage.model_validate(defaults)


# --- has_message / record_message ---------------------------------------------------------


def test_has_message_false_before_recording(store: Store) -> None:
    assert store.has_message("msg-0001") is False


def test_record_message_then_has_message(store: Store) -> None:
    store.record_message(_message())
    assert store.has_message("msg-0001") is True


def test_record_message_is_idempotent(store: Store) -> None:
    """A second call for the same id is a no-op, not an error (I1)."""
    message = _message()
    store.record_message(message)
    store.record_message(message)
    count = store._connection.execute("SELECT COUNT(*) AS n FROM messages").fetchone()
    assert count["n"] == 1


# --- record_classification -----------------------------------------------------------------


def test_record_classification_round_trips(
    store: Store, make_classification: Callable[..., Classification]
) -> None:
    store.record_message(_message())
    store.record_classification(make_classification())
    row = store._connection.execute(
        "SELECT company, event_type FROM classifications WHERE message_id = ?", ("msg-0001",)
    ).fetchone()
    assert row["company"] == "Acme Robotics"
    assert row["event_type"] == "application_received"


def test_record_classification_upserts_on_reclassify(
    store: Store, make_classification: Callable[..., Classification]
) -> None:
    """A second record_classification for the same message_id replaces the first."""
    store.record_message(_message())
    store.record_classification(make_classification(event_type=EventType.APPLICATION_RECEIVED))
    store.record_classification(make_classification(event_type=EventType.REJECTION))
    row = store._connection.execute(
        "SELECT event_type FROM classifications WHERE message_id = ?", ("msg-0001",)
    ).fetchone()
    assert row["event_type"] == "rejection"
    count = store._connection.execute("SELECT COUNT(*) AS n FROM classifications").fetchone()
    assert count["n"] == 1


# --- link_and_record_event: idempotency (I1) and append-only (I5) --------------------------


def test_link_and_record_event_creates_application_and_event(
    store: Store, now: datetime, make_classification: Callable[..., Classification]
) -> None:
    message = _message()
    store.record_message(message)
    classification = make_classification()

    event = store.link_and_record_event(message, classification, now=now)

    assert event.application_id is not None
    assert event.event_type == EventType.APPLICATION_RECEIVED
    assert event.message_id == "msg-0001"
    apps = store.list_applications(now=now)
    assert len(apps) == 1
    assert apps[0].application_id == event.application_id
    assert apps[0].company == "Acme Robotics"


def test_link_and_record_event_unknown_has_no_application(
    store: Store, now: datetime, make_classification: Callable[..., Classification]
) -> None:
    message = _message()
    store.record_message(message)
    classification = make_classification(
        event_type=EventType.UNKNOWN, company=None, company_key=None, confidence=0.0
    )

    event = store.link_and_record_event(message, classification, now=now)

    assert event.application_id is None
    assert store.list_applications(now=now) == []
    all_events = store.list_events()
    assert len(all_events) == 1


def test_link_and_record_event_is_idempotent_on_message_id(
    store: Store, now: datetime, make_classification: Callable[..., Classification]
) -> None:
    """Processing the same message twice produces no second event, no second application (I1)."""
    message = _message()
    store.record_message(message)
    classification = make_classification()

    first = store.link_and_record_event(message, classification, now=now)
    second = store.link_and_record_event(message, classification, now=now)

    assert first.event_id == second.event_id
    assert first.application_id == second.application_id
    assert len(store.list_events()) == 1
    assert len(store.list_applications(now=now)) == 1


def test_events_are_append_only_no_update_or_delete_path(
    store: Store, now: datetime, make_classification: Callable[..., Classification]
) -> None:
    """There is no Store method that mutates or deletes an event row (I5): a correction is
    only ever visible through an Override, joined in at read time."""
    message = _message()
    store.record_message(message)
    classification = make_classification(event_type=EventType.APPLICATION_RECEIVED)
    event = store.link_and_record_event(message, classification, now=now)

    override = Override(
        message_id=message.message_id,
        event_type=EventType.REJECTION,
        corrected_at=now,
    )
    store.set_override(override)

    stored_event_type = store._connection.execute(
        "SELECT event_type FROM events WHERE event_id = ?", (event.event_id,)
    ).fetchone()["event_type"]
    assert stored_event_type == "application_received", "raw event row must never be mutated"

    events_after = store.list_events()
    assert events_after[0].event_type == EventType.REJECTION, "override wins at read time"
    assert events_after[0].is_overridden is True


def test_second_application_created_for_different_company(
    store: Store, now: datetime, make_classification: Callable[..., Classification]
) -> None:
    msg_a = _message(message_id="msg-a", thread_id="thread-a")
    msg_b = _message(message_id="msg-b", thread_id="thread-b", from_email="no-reply@lever.co")
    store.record_message(msg_a)
    store.record_message(msg_b)
    store.link_and_record_event(
        msg_a, make_classification(message_id="msg-a", company_key="acme robotics"), now=now
    )
    store.link_and_record_event(
        msg_b,
        make_classification(
            message_id="msg-b",
            company="Wayne Enterprises",
            company_key="wayne enterprises",
        ),
        now=now,
    )

    apps = store.list_applications(now=now)
    assert len(apps) == 2


def test_same_thread_links_to_same_application(
    store: Store, now: datetime, make_classification: Callable[..., Classification]
) -> None:
    """A follow-up message in the same thread links to the existing application (linker rule 1)."""
    first_msg = _message(message_id="msg-1", thread_id="thread-shared")
    second_msg = _message(
        message_id="msg-2",
        thread_id="thread-shared",
        received_at=datetime(2026, 8, 5, 9, 0, 0, tzinfo=UTC),
        subject="Interview invitation",
    )
    store.record_message(first_msg)
    store.record_message(second_msg)

    first_event = store.link_and_record_event(
        first_msg, make_classification(message_id="msg-1"), now=now
    )
    second_event = store.link_and_record_event(
        second_msg,
        make_classification(message_id="msg-2", event_type=EventType.INTERVIEW),
        now=now,
    )

    assert second_event.application_id == first_event.application_id
    assert len(store.list_applications(now=now)) == 1


# --- overrides survive reclassify (I6) ------------------------------------------------------


def test_override_survives_clear_and_reclassify(
    store: Store, now: datetime, make_classification: Callable[..., Classification]
) -> None:
    """Simulated reclassify: clear classifications, record a new one, override still wins."""
    message = _message()
    store.record_message(message)
    classification = make_classification(event_type=EventType.APPLICATION_RECEIVED)
    event = store.link_and_record_event(message, classification, now=now)

    override = Override(
        message_id=message.message_id,
        event_type=EventType.OFFER,
        corrected_at=now,
        note="actually got an offer",
    )
    store.set_override(override)

    # Simulate `jobtrack reclassify`: drop and re-record the classification.
    store.clear_classifications(only_unreviewed=False)
    store.record_classification(make_classification(event_type=EventType.REJECTION))

    events_after = store.list_events()
    assert len(events_after) == 1
    assert events_after[0].event_id == event.event_id
    assert events_after[0].event_type == EventType.OFFER, "override must survive reclassify"


# --- pending_review / accept_classification / clear_classifications ------------------------


def test_pending_review_lists_unreviewed_low_confidence_messages(
    store: Store, now: datetime, make_classification: Callable[..., Classification]
) -> None:
    message = _message()
    store.record_message(message)
    store.record_classification(make_classification(needs_review=True, confidence=0.2))

    items = store.pending_review()

    assert len(items) == 1
    assert items[0].message.message_id == "msg-0001"
    assert items[0].classification.needs_review is True


def test_pending_review_excludes_messages_with_an_override(
    store: Store, now: datetime, make_classification: Callable[..., Classification]
) -> None:
    message = _message()
    store.record_message(message)
    store.record_classification(make_classification(needs_review=True, confidence=0.2))
    store.set_override(
        Override(message_id=message.message_id, event_type=EventType.REJECTION, corrected_at=now)
    )

    assert store.pending_review() == []


def test_pending_review_respects_limit_and_ordering(
    store: Store, now: datetime, make_classification: Callable[..., Classification]
) -> None:
    for i in range(3):
        message = _message(
            message_id=f"msg-{i}",
            received_at=datetime(2026, 8, 1 + i, 9, 0, 0, tzinfo=UTC),
        )
        store.record_message(message)
        store.record_classification(
            make_classification(message_id=f"msg-{i}", needs_review=True, confidence=0.1)
        )

    items = store.pending_review(limit=2)

    assert [item.message.message_id for item in items] == ["msg-0", "msg-1"]


def test_accept_classification_clears_needs_review_without_altering_fields(
    store: Store, now: datetime, make_classification: Callable[..., Classification]
) -> None:
    message = _message()
    store.record_message(message)
    store.record_classification(
        make_classification(needs_review=True, confidence=0.2, company="Acme Robotics")
    )

    store.accept_classification(message.message_id, now=now)

    row = store._connection.execute(
        "SELECT needs_review, company, confidence FROM classifications WHERE message_id = ?",
        (message.message_id,),
    ).fetchone()
    assert row["needs_review"] == 0
    assert row["company"] == "Acme Robotics"
    assert row["confidence"] == pytest.approx(0.2)
    assert store.pending_review() == []


def test_accept_classification_raises_for_unknown_message(store: Store, now: datetime) -> None:
    with pytest.raises(StoreError):
        store.accept_classification("does-not-exist", now=now)


def test_clear_classifications_only_unreviewed_preserves_accepted(
    store: Store, now: datetime, make_classification: Callable[..., Classification]
) -> None:
    accepted_msg = _message(message_id="msg-accepted")
    pending_msg = _message(message_id="msg-pending")
    store.record_message(accepted_msg)
    store.record_message(pending_msg)
    store.record_classification(make_classification(message_id="msg-accepted", needs_review=True))
    store.record_classification(make_classification(message_id="msg-pending", needs_review=True))
    store.accept_classification("msg-accepted", now=now)

    cleared = store.clear_classifications(only_unreviewed=True)

    assert cleared == 1
    remaining = {
        row["message_id"]
        for row in store._connection.execute("SELECT message_id FROM classifications").fetchall()
    }
    assert remaining == {"msg-accepted"}


def test_clear_classifications_all_clears_everything_but_not_messages(
    store: Store, now: datetime, make_classification: Callable[..., Classification]
) -> None:
    message = _message()
    store.record_message(message)
    store.record_classification(make_classification())

    cleared = store.clear_classifications(only_unreviewed=False)

    assert cleared == 1
    count = store._connection.execute("SELECT COUNT(*) AS n FROM classifications").fetchone()
    assert count["n"] == 0
    assert store.has_message(message.message_id) is True


# --- sync cursor -----------------------------------------------------------------------------


def test_get_cursor_none_before_any_sync(store: Store) -> None:
    assert store.get_cursor("gmail") is None


def test_set_and_get_cursor_round_trips(store: Store, now: datetime) -> None:
    store.set_cursor("gmail", "history-123", synced_at=now)
    assert store.get_cursor("gmail") == "history-123"


def test_set_cursor_upserts(store: Store, now: datetime) -> None:
    store.set_cursor("gmail", "history-123", synced_at=now)
    store.set_cursor("gmail", "history-456", synced_at=now + timedelta(days=1))
    assert store.get_cursor("gmail") == "history-456"
    count = store._connection.execute("SELECT COUNT(*) AS n FROM sync_state").fetchone()
    assert count["n"] == 1


# --- list_applications / get_application filters ---------------------------------------------


def test_list_applications_filters_by_company_key(
    store: Store, now: datetime, make_classification: Callable[..., Classification]
) -> None:
    msg_a = _message(message_id="msg-a", thread_id="thread-a")
    msg_b = _message(message_id="msg-b", thread_id="thread-b")
    store.record_message(msg_a)
    store.record_message(msg_b)
    store.link_and_record_event(
        msg_a, make_classification(message_id="msg-a", company_key="acme robotics"), now=now
    )
    store.link_and_record_event(
        msg_b,
        make_classification(
            message_id="msg-b", company="Wayne Enterprises", company_key="wayne enterprises"
        ),
        now=now,
    )

    filtered = store.list_applications(now=now, company="wayne enterprises")

    assert len(filtered) == 1
    assert filtered[0].company == "Wayne Enterprises"


def test_get_application_returns_none_for_unknown_id(store: Store, now: datetime) -> None:
    assert store.get_application("does-not-exist", now=now) is None


def test_get_application_computes_days_since_last_event(
    store: Store, now: datetime, make_classification: Callable[..., Classification]
) -> None:
    message = _message(received_at=now - timedelta(days=7))
    store.record_message(message)
    event = store.link_and_record_event(message, make_classification(), now=now - timedelta(days=7))

    app = store.get_application(event.application_id, now=now)  # type: ignore[arg-type]

    assert app is not None
    assert app.days_since_last_event == 7


# --- match_candidates -------------------------------------------------------------------------


def test_match_candidates_by_company_key(
    store: Store, now: datetime, make_classification: Callable[..., Classification]
) -> None:
    msg = _message()
    store.record_message(msg)
    event = store.link_and_record_event(msg, make_classification(), now=now)

    candidates = store.match_candidates("acme robotics", "unrelated-thread", within_days=180)

    assert len(candidates) == 1
    assert candidates[0].application_id == event.application_id
    assert "thread-0001" in candidates[0].thread_ids


def test_match_candidates_by_thread_id_regardless_of_company_key(
    store: Store, now: datetime, make_classification: Callable[..., Classification]
) -> None:
    msg = _message()
    store.record_message(msg)
    event = store.link_and_record_event(msg, make_classification(), now=now)

    candidates = store.match_candidates("some other company", "thread-0001", within_days=180)

    assert any(c.application_id == event.application_id for c in candidates)


def test_match_candidates_empty_when_nothing_matches(store: Store) -> None:
    assert store.match_candidates("nobody", "no-thread", within_days=180) == []
