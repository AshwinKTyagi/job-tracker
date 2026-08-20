"""Tests for the reclassify read-back and in-place refresh added for M6.

``link_and_record_event`` is a deliberate no-op on replay (I1), so ``reclassify`` needs a
separate path. These cover what that path must guarantee: the classifier's own output is
refreshed, a human ``Override`` is never clobbered (I6), and the application link follows
the corrected history (I4).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

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
    company: str | None = "Acme Robotics",
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
        evidence=["ack.body.application_received"],
        classifier_name="rules",
        classifier_version="1.0.0",
    )


def test_list_messages_returns_stored_messages_oldest_first(
    store: Store, make_message: MessageFactory
) -> None:
    """The read-back reclassify needs returns every message in receipt order."""
    late = make_message(message_id="late", received_at=NOW)
    early = make_message(message_id="early", received_at=datetime(2026, 1, 1, tzinfo=UTC))
    for message in (late, early):
        store.link_and_record_event(message, classify(message), now=NOW)

    assert [m.message_id for m in store.list_messages()] == ["early", "late"]


def test_list_messages_round_trips_every_field(store: Store, make_message: MessageFactory) -> None:
    """A rebuilt RawMessage must equal the one that was stored — the classifier is pure (I2)."""
    original = make_message(
        subject="Thanks for applying",
        body_text="We received your application.",
        labels=["INBOX", "CATEGORY_UPDATES"],
        headers={"List-Unsubscribe": "<mailto:x@greenhouse.io>"},
    )
    store.link_and_record_event(original, classify(original), now=NOW)

    assert store.list_messages() == [original]


def test_list_messages_is_empty_on_a_fresh_store(store: Store) -> None:
    """An empty mailbox reads back as an empty list, not an error."""
    assert store.list_messages() == []


def test_reapply_updates_the_stored_event_type(store: Store, make_message: MessageFactory) -> None:
    """A rules change that retypes a message must actually move the stored event."""
    message = make_message()
    store.link_and_record_event(message, classify(message), now=NOW)

    refreshed = store.reapply_classification(
        message, classify(message, event_type=EventType.REJECTION), now=NOW
    )

    assert refreshed.event_type is EventType.REJECTION
    assert [e.event_type for e in store.list_events()] == [EventType.REJECTION]


def test_reapply_does_not_duplicate_the_event(store: Store, make_message: MessageFactory) -> None:
    """Refreshing is an update, not a second append — one message is still one event (I1)."""
    message = make_message()
    store.link_and_record_event(message, classify(message), now=NOW)
    first_id = store.list_events()[0].event_id

    store.reapply_classification(message, classify(message, confidence=0.4), now=NOW)

    events = store.list_events()
    assert len(events) == 1
    assert events[0].event_id == first_id


def test_reapply_refreshes_confidence_and_review_flag(
    store: Store, make_message: MessageFactory
) -> None:
    """The classification row is rewritten, so a newly-unsure verdict re-enters the queue."""
    message = make_message()
    store.link_and_record_event(message, classify(message), now=NOW)

    store.reapply_classification(
        message, classify(message, confidence=0.2, needs_review=True), now=NOW
    )

    assert [item.message.message_id for item in store.pending_review()] == [message.message_id]


def test_reapply_never_clobbers_an_override(store: Store, make_message: MessageFactory) -> None:
    """I6: a human correction still wins after the classifier is re-run."""
    message = make_message()
    store.link_and_record_event(message, classify(message), now=NOW)
    store.set_override(
        Override(
            message_id=message.message_id,
            event_type=EventType.OFFER,
            corrected_at=NOW,
        )
    )

    refreshed = store.reapply_classification(
        message, classify(message, event_type=EventType.REJECTION), now=NOW
    )

    assert refreshed.event_type is EventType.OFFER
    assert refreshed.is_overridden is True


def test_reapply_keeps_an_overridden_status_derivable(
    store: Store, make_message: MessageFactory
) -> None:
    """The derived status (I4) follows the override, not the fresh classification."""
    message = make_message()
    store.link_and_record_event(message, classify(message), now=NOW)
    store.set_override(
        Override(message_id=message.message_id, event_type=EventType.OFFER, corrected_at=NOW)
    )

    store.reapply_classification(
        message, classify(message, event_type=EventType.REJECTION), now=NOW
    )

    rows = store.list_applications(now=NOW)
    assert [row.status for row in rows] == [ApplicationStatus.OFFER]


def test_reapply_records_a_message_that_has_no_event_yet(
    store: Store, make_message: MessageFactory
) -> None:
    """Calling over the whole mailbox is safe: an unrecorded message is simply recorded."""
    message = make_message()

    recorded = store.reapply_classification(message, classify(message), now=NOW)

    assert recorded.message_id == message.message_id
    assert store.has_message(message.message_id) is True


def test_reapply_unlinks_when_the_new_verdict_is_unknown(
    store: Store, make_message: MessageFactory
) -> None:
    """Retyping to UNKNOWN drops the application link and cleans up the orphan."""
    message = make_message()
    store.link_and_record_event(message, classify(message), now=NOW)
    assert store.list_applications(now=NOW) != []

    refreshed = store.reapply_classification(
        message, classify(message, event_type=EventType.UNKNOWN), now=NOW
    )

    assert refreshed.application_id is None
    assert store.list_applications(now=NOW) == []


def test_reapply_is_deterministic(store: Store, make_message: MessageFactory) -> None:
    """Running it twice with the same input leaves identical state."""
    message = make_message()
    store.link_and_record_event(message, classify(message), now=NOW)

    first = store.reapply_classification(message, classify(message), now=NOW)
    second = store.reapply_classification(message, classify(message), now=NOW)

    assert first == second
