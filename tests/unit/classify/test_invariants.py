"""The cross-module invariants M2 is responsible for: I2 purity and I3 precedence.

These are the properties other modules are allowed to assume. A failure here is not a bug in
the classifier so much as a broken promise to M3 and M6.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from jobtrack.classify.rules import RulesClassifier, resolve_event_type, score_event_types
from jobtrack.constants import EVENT_PRECEDENCE
from jobtrack.models import Classification, EventType, RawMessage

# --------------------------------------------------------------------------------------
# I2 — the classifier is pure
# --------------------------------------------------------------------------------------


def test_classifying_twice_is_byte_identical(
    email_fixtures: list[tuple[str, RawMessage]],
) -> None:
    """I2: same RawMessage in ⇒ byte-identical Classification out.

    Compared as serialized JSON rather than by field, because that is the form the store
    persists and the form a difference would actually corrupt.
    """
    classifier = RulesClassifier()
    for stem, message in email_fixtures:
        first = classifier.classify(message).model_dump_json()
        second = classifier.classify(message).model_dump_json()
        assert first == second, f"{stem} is not deterministic"


def test_a_fresh_classifier_produces_the_same_output(
    email_fixtures: list[tuple[str, RawMessage]],
) -> None:
    """No instance state leaks between calls — a second classifier agrees with the first."""
    for stem, message in email_fixtures:
        first = RulesClassifier().classify(message).model_dump_json()
        second = RulesClassifier().classify(message).model_dump_json()
        assert first == second, f"{stem} depends on classifier instance state"


def test_evidence_order_is_stable(email_fixtures: list[tuple[str, RawMessage]]) -> None:
    """Evidence is a list, not a set: its order is part of the byte-identical guarantee."""
    classifier = RulesClassifier()
    for _stem, message in email_fixtures:
        assert classifier.classify(message).evidence == classifier.classify(message).evidence


def test_classification_does_not_depend_on_the_clock(
    make_message: Callable[..., RawMessage],
) -> None:
    """received_at must not influence the classification — only the text does.

    The classifier takes no ``now`` because it needs no clock at all. Two otherwise identical
    messages a year apart must classify the same.
    """
    from datetime import UTC, datetime

    common = {
        "subject": "Thanks for applying to Acme Robotics",
        "body_text": "We have received your application for the Data Engineer role.",
        "from_email": "no-reply@greenhouse.io",
        "from_name": "Acme Robotics",
        "message_id": "same-id",
    }
    early = make_message(received_at=datetime(2024, 1, 1, tzinfo=UTC), **common)
    late = make_message(received_at=datetime(2026, 12, 31, tzinfo=UTC), **common)

    classifier = RulesClassifier()
    assert (
        classifier.classify(early).model_dump_json() == classifier.classify(late).model_dump_json()
    )


def test_classify_batch_preserves_order(email_fixtures: list[tuple[str, RawMessage]]) -> None:
    """classify_batch is order-preserving and agrees with classify one-by-one."""
    messages = [message for _stem, message in email_fixtures]
    classifier = RulesClassifier()

    batch = classifier.classify_batch(messages)
    singly = [classifier.classify(message) for message in messages]

    assert [c.message_id for c in batch] == [m.message_id for m in messages]
    assert [c.model_dump_json() for c in batch] == [c.model_dump_json() for c in singly]


def test_classify_batch_of_nothing_is_empty() -> None:
    """An empty batch is an empty list, not an error."""
    assert RulesClassifier().classify_batch([]) == []


# --------------------------------------------------------------------------------------
# I3 — precedence, not first-match
# --------------------------------------------------------------------------------------


def test_precedence_covers_every_event_type() -> None:
    """resolve_event_type indexes into EVENT_PRECEDENCE, so a missing member is a crash."""
    assert set(EVENT_PRECEDENCE) == set(EventType)
    assert len(EVENT_PRECEDENCE) == len(set(EVENT_PRECEDENCE))


def test_rejection_outranks_application_received(
    make_message: Callable[..., RawMessage],
) -> None:
    """I3, the case the whole design exists for.

    A message that restates the application language AND rejects must resolve to REJECTION.
    First-match-wins over the subject would answer APPLICATION_RECEIVED.
    """
    message = make_message(
        subject="Thanks for applying to Acme Robotics",
        body_text=(
            "Thanks for applying to Acme Robotics. We have received your application. "
            "Unfortunately we have decided not to move forward with other candidates at "
            "this time."
        ),
    )

    scores = score_event_types(message)
    assert EventType.APPLICATION_RECEIVED in scores, "the confirmation half must really match"
    assert EventType.REJECTION in scores, "the rejection half must really match"

    winner, rule_ids = resolve_event_type(scores)
    assert winner is EventType.REJECTION
    assert rule_ids == scores[EventType.REJECTION]


def test_scoring_never_stops_at_the_first_match(
    make_message: Callable[..., RawMessage],
) -> None:
    """Every type is scored, not just the winner — that is what makes precedence possible."""
    message = make_message(
        subject="Thanks for applying to Acme Robotics",
        body_text=(
            "Thanks for applying. We have received your application. Unfortunately we are "
            "moving ahead with other candidates."
        ),
    )
    scores = score_event_types(message)
    assert len(scores) >= 2, f"expected several types to match, got {list(scores)}"


@pytest.mark.parametrize(
    ("higher", "lower"),
    [
        (EventType.WITHDRAWN, EventType.REJECTION),
        (EventType.REJECTION, EventType.OFFER),
        (EventType.OFFER, EventType.INTERVIEW),
        (EventType.INTERVIEW, EventType.ASSESSMENT),
        (EventType.ASSESSMENT, EventType.APPLICATION_RECEIVED),
        (EventType.APPLICATION_RECEIVED, EventType.RECRUITER_OUTREACH),
        (EventType.RECRUITER_OUTREACH, EventType.UNKNOWN),
    ],
)
def test_resolve_respects_every_precedence_step(higher: EventType, lower: EventType) -> None:
    """Each adjacent pair in EVENT_PRECEDENCE resolves the documented way."""
    scores = {lower: ["lower.rule"], higher: ["higher.rule"]}
    winner, rule_ids = resolve_event_type(scores)
    assert winner is higher
    assert rule_ids == ["higher.rule"]


def test_resolve_of_nothing_is_unknown() -> None:
    """No match at all is UNKNOWN with no evidence, not an exception."""
    assert resolve_event_type({}) == (EventType.UNKNOWN, [])


def test_resolve_ignores_types_with_no_rules() -> None:
    """An empty rule list is not a match, even for a high-precedence type."""
    scores: dict[EventType, list[str]] = {
        EventType.REJECTION: [],
        EventType.APPLICATION_RECEIVED: ["ack.subject.thanks_for_applying"],
    }
    winner, _ = resolve_event_type(scores)
    assert winner is EventType.APPLICATION_RECEIVED


def test_resolve_does_not_mutate_its_input() -> None:
    """The returned rule list is a copy; callers must not be able to corrupt the score map."""
    scores = {EventType.REJECTION: ["rej.body.unfortunately"]}
    _winner, rule_ids = resolve_event_type(scores)
    rule_ids.append("tampered")
    assert scores[EventType.REJECTION] == ["rej.body.unfortunately"]


def test_score_map_is_keyed_in_precedence_order(
    make_message: Callable[..., RawMessage],
) -> None:
    """Canonical key order keeps the output byte-identical regardless of table order (I2)."""
    message = make_message(
        subject="Thanks for applying to Acme Robotics",
        body_text="Unfortunately we are moving ahead with other candidates.",
    )
    keys = list(score_event_types(message))
    positions = [EVENT_PRECEDENCE.index(key) for key in keys]
    assert positions == sorted(positions)


# --------------------------------------------------------------------------------------
# The Protocol contract
# --------------------------------------------------------------------------------------


def test_unparseable_message_is_unknown_not_an_exception(
    make_message: Callable[..., RawMessage],
) -> None:
    """The Classifier Protocol forbids raising for ordinary input."""
    result = RulesClassifier().classify(make_message(subject="", body_text=""))
    assert result.event_type is EventType.UNKNOWN
    assert result.confidence == 0.0
    assert result.needs_review is True
    assert result.evidence == []


def test_unknown_carries_no_extracted_fields(
    make_message: Callable[..., RawMessage],
) -> None:
    """Extraction is skipped for UNKNOWN, so a newsletter's sender never becomes a company."""
    result = RulesClassifier().classify(
        make_message(
            subject="This week in tech",
            body_text="A weekly digest of industry news.",
            from_name="Tech Weekly",
            from_email="digest@techweekly.example",
        )
    )
    assert result.event_type is EventType.UNKNOWN
    assert result.company is None
    assert result.company_key is None
    assert result.role is None
    assert result.location is None


def test_classification_is_attributed_to_the_classifier(
    make_message: Callable[..., RawMessage],
) -> None:
    """Stored rows must stay attributable to the exact backend and version that wrote them."""
    classifier = RulesClassifier()
    result: Classification = classifier.classify(make_message(subject="Thanks for applying"))
    assert result.classifier_name == classifier.name == "rules"
    assert result.classifier_version == classifier.version
