"""Tests for the M7 eval harness.

The harness is how a model gets chosen (PLAN.md §8), so its arithmetic has to be right:
a metric that quietly reports 100% on an empty population would let a bad backend through.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from jobtrack.classify.evaluate import evaluate
from jobtrack.models import Classification, EventType, RawMessage

MessageFactory = Callable[..., RawMessage]


def label(message: RawMessage, **overrides: object) -> Classification:
    """Build a human-confirmed Classification for a message."""
    fields: dict[str, object] = {
        "message_id": message.message_id,
        "event_type": EventType.REJECTION,
        "company": "2K",
        "role": "Engineering Graduate Program",
        "confidence": 1.0,
        "classifier_name": "human",
        "classifier_version": "1",
    }
    fields.update(overrides)
    return Classification.model_validate(fields)


class Fixed:
    """A classifier that always returns the Classification it was given."""

    name = "fixed"
    version = "1"

    def __init__(self, answer: Classification) -> None:
        """Args:
        answer: The verdict to return for every message.
        """
        self._answer = answer

    def classify(self, message: RawMessage) -> Classification:
        """Return the fixed answer, re-keyed to the message."""
        return self._answer.model_copy(update={"message_id": message.message_id})

    def classify_batch(self, messages: Sequence[RawMessage]) -> list[Classification]:
        """Map classify over the input."""
        return [self.classify(m) for m in messages]


def test_a_perfect_classifier_scores_one(make_message: MessageFactory) -> None:
    """Every metric maxes out when the backend agrees with every label."""
    message = make_message()
    expected = label(message)

    scores = evaluate(Fixed(expected), [(message, expected)])

    assert scores["event_type_accuracy"] == 1.0
    assert scores["confusable_pair_accuracy"] == 1.0
    assert scores["company_exact_match"] == 1.0
    assert scores["role_exact_match"] == 1.0
    assert scores["schema_compliance"] == 1.0


def test_an_empty_population_scores_zero_not_one(make_message: MessageFactory) -> None:
    """The trap this guards: 0/0 must not read as perfect."""
    scores = evaluate(Fixed(label(make_message())), [])

    assert scores["n"] == 0.0
    assert scores["event_type_accuracy"] == 0.0
    assert scores["confusable_pair_accuracy"] == 0.0


def test_the_sentinel_counts_as_a_compliance_failure(make_message: MessageFactory) -> None:
    """A zero-confidence result means the backend produced nothing usable."""
    message = make_message()
    expected = label(message)
    sentinel = expected.model_copy(
        update={"event_type": EventType.UNKNOWN, "confidence": 0.0, "company": None, "role": None}
    )

    scores = evaluate(Fixed(sentinel), [(message, expected)])

    assert scores["schema_compliance"] == 0.0
    assert scores["event_type_accuracy"] == 0.0


def test_the_confusable_pair_is_scored_separately(make_message: MessageFactory) -> None:
    """Calling a rejection an acknowledgement is the failure that disqualifies a model."""
    rejection = make_message(message_id="a")
    interview = make_message(message_id="b")
    labels = [
        (rejection, label(rejection, event_type=EventType.REJECTION)),
        (interview, label(interview, event_type=EventType.INTERVIEW)),
    ]
    always_ack = label(rejection, event_type=EventType.APPLICATION_RECEIVED)

    scores = evaluate(Fixed(always_ack), labels)

    # Only the rejection is in the confusable pair, and it was missed.
    assert scores["confusable_pair_accuracy"] == 0.0
    assert scores["event_type_accuracy"] == 0.0


def test_company_match_is_normalization_insensitive(make_message: MessageFactory) -> None:
    """'Acme Robotics, Inc.' and 'acme robotics' are the same company (I8)."""
    message = make_message()
    expected = label(message, company="Acme Robotics, Inc.")
    actual = label(message, company="acme robotics")

    scores = evaluate(Fixed(actual), [(message, expected)])

    assert scores["company_exact_match"] == 1.0


def test_fields_the_label_omits_are_not_scored(make_message: MessageFactory) -> None:
    """An unlabeled company must not count against the backend."""
    message = make_message()
    expected = label(message, company=None, role=None)

    scores = evaluate(Fixed(label(message)), [(message, expected)])

    assert scores["company_exact_match"] == 0.0
    assert scores["role_exact_match"] == 0.0
    assert scores["event_type_accuracy"] == 1.0


def test_latency_is_reported(make_message: MessageFactory) -> None:
    """Latency is a real selection criterion, so it has to be measured."""
    message = make_message()
    scores = evaluate(Fixed(label(message)), [(message, label(message))])

    assert scores["median_latency_seconds"] >= 0.0
