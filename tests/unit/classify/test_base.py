"""Tests for jobtrack.classify.base: the Classifier protocol and CompositeClassifier."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from jobtrack.classify.base import Classifier, CompositeClassifier
from jobtrack.classify.rules import RulesClassifier
from jobtrack.models import Classification, EventType, RawMessage

MakeMessage = Callable[..., RawMessage]


class _StubClassifier:
    """A minimal Classifier double: always returns a fixed Classification."""

    name = "stub"
    version = "0.0.0"

    def __init__(self, event_type: EventType, confidence: float) -> None:
        self._event_type = event_type
        self._confidence = confidence
        self.calls: list[str] = []

    def classify(self, message: RawMessage) -> Classification:
        self.calls.append(message.message_id)
        return Classification(
            message_id=message.message_id,
            event_type=self._event_type,
            confidence=self._confidence,
            classifier_name=self.name,
            classifier_version=self.version,
        )

    def classify_batch(self, messages: Sequence[RawMessage]) -> list[Classification]:
        return [self.classify(message) for message in messages]


def test_rules_classifier_satisfies_classifier_protocol() -> None:
    classifier: Classifier = RulesClassifier()
    assert classifier.name == "rules"
    assert classifier.version == "1.0.0"


def test_composite_pass_through_with_no_fallback_returns_primary_result(
    make_message: MakeMessage,
) -> None:
    """Phase 1: fallback=None is a pass-through, regardless of primary confidence."""
    message = make_message()
    primary = _StubClassifier(EventType.UNKNOWN, confidence=0.0)
    composite = CompositeClassifier(primary, None, min_confidence=0.6)

    result = composite.classify(message)

    assert result.event_type is EventType.UNKNOWN
    assert result.confidence == 0.0
    assert result.classifier_name == "stub"


def test_composite_defers_to_fallback_below_threshold(make_message: MakeMessage) -> None:
    message = make_message()
    primary = _StubClassifier(EventType.UNKNOWN, confidence=0.1)
    fallback = _StubClassifier(EventType.APPLICATION_RECEIVED, confidence=0.9)
    composite = CompositeClassifier(primary, fallback, min_confidence=0.6)

    result = composite.classify(message)

    assert result.event_type is EventType.APPLICATION_RECEIVED
    assert result.classifier_name == "stub"
    assert primary.calls == [message.message_id]
    assert fallback.calls == [message.message_id]


def test_composite_does_not_call_fallback_when_primary_confidence_meets_threshold(
    make_message: MakeMessage,
) -> None:
    message = make_message()
    primary = _StubClassifier(EventType.OFFER, confidence=0.8)
    fallback = _StubClassifier(EventType.APPLICATION_RECEIVED, confidence=0.9)
    composite = CompositeClassifier(primary, fallback, min_confidence=0.6)

    result = composite.classify(message)

    assert result.event_type is EventType.OFFER
    assert fallback.calls == []


def test_composite_classify_batch_is_order_preserving(make_message: MakeMessage) -> None:
    messages = [make_message() for _ in range(3)]
    primary = _StubClassifier(EventType.OFFER, confidence=0.9)
    composite = CompositeClassifier(primary, None)

    results = composite.classify_batch(messages)

    assert [r.message_id for r in results] == [m.message_id for m in messages]
