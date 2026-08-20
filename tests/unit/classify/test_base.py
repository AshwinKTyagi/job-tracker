"""The Classifier seam.

CompositeClassifier is the reason a local-LLM backend can land in Phase 3 without touching a
caller. These tests pin the escalation behaviour now, while the fallback is still None, so M7
inherits a specification rather than a guess.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest

from jobtrack.classify.base import Classifier, CompositeClassifier
from jobtrack.classify.rules import RulesClassifier
from jobtrack.errors import ClassificationError
from jobtrack.models import Classification, EventType, RawMessage


class StubClassifier:
    """A fake backend that answers with a fixed confidence. Stands in for M7's Ollama."""

    def __init__(self, name: str, confidence: float, event_type: EventType) -> None:
        self.name = name
        self.version = "stub-1"
        self.confidence = confidence
        self.event_type = event_type
        self.seen: list[str] = []

    def classify(self, message: RawMessage) -> Classification:
        """Answer with this stub's fixed verdict and record that it was consulted."""
        self.seen.append(message.message_id)
        return Classification(
            message_id=message.message_id,
            event_type=self.event_type,
            company="Stub Co",
            company_key="stub co",
            confidence=self.confidence,
            evidence=[f"stub.{self.name}"],
            classifier_name=self.name,
            classifier_version=self.version,
        )

    def classify_batch(self, messages: Sequence[RawMessage]) -> list[Classification]:
        """Order-preserving map of classify over the input."""
        return [self.classify(m) for m in messages]


class ExplodingClassifier(StubClassifier):
    """A backend that fails the way a dead Ollama host would."""

    def classify(self, message: RawMessage) -> Classification:
        """Always raise a JobTrackError, as a failing backend must."""
        raise ClassificationError("backend unavailable")


def test_rules_classifier_satisfies_the_protocol() -> None:
    """Callers depend on the Protocol, never on the concrete class."""
    assert isinstance(RulesClassifier(), Classifier)


def test_composite_satisfies_the_protocol() -> None:
    """The composite is itself a Classifier, so it can nest or be swapped in."""
    assert isinstance(CompositeClassifier(RulesClassifier()), Classifier)


def test_phase_one_composite_is_a_pass_through(
    email_fixtures: list[tuple[str, RawMessage]],
) -> None:
    """With fallback=None the composite must not change a single answer."""
    rules = RulesClassifier()
    composite = CompositeClassifier(rules)
    for _stem, message in email_fixtures:
        assert (
            composite.classify(message).model_dump_json()
            == rules.classify(message).model_dump_json()
        )


def test_a_confident_primary_is_not_escalated(
    make_message: Callable[..., RawMessage],
) -> None:
    """The fallback exists to see only what the rules were unsure about."""
    fallback = StubClassifier("ollama", 0.99, EventType.OFFER)
    composite = CompositeClassifier(
        StubClassifier("rules", 0.90, EventType.REJECTION), fallback, min_confidence=0.60
    )
    result = composite.classify(make_message())

    assert fallback.seen == [], "a confident primary must not cost an LLM call"
    assert result.event_type is EventType.REJECTION


def test_an_unsure_primary_is_escalated(make_message: Callable[..., RawMessage]) -> None:
    """Below the threshold, the fallback gets a look."""
    fallback = StubClassifier("ollama", 0.95, EventType.OFFER)
    composite = CompositeClassifier(
        StubClassifier("rules", 0.20, EventType.UNKNOWN), fallback, min_confidence=0.60
    )
    result = composite.classify(make_message())

    assert fallback.seen != []
    assert result.event_type is EventType.OFFER


def test_a_hesitant_fallback_cannot_make_things_worse(
    make_message: Callable[..., RawMessage],
) -> None:
    """The fallback wins only when strictly more confident."""
    composite = CompositeClassifier(
        StubClassifier("rules", 0.50, EventType.REJECTION),
        StubClassifier("ollama", 0.10, EventType.OFFER),
        min_confidence=0.60,
    )
    assert composite.classify(make_message()).event_type is EventType.REJECTION


def test_an_equally_confident_fallback_does_not_win(
    make_message: Callable[..., RawMessage],
) -> None:
    """Ties go to the deterministic primary, which keeps the composite reproducible."""
    composite = CompositeClassifier(
        StubClassifier("rules", 0.50, EventType.REJECTION),
        StubClassifier("ollama", 0.50, EventType.OFFER),
        min_confidence=0.60,
    )
    assert composite.classify(make_message()).classifier_name == "rules"


def test_a_failing_fallback_degrades_to_the_primary(
    make_message: Callable[..., RawMessage],
) -> None:
    """A dead Ollama host must degrade, never raise — sync has to survive it."""
    composite = CompositeClassifier(
        StubClassifier("rules", 0.20, EventType.REJECTION),
        ExplodingClassifier("ollama", 0.99, EventType.OFFER),
        min_confidence=0.60,
    )
    result = composite.classify(make_message())
    assert result.event_type is EventType.REJECTION
    assert result.classifier_name == "rules"


def test_results_stay_attributable_to_the_producing_backend(
    make_message: Callable[..., RawMessage],
) -> None:
    """Stored rows must name the backend and version that actually wrote them, not 'composite'.

    This is what makes M7's prompt-SHA versioning meaningful: old rows stay attributable to
    the prompt that produced them.
    """
    composite = CompositeClassifier(
        StubClassifier("rules", 0.10, EventType.UNKNOWN),
        StubClassifier("ollama", 0.90, EventType.OFFER),
        min_confidence=0.60,
    )
    result = composite.classify(make_message())
    assert result.classifier_name == "ollama"
    assert result.classifier_version == "stub-1"


def test_composite_batch_preserves_order(
    email_fixtures: list[tuple[str, RawMessage]],
) -> None:
    """Order preservation is part of the Protocol, composite included."""
    messages = [message for _stem, message in email_fixtures]
    results = CompositeClassifier(RulesClassifier()).classify_batch(messages)
    assert [r.message_id for r in results] == [m.message_id for m in messages]


def test_composite_batch_of_nothing_is_empty() -> None:
    """An empty batch is an empty list."""
    assert CompositeClassifier(RulesClassifier()).classify_batch([]) == []


def test_composite_is_deterministic(email_fixtures: list[tuple[str, RawMessage]]) -> None:
    """I2 holds through the composite, not just the rules engine."""
    composite = CompositeClassifier(RulesClassifier())
    for _stem, message in email_fixtures:
        assert (
            composite.classify(message).model_dump_json()
            == composite.classify(message).model_dump_json()
        )


@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
def test_the_escalation_threshold_is_configurable(
    threshold: float, make_message: Callable[..., RawMessage]
) -> None:
    """min_confidence decides what the fallback ever sees."""
    fallback = StubClassifier("ollama", 0.99, EventType.OFFER)
    composite = CompositeClassifier(
        StubClassifier("rules", 0.50, EventType.REJECTION), fallback, min_confidence=threshold
    )
    composite.classify(make_message())
    assert bool(fallback.seen) is (threshold > 0.50)
