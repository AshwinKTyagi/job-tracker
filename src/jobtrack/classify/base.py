"""The Classifier interface and the primary/fallback composite.

This is the seam the future Ollama backend (M7) slots into: ``CompositeClassifier`` never
changes when a real fallback replaces ``None``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from jobtrack.models import Classification, RawMessage


class Classifier(Protocol):
    """RawMessage -> Classification. THE plug point for a future Ollama backend.

    Implementations must be deterministic (I2) and must never raise for ordinary input:
    an unparseable message is EventType.UNKNOWN with confidence 0.0, not an exception.
    """

    name: str
    version: str

    def classify(self, message: RawMessage) -> Classification:
        """Classify one message.

        Args:
            message: The normalized email to classify.

        Returns:
            The resulting Classification.
        """
        ...

    def classify_batch(self, messages: Sequence[RawMessage]) -> list[Classification]:
        """Classify many messages, preserving input order.

        Args:
            messages: Messages to classify, in order.

        Returns:
            One Classification per input message, in the same order. Default implementation
            may map classify() over the input.
        """
        ...


class CompositeClassifier:
    """Primary classifier with a fallback for low-confidence results.

    Phase 1 ships this with fallback=None (a pass-through). Phase 3 constructs it as
    CompositeClassifier(RulesClassifier(), OllamaClassifier(), min_confidence=0.60) —
    and no caller changes.
    """

    name: str = "composite"
    version: str = "1.0.0"

    def __init__(
        self,
        primary: Classifier,
        fallback: Classifier | None = None,
        *,
        min_confidence: float = 0.60,
    ) -> None:
        """Args:
        primary: Classifier consulted first for every message.
        fallback: Classifier consulted when primary's confidence is below
            min_confidence. None (Phase 1) makes this a pass-through.
        min_confidence: Confidence threshold below which the fallback is consulted.
        """
        self.primary = primary
        self.fallback = fallback
        self.min_confidence = min_confidence

    def classify(self, message: RawMessage) -> Classification:
        """Classify one message via the primary, deferring to the fallback when unsure.

        Args:
            message: The normalized email to classify.

        Returns:
            The primary's Classification, or the fallback's when the primary's confidence
            is below min_confidence and a fallback is configured.
        """
        result = self.primary.classify(message)
        if self.fallback is not None and result.confidence < self.min_confidence:
            return self.fallback.classify(message)
        return result

    def classify_batch(self, messages: Sequence[RawMessage]) -> list[Classification]:
        """Classify many messages, preserving input order.

        Args:
            messages: Messages to classify, in order.

        Returns:
            One Classification per input message, in the same order.
        """
        return [self.classify(message) for message in messages]


__all__ = ["Classifier", "CompositeClassifier"]
