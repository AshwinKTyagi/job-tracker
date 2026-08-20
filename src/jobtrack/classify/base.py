"""The ``Classifier`` seam.

Every consumer depends on the ``Classifier`` Protocol, never on ``RulesClassifier``. That is
what lets Phase 3 drop in a local-LLM backend by constructing a different object in
``cli.py`` and changing nothing else (PLAN.md §7, CONTRACTS.md §10).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from jobtrack.classify.confidence import DEFAULT_MIN_CONFIDENCE
from jobtrack.errors import JobTrackError
from jobtrack.models import Classification, RawMessage

logger = logging.getLogger(__name__)


@runtime_checkable
class Classifier(Protocol):
    """RawMessage → Classification. THE plug point for a future Ollama backend.

    Implementations must be deterministic (I2) and must never raise for ordinary input —
    an unparseable message is EventType.UNKNOWN with confidence 0.0, not an exception.
    """

    name: str
    version: str

    def classify(self, message: RawMessage) -> Classification: ...

    def classify_batch(self, messages: Sequence[RawMessage]) -> list[Classification]:
        """Order-preserving. Default implementation may map classify() over the input."""
        ...


class CompositeClassifier:
    """Primary classifier with a fallback for low-confidence results.

    Phase 1 ships this with fallback=None (a pass-through). Phase 3 constructs it as
    CompositeClassifier(RulesClassifier(), OllamaClassifier(), min_confidence=0.60) —
    and no caller changes.

    The returned Classification always carries the ``classifier_name``/``classifier_version``
    of whichever backend actually produced it, never "composite". Stored rows have to stay
    attributable to the exact backend and prompt that wrote them, which is the whole point of
    the M7 reproducibility contract.
    """

    name = "composite"
    version = "1.0.0"

    def __init__(
        self,
        primary: Classifier,
        fallback: Classifier | None = None,
        *,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ) -> None:
        """Wire a primary classifier to an optional low-confidence fallback.

        Args:
            primary: The classifier consulted first — the rules engine in Phase 1.
            fallback: Consulted only when the primary is unsure. None makes this a
                pass-through.
            min_confidence: Primary results at or above this score are returned as-is.
        """
        self._primary = primary
        self._fallback = fallback
        self._min_confidence = min_confidence

    def classify(self, message: RawMessage) -> Classification:
        """Classify one message, escalating to the fallback when the primary is unsure.

        The fallback's answer is taken only when it is strictly more confident than the
        primary's, so a hesitant fallback can never make the result worse. A fallback that
        fails is logged and ignored — degrading to the rules result, never raising.

        Args:
            message: The normalized email to classify.

        Returns:
            The winning Classification, attributed to the backend that produced it.
        """
        result = self._primary.classify(message)
        if self._fallback is None or result.confidence >= self._min_confidence:
            return result

        try:
            alternative = self._fallback.classify(message)
        except JobTrackError:
            logger.warning(
                "fallback classifier %s failed for message %s; keeping %s result",
                self._fallback.name,
                message.message_id,
                self._primary.name,
                exc_info=True,
            )
            return result

        if alternative.confidence > result.confidence:
            return alternative
        return result

    def classify_batch(self, messages: Sequence[RawMessage]) -> list[Classification]:
        """Classify a sequence, preserving input order.

        Args:
            messages: Messages to classify.

        Returns:
            One Classification per input message, in the same order.
        """
        return [self.classify(message) for message in messages]


__all__ = ["Classifier", "CompositeClassifier"]
