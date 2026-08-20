"""The confidence rubric: one additive table, applied in one place.

See PLAN.md §7 "Confidence rubric". No magic numbers belong in rules.py — everything that
moves the score lives in ``CONFIDENCE_WEIGHTS`` below.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Final

from jobtrack.constants import EVENT_PRECEDENCE
from jobtrack.models import EventType

CONFIDENCE_WEIGHTS: dict[str, float] = {
    "ats_detected": 0.35,
    "subject_pattern": 0.40,
    "body_pattern": 0.20,
    "company_extracted": 0.05,
    "ambiguous_penalty": -0.20,
}
"""The rubric lives HERE, as one table. No magic numbers scattered through rules.py."""

DEFAULT_MIN_CONFIDENCE: Final[float] = 0.60
"""Mirrors ``ClassifyConfig.min_confidence``'s default. RulesClassifier has no Config
parameter (CONTRACTS.md's frozen signature), so it needs a fixed fallback threshold for its
own ``needs_review`` call; a caller that wants a different threshold re-derives
``needs_review`` from ``Classification.confidence`` using ``config.classify.min_confidence``."""

_SUBJECT_MARKER: Final[str] = ".subject."
_BODY_MARKER: Final[str] = ".body."


def _fired(evidence: list[str], marker: str) -> bool:
    """True if any rule id in `evidence` targets the given signal (subject or body)."""
    return any(marker in rule_id for rule_id in evidence)


def _has_adjacent_ambiguity(all_scores: dict[EventType, list[str]]) -> bool:
    """True if two matched event types sit next to each other in EVENT_PRECEDENCE."""
    matched_indices = sorted(
        EVENT_PRECEDENCE.index(event_type)
        for event_type, rule_ids in all_scores.items()
        if rule_ids
    )
    return any(b - a == 1 for a, b in pairwise(matched_indices))


def score_confidence(
    *,
    ats: str | None,
    winning_type: EventType,
    evidence: list[str],
    company: str | None,
    all_scores: dict[EventType, list[str]],
) -> float:
    """Additive rubric from CONFIDENCE_WEIGHTS, clamped to [0.0, 1.0].

    Args:
        ats: The detected ATS slug, or None.
        winning_type: The event type resolve_event_type picked.
        evidence: The rule ids that fired for `winning_type` specifically (as returned by
            resolve_event_type).
        company: The extracted display-form company name, or None.
        all_scores: Every event type's matched rule ids, as returned by score_event_types.
            Used only to detect ambiguity between adjacent-precedence types.

    Returns:
        The clamped confidence score.
    """
    total = 0.0
    if ats is not None:
        total += CONFIDENCE_WEIGHTS["ats_detected"]
    if _fired(evidence, _SUBJECT_MARKER):
        total += CONFIDENCE_WEIGHTS["subject_pattern"]
    if _fired(evidence, _BODY_MARKER):
        total += CONFIDENCE_WEIGHTS["body_pattern"]
    if company is not None:
        total += CONFIDENCE_WEIGHTS["company_extracted"]
    if _has_adjacent_ambiguity(all_scores):
        total += CONFIDENCE_WEIGHTS["ambiguous_penalty"]
    return max(0.0, min(1.0, total))


def needs_review(confidence: float, company: str | None, *, threshold: float) -> bool:
    """True when confidence < threshold or company is None.

    Args:
        confidence: The classifier's confidence score.
        company: The extracted display-form company name, or None.
        threshold: The minimum confidence that does not require review.

    Returns:
        Whether the classification should be queued for human review.
    """
    return confidence < threshold or company is None


__all__ = ["CONFIDENCE_WEIGHTS", "DEFAULT_MIN_CONFIDENCE", "needs_review", "score_confidence"]
