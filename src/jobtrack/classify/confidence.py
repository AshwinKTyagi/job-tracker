"""The confidence rubric — one table, no magic numbers anywhere else.

Every number the classifier uses to score its own certainty lives in ``CONFIDENCE_WEIGHTS``.
``rules.py`` never adds a float of its own; it hands the evidence to ``score_confidence`` and
takes back a number. Tuning the classifier therefore means editing this table (and bumping
``RulesClassifier.version``), not hunting through the engine.

The rubric is additive and clamped to [0.0, 1.0], per PLAN.md §7.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Final

from jobtrack.classify.patterns import HIGH_PRECISION_COMPANY_RULES, RULE_INDEX
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

CONFIDENCE_FLOOR: Final[float] = 0.0
CONFIDENCE_CEILING: Final[float] = 1.0

CONFIDENCE_PRECISION: Final[int] = 4
"""Decimal places the final score is rounded to.

Float addition of the weights produces values like 0.6000000000000001, which would make a
score that should be exactly at the threshold land on the wrong side of it — and would break
the byte-identical-output guarantee (I2) as soon as the weights are reordered. Rounding at a
fixed precision makes the arithmetic reproducible."""

UNKNOWN_CONFIDENCE: Final[float] = 0.0
"""An unrecognized message scores 0.0 by definition. The Classifier Protocol promises that an
unparseable message is UNKNOWN with confidence 0.0 rather than an exception, and a confident
"I have no idea" is a contradiction."""

DEFAULT_MIN_CONFIDENCE: Final[float] = 0.60
"""Mirrors ``ClassifyConfig.min_confidence``. Used only when a caller constructs a classifier
without passing a threshold; ``cli.py`` passes the configured value."""

ADJACENT_PRECEDENCE_DISTANCE: Final[int] = 1
"""Two matched types this far apart in EVENT_PRECEDENCE count as genuine ambiguity.

Distant pairs are NOT ambiguous: a rejection that restates the application language matches
both REJECTION and APPLICATION_RECEIVED, and that combination is the expected, well-understood
shape of a rejection email rather than a sign the classifier is confused."""


def _has_high_precision_subject_rule(evidence: list[str], winning_type: EventType) -> bool:
    """True when a high-precision subject pattern for the winning type fired."""
    return any(
        (pattern := RULE_INDEX.get(rule_id)) is not None
        and pattern.event_type is winning_type
        and pattern.scope == "subject"
        and pattern.high_precision
        for rule_id in evidence
    )


def _has_body_rule(evidence: list[str], winning_type: EventType) -> bool:
    """True when a body pattern for the winning type corroborated the subject."""
    return any(
        (pattern := RULE_INDEX.get(rule_id)) is not None
        and pattern.event_type is winning_type
        and pattern.scope == "body"
        for rule_id in evidence
    )


def _has_adjacent_ambiguity(all_scores: dict[EventType, list[str]]) -> bool:
    """True when two matched types sit next to each other in EVENT_PRECEDENCE."""
    positions = sorted(
        EVENT_PRECEDENCE.index(event_type)
        for event_type, rules in all_scores.items()
        if rules and event_type is not EventType.UNKNOWN
    )
    return any(
        second - first == ADJACENT_PRECEDENCE_DISTANCE for first, second in pairwise(positions)
    )


def score_confidence(
    *,
    ats: str | None,
    winning_type: EventType,
    evidence: list[str],
    company: str | None,
    all_scores: dict[EventType, list[str]],
) -> float:
    """Score the classifier's certainty from CONFIDENCE_WEIGHTS, clamped to [0.0, 1.0].

    Additive: a known ATS sender, a high-precision subject pattern for the winning type, a
    corroborating body pattern, and a high-precision company extraction each contribute their
    weight. ``ambiguous_penalty`` is subtracted when two types matched at adjacent precedence,
    which is the signal that the message genuinely straddles two stages.

    Args:
        ats: Detected ATS slug, or None.
        winning_type: The type that won ``resolve_event_type``.
        evidence: Every rule id that fired, event and extraction alike.
        company: The extracted display company, or None.
        all_scores: The full per-type score map, used for the ambiguity check.

    Returns:
        A score in [0.0, 1.0], rounded to CONFIDENCE_PRECISION decimal places. UNKNOWN always
        scores UNKNOWN_CONFIDENCE.
    """
    if winning_type is EventType.UNKNOWN:
        return UNKNOWN_CONFIDENCE

    total = 0.0
    if ats is not None:
        total += CONFIDENCE_WEIGHTS["ats_detected"]
    if _has_high_precision_subject_rule(evidence, winning_type):
        total += CONFIDENCE_WEIGHTS["subject_pattern"]
    if _has_body_rule(evidence, winning_type):
        total += CONFIDENCE_WEIGHTS["body_pattern"]
    if company is not None and any(rule in HIGH_PRECISION_COMPANY_RULES for rule in evidence):
        total += CONFIDENCE_WEIGHTS["company_extracted"]
    if _has_adjacent_ambiguity(all_scores):
        total += CONFIDENCE_WEIGHTS["ambiguous_penalty"]

    clamped = min(max(total, CONFIDENCE_FLOOR), CONFIDENCE_CEILING)
    return round(clamped, CONFIDENCE_PRECISION)


def needs_review(confidence: float, company: str | None, *, threshold: float) -> bool:
    """Decide whether a classification goes to the human review queue.

    Args:
        confidence: The score from ``score_confidence``.
        company: The extracted display company, or None.
        threshold: ``config.classify.min_confidence``.

    Returns:
        True when confidence is below the threshold or no company could be extracted. A
        classification with no company cannot be linked to an application, so it always needs
        a human regardless of how confident the event typing was.
    """
    return confidence < threshold or company is None


__all__ = [
    "ADJACENT_PRECEDENCE_DISTANCE",
    "CONFIDENCE_CEILING",
    "CONFIDENCE_FLOOR",
    "CONFIDENCE_PRECISION",
    "CONFIDENCE_WEIGHTS",
    "DEFAULT_MIN_CONFIDENCE",
    "UNKNOWN_CONFIDENCE",
    "needs_review",
    "score_confidence",
]
