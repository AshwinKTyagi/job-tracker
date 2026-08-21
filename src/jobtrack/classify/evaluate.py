"""M7 — scoring a classifier against human-labeled review items (CONTRACTS.md §10).

Model choice is a measurement, not a preference. Every item accepted or corrected in
``jobtrack review`` is labeled data; this harness scores a candidate backend against it so
the smallest model that clears the bar can be picked on evidence.

The metric order mirrors PLAN.md §8: schema compliance first (a backend that needs retries
is disqualified, because retries break reproducibility), then accuracy on the
confirmation-versus-rejection pair, then field exactness, then latency.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Sequence
from typing import Final

from jobtrack.classify.base import Classifier
from jobtrack.classify.normalize import normalize_company, normalize_role
from jobtrack.models import Classification, EventType, RawMessage

#: The pair the LLM backend exists to get right: a "thanks for applying" acknowledgement
#: versus a rejection that restates the same language before declining.
CONFUSABLE_PAIR: Final[frozenset[EventType]] = frozenset(
    {EventType.APPLICATION_RECEIVED, EventType.REJECTION}
)


def _ratio(hits: int, total: int) -> float:
    """Hits over total, with an empty population scoring 0.0 rather than dividing by zero."""
    return 0.0 if total == 0 else hits / total


def evaluate(
    classifier: Classifier, labeled: Sequence[tuple[RawMessage, Classification]]
) -> dict[str, float]:
    """Score a classifier against review-queue labels.

    Args:
        classifier: The backend under test.
        labeled: Pairs of message and the human-confirmed Classification for it.

    Returns:
        A mapping with these keys:

        * ``n`` — how many labeled items were scored
        * ``schema_compliance`` — fraction that produced a usable, non-sentinel verdict
        * ``event_type_accuracy`` — fraction whose event_type matched the label
        * ``confusable_pair_accuracy`` — the same, restricted to acknowledgement-versus-
          rejection items, which is the metric that actually decides the model
        * ``company_exact_match`` — fraction matching after ``normalize_company``, over the
          items whose label names a company
        * ``role_exact_match`` — the same for ``normalize_role``
        * ``median_latency_seconds`` — per-message wall clock
    """
    if not labeled:
        return {
            "n": 0.0,
            "schema_compliance": 0.0,
            "event_type_accuracy": 0.0,
            "confusable_pair_accuracy": 0.0,
            "company_exact_match": 0.0,
            "role_exact_match": 0.0,
            "median_latency_seconds": 0.0,
        }

    latencies: list[float] = []
    compliant = 0
    type_hits = 0
    pair_total = 0
    pair_hits = 0
    company_total = 0
    company_hits = 0
    role_total = 0
    role_hits = 0

    for message, expected in labeled:
        started = time.perf_counter()
        actual = classifier.classify(message)
        latencies.append(time.perf_counter() - started)

        # A zero-confidence result is the sentinel every backend returns when it could not
        # produce a verdict at all; it is a compliance failure, not a wrong answer.
        if actual.confidence > 0.0:
            compliant += 1

        if actual.event_type is expected.event_type:
            type_hits += 1

        if expected.event_type in CONFUSABLE_PAIR:
            pair_total += 1
            if actual.event_type is expected.event_type:
                pair_hits += 1

        if expected.company is not None:
            company_total += 1
            if normalize_company(actual.company) == normalize_company(expected.company):
                company_hits += 1

        if expected.role is not None:
            role_total += 1
            if normalize_role(actual.role) == normalize_role(expected.role):
                role_hits += 1

    return {
        "n": float(len(labeled)),
        "schema_compliance": _ratio(compliant, len(labeled)),
        "event_type_accuracy": _ratio(type_hits, len(labeled)),
        "confusable_pair_accuracy": _ratio(pair_hits, pair_total),
        "company_exact_match": _ratio(company_hits, company_total),
        "role_exact_match": _ratio(role_hits, role_total),
        "median_latency_seconds": statistics.median(latencies),
    }


__all__ = ["CONFUSABLE_PAIR", "evaluate"]
