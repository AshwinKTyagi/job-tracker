"""classify — RawMessage -> Classification, pure and deterministic (I2).

Public surface re-exported here for convenience; see CONTRACTS.md §5 for the frozen
signatures.
"""

from __future__ import annotations

from jobtrack.classify.base import Classifier, CompositeClassifier
from jobtrack.classify.confidence import CONFIDENCE_WEIGHTS, needs_review, score_confidence
from jobtrack.classify.normalize import normalize_company, normalize_role, role_similarity
from jobtrack.classify.rules import (
    RulesClassifier,
    detect_ats,
    extract_company,
    extract_location,
    extract_role,
    resolve_event_type,
    score_event_types,
)

__all__ = [
    "CONFIDENCE_WEIGHTS",
    "Classifier",
    "CompositeClassifier",
    "RulesClassifier",
    "detect_ats",
    "extract_company",
    "extract_location",
    "extract_role",
    "needs_review",
    "normalize_company",
    "normalize_role",
    "resolve_event_type",
    "role_similarity",
    "score_confidence",
    "score_event_types",
]
