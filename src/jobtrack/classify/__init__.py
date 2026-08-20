"""M2 classify — RawMessage → Classification, deterministically.

Import the Protocol, not the implementation: everything downstream depends on
``Classifier`` so a future local-LLM backend drops in without touching a caller
(CONTRACTS.md §5, §10).
"""

from __future__ import annotations

from jobtrack.classify.base import Classifier, CompositeClassifier
from jobtrack.classify.confidence import (
    CONFIDENCE_WEIGHTS,
    DEFAULT_MIN_CONFIDENCE,
    needs_review,
    score_confidence,
)
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
    "DEFAULT_MIN_CONFIDENCE",
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
