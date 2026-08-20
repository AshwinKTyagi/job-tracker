"""Company and role normalization.

``normalize_company`` is the sole producer of ``company_key`` (I8): deterministic and
idempotent, so ``normalize_company(normalize_company(x)) == normalize_company(x)``.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
"""Anything that is not a word character or whitespace, dropped during normalization."""

_WHITESPACE = re.compile(r"\s+")

_LEGAL_SUFFIXES: frozenset[str] = frozenset(
    {"inc", "llc", "ltd", "corp", "corporation", "gmbh", "pbc", "co"}
)
"""Legal-entity suffixes stripped from the tail of a normalized company name."""

_SENIORITY_TERMS: frozenset[str] = frozenset(
    {
        "senior",
        "sr",
        "staff",
        "principal",
        "lead",
        "junior",
        "jr",
        "associate",
        "entry",
        "level",
    }
)
"""Seniority noise stripped from a normalized role title before comparison."""

_ROLE_ABBREVIATIONS: dict[str, str] = {
    "swe": "software engineer",
    "sde": "software development engineer",
    "sre": "site reliability engineer",
    "pm": "product manager",
    "tpm": "technical program manager",
    "eng": "engineer",
    "mgr": "manager",
    "dev": "developer",
    "qa": "quality assurance",
}
"""Common job-title abbreviations expanded during role normalization."""

_REQ_ID = re.compile(r"\b(?:req|r|job)[\s#-]*\d{3,}\b", re.IGNORECASE)
"""Requisition/job-posting ids embedded in a title, e.g. 'Req-12345' or 'R12345'."""


def normalize_company(name: str | None) -> str | None:
    """Produce the stable matching key for a company (I8).

    Casefolds, strips legal suffixes (Inc, LLC, Ltd, Corp, GmbH, PBC, Co), drops
    punctuation, and collapses whitespace. Deterministic and idempotent.

    Args:
        name: The display-form company name, or None.

    Returns:
        The normalized key, or None if `name` is None or normalizes to an empty string.

    Examples:
        >>> normalize_company("Acme Robotics, Inc.") == normalize_company("acme robotics")
        True
    """
    if name is None:
        return None
    folded = name.casefold()
    stripped = _PUNCTUATION.sub(" ", folded)
    tokens = _WHITESPACE.split(stripped.strip())
    tokens = [t for t in tokens if t]
    while tokens and tokens[-1] in _LEGAL_SUFFIXES:
        tokens.pop()
    if not tokens:
        return None
    return " ".join(tokens)


def normalize_role(title: str | None) -> str | None:
    """Canonicalize a job title for fuzzy comparison.

    Casefolds, strips requisition ids and seniority noise, and expands common abbreviations
    (SWE -> software engineer).

    Args:
        title: The raw job title, or None.

    Returns:
        The normalized title, or None if `title` is None or normalizes to an empty string.
    """
    if title is None:
        return None
    folded = title.casefold()
    without_req = _REQ_ID.sub(" ", folded)
    stripped_punct = _PUNCTUATION.sub(" ", without_req)
    tokens = [t for t in _WHITESPACE.split(stripped_punct.strip()) if t]
    tokens = [t for t in tokens if t not in _SENIORITY_TERMS]
    tokens = [_ROLE_ABBREVIATIONS.get(t, t) for t in tokens]
    expanded = " ".join(tokens)
    tokens = [t for t in _WHITESPACE.split(expanded.strip()) if t]
    if not tokens:
        return None
    return " ".join(tokens)


def role_similarity(a: str | None, b: str | None) -> float:
    """Similarity in [0, 1] between two normalized titles.

    Used by the linker's fuzzy match. Deterministic; no external NLP dependency.

    Args:
        a: A job title, raw or already normalized.
        b: Another job title, raw or already normalized.

    Returns:
        1.0 for an exact match after normalization, 0.0 if either normalizes to None,
        otherwise a difflib.SequenceMatcher ratio.
    """
    normalized_a = normalize_role(a)
    normalized_b = normalize_role(b)
    if normalized_a is None or normalized_b is None:
        return 0.0
    if normalized_a == normalized_b:
        return 1.0
    return SequenceMatcher(None, normalized_a, normalized_b).ratio()


__all__ = ["normalize_company", "normalize_role", "role_similarity"]
