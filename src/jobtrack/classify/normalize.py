"""Company and role normalization.

``normalize_company`` is the sole producer of ``Classification.company_key`` (I8). Matching
in ``store/linker.py`` runs entirely on that key, while the verbatim ``company`` string is
what the spreadsheet and the dashboard display. The two must never be conflated.

Both normalizers are **deterministic and idempotent**: ``f(f(x)) == f(x)``. Idempotence is
what makes a key stable across re-classification — a key that drifted on a second pass would
split one application into two.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Final

LEGAL_SUFFIXES: Final[frozenset[str]] = frozenset(
    {
        "ab",
        "ag",
        "aps",
        "as",
        "bv",
        "co",
        "company",
        "corp",
        "corporation",
        "gmbh",
        "inc",
        "incorporated",
        "kk",
        "limited",
        "llc",
        "llp",
        "lp",
        "ltd",
        "nv",
        "oy",
        "oyj",
        "pbc",
        "plc",
        "pty",
        "sa",
        "sarl",
        "sas",
        "spa",
        "srl",
    }
)
"""Legal-form tokens stripped from the end of a company name (CONTRACTS.md §5)."""

SENIORITY_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "assoc",
        "associate",
        "entry",
        "i",
        "ii",
        "iii",
        "iv",
        "junior",
        "lead",
        "level",
        "mid",
        "principal",
        "senior",
        "staff",
        "v",
        "1",
        "2",
        "3",
        "4",
        "5",
    }
)
"""Level noise dropped from a role before comparison, so 'Senior Software Engineer II' and
'Software Engineer' compare equal. Titles are matched to link applications, not to grade them."""

ROLE_ABBREVIATIONS: Final[dict[str, str]] = {
    "dev": "developer",
    "devops": "development operations",
    "em": "engineering manager",
    "eng": "engineer",
    "engr": "engineer",
    "mgr": "manager",
    "ml": "machine learning",
    "mle": "machine learning engineer",
    "ops": "operations",
    "pm": "product manager",
    "qa": "quality assurance",
    "sde": "software development engineer",
    "sr": "senior",
    "sre": "site reliability engineer",
    "swe": "software engineer",
    "ui": "user interface",
    "ux": "user experience",
}
"""Single-token expansions applied before seniority stripping. No expansion produces a token
that is itself a key, so one pass is enough and the result is idempotent."""

_COMBINING_MARK_CATEGORY: Final[str] = "Mn"

_NON_ALNUM_RE: Final[re.Pattern[str]] = re.compile(r"[^0-9a-z]+")
"""Everything that is not a lower-case alphanumeric becomes a space. Runs after casefolding,
so the class only needs the lower-case range."""

_REQ_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    (?:
        \(\s*(?:req|job|requisition)?\s*[#\-]?\s*[a-z]*\d[\w\-]*\s*\)  # "(REQ-12345)"
      | \[\s*[\w\-]*\d[\w\-]*\s*\]                                    # "[R-1234]", "[1234]"
      | \#\s*\w*\d\w*                                                 # "#12345"
      | \b(?:req|requisition|job)\s*(?:id|number|no)?\s*[#:\-]?\s*\w*\d\w*
      | \b[a-z]{1,4}\d{4,}\b                                          # "JR0012345"
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
"""Requisition identifiers glued onto a job title. Removed before token processing."""

_LEADING_THE_RE: Final[re.Pattern[str]] = re.compile(r"^the\s+")

_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")


def _strip_accents(text: str) -> str:
    """Decompose to NFKD and drop combining marks, so 'Zürich' and 'Zurich' share a key."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != _COMBINING_MARK_CATEGORY)


def _tokenize(text: str) -> list[str]:
    """Casefold, fold accents, replace punctuation with spaces, and split into tokens."""
    folded = _strip_accents(text).casefold().replace("&", " and ")
    return _NON_ALNUM_RE.sub(" ", folded).split()


def normalize_company(name: str | None) -> str | None:
    """Produce the stable matching key for a company (I8).

    Casefolds, folds accents, strips legal suffixes (Inc, LLC, Ltd, Corp, GmbH, PBC, Co and
    friends), drops punctuation, and collapses whitespace. Deterministic and idempotent.

    A name made up entirely of legal-form tokens keeps its last token rather than reducing to
    nothing, so a company literally called "Corp" still gets a usable key.

    Args:
        name: Display company name, or None.

    Returns:
        The lower-case matching key, or None when the input is None or holds no usable
        characters.

    >>> normalize_company("Acme Robotics, Inc.") == normalize_company("acme robotics")
    True
    """
    if name is None:
        return None
    tokens = _tokenize(name)
    if not tokens:
        return None
    if len(tokens) > 1 and tokens[0] == "the":
        tokens = tokens[1:]
    while len(tokens) > 1 and tokens[-1] in LEGAL_SUFFIXES:
        tokens = tokens[:-1]
    return " ".join(tokens) or None


def normalize_role(title: str | None) -> str | None:
    """Canonicalize a job title for fuzzy comparison.

    Casefolds, removes requisition ids, expands common abbreviations (SWE → software
    engineer), and strips seniority and level noise. Deterministic and idempotent.

    Stripping every token would destroy the title, so a title made only of seniority words
    (for example "Intern I") keeps its pre-strip form.

    Args:
        title: Raw job title, or None.

    Returns:
        The canonical lower-case title, or None when the input holds no usable characters.
    """
    if title is None:
        return None
    without_ids = _REQ_ID_RE.sub(" ", title)
    tokens = _tokenize(without_ids)
    if not tokens:
        return None

    expanded: list[str] = []
    for token in tokens:
        expanded.extend(ROLE_ABBREVIATIONS.get(token, token).split())

    stripped = [t for t in expanded if t not in SENIORITY_TOKENS]
    final = stripped or expanded
    joined = _LEADING_THE_RE.sub("", _WHITESPACE_RE.sub(" ", " ".join(final)).strip())
    return joined or None


def _jaccard(a: set[str], b: set[str]) -> float:
    """Token-set overlap: intersection size over union size, 0.0 when both sides are empty."""
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def role_similarity(a: str | None, b: str | None) -> float:
    """Similarity in [0,1] between two job titles, used by the linker's fuzzy match.

    Normalizes both sides (``normalize_role`` is idempotent, so pre-normalized input is
    fine), then averages token-set overlap with a character-level sequence ratio. Token
    overlap alone would miss word-order and inflection differences; the sequence ratio alone
    would over-reward titles that merely share a common suffix like "engineer".

    Deterministic, stdlib only — no external NLP dependency.

    Args:
        a: First title, raw or normalized, or None.
        b: Second title, raw or normalized, or None.

    Returns:
        1.0 for titles that normalize identically, 0.0 when either side is None or empty,
        otherwise a blended score rounded to 4 decimal places.
    """
    left = normalize_role(a)
    right = normalize_role(b)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    overlap = _jaccard(set(left.split()), set(right.split()))
    ratio = SequenceMatcher(None, left, right).ratio()
    return round((overlap + ratio) / 2.0, 4)


__all__ = [
    "LEGAL_SUFFIXES",
    "ROLE_ABBREVIATIONS",
    "SENIORITY_TOKENS",
    "normalize_company",
    "normalize_role",
    "role_similarity",
]
