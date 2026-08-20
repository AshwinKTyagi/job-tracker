"""Company and role normalization.

``normalize_company`` is the sole producer of ``company_key`` (I8), and the linker matches
entirely on that key. A key that drifted between runs would split one application into two, so
idempotence is tested as a property over the whole corpus rather than on a couple of examples.
"""

from __future__ import annotations

import pytest

from jobtrack.classify.normalize import (
    normalize_company,
    normalize_role,
    role_similarity,
)
from jobtrack.models import RawMessage

# A deliberately nasty spread: legal forms, punctuation, accents, ampersands, casing,
# all-suffix names, and empty-ish input.
COMPANY_SAMPLES: list[str | None] = [
    None,
    "",
    "   ",
    ".",
    "Acme Robotics, Inc.",
    "acme robotics",
    "ACME ROBOTICS INC",
    "Acme Robotics Inc. LLC",
    "The New York Times",
    "The Co",
    "Corp",
    "Inc",
    "Zürich Insurance",
    "Zurich Insurance",
    "Ben & Jerry's",
    "Ben and Jerry's",
    "Foo-Bar Ltd",
    "Société Générale S.A.",
    "  Spacey   Name  ",
    "7Eleven",
    "X",
]


@pytest.mark.parametrize("value", COMPANY_SAMPLES)
def test_normalize_company_is_idempotent(value: str | None) -> None:
    """I8: normalize_company(normalize_company(x)) == normalize_company(x)."""
    once = normalize_company(value)
    twice = normalize_company(once)
    assert twice == once


@pytest.mark.parametrize("value", COMPANY_SAMPLES)
def test_normalize_company_is_deterministic(value: str | None) -> None:
    """Same input, same key, every time."""
    assert normalize_company(value) == normalize_company(value)


def test_normalize_company_strips_legal_suffixes() -> None:
    """The documented example from CONTRACTS.md §5."""
    assert normalize_company("Acme Robotics, Inc.") == normalize_company("acme robotics")
    assert normalize_company("Acme Robotics, Inc.") == "acme robotics"


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Acme Robotics, Inc.", "ACME ROBOTICS"),
        ("Foo-Bar Ltd", "Foo Bar"),
        ("Zürich Insurance", "Zurich Insurance"),
        ("Ben & Jerry's", "Ben and Jerry's"),
        ("  Spacey   Name  ", "Spacey Name"),
        ("The Acme Corporation", "Acme"),
    ],
)
def test_normalize_company_collapses_display_variants(left: str, right: str) -> None:
    """Two spellings of one employer must land on one key, or the linker splits them."""
    assert normalize_company(left) == normalize_company(right)


def test_normalize_company_of_none_is_none() -> None:
    """None in, None out — the classifier passes company through unconditionally."""
    assert normalize_company(None) is None


@pytest.mark.parametrize("value", ["", "   ", ".", "!!!"])
def test_normalize_company_of_junk_is_none(value: str) -> None:
    """A string with no usable characters yields no key rather than an empty one."""
    assert normalize_company(value) is None


@pytest.mark.parametrize("value", ["Corp", "Inc", "LLC"])
def test_a_company_named_only_for_its_legal_form_keeps_a_key(value: str) -> None:
    """Stripping must not erase the whole name — a lone 'Corp' still needs a key."""
    assert normalize_company(value) == value.casefold()


def test_distinct_companies_keep_distinct_keys() -> None:
    """Over-normalizing is as bad as under-normalizing."""
    assert normalize_company("Acme Robotics") != normalize_company("Acme Analytics")
    assert normalize_company("Northwind") != normalize_company("Northwood")


# --------------------------------------------------------------------------------------
# Roles
# --------------------------------------------------------------------------------------

ROLE_SAMPLES: list[str | None] = [
    None,
    "",
    "Senior Software Engineer, Platform",
    "SWE II",
    "Sr. SWE",
    "Software Engineer (REQ-12345)",
    "Backend Engineer #4471",
    "Staff Data Scientist [R-1234]",
    "Product Manager JR0012345",
    "Intern I",
    "principal engineer",
    "SRE",
]


@pytest.mark.parametrize("value", ROLE_SAMPLES)
def test_normalize_role_is_idempotent(value: str | None) -> None:
    """A drifting role key would break the linker's fuzzy match on the second pass."""
    once = normalize_role(value)
    assert normalize_role(once) == once


def test_normalize_role_strips_seniority() -> None:
    """Seniority is level, not identity: the linker matches jobs, it does not grade them."""
    assert normalize_role("Senior Software Engineer") == normalize_role("Software Engineer")
    assert normalize_role("Software Engineer II") == normalize_role("Software Engineer")


def test_normalize_role_expands_abbreviations() -> None:
    """SWE → software engineer, per CONTRACTS.md §5."""
    assert normalize_role("SWE") == "software engineer"
    assert normalize_role("Sr. SWE") == "software engineer"


@pytest.mark.parametrize(
    "value",
    [
        "Software Engineer (REQ-12345)",
        "Software Engineer #4471",
        "Software Engineer [R-1234]",
        "Software Engineer JR0012345",
    ],
)
def test_normalize_role_strips_requisition_ids(value: str) -> None:
    """Two postings of one job differ only by req id; they must not become two applications."""
    assert normalize_role(value) == normalize_role("Software Engineer")


def test_normalize_role_of_pure_seniority_keeps_something() -> None:
    """Stripping every token would destroy the title, so the pre-strip form survives."""
    assert normalize_role("Intern I") == "intern"
    assert normalize_role("Senior") == "senior"


def test_normalize_role_of_none_is_none() -> None:
    """None in, None out."""
    assert normalize_role(None) is None
    assert normalize_role("  ") is None


# --------------------------------------------------------------------------------------
# Similarity
# --------------------------------------------------------------------------------------


def test_identical_roles_score_one() -> None:
    """Titles that normalize the same are the same job."""
    assert (
        role_similarity("Senior Software Engineer, Platform", "Software Engineer, Platform") == 1.0
    )
    assert role_similarity("SWE", "Software Engineer") == 1.0


def test_missing_roles_score_zero() -> None:
    """None is not a match — the linker has a separate rule for a missing role."""
    assert role_similarity(None, "Software Engineer") == 0.0
    assert role_similarity("Software Engineer", None) == 0.0
    assert role_similarity(None, None) == 0.0
    assert role_similarity("", "") == 0.0


def test_unrelated_roles_score_low() -> None:
    """Well below the linker's 0.75 threshold, or unrelated jobs would merge."""
    assert role_similarity("Software Engineer", "Chief Financial Officer") < 0.5
    assert role_similarity("Backend Engineer", "Product Designer") < 0.5


def test_related_but_distinct_roles_stay_below_threshold() -> None:
    """The hard case: same suffix, different job. Must not clear 0.75."""
    assert role_similarity("Backend Engineer", "Data Engineer") < 0.75


def test_similarity_is_symmetric_and_bounded() -> None:
    """A similarity that depended on argument order would make linking order-dependent."""
    pairs = [
        ("Software Engineer", "Senior Software Engineer"),
        ("Data Scientist", "Data Analyst"),
        ("Product Manager", "Engineering Manager"),
        ("SRE", "Site Reliability Engineer"),
    ]
    for left, right in pairs:
        forward = role_similarity(left, right)
        backward = role_similarity(right, left)
        assert forward == backward
        assert 0.0 <= forward <= 1.0


def test_similarity_is_deterministic() -> None:
    """No hashing or set-ordering may leak into the score."""
    for _ in range(3):
        assert role_similarity("Backend Engineer", "Data Engineer") == role_similarity(
            "Backend Engineer", "Data Engineer"
        )


def test_fixture_company_keys_are_idempotent(
    email_fixtures: list[tuple[str, RawMessage]],
) -> None:
    """I8 as a property of the real corpus, not just of hand-picked strings."""
    from jobtrack.classify.rules import RulesClassifier

    classifier = RulesClassifier()
    for stem, message in email_fixtures:
        key = classifier.classify(message).company_key
        assert normalize_company(key) == key, f"{stem} has a non-idempotent company_key"
