"""Integrity of the pattern tables themselves.

Rule ids are persisted in ``classifications.evidence_json`` and shown by ``jobtrack review``,
so they are data, not internal names. A duplicate or a typo'd prefix is a data bug.
"""

from __future__ import annotations

import re

import pytest

from jobtrack.classify.patterns import (
    ATS_DOMAIN_ORDER,
    ATS_DOMAINS,
    COMPANY_PATTERNS,
    EVENT_PATTERNS,
    HIGH_PRECISION_COMPANY_RULES,
    LOCATION_PATTERNS,
    ROLE_PATTERNS,
    RULE_INDEX,
    EventPattern,
    ExtractionPattern,
    normalize_text,
)
from jobtrack.models import EventType

ALL_EXTRACTORS: list[ExtractionPattern] = [
    *COMPANY_PATTERNS,
    *ROLE_PATTERNS,
    *LOCATION_PATTERNS,
]

RULE_ID_RE = re.compile(r"^[a-z]{2,4}\.(?:subject|body|ats|sender)\.[a-z0-9_]+$")

EVENT_PREFIXES: dict[str, EventType] = {
    "wdr": EventType.WITHDRAWN,
    "rej": EventType.REJECTION,
    "off": EventType.OFFER,
    "itv": EventType.INTERVIEW,
    "asm": EventType.ASSESSMENT,
    "ack": EventType.APPLICATION_RECEIVED,
    "rec": EventType.RECRUITER_OUTREACH,
}


def test_event_rule_ids_are_unique() -> None:
    """A duplicate id would silently shadow a pattern in RULE_INDEX."""
    ids = [p.rule_id for p in EVENT_PATTERNS]
    assert len(ids) == len(set(ids))
    assert len(RULE_INDEX) == len(EVENT_PATTERNS)


def test_all_rule_ids_are_globally_unique() -> None:
    """Event and extraction ids share one namespace inside Classification.evidence."""
    ids = [p.rule_id for p in EVENT_PATTERNS] + [p.rule_id for p in ALL_EXTRACTORS]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    assert duplicates == []


@pytest.mark.parametrize("pattern", EVENT_PATTERNS, ids=lambda p: p.rule_id)
def test_event_rule_id_matches_its_type_and_scope(pattern: EventPattern) -> None:
    """The id encodes what the rule is; a mismatch makes stored evidence misleading."""
    assert RULE_ID_RE.match(pattern.rule_id), pattern.rule_id
    prefix, scope, _name = pattern.rule_id.split(".", 2)
    assert EVENT_PREFIXES[prefix] is pattern.event_type
    assert scope == pattern.scope


@pytest.mark.parametrize("pattern", ALL_EXTRACTORS, ids=lambda p: p.rule_id)
def test_extractor_captures_the_group_it_declares(pattern: ExtractionPattern) -> None:
    """Every extractor must actually define the named group the engine reads."""
    assert pattern.group in pattern.regex.groupindex, pattern.rule_id
    assert pattern.rule_id.split(".", 2)[1] == pattern.scope


def test_high_precision_company_rules_all_exist() -> None:
    """A typo here would silently withhold the company confidence weight forever."""
    known = {p.rule_id for p in COMPANY_PATTERNS} | {"co.ats.sender_name"}
    assert known >= HIGH_PRECISION_COMPANY_RULES


def test_every_event_type_except_unknown_has_a_pattern() -> None:
    """UNKNOWN is the absence of a match, so it is the only type with no patterns."""
    covered = {p.event_type for p in EVENT_PATTERNS}
    assert covered == set(EventType) - {EventType.UNKNOWN}


def test_only_subject_patterns_are_marked_high_precision() -> None:
    """The subject weight is defined for subject patterns; a flagged body rule is a mistake."""
    for pattern in EVENT_PATTERNS:
        if pattern.high_precision:
            assert pattern.scope == "subject", pattern.rule_id


def test_every_ats_domain_maps_back_to_its_slug() -> None:
    """ATS_DOMAIN_ORDER is derived from ATS_DOMAINS; the two must not drift."""
    flat = {(domain, slug) for slug, domains in ATS_DOMAINS.items() for domain in domains}
    assert set(ATS_DOMAIN_ORDER) == flat


def test_ats_domains_are_ordered_longest_first() -> None:
    """Longest-first is what makes ``greenhouse-mail.io`` beat a shorter suffix (I2)."""
    lengths = [len(domain) for domain, _slug in ATS_DOMAIN_ORDER]
    assert lengths == sorted(lengths, reverse=True)


def test_ats_domain_order_is_independent_of_dict_order() -> None:
    """Sorted, not table-ordered, so detection cannot depend on dict iteration order (I2)."""
    expected = sorted(
        ((d, s) for s, ds in ATS_DOMAINS.items() for d in ds),
        key=lambda pair: (-len(pair[0]), pair[0]),
    )
    assert list(ATS_DOMAIN_ORDER) == expected


def test_the_plan_ats_roster_is_covered() -> None:
    """PLAN.md §7 names the systems that must be recognized."""
    required = {
        "greenhouse",
        "lever",
        "workday",
        "ashby",
        "smartrecruiters",
        "icims",
        "taleo",
        "jobvite",
        "workable",
        "breezy",
        "bamboohr",
        "recruitee",
        "teamtailor",
        "jazzhr",
        "dover",
        "rippling",
        "wellfound",
        "linkedin",
        "indeed",
    }
    assert required <= set(ATS_DOMAINS)


# --------------------------------------------------------------------------------------
# normalize_text
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("  Hello   World  ", "hello world"),
        ("Line one\nLine two", "line one line two"),
        (
            "We\u2019ve got it",
            "we've got it",
        ),
        ("A \u2014 B", "a - b"),
        ("A \u2013 B", "a - b"),
        (
            "Non\u00a0breaking",
            "non breaking",
        ),
        ("SHOUTING", "shouting"),
        ("", ""),
    ],
)
def test_normalize_text_folds_to_the_canonical_form(raw: str, want: str) -> None:
    """Event patterns are written against this form, so its output is part of the contract."""
    assert normalize_text(raw) == want


def test_normalize_text_is_idempotent() -> None:
    """Folding twice changes nothing, which keeps scoring reproducible."""
    for raw in [
        "  A\u2019s  \u2014 B  ",
        "plain",
        "",
        "Tabs\tand\nnewlines",
    ]:
        once = normalize_text(raw)
        assert normalize_text(once) == once
