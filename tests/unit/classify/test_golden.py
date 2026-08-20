"""Golden tests: every recorded fixture must classify exactly as expected.jsonl says.

This is the regression net for the whole module. A pattern change that improves one email and
quietly breaks another shows up here, which is why CLAUDE.md requires a fixture for every
pattern and why ``test_every_pattern_is_exercised_by_a_fixture`` enforces that mechanically.
"""

from __future__ import annotations

from typing import Any

import pytest

from jobtrack.classify.normalize import normalize_company
from jobtrack.classify.patterns import COMPANY_PATTERNS, EVENT_PATTERNS, ROLE_PATTERNS
from jobtrack.classify.rules import RulesClassifier, score_event_types
from jobtrack.models import EventType, RawMessage
from tests.conftest import all_email_fixtures


def test_fixture_corpus_is_substantial(email_fixtures: list[tuple[str, RawMessage]]) -> None:
    """The corpus has to be big enough to be a real net, not three synthetic seeds."""
    assert len(email_fixtures) >= 20


def test_every_fixture_has_an_expectation(
    email_fixtures: list[tuple[str, RawMessage]], expected: dict[str, dict[str, Any]]
) -> None:
    """An unlisted fixture is a silent hole in the net."""
    missing = sorted({stem for stem, _ in email_fixtures} - set(expected))
    assert missing == []


def test_every_expectation_has_a_fixture(
    email_fixtures: list[tuple[str, RawMessage]], expected: dict[str, dict[str, Any]]
) -> None:
    """A stale expectation for a deleted fixture would never be checked."""
    stale = sorted(set(expected) - {stem for stem, _ in email_fixtures})
    assert stale == []


FIXTURE_STEMS = [stem for stem, _ in all_email_fixtures()]


@pytest.mark.parametrize("stem", FIXTURE_STEMS)
def test_fixture_classifies_as_expected(
    stem: str, email_fixtures: list[tuple[str, RawMessage]], expected: dict[str, dict[str, Any]]
) -> None:
    """Each fixture's event type, company, role, and ATS match the golden record.

    Parametrized per fixture rather than looped so a failure names the offending email.
    """
    message = dict(email_fixtures)[stem]
    want = expected[stem]
    got = RulesClassifier().classify(message)

    assert got.event_type == EventType(want["event_type"]), f"{stem}: evidence={got.evidence}"
    assert got.company == want["company"], f"{stem}: evidence={got.evidence}"
    assert got.role == want["role"], f"{stem}: evidence={got.evidence}"
    assert got.ats == want["ats"], f"{stem}: evidence={got.evidence}"


def test_company_key_is_derived_from_company(
    email_fixtures: list[tuple[str, RawMessage]],
) -> None:
    """I8: company_key is always normalize_company(company), never anything else."""
    classifier = RulesClassifier()
    for _stem, message in email_fixtures:
        result = classifier.classify(message)
        assert result.company_key == normalize_company(result.company)


def test_every_pattern_is_exercised_by_a_fixture(
    email_fixtures: list[tuple[str, RawMessage]],
) -> None:
    """CLAUDE.md: adding a pattern REQUIRES adding a fixture that exercises it.

    Checked against ``score_event_types`` rather than the final evidence list, because a
    pattern for a type that loses on precedence still fired and is still covered — the
    rejection fixtures deliberately match APPLICATION_RECEIVED too.
    """
    classifier = RulesClassifier()
    fired: set[str] = set()
    for _stem, message in email_fixtures:
        fired.update(classifier.classify(message).evidence)
        for rule_ids in score_event_types(message).values():
            fired.update(rule_ids)

    declared = {p.rule_id for p in EVENT_PATTERNS}
    declared |= {p.rule_id for p in COMPANY_PATTERNS}
    declared |= {p.rule_id for p in ROLE_PATTERNS}

    unexercised = sorted(declared - fired)
    assert unexercised == [], f"patterns with no fixture: {unexercised}"


def test_confirmation_and_rejection_share_a_subject(
    email_fixtures: list[tuple[str, RawMessage]],
) -> None:
    """The confusable pair really is confusable — otherwise the pair proves nothing.

    If someone "fixes" a failing test by editing the rejection fixture's subject, this guard
    fails and says why.
    """
    by_stem = dict(email_fixtures)
    confirmation = by_stem["greenhouse_confirmation"]
    rejection = by_stem["greenhouse_rejection"]

    assert confirmation.subject == rejection.subject
    assert confirmation.body_text.lower().startswith("hi alex,\n\nthanks for applying")
    assert rejection.body_text.lower().startswith("hi alex,\n\nthanks for applying")

    classifier = RulesClassifier()
    assert classifier.classify(confirmation).event_type is EventType.APPLICATION_RECEIVED
    assert classifier.classify(rejection).event_type is EventType.REJECTION
