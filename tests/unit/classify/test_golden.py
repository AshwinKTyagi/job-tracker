"""Golden classifier tests against tests/fixtures/emails/*.json + expected.jsonl.

CLAUDE.md: every fixture must pass, and adding a pattern requires a fixture exercising it,
including at least one negative. Determinism (I2) is asserted explicitly: classifying the
same fixture twice must produce byte-identical output.
"""

from __future__ import annotations

from typing import Any

import pytest

from jobtrack.classify.rules import RulesClassifier
from jobtrack.constants import EVENT_PRECEDENCE
from jobtrack.models import EventType, RawMessage

_SKIP_KEYS = frozenset({"fixture", "note"})


def test_expected_covers_every_fixture(
    email_fixtures: list[tuple[str, RawMessage]],
    expected: dict[str, dict[str, Any]],
) -> None:
    """Every recorded fixture has a golden expectation and vice versa."""
    fixture_stems = {stem for stem, _ in email_fixtures}
    assert fixture_stems == set(expected.keys())


def test_event_precedence_covers_every_event_type() -> None:
    """A guard against the regression Phase 0 already found once (missing WITHDRAWN)."""
    assert set(EVENT_PRECEDENCE) == set(EventType)
    assert len(EVENT_PRECEDENCE) == len(set(EVENT_PRECEDENCE))


@pytest.mark.parametrize(
    "stem",
    [
        "greenhouse_confirmation",
        "greenhouse_rejection",
        "newsletter_unknown",
        "lever_interview",
        "workday_assessment",
        "ashby_offer",
        "withdrawn_confirmation",
        "recruiter_outreach",
    ],
)
def test_golden_fixture_matches_expected(
    stem: str,
    email_fixtures: list[tuple[str, RawMessage]],
    expected: dict[str, dict[str, Any]],
) -> None:
    """Each fixture must classify to the fields recorded in expected.jsonl."""
    fixtures_by_stem = dict(email_fixtures)
    message = fixtures_by_stem[stem]
    golden = expected[stem]

    classification = RulesClassifier().classify(message)

    for field, want in golden.items():
        if field in _SKIP_KEYS:
            continue
        got = getattr(classification, field)
        assert got == want, f"{stem}: field {field!r}: expected {want!r}, got {got!r}"


def test_confusable_pair_is_not_confused(email_fixtures: list[tuple[str, RawMessage]]) -> None:
    """The core hard case (PLAN.md §8): identical subjects, opposite outcomes."""
    fixtures_by_stem = dict(email_fixtures)
    confirmation = RulesClassifier().classify(fixtures_by_stem["greenhouse_confirmation"])
    rejection = RulesClassifier().classify(fixtures_by_stem["greenhouse_rejection"])

    assert fixtures_by_stem["greenhouse_confirmation"].subject == (
        fixtures_by_stem["greenhouse_rejection"].subject
    )
    assert confirmation.event_type is EventType.APPLICATION_RECEIVED
    assert rejection.event_type is EventType.REJECTION
    assert rejection.event_type is not EventType.APPLICATION_RECEIVED


def test_golden_fixtures_classify_deterministically(
    email_fixtures: list[tuple[str, RawMessage]],
) -> None:
    """Same RawMessage in, byte-identical Classification out (I2), for every fixture."""
    classifier = RulesClassifier()
    for stem, message in email_fixtures:
        first = classifier.classify(message)
        second = classifier.classify(message)
        assert first.model_dump() == second.model_dump(), f"{stem} is not deterministic"


def test_classify_batch_matches_individual_classify_and_preserves_order(
    email_fixtures: list[tuple[str, RawMessage]],
) -> None:
    """classify_batch is order-preserving and agrees with per-message classify()."""
    classifier = RulesClassifier()
    messages = [message for _, message in email_fixtures]
    batch_result = classifier.classify_batch(messages)
    individual_result = [classifier.classify(message) for message in messages]
    assert [c.model_dump() for c in batch_result] == [c.model_dump() for c in individual_result]
    assert [c.message_id for c in batch_result] == [m.message_id for m in messages]
