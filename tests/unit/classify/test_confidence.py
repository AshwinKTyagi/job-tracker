"""The confidence rubric and the review threshold.

The rubric is a table, so these tests read the table rather than hard-coding its numbers: a
deliberate re-tune should not have to touch the arithmetic tests, only the expectations that
are genuinely about behaviour.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from jobtrack.classify.confidence import (
    CONFIDENCE_WEIGHTS,
    DEFAULT_MIN_CONFIDENCE,
    needs_review,
    score_confidence,
)
from jobtrack.classify.rules import RulesClassifier
from jobtrack.models import EventType, RawMessage

W = CONFIDENCE_WEIGHTS

ACK_SUBJECT = "ack.subject.thanks_for_applying"
ACK_BODY = "ack.body.application_received"
REJ_BODY = "rej.body.unfortunately"
HP_COMPANY = "co.ats.sender_name"


def test_the_rubric_has_exactly_the_documented_signals() -> None:
    """CONTRACTS.md §5 freezes the key set; a stray key means a number escaped the table."""
    assert set(W) == {
        "ats_detected",
        "subject_pattern",
        "body_pattern",
        "company_extracted",
        "ambiguous_penalty",
    }


def test_only_the_penalty_is_negative() -> None:
    """An additive rubric with a second negative term would be a different design."""
    assert W["ambiguous_penalty"] < 0
    assert all(value > 0 for key, value in W.items() if key != "ambiguous_penalty")


def test_every_signal_together_saturates_at_one() -> None:
    """The weights are chosen so a fully corroborated message reads as certain."""
    score = score_confidence(
        ats="greenhouse",
        winning_type=EventType.APPLICATION_RECEIVED,
        evidence=[ACK_SUBJECT, ACK_BODY, HP_COMPANY],
        company="Acme Robotics",
        all_scores={EventType.APPLICATION_RECEIVED: [ACK_SUBJECT, ACK_BODY]},
    )
    assert score == 1.0


def test_signals_are_additive() -> None:
    """Each signal contributes exactly its tabled weight."""
    base = {
        "winning_type": EventType.APPLICATION_RECEIVED,
        "all_scores": {EventType.APPLICATION_RECEIVED: [ACK_BODY]},
    }
    body_only = score_confidence(ats=None, evidence=[ACK_BODY], company=None, **base)
    with_ats = score_confidence(ats="greenhouse", evidence=[ACK_BODY], company=None, **base)

    assert body_only == pytest.approx(W["body_pattern"])
    assert with_ats == pytest.approx(W["body_pattern"] + W["ats_detected"])


def test_a_low_precision_subject_pattern_earns_nothing() -> None:
    """Only HIGH-precision subject patterns earn the subject weight."""
    score = score_confidence(
        ats=None,
        winning_type=EventType.INTERVIEW,
        evidence=["itv.subject.next_steps"],
        company=None,
        all_scores={EventType.INTERVIEW: ["itv.subject.next_steps"]},
    )
    assert score == 0.0


def test_a_subject_pattern_for_a_losing_type_earns_nothing() -> None:
    """The weight is for the WINNING type's subject pattern, not any subject pattern."""
    score = score_confidence(
        ats=None,
        winning_type=EventType.REJECTION,
        evidence=[ACK_SUBJECT, REJ_BODY],
        company=None,
        all_scores={EventType.REJECTION: [REJ_BODY], EventType.APPLICATION_RECEIVED: [ACK_SUBJECT]},
    )
    assert score == pytest.approx(W["body_pattern"])


def test_company_weight_needs_a_high_precision_extractor() -> None:
    """A company guessed from a sender domain is not worth the same as one read from text."""
    common = {
        "ats": None,
        "winning_type": EventType.APPLICATION_RECEIVED,
        "all_scores": {EventType.APPLICATION_RECEIVED: [ACK_BODY]},
        "company": "Acme",
    }
    weak = score_confidence(evidence=[ACK_BODY, "co.sender.domain"], **common)
    strong = score_confidence(evidence=[ACK_BODY, HP_COMPANY], **common)

    assert weak == pytest.approx(W["body_pattern"])
    assert strong == pytest.approx(W["body_pattern"] + W["company_extracted"])


def test_company_weight_needs_an_actual_company() -> None:
    """A high-precision rule that produced nothing earns nothing."""
    score = score_confidence(
        ats=None,
        winning_type=EventType.APPLICATION_RECEIVED,
        evidence=[ACK_BODY, HP_COMPANY],
        company=None,
        all_scores={EventType.APPLICATION_RECEIVED: [ACK_BODY]},
    )
    assert score == pytest.approx(W["body_pattern"])


def test_adjacent_types_are_penalized_as_ambiguous() -> None:
    """ASSESSMENT and APPLICATION_RECEIVED are neighbours: matching both is genuine doubt."""
    scores = {
        EventType.ASSESSMENT: ["asm.body.complete_the_assessment"],
        EventType.APPLICATION_RECEIVED: [ACK_BODY],
    }
    penalized = score_confidence(
        ats="greenhouse",
        winning_type=EventType.ASSESSMENT,
        evidence=["asm.body.complete_the_assessment"],
        company=None,
        all_scores=scores,
    )
    alone = score_confidence(
        ats="greenhouse",
        winning_type=EventType.ASSESSMENT,
        evidence=["asm.body.complete_the_assessment"],
        company=None,
        all_scores={EventType.ASSESSMENT: ["asm.body.complete_the_assessment"]},
    )
    assert penalized == pytest.approx(alone + W["ambiguous_penalty"])


def test_distant_types_are_not_ambiguous() -> None:
    """A rejection restating the application language is expected, not confusing.

    REJECTION and APPLICATION_RECEIVED are far apart in EVENT_PRECEDENCE precisely because
    that combination is the normal shape of a rejection email.
    """
    scores = {EventType.REJECTION: [REJ_BODY], EventType.APPLICATION_RECEIVED: [ACK_SUBJECT]}
    penalized = score_confidence(
        ats=None,
        winning_type=EventType.REJECTION,
        evidence=[REJ_BODY],
        company=None,
        all_scores=scores,
    )
    assert penalized == pytest.approx(W["body_pattern"])


def test_scores_are_clamped_to_the_unit_interval() -> None:
    """Classification.confidence is a pydantic field with ge=0 le=1; a breach would raise."""
    floor = score_confidence(
        ats=None,
        winning_type=EventType.ASSESSMENT,
        evidence=[],
        company=None,
        all_scores={
            EventType.ASSESSMENT: ["asm.body.complete_the_assessment"],
            EventType.APPLICATION_RECEIVED: [ACK_BODY],
        },
    )
    assert floor == 0.0


def test_unknown_is_always_zero() -> None:
    """A confident 'I have no idea' is a contradiction, even from an ATS sender."""
    score = score_confidence(
        ats="greenhouse",
        winning_type=EventType.UNKNOWN,
        evidence=["ats.sender.greenhouse"],
        company="Acme",
        all_scores={},
    )
    assert score == 0.0


def test_scores_are_rounded_for_reproducibility() -> None:
    """Float addition of the weights must not leak 0.6000000000000001 into stored data (I2)."""
    score = score_confidence(
        ats="greenhouse",
        winning_type=EventType.REJECTION,
        evidence=[REJ_BODY, HP_COMPANY],
        company="Acme",
        all_scores={EventType.REJECTION: [REJ_BODY]},
    )
    assert score == 0.6
    assert repr(score) == "0.6"


# --------------------------------------------------------------------------------------
# needs_review
# --------------------------------------------------------------------------------------


def test_below_threshold_needs_review() -> None:
    """The documented rule: confidence < min_confidence."""
    assert needs_review(0.59, "Acme", threshold=0.60) is True
    assert needs_review(0.60, "Acme", threshold=0.60) is False
    assert needs_review(0.61, "Acme", threshold=0.60) is False


def test_a_missing_company_always_needs_review() -> None:
    """Without a company the store cannot link the message, however sure the typing was."""
    assert needs_review(1.0, None, threshold=0.60) is True


def test_the_default_threshold_matches_the_config_default() -> None:
    """A classifier built without a threshold must behave like the shipped config."""
    assert DEFAULT_MIN_CONFIDENCE == 0.60


def test_the_threshold_is_configurable(make_message: Callable[..., RawMessage]) -> None:
    """cli.py passes config.classify.min_confidence; a stricter one flags more."""
    message = make_message(
        subject="Thanks for applying to Acme Robotics",
        body_text="We have received your application.",
        from_email="careers@acme.com",
        from_name="Acme Robotics",
    )
    lenient = RulesClassifier(min_confidence=0.10).classify(message)
    strict = RulesClassifier(min_confidence=0.999).classify(message)

    assert lenient.confidence == strict.confidence, "the threshold must not move the score"
    assert lenient.needs_review is False
    assert strict.needs_review is True


def test_fixture_confidences_are_in_range(
    email_fixtures: list[tuple[str, RawMessage]],
) -> None:
    """Every real fixture produces a storable score."""
    classifier = RulesClassifier()
    for stem, message in email_fixtures:
        result = classifier.classify(message)
        assert 0.0 <= result.confidence <= 1.0, stem


def test_high_confidence_fixtures_skip_the_review_queue(
    email_fixtures: list[tuple[str, RawMessage]],
) -> None:
    """The queue is only useful if confident, well-formed ATS mail stays out of it."""
    by_stem = dict(email_fixtures)
    classifier = RulesClassifier()
    for stem in ("greenhouse_confirmation", "lever_rejection_not_moving_forward"):
        result = classifier.classify(by_stem[stem])
        assert result.needs_review is False, f"{stem} should not need review"
