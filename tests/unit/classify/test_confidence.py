"""Tests for jobtrack.classify.confidence."""

from __future__ import annotations

from jobtrack.classify.confidence import (
    CONFIDENCE_WEIGHTS,
    DEFAULT_MIN_CONFIDENCE,
    needs_review,
    score_confidence,
)
from jobtrack.models import EventType


def test_confidence_weights_table_matches_contract() -> None:
    """CONTRACTS.md §5 pins these exact keys and values."""
    assert CONFIDENCE_WEIGHTS == {
        "ats_detected": 0.35,
        "subject_pattern": 0.40,
        "body_pattern": 0.20,
        "company_extracted": 0.05,
        "ambiguous_penalty": -0.20,
    }


def test_score_confidence_no_signals_is_zero() -> None:
    score = score_confidence(
        ats=None,
        winning_type=EventType.UNKNOWN,
        evidence=[],
        company=None,
        all_scores={},
    )
    assert score == 0.0


def test_score_confidence_adds_ats_detected() -> None:
    score = score_confidence(
        ats="greenhouse",
        winning_type=EventType.APPLICATION_RECEIVED,
        evidence=[],
        company=None,
        all_scores={EventType.APPLICATION_RECEIVED: ["ack.body.application_received"]},
    )
    assert score == CONFIDENCE_WEIGHTS["ats_detected"] + CONFIDENCE_WEIGHTS["body_pattern"]


def test_score_confidence_adds_subject_and_body_pattern_once_each() -> None:
    evidence = ["ack.subject.thanks_for_applying", "ack.body.application_received"]
    score = score_confidence(
        ats=None,
        winning_type=EventType.APPLICATION_RECEIVED,
        evidence=evidence,
        company=None,
        all_scores={EventType.APPLICATION_RECEIVED: evidence},
    )
    assert score == CONFIDENCE_WEIGHTS["subject_pattern"] + CONFIDENCE_WEIGHTS["body_pattern"]


def test_score_confidence_multiple_subject_rules_do_not_double_count() -> None:
    evidence = ["ack.subject.thanks_for_applying", "ack.subject.application_received"]
    score = score_confidence(
        ats=None,
        winning_type=EventType.APPLICATION_RECEIVED,
        evidence=evidence,
        company=None,
        all_scores={EventType.APPLICATION_RECEIVED: evidence},
    )
    assert score == CONFIDENCE_WEIGHTS["subject_pattern"]


def test_score_confidence_adds_company_extracted() -> None:
    score = score_confidence(
        ats=None,
        winning_type=EventType.APPLICATION_RECEIVED,
        evidence=[],
        company="Acme Robotics",
        all_scores={EventType.APPLICATION_RECEIVED: ["ack.body.application_received"]},
    )
    assert score == CONFIDENCE_WEIGHTS["company_extracted"]


def test_score_confidence_applies_ambiguous_penalty_for_adjacent_matches() -> None:
    # REJECTION (index 1) and OFFER (index 2) are adjacent in EVENT_PRECEDENCE.
    all_scores = {
        EventType.REJECTION: ["rej.body.not_moving_forward"],
        EventType.OFFER: ["off.body.extend_offer"],
    }
    score = score_confidence(
        ats=None,
        winning_type=EventType.REJECTION,
        evidence=["rej.body.not_moving_forward"],
        company=None,
        all_scores=all_scores,
    )
    assert score == max(
        0.0, CONFIDENCE_WEIGHTS["body_pattern"] + CONFIDENCE_WEIGHTS["ambiguous_penalty"]
    )


def test_score_confidence_no_penalty_for_non_adjacent_matches() -> None:
    # WITHDRAWN (index 0) and APPLICATION_RECEIVED (index 5) are not adjacent.
    all_scores = {
        EventType.WITHDRAWN: ["wd.body.confirmed_withdrawal"],
        EventType.APPLICATION_RECEIVED: ["ack.body.application_received"],
    }
    score = score_confidence(
        ats=None,
        winning_type=EventType.WITHDRAWN,
        evidence=["wd.body.confirmed_withdrawal"],
        company=None,
        all_scores=all_scores,
    )
    assert score == CONFIDENCE_WEIGHTS["body_pattern"]


def test_score_confidence_clamped_to_one() -> None:
    evidence = ["x.subject.a", "x.body.a"]
    score = score_confidence(
        ats="greenhouse",
        winning_type=EventType.OFFER,
        evidence=evidence,
        company="Acme Robotics",
        all_scores={EventType.OFFER: evidence},
    )
    assert score == 1.0


def test_score_confidence_clamped_to_zero() -> None:
    # ambiguous_penalty alone (no other positive signal) must not go negative.
    all_scores = {
        EventType.OFFER: ["off.body.extend_offer"],
        EventType.INTERVIEW: ["int.subject.schedule_interview"],
    }
    score = score_confidence(
        ats=None,
        winning_type=EventType.OFFER,
        evidence=[],
        company=None,
        all_scores=all_scores,
    )
    assert score == 0.0


def test_needs_review_true_below_threshold() -> None:
    assert needs_review(0.5, "Acme Robotics", threshold=0.6) is True


def test_needs_review_false_at_or_above_threshold_with_company() -> None:
    assert needs_review(0.6, "Acme Robotics", threshold=0.6) is False
    assert needs_review(0.9, "Acme Robotics", threshold=0.6) is False


def test_needs_review_true_when_company_missing_even_with_high_confidence() -> None:
    assert needs_review(0.95, None, threshold=0.6) is True


def test_default_min_confidence_matches_classify_config_default() -> None:
    assert DEFAULT_MIN_CONFIDENCE == 0.60
