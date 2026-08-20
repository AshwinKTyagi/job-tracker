"""Tests for jobtrack.classify.rules: detect_ats, scoring, resolution, extraction."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from jobtrack.classify.rules import (
    RulesClassifier,
    detect_ats,
    extract_company,
    extract_location,
    extract_role,
    resolve_event_type,
    score_event_types,
)
from jobtrack.models import EventType, RawMessage

MakeMessage = Callable[..., RawMessage]


# --------------------------------------------------------------------------------------
# detect_ats
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("slug", "domain"),
    [
        ("greenhouse", "boards.greenhouse.io"),
        ("lever", "hire.lever.co"),
        ("workday", "myworkday.com"),
        ("ashby", "ashbyhq.com"),
        ("smartrecruiters", "smartrecruiters.com"),
        ("icims", "icims.com"),
        ("taleo", "taleo.net"),
        ("jobvite", "jobvite.com"),
        ("workable", "workable.com"),
        ("breezy", "breezy.hr"),
        ("bamboohr", "bamboohr.com"),
        ("recruitee", "recruitee.com"),
        ("teamtailor", "teamtailor.com"),
        ("jazzhr", "jazzhr.com"),
        ("dover", "dover.com"),
        ("rippling", "rippling.com"),
        ("wellfound", "wellfound.com"),
        ("linkedin", "linkedin.com"),
        ("indeed", "indeed.com"),
    ],
)
def test_detect_ats_by_sender_domain(
    make_message: MakeMessage, slug: str, domain: str
) -> None:
    message = make_message(from_email=f"no-reply@{domain}")
    ats, evidence = detect_ats(message)
    assert ats == slug
    assert evidence == [f"ats.sender.{slug}"]


def test_detect_ats_by_reply_to_header(make_message: MakeMessage) -> None:
    message = make_message(
        from_email="jobs@acme.example",
        headers={"reply-to": "no-reply@greenhouse.io"},
    )
    ats, evidence = detect_ats(message)
    assert ats == "greenhouse"
    assert evidence == ["ats.reply_to.greenhouse"]


def test_detect_ats_by_list_unsubscribe_header(make_message: MakeMessage) -> None:
    message = make_message(
        from_email="jobs@acme.example",
        headers={"list-unsubscribe": "<https://jobs.lever.co/unsubscribe/xyz>"},
    )
    ats, evidence = detect_ats(message)
    assert ats == "lever"
    assert evidence == ["ats.list_unsubscribe.lever"]


def test_detect_ats_combines_multiple_signals_for_same_ats(make_message: MakeMessage) -> None:
    message = make_message(
        from_email="no-reply@us.greenhouse-mail.io",
        headers={
            "reply-to": "no-reply@greenhouse.io",
            "list-unsubscribe": "<https://boards.greenhouse.io/unsubscribe/abc>",
        },
    )
    ats, evidence = detect_ats(message)
    assert ats == "greenhouse"
    assert evidence == [
        "ats.sender.greenhouse",
        "ats.reply_to.greenhouse",
        "ats.list_unsubscribe.greenhouse",
    ]


def test_detect_ats_none_when_no_known_domain(make_message: MakeMessage) -> None:
    message = make_message(from_email="careers@acme.example")
    assert detect_ats(message) == (None, [])


def test_detect_ats_is_case_insensitive(make_message: MakeMessage) -> None:
    message = make_message(from_email="No-Reply@Greenhouse.IO")
    ats, _ = detect_ats(message)
    assert ats == "greenhouse"


# --------------------------------------------------------------------------------------
# score_event_types / resolve_event_type
# --------------------------------------------------------------------------------------


def test_score_event_types_empty_for_unrelated_message(make_message: MakeMessage) -> None:
    message = make_message(subject="Your weekly newsletter", body_text="Nothing job related.")
    assert score_event_types(message) == {}


def test_score_event_types_scores_every_matching_type_not_just_first(
    make_message: MakeMessage,
) -> None:
    """I3: score against ALL event types, never stop at first hit."""
    message = make_message(
        subject="Thanks for applying to Acme Robotics",
        body_text=(
            "Thanks for applying to Acme Robotics. We have decided not to move forward "
            "with your application; we are moving ahead with other candidates."
        ),
    )
    scores = score_event_types(message)
    assert EventType.APPLICATION_RECEIVED in scores
    assert EventType.REJECTION in scores


def test_resolve_event_type_empty_scores_is_unknown() -> None:
    assert resolve_event_type({}) == (EventType.UNKNOWN, [])


def test_resolve_event_type_picks_highest_precedence() -> None:
    scores = {
        EventType.APPLICATION_RECEIVED: ["ack.subject.thanks_for_applying"],
        EventType.REJECTION: ["rej.body.not_moving_forward"],
    }
    winner, evidence = resolve_event_type(scores)
    assert winner is EventType.REJECTION
    assert evidence == ["rej.body.not_moving_forward"]


def test_resolve_event_type_withdrawn_beats_everything(
    make_message: MakeMessage,
) -> None:
    """WITHDRAWN leads EVENT_PRECEDENCE (Phase 0 correction #1)."""
    scores = {
        EventType.WITHDRAWN: ["wd.body.confirmed_withdrawal"],
        EventType.REJECTION: ["rej.body.not_moving_forward"],
        EventType.OFFER: ["off.body.extend_offer"],
    }
    winner, _ = resolve_event_type(scores)
    assert winner is EventType.WITHDRAWN


def test_score_event_types_does_not_score_unknown_as_a_key(make_message: MakeMessage) -> None:
    message = make_message(subject="", body_text="")
    scores = score_event_types(message)
    assert EventType.UNKNOWN not in scores


# --------------------------------------------------------------------------------------
# extract_company
# --------------------------------------------------------------------------------------


def test_extract_company_uses_ats_sender_display_name(make_message: MakeMessage) -> None:
    message = make_message(from_name="Acme Robotics", from_email="no-reply@greenhouse.io")
    company, evidence = extract_company(message, "greenhouse")
    assert company == "Acme Robotics"
    assert evidence == ["company.ats_sender.greenhouse"]


def test_extract_company_skips_ats_brand_display_name(make_message: MakeMessage) -> None:
    """A sender literally named after the ATS platform is not the employer."""
    message = make_message(
        from_name="Greenhouse",
        from_email="no-reply@greenhouse.io",
        subject="Thank you for applying to Acme Robotics",
    )
    company, evidence = extract_company(message, "greenhouse")
    assert company == "Acme Robotics"
    assert evidence == ["company.subject.applying_to"]


def test_extract_company_subject_capture_opportunity_at(make_message: MakeMessage) -> None:
    message = make_message(
        from_name="Jordan Lee",
        subject="Exciting opportunity at Vertex Cloud",
    )
    company, evidence = extract_company(message, None)
    assert company == "Vertex Cloud"
    assert evidence == ["company.subject.opportunity_at"]


def test_extract_company_subject_capture_offer_from(make_message: MakeMessage) -> None:
    message = make_message(from_name=None, subject="Your offer from Lumen Robotics!")
    company, evidence = extract_company(message, None)
    assert company == "Lumen Robotics"
    assert evidence == ["company.subject.offer_from"]


def test_extract_company_body_signoff(make_message: MakeMessage) -> None:
    message = make_message(
        from_name=None,
        subject="An update",
        body_text="Hi Alex,\n\nDetails inside.\n\nBest,\nThe Contoso Devices Recruiting Team",
    )
    company, evidence = extract_company(message, None)
    assert company == "Contoso Devices"
    assert evidence == ["company.body.signoff"]


def test_extract_company_falls_back_to_sender_display_name(make_message: MakeMessage) -> None:
    message = make_message(from_name="BrightPath Analytics", subject="An update", body_text="")
    company, evidence = extract_company(message, None)
    assert company == "BrightPath Analytics"
    assert evidence == ["company.sender_display_name"]


def test_extract_company_none_when_nothing_matches(make_message: MakeMessage) -> None:
    message = make_message(from_name=None, subject="An update", body_text="No signature here.")
    assert extract_company(message, None) == (None, [])


def test_extract_company_recruiter_prefers_subject_over_sender_name(
    make_message: MakeMessage,
) -> None:
    """The recruiter's own name (from_name) must not be mistaken for the employer."""
    message = make_message(
        from_name="Jordan Lee",
        subject="Exciting opportunity at Vertex Cloud",
        body_text="I came across your profile...",
    )
    company, _ = extract_company(message, None)
    assert company == "Vertex Cloud"
    assert company != "Jordan Lee"


# --------------------------------------------------------------------------------------
# extract_role
# --------------------------------------------------------------------------------------


def test_extract_role_from_body_for_the_role(make_message: MakeMessage) -> None:
    message = make_message(
        body_text="We have received your application for the Senior Software Engineer role."
    )
    role, evidence = extract_role(message, None)
    assert role == "Senior Software Engineer"
    assert evidence == ["role.body.for_the_role"]


def test_extract_role_from_body_position_of(make_message: MakeMessage) -> None:
    message = make_message(body_text="We are offering you the position of Senior Data Scientist.")
    role, evidence = extract_role(message, None)
    assert role == "Senior Data Scientist"
    assert evidence == ["role.body.position_of"]


def test_extract_role_from_subject(make_message: MakeMessage) -> None:
    message = make_message(subject="Update on your application for the Backend Engineer role")
    role, evidence = extract_role(message, None)
    assert role == "Backend Engineer"
    assert evidence == ["role.subject.for_the_role"]


def test_extract_role_none_when_nothing_matches(make_message: MakeMessage) -> None:
    message = make_message(subject="An update", body_text="No role mentioned here.")
    assert extract_role(message, None) == (None, [])


# --------------------------------------------------------------------------------------
# extract_location
# --------------------------------------------------------------------------------------


def test_extract_location_label(make_message: MakeMessage) -> None:
    message = make_message(body_text="Role details.\nLocation: Boston, MA\nMore details.")
    assert extract_location(message) == "Boston, MA"


def test_extract_location_based_in(make_message: MakeMessage) -> None:
    message = make_message(body_text="This role is based in Austin, TX and reports to the CTO.")
    assert extract_location(message) == "Austin, TX"


def test_extract_location_parenthetical_arrangement(make_message: MakeMessage) -> None:
    message = make_message(subject="Senior Engineer (Remote)", body_text="")
    assert extract_location(message) == "Remote"


def test_extract_location_none_when_absent(make_message: MakeMessage) -> None:
    message = make_message(subject="An update", body_text="No location mentioned.")
    assert extract_location(message) is None


# --------------------------------------------------------------------------------------
# RulesClassifier orchestration
# --------------------------------------------------------------------------------------


def test_rules_classifier_unknown_message_has_no_extracted_fields(
    make_message: MakeMessage,
) -> None:
    message = make_message(
        from_name="Tech Weekly",
        subject="This week: 12 companies hiring",
        body_text="A newsletter, not a job application email.",
    )
    result = RulesClassifier().classify(message)
    assert result.event_type is EventType.UNKNOWN
    assert result.company is None
    assert result.company_key is None
    assert result.role is None
    assert result.location is None
    assert result.confidence == 0.0
    assert result.needs_review is True


def test_rules_classifier_never_raises_on_empty_message(make_message: MakeMessage) -> None:
    message = make_message(subject="", body_text="", from_name=None, headers={})
    result = RulesClassifier().classify(message)
    assert result.event_type is EventType.UNKNOWN
    assert result.confidence == 0.0


def test_rules_classifier_sets_classifier_identity(make_message: MakeMessage) -> None:
    result = RulesClassifier().classify(make_message())
    assert result.classifier_name == "rules"
    assert result.classifier_version == "1.0.0"


def test_rules_classifier_company_key_matches_normalize_company(
    make_message: MakeMessage,
) -> None:
    message = make_message(
        from_name="Acme Robotics, Inc.",
        from_email="no-reply@greenhouse.io",
        subject="Thanks for applying to Acme Robotics",
        body_text="We have received your application for the Engineer role.",
    )
    result = RulesClassifier().classify(message)
    assert result.company == "Acme Robotics, Inc."
    assert result.company_key == "acme robotics"


def test_rules_classifier_evidence_includes_ats_and_type_rule_ids(
    make_message: MakeMessage,
) -> None:
    message = make_message(
        from_name="Acme Robotics",
        from_email="no-reply@greenhouse.io",
        subject="Thanks for applying to Acme Robotics",
        body_text="We have received your application for the Engineer role.",
    )
    result = RulesClassifier().classify(message)
    assert "ats.sender.greenhouse" in result.evidence
    assert "ack.subject.thanks_for_applying" in result.evidence
    assert "ack.body.application_received" in result.evidence


def test_rules_classifier_no_standalone_unfortunately_false_positive(
    make_message: MakeMessage,
) -> None:
    """Regression guard: a bare 'unfortunately' must not, by itself, read as a rejection."""
    message = make_message(
        subject="A quick note",
        body_text="Unfortunately our office coffee machine is broken again today.",
    )
    result = RulesClassifier().classify(message)
    assert result.event_type is not EventType.REJECTION
