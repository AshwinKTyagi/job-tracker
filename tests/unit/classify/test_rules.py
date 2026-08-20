"""The three pipeline stages in isolation: ATS detection, event typing, field extraction."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from jobtrack.classify.patterns import ATS_DOMAINS
from jobtrack.classify.rules import (
    RulesClassifier,
    detect_ats,
    extract_company,
    extract_location,
    extract_role,
    score_event_types,
)
from jobtrack.models import EventType, RawMessage

# --------------------------------------------------------------------------------------
# Stage 1 — ATS detection
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("slug", sorted(ATS_DOMAINS))
def test_every_known_ats_is_detected_from_the_sender(
    slug: str, make_message: Callable[..., RawMessage]
) -> None:
    """Every slug in the table is reachable — an unreachable row is dead config."""
    domain = ATS_DOMAINS[slug][0]
    message = make_message(from_email=f"no-reply@{domain}")
    detected, rule_ids = detect_ats(message)
    assert detected == slug
    assert rule_ids == [f"ats.sender.{slug}"]


@pytest.mark.parametrize("slug", sorted(ATS_DOMAINS))
def test_every_known_ats_is_detected_from_a_subdomain(
    slug: str, make_message: Callable[..., RawMessage]
) -> None:
    """ATS mail arrives from regional subdomains like ``us.greenhouse-mail.io``."""
    domain = ATS_DOMAINS[slug][0]
    message = make_message(from_email=f"no-reply@eu.mail.{domain}")
    detected, _ = detect_ats(message)
    assert detected == slug


def test_ats_detected_from_reply_to(make_message: Callable[..., RawMessage]) -> None:
    """ATS mail relayed through a customer domain still names the vendor in Reply-To."""
    message = make_message(
        from_email="careers@acme.com",
        headers={"reply-to": "no-reply@greenhouse.io"},
    )
    detected, rule_ids = detect_ats(message)
    assert detected == "greenhouse"
    assert rule_ids == ["ats.replyto.greenhouse"]


def test_ats_detected_from_list_unsubscribe_url(
    make_message: Callable[..., RawMessage],
) -> None:
    """List-Unsubscribe carries a URL, not an address — the host must still be read out."""
    message = make_message(
        from_email="careers@acme.com",
        headers={"list-unsubscribe": "<https://boards.greenhouse.io/unsubscribe/abc123>"},
    )
    detected, rule_ids = detect_ats(message)
    assert detected == "greenhouse"
    assert rule_ids == ["ats.unsubscribe.greenhouse"]


def test_sender_wins_over_headers(make_message: Callable[..., RawMessage]) -> None:
    """The From address decides; headers only corroborate or fill a gap."""
    message = make_message(
        from_email="no-reply@hire.lever.co",
        headers={"reply-to": "no-reply@greenhouse.io"},
    )
    detected, rule_ids = detect_ats(message)
    assert detected == "lever"
    assert rule_ids == ["ats.sender.lever"]


def test_corroborating_sources_all_contribute_evidence(
    make_message: Callable[..., RawMessage],
) -> None:
    """Agreement across sender, Reply-To and unsubscribe is recorded, not collapsed."""
    message = make_message(
        from_email="no-reply@us.greenhouse-mail.io",
        headers={
            "reply-to": "no-reply@greenhouse.io",
            "list-unsubscribe": "<https://boards.greenhouse.io/unsubscribe/abc123>",
        },
    )
    _detected, rule_ids = detect_ats(message)
    assert rule_ids == [
        "ats.sender.greenhouse",
        "ats.replyto.greenhouse",
        "ats.unsubscribe.greenhouse",
    ]


def test_no_ats_for_an_ordinary_sender(make_message: Callable[..., RawMessage]) -> None:
    """A company's own mail server is not an ATS."""
    assert detect_ats(make_message(from_email="careers@acme.com")) == (None, [])


def test_a_lookalike_domain_is_not_an_ats(make_message: Callable[..., RawMessage]) -> None:
    """Suffix matching must be anchored at a label boundary, not a substring."""
    detected, _ = detect_ats(make_message(from_email="hi@notgreenhouse.io"))
    assert detected is None


def test_empty_headers_are_skipped(make_message: Callable[..., RawMessage]) -> None:
    """A present-but-blank header must not crash or match."""
    message = make_message(from_email="careers@acme.com", headers={"reply-to": ""})
    assert detect_ats(message) == (None, [])


# --------------------------------------------------------------------------------------
# Stage 2 — event typing
# --------------------------------------------------------------------------------------


def test_scoring_reads_subject_and_body_separately(
    make_message: Callable[..., RawMessage],
) -> None:
    """A subject pattern must not match on body text, or precision collapses."""
    body_only = make_message(subject="Hello", body_text="Thanks for applying to Acme.")
    scores = score_event_types(body_only)
    assert scores[EventType.APPLICATION_RECEIVED] == ["ack.body.thanks_for_applying"]


def test_scoring_normalizes_smart_punctuation(
    make_message: Callable[..., RawMessage],
) -> None:
    """Mail clients insert curly apostrophes; one pattern must match both forms."""
    curly = make_message(body_text="We\u2019ve received your application.")
    straight = make_message(body_text="We've received your application.")
    assert score_event_types(curly) == score_event_types(straight)


def test_scoring_is_case_insensitive(make_message: Callable[..., RawMessage]) -> None:
    """Subject casing is a styling choice, not a signal."""
    shouty = make_message(subject="THANKS FOR APPLYING TO ACME")
    quiet = make_message(subject="thanks for applying to acme")
    assert score_event_types(shouty) == score_event_types(quiet)


def test_scoring_collapses_wrapped_lines(make_message: Callable[..., RawMessage]) -> None:
    """A phrase split across a hard line wrap must still match."""
    wrapped = make_message(body_text="We have received\nyour application today.")
    assert EventType.APPLICATION_RECEIVED in score_event_types(wrapped)


def test_unrelated_mail_scores_nothing(make_message: Callable[..., RawMessage]) -> None:
    """An empty score map is how UNKNOWN happens."""
    message = make_message(subject="Lunch tomorrow?", body_text="Are you free at noon?")
    assert score_event_types(message) == {}


def test_benign_unfortunately_is_not_a_rejection(
    make_message: Callable[..., RawMessage],
) -> None:
    """The word alone is not a rejection — many confirmations use it apologetically.

    This is the false positive that made rej.body.unfortunately require a negative-outcome
    word in the same clause.
    """
    message = make_message(
        body_text=(
            "Thanks for applying. Unfortunately we cannot respond to every applicant "
            "individually, but our team is reviewing your application now."
        )
    )
    scores = score_event_types(message)
    assert EventType.REJECTION not in scores
    assert EventType.APPLICATION_RECEIVED in scores


def test_a_promise_to_schedule_is_not_an_interview(
    make_message: Callable[..., RawMessage],
) -> None:
    """ "someone will be in touch to schedule" is a confirmation, not an invitation."""
    message = make_message(
        subject="Thanks for applying to Acme",
        body_text=(
            "We have received your application. If your background is a match, someone from "
            "our team will be in touch to schedule a first conversation."
        ),
    )
    scores = score_event_types(message)
    assert EventType.INTERVIEW not in scores


# --------------------------------------------------------------------------------------
# Stage 3 — extraction
# --------------------------------------------------------------------------------------


def test_company_from_ats_sender_name(make_message: Callable[..., RawMessage]) -> None:
    """ATS relays put the customer's name in the display name and nothing in the envelope."""
    message = make_message(from_email="no-reply@greenhouse.io", from_name="Acme Robotics")
    assert extract_company(message, "greenhouse") == ("Acme Robotics", ["co.ats.sender_name"])


@pytest.mark.parametrize(
    "display",
    ["Acme Robotics Recruiting", "Acme Robotics Careers", "Acme Robotics Talent Acquisition"],
)
def test_department_noise_is_stripped_from_the_sender_name(
    display: str, make_message: Callable[..., RawMessage]
) -> None:
    """'Acme Robotics Recruiting' and 'Acme Robotics' are one employer, not two."""
    message = make_message(from_email="no-reply@greenhouse.io", from_name=display)
    company, _ = extract_company(message, "greenhouse")
    assert company == "Acme Robotics"


@pytest.mark.parametrize("display", ["no-reply", "Careers", "Recruiting Team", "Notifications"])
def test_a_generic_sender_name_is_not_a_company(
    display: str, make_message: Callable[..., RawMessage]
) -> None:
    """A generic display name must fall through, not become the employer."""
    message = make_message(
        from_email="no-reply@greenhouse.io", from_name=display, subject="", body_text=""
    )
    assert extract_company(message, "greenhouse") == (None, [])


def test_the_ats_vendor_is_not_the_company(make_message: Callable[..., RawMessage]) -> None:
    """'LinkedIn <messages-noreply@linkedin.com>' is the relay, not the employer."""
    message = make_message(
        from_email="messages-noreply@linkedin.com", from_name="LinkedIn", subject="", body_text=""
    )
    assert extract_company(message, "linkedin") == (None, [])


def test_company_from_a_person_at_company_display_name(
    make_message: Callable[..., RawMessage],
) -> None:
    """'Jane from Acme Robotics' names a company through a person."""
    message = make_message(from_email="no-reply@greenhouse.io", from_name="Jane from Acme Robotics")
    company, _ = extract_company(message, "greenhouse")
    assert company == "Acme Robotics"


def test_a_human_sender_falls_through_to_the_domain(
    make_message: Callable[..., RawMessage],
) -> None:
    """A recruiter's own name is not their employer; the domain is the better guess."""
    message = make_message(
        from_email="jordan@sparkloom.com", from_name="Jordan Avery", subject="", body_text=""
    )
    assert extract_company(message, None) == ("Sparkloom", ["co.sender.domain"])


def test_a_free_mail_domain_yields_no_company(
    make_message: Callable[..., RawMessage],
) -> None:
    """gmail.com is not an employer."""
    message = make_message(
        from_email="jordan@gmail.com", from_name="Jordan Avery", subject="", body_text=""
    )
    assert extract_company(message, None) == (None, [])


def test_functional_subdomains_are_skipped(make_message: Callable[..., RawMessage]) -> None:
    """``careers.acme.com`` names the function in one label and the company in the next."""
    message = make_message(
        from_email="hi@careers.acme.com", from_name="Jordan Avery", subject="", body_text=""
    )
    company, _ = extract_company(message, None)
    assert company == "Acme"


def test_company_capture_stops_at_the_trailing_clause(
    make_message: Callable[..., RawMessage],
) -> None:
    """A subject naming both company and role must not swallow the role into the company."""
    message = make_message(
        subject="Your application to Acme Robotics for the Senior Engineer role",
        from_email="careers@example.org",
        from_name=None,
    )
    company, rule_ids = extract_company(message, None)
    assert company == "Acme Robotics"
    assert rule_ids == ["co.subject.application_at"]


def test_role_capture_keeps_internal_commas(make_message: Callable[..., RawMessage]) -> None:
    """Real titles carry commas: 'Senior Software Engineer, Platform' is one title."""
    message = make_message(
        body_text="We received your application for the Senior Software Engineer, Platform role."
    )
    role, rule_ids = extract_role(message, None)
    assert role == "Senior Software Engineer, Platform"
    assert rule_ids == ["role.body.application_for_role"]


@pytest.mark.parametrize(
    "subject",
    [
        "Your offer at Northwind Analytics",
        "Offer of employment at Ridgeline Robotics",
        "Exciting opportunity at Vantage Grid",
        "Update on your application at Acme",
    ],
)
def test_subject_boilerplate_is_not_read_as_a_job_title(
    subject: str, make_message: Callable[..., RawMessage]
) -> None:
    """'<something> at <Company>' only names a role when the prefix is actually a title."""
    role, _ = extract_role(make_message(subject=subject, body_text=""), None)
    assert role is None


def test_a_real_title_at_a_company_is_read_as_a_role(
    make_message: Callable[..., RawMessage],
) -> None:
    """The job-board subject form still works after the boilerplate guard."""
    message = make_message(subject="Software Engineer at Cinderwood Games", body_text="")
    role, rule_ids = extract_role(message, None)
    assert role == "Software Engineer"
    assert rule_ids == ["role.subject.role_at_company"]


def test_filler_is_not_a_role(make_message: Callable[..., RawMessage]) -> None:
    """'your application for this position' names no title."""
    message = make_message(
        subject="", body_text="We have received your application for this position."
    )
    assert extract_role(message, None) == (None, [])


def test_an_ats_message_prefers_the_body_for_the_role(
    make_message: Callable[..., RawMessage],
) -> None:
    """ATS subjects are templated boilerplate; the body names the requisition."""
    message = make_message(
        subject="Data Analyst at Acme Robotics",
        body_text="Your application for the Senior Platform Engineer role is in review.",
    )
    ats_role, _ = extract_role(message, "greenhouse")
    plain_role, _ = extract_role(message, None)
    assert ats_role == "Senior Platform Engineer"
    assert plain_role == "Data Analyst"


@pytest.mark.parametrize(
    ("body", "subject", "want"),
    [
        ("Location: Boston, MA\nRole: Engineer", "", "Boston"),
        ("Our team is based in Berlin.", "", "Berlin"),
        ("", "Software Engineer (Remote)", "Remote"),
    ],
)
def test_location_is_extracted_when_stated(
    body: str, subject: str, want: str, make_message: Callable[..., RawMessage]
) -> None:
    """Location is read only when the email says it outright."""
    message = make_message(subject=subject, body_text=body)
    assert extract_location(message) == want


def test_missing_location_is_none(make_message: Callable[..., RawMessage]) -> None:
    """Most job mail omits location; that is not a defect."""
    message = make_message(subject="Thanks for applying", body_text="We got your application.")
    assert extract_location(message) is None


def test_extraction_survives_an_empty_message(
    make_message: Callable[..., RawMessage],
) -> None:
    """Every extractor tolerates a blank message rather than raising."""
    blank = make_message(subject="", body_text="", from_name=None, from_email="")
    assert extract_company(blank, None) == (None, [])
    assert extract_role(blank, None) == (None, [])
    assert extract_location(blank) is None


def test_classifier_exposes_the_contract_surface() -> None:
    """name and version are part of the Protocol and are persisted with every row."""
    classifier = RulesClassifier()
    assert classifier.name == "rules"
    assert classifier.version == "1.0.0"
