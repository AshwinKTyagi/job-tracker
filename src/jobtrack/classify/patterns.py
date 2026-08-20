"""Named, pre-compiled pattern tables for the rules classifier.

Every regex here is compiled once at import time and carries a stable rule id (documented
inline). Rule ids flow straight into ``Classification.evidence`` — treat them as part of the
public contract: renaming one changes what downstream review tooling displays, so prefer
adding a new id over repurposing an old one.

Naming scheme for event-type rule ids: ``<prefix>.<subject|body>.<slug>`` where prefix is
``wd`` (withdrawn), ``rej`` (rejection), ``off`` (offer), ``int`` (interview), ``asmt``
(assessment), ``ack`` (application_received), ``rec`` (recruiter_outreach). ATS detection
uses ``ats.<signal>.<slug>``; field extraction uses ``company.<subject|body|...>.<slug>`` and
``role.<subject|body>.<slug>``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from jobtrack.models import EventType


@dataclass(frozen=True)
class RulePattern:
    """One named, compiled pattern and the rule id it reports when it fires."""

    rule_id: str
    regex: re.Pattern[str]


def _p(rule_id: str, pattern: str, *, verbose: bool = False) -> RulePattern:
    """Compile one case-insensitive rule pattern.

    Private helper: keeps the tables below terse while still naming every regex.
    """
    flags = re.IGNORECASE | (re.VERBOSE if verbose else 0)
    return RulePattern(rule_id, re.compile(pattern, flags))


# --------------------------------------------------------------------------------------
# ATS / sender detection
# --------------------------------------------------------------------------------------

ATS_DOMAINS: dict[str, tuple[str, ...]] = {
    "greenhouse": ("greenhouse.io", "greenhouse-mail.io"),
    "lever": ("lever.co",),
    "workday": ("myworkday.com", "workday.com"),
    "ashby": ("ashbyhq.com",),
    "smartrecruiters": ("smartrecruiters.com",),
    "icims": ("icims.com",),
    "taleo": ("taleo.net",),
    "jobvite": ("jobvite.com",),
    "workable": ("workable.com",),
    "breezy": ("breezy.hr",),
    "bamboohr": ("bamboohr.com",),
    "recruitee": ("recruitee.com",),
    "teamtailor": ("teamtailor.com",),
    "jazzhr": ("jazzhr.com",),
    "dover": ("dover.com", "dover.io"),
    "rippling": ("rippling.com",),
    "wellfound": ("wellfound.com",),
    "linkedin": ("linkedin.com",),
    "indeed": ("indeed.com",),
}
"""Slug -> sender/reply-to/list-unsubscribe domain substrings. Order is the detection order
(``detect_ats`` walks this table top to bottom), so it is fixed and deliberate — not an
incidental dict-ordering dependency (I2)."""

ATS_BRAND_NAMES: frozenset[str] = frozenset(
    {
        "greenhouse",
        "lever",
        "workday",
        "ashby",
        "ashbyhq",
        "smartrecruiters",
        "icims",
        "taleo",
        "jobvite",
        "workable",
        "breezy",
        "breezy hr",
        "bamboohr",
        "recruitee",
        "teamtailor",
        "jazzhr",
        "dover",
        "rippling",
        "wellfound",
        "linkedin",
        "indeed",
        "no reply",
        "noreply",
        "notifications",
    }
)
"""Casefolded sender display names that name the ATS platform itself, not the employer —
excluded as a company guess."""


# --------------------------------------------------------------------------------------
# Event-type patterns. Every EventType except UNKNOWN gets a subject and a body table;
# UNKNOWN is the absence of any match, never a pattern of its own.
# --------------------------------------------------------------------------------------

WITHDRAWN_SUBJECT: tuple[RulePattern, ...] = (
    _p(
        "wd.subject.withdrawn_application",
        r"withdr\w*.{0,40}application|application.{0,40}withdr\w*",
    ),
)

WITHDRAWN_BODY: tuple[RulePattern, ...] = (
    _p("wd.body.confirmed_withdrawal", r"(?:you|we) have withdrawn your application"),
    _p("wd.body.per_request", r"per your request,? we have withdrawn"),
    _p("wd.body.withdrawal_confirmed", r"your withdrawal (?:has been|is) confirmed"),
)

REJECTION_SUBJECT: tuple[RulePattern, ...] = (
    _p("rej.subject.not_moving_forward", r"not moving forward"),
    _p("rej.subject.regret_to_inform", r"regret to inform"),
    _p("rej.subject.unsuccessful", r"application (?:has been )?(?:unsuccessful|declined)"),
)

REJECTION_BODY: tuple[RulePattern, ...] = (
    _p("rej.body.not_moving_forward", r"not moving forward"),
    _p("rej.body.decided_not_to_proceed", r"decided not to (?:move forward|proceed)"),
    _p("rej.body.other_candidates", r"(?:pursuing|moving (?:ahead|forward) with) other candidates"),
    _p("rej.body.will_not_be", r"will not be (?:moving forward|proceeding|selected)"),
    _p("rej.body.not_selected", r"not been selected"),
    _p("rej.body.position_filled", r"filled the position"),
    _p("rej.body.regret_to_inform", r"regret to inform"),
)

OFFER_SUBJECT: tuple[RulePattern, ...] = (
    _p("off.subject.your_offer", r"\byour offer\b"),
    _p("off.subject.offer_letter", r"\boffer letter\b"),
    _p("off.subject.pleased_to_offer", r"(?:excited|pleased) to offer"),
)

OFFER_BODY: tuple[RulePattern, ...] = (
    _p(
        "off.body.pleased_to_offer",
        r"(?:pleased|thrilled|excited) to (?:formally )?(?:offer|extend)",
    ),
    _p("off.body.extend_offer", r"extend(?:ing)? (?:you )?(?:an |the )?offer"),
    _p("off.body.offer_of_employment", r"offer of employment"),
    _p("off.body.offer_letter", r"\boffer letter\b"),
)

INTERVIEW_SUBJECT: tuple[RulePattern, ...] = (
    _p("int.subject.invitation_to_interview", r"invit\w* to interview"),
    _p("int.subject.schedule_interview", r"schedule (?:your |an )?interview"),
    _p("int.subject.interview_confirmation", r"interview (?:confirmation|invite)"),
)

INTERVIEW_BODY: tuple[RulePattern, ...] = (
    _p(
        "int.body.schedule_interview",
        r"schedule (?:a |your )?(?:phone|video|onsite|virtual)?\s*interview",
    ),
    _p("int.body.would_like_to_invite", r"we(?:'d| would) like to (?:invite|schedule) you"),
    _p("int.body.next_steps_interview", r"next steps.{0,40}interview"),
    _p("int.body.book_a_time", r"book a time"),
    _p("int.body.interview_scheduled", r"interview (?:has been )?scheduled"),
)

ASSESSMENT_SUBJECT: tuple[RulePattern, ...] = (
    _p("asmt.subject.assessment", r"(?:online |technical )?assessment"),
    _p("asmt.subject.coding_challenge", r"coding challenge"),
    _p("asmt.subject.take_home", r"take[- ]home"),
)

ASSESSMENT_BODY: tuple[RulePattern, ...] = (
    _p("asmt.body.complete_assessment", r"complete (?:the |an )?(?:online )?assessment"),
    _p("asmt.body.coding_challenge", r"coding challenge"),
    _p("asmt.body.take_home_project", r"take[- ]home (?:project|assignment|exercise)"),
    _p("asmt.body.technical_screen", r"technical (?:screen|assessment)"),
)

APPLICATION_RECEIVED_SUBJECT: tuple[RulePattern, ...] = (
    _p("ack.subject.thanks_for_applying", r"thank(?:s| you) for applying"),
    _p("ack.subject.application_received", r"application received"),
)

APPLICATION_RECEIVED_BODY: tuple[RulePattern, ...] = (
    _p("ack.body.application_received", r"(?:we )?(?:have )?received your application"),
    _p("ack.body.thanks_for_applying", r"thank(?:s| you) for (?:your interest|applying)"),
    _p("ack.body.reviewing", r"our (?:recruiting|hiring) team is reviewing"),
    _p("ack.body.submitted_successfully", r"application (?:has been )?submitted successfully"),
)

RECRUITER_OUTREACH_SUBJECT: tuple[RulePattern, ...] = (
    _p("rec.subject.opportunity_at", r"(?:exciting )?opportunity at"),
    _p("rec.subject.interested_in_connecting", r"interested in connecting"),
    _p("rec.subject.open_to_opportunities", r"open to (?:new )?opportunities"),
)

RECRUITER_OUTREACH_BODY: tuple[RulePattern, ...] = (
    _p("rec.body.came_across_profile", r"came across your profile"),
    _p("rec.body.im_a_recruiter", r"i'?m a (?:technical )?recruiter"),
    _p("rec.body.open_to_opportunities", r"are you open to (?:new )?opportunities"),
    _p("rec.body.background_caught_eye", r"your background (?:caught my eye|stood out)"),
)

SUBJECT_PATTERNS: dict[EventType, tuple[RulePattern, ...]] = {
    EventType.WITHDRAWN: WITHDRAWN_SUBJECT,
    EventType.REJECTION: REJECTION_SUBJECT,
    EventType.OFFER: OFFER_SUBJECT,
    EventType.INTERVIEW: INTERVIEW_SUBJECT,
    EventType.ASSESSMENT: ASSESSMENT_SUBJECT,
    EventType.APPLICATION_RECEIVED: APPLICATION_RECEIVED_SUBJECT,
    EventType.RECRUITER_OUTREACH: RECRUITER_OUTREACH_SUBJECT,
}
"""Subject pattern table, keyed by EventType. UNKNOWN intentionally absent — it is what's
left when nothing else matches."""

BODY_PATTERNS: dict[EventType, tuple[RulePattern, ...]] = {
    EventType.WITHDRAWN: WITHDRAWN_BODY,
    EventType.REJECTION: REJECTION_BODY,
    EventType.OFFER: OFFER_BODY,
    EventType.INTERVIEW: INTERVIEW_BODY,
    EventType.ASSESSMENT: ASSESSMENT_BODY,
    EventType.APPLICATION_RECEIVED: APPLICATION_RECEIVED_BODY,
    EventType.RECRUITER_OUTREACH: RECRUITER_OUTREACH_BODY,
}
"""Body pattern table, keyed by EventType. Same rationale as SUBJECT_PATTERNS."""


# --------------------------------------------------------------------------------------
# Field extraction: company
# --------------------------------------------------------------------------------------

COMPANY_SUBJECT_PATTERNS: tuple[RulePattern, ...] = (
    _p(
        "company.subject.applying_to",
        r"(?:applying to|application to|thank you for applying to)\s+"
        r"(?P<company>[A-Z][\w&.,' -]+?)(?:[!.]|$)",
    ),
    _p(
        "company.subject.opportunity_at",
        r"opportunity at\s+(?P<company>[A-Z][\w&.,' -]+?)(?:[!.]|$)",
    ),
    _p(
        "company.subject.offer_from",
        r"offer from\s+(?P<company>[A-Z][\w&.,' -]+?)(?:[!.]|$)",
    ),
)
"""Subject capture groups, tried after ATS-specific extraction and before body signatures."""

COMPANY_BODY_PATTERNS: tuple[RulePattern, ...] = (
    _p(
        "company.body.signoff",
        r"""
        \bThe\s+                                                     # sign-off preamble
        (?P<company>[A-Z][\w&.,'-]*(?:\s+[A-Z&][\w&.,'-]*){0,3})     # 1-4 capitalized words
        \s+(?:Recruiting|Hiring|Talent\s+Acquisition)\s+Team\b
        """,
        verbose=True,
    ),
)
"""Body signature (letter sign-off) patterns, tried after subject capture and before the
sender-display-name fallback."""


# --------------------------------------------------------------------------------------
# Field extraction: role
# --------------------------------------------------------------------------------------

ROLE_SUBJECT_PATTERNS: tuple[RulePattern, ...] = (
    _p(
        "role.subject.for_the_role",
        r"(?:for|in) the (?P<role>.+?) (?:role|position)\b",
    ),
)

ROLE_BODY_PATTERNS: tuple[RulePattern, ...] = (
    _p(
        "role.body.for_the_role",
        r"(?:for|in) the (?P<role>.+?) (?:role|position)\b",
    ),
    _p(
        "role.body.position_of",
        r"position of (?P<role>.+?)(?:\s+at\b|[.,]|$)",
    ),
)


# --------------------------------------------------------------------------------------
# Field extraction: location
# --------------------------------------------------------------------------------------

LOCATION_LABEL = re.compile(r"^location:\s*(?P<location>.+)$", re.IGNORECASE | re.MULTILINE)
LOCATION_BASED_IN = re.compile(r"\bbased in\s+(?P<location>[A-Z][\w.,\s]+?)(?:[.\n]|$)")
LOCATION_ARRANGEMENT = re.compile(r"\((?P<location>Remote|Hybrid|On-?site)\)", re.IGNORECASE)


__all__ = [
    "APPLICATION_RECEIVED_BODY",
    "APPLICATION_RECEIVED_SUBJECT",
    "ASSESSMENT_BODY",
    "ASSESSMENT_SUBJECT",
    "ATS_BRAND_NAMES",
    "ATS_DOMAINS",
    "BODY_PATTERNS",
    "COMPANY_BODY_PATTERNS",
    "COMPANY_SUBJECT_PATTERNS",
    "INTERVIEW_BODY",
    "INTERVIEW_SUBJECT",
    "LOCATION_ARRANGEMENT",
    "LOCATION_BASED_IN",
    "LOCATION_LABEL",
    "OFFER_BODY",
    "OFFER_SUBJECT",
    "RECRUITER_OUTREACH_BODY",
    "RECRUITER_OUTREACH_SUBJECT",
    "REJECTION_BODY",
    "REJECTION_SUBJECT",
    "ROLE_BODY_PATTERNS",
    "ROLE_SUBJECT_PATTERNS",
    "SUBJECT_PATTERNS",
    "WITHDRAWN_BODY",
    "WITHDRAWN_SUBJECT",
    "RulePattern",
]
