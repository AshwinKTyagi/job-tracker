"""Pattern tables for the rules classifier.

Everything the classifier knows about the *shape* of job mail lives here, as data. The
engine in ``rules.py`` only walks these tables; it never hard-codes a phrase.

Three tables:

* ``ATS_DOMAINS`` — applicant-tracking-system slug to the mail hosts it sends from.
* ``EVENT_PATTERNS`` — one row per phrase the classifier recognizes, each tagged with the
  ``EventType`` it argues for and a **stable rule id** that is recorded verbatim in
  ``Classification.evidence``.
* ``COMPANY_PATTERNS`` / ``ROLE_PATTERNS`` / ``LOCATION_PATTERNS`` — ordered extraction
  chains.

Rule ids are ``<prefix>.<scope>.<name>`` and are part of the stored data: they end up in
``classifications.evidence_json`` and in the ``jobtrack review`` UI. Renaming one is a
breaking change, so treat them as append-only and bump ``RulesClassifier.version`` when the
tables change.

Every rule id in ``EVENT_PATTERNS``, ``COMPANY_PATTERNS`` and ``ROLE_PATTERNS`` must be
exercised by at least one fixture in ``tests/fixtures/emails/`` — ``test_patterns.py``
asserts it (CLAUDE.md: "adding a pattern requires adding a fixture that exercises it").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Literal

from jobtrack.models import EventType

# --------------------------------------------------------------------------------------
# Text normalization
# --------------------------------------------------------------------------------------

_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")

_PUNCTUATION_FOLD: Final[dict[int, str]] = str.maketrans(
    {
        "\u2018": "'",  # left single quote
        "\u2019": "'",  # right single quote / apostrophe
        "\u201c": '"',  # left double quote
        "\u201d": '"',  # right double quote
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u00a0": " ",  # non-breaking space
        "\u200b": "",  # zero-width space
    }
)
"""Fold the typographic characters mail clients insert, so one regex matches both forms."""


def normalize_text(text: str) -> str:
    """Fold a subject or body into the canonical form the event patterns are written against.

    Casefolds, replaces smart quotes and dashes with ASCII, and collapses every run of
    whitespace (including newlines) to a single space. Deterministic — I2 depends on it.

    Args:
        text: Raw subject or body text.

    Returns:
        The normalized single-line, lower-case form.
    """
    return _WHITESPACE_RE.sub(" ", text.translate(_PUNCTUATION_FOLD).casefold()).strip()


# --------------------------------------------------------------------------------------
# Stage 1 — ATS / sender detection
# --------------------------------------------------------------------------------------

ATS_DOMAINS: Final[dict[str, tuple[str, ...]]] = {
    "greenhouse": ("greenhouse.io", "greenhouse-mail.io", "greenhousemail.io"),
    "lever": ("lever.co", "hire.lever.co", "levermail.com"),
    "workday": ("myworkday.com", "myworkdayjobs.com", "workday.com"),
    "ashby": ("ashbyhq.com", "ashbyhq.com.mail"),
    "smartrecruiters": ("smartrecruiters.com", "smartrecruiters.net"),
    "icims": ("icims.com", "icimsmail.com"),
    "taleo": ("taleo.net", "taleo.com"),
    "jobvite": ("jobvite.com", "jobvitemail.com"),
    "workable": ("workable.com", "workablemail.com"),
    "breezy": ("breezy.hr",),
    "bamboohr": ("bamboohr.com",),
    "recruitee": ("recruitee.com",),
    "teamtailor": ("teamtailor.com", "teamtailormail.com"),
    "jazzhr": ("jazzhr.com", "applytojob.com"),
    "dover": ("dover.com", "dover.io"),
    "rippling": ("rippling.com", "rippling-ats.com"),
    "wellfound": ("wellfound.com", "angel.co"),
    "linkedin": ("linkedin.com", "e.linkedin.com"),
    "indeed": ("indeed.com", "indeedemail.com"),
}
"""Known applicant-tracking systems and the hosts their mail comes from (PLAN.md §7)."""

ATS_DOMAIN_ORDER: Final[tuple[tuple[str, str], ...]] = tuple(
    sorted(
        ((domain, slug) for slug, domains in ATS_DOMAINS.items() for domain in domains),
        key=lambda pair: (-len(pair[0]), pair[0]),
    )
)
"""(domain, slug) pairs, longest domain first so ``greenhouse-mail.io`` beats a shorter
suffix. Sorted rather than table-ordered so detection does not depend on dict order (I2)."""

ATS_HEADER_SOURCES: Final[tuple[tuple[str, str], ...]] = (
    ("reply-to", "replyto"),
    ("list-unsubscribe", "unsubscribe"),
    ("return-path", "returnpath"),
    ("x-original-sender", "originalsender"),
)
"""(header name, rule-id token) pairs, checked in this order after the From address."""

FREE_MAIL_DOMAINS: Final[frozenset[str]] = frozenset(
    {
        "gmail.com",
        "googlemail.com",
        "yahoo.com",
        "hotmail.com",
        "outlook.com",
        "live.com",
        "icloud.com",
        "me.com",
        "aol.com",
        "proton.me",
        "protonmail.com",
    }
)
"""Consumer mail hosts. A company is never inferred from one of these domains."""

DOMAIN_RE: Final[re.Pattern[str]] = re.compile(r"@([A-Za-z0-9.\-]+)")
"""Pull the host out of an email address."""

HOST_RE: Final[re.Pattern[str]] = re.compile(r"(?:@|//)([A-Za-z0-9.\-]+)")
"""Pull every host out of a header value. Matches both the address form
(``@greenhouse.io``) and the URL form (``//boards.greenhouse.io``), because
``List-Unsubscribe`` carries URLs while ``Reply-To`` carries addresses."""

DOMAIN_LABEL_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "careers",
        "email",
        "hr",
        "jobs",
        "mail",
        "mailer",
        "no-reply",
        "noreply",
        "notifications",
        "recruiting",
        "smtp",
        "talent",
        "www",
    }
)
"""Sub-domain labels that name a function, not the company: ``careers.acme.com`` → ``acme``."""

# --------------------------------------------------------------------------------------
# Stage 2 — event typing
# --------------------------------------------------------------------------------------

PatternScope = Literal["subject", "body"]
"""Which part of the message a pattern is matched against."""


@dataclass(frozen=True)
class EventPattern:
    """One phrase that argues for one ``EventType``.

    Attributes:
        rule_id: Stable identifier recorded in ``Classification.evidence``.
        event_type: The type this phrase argues for.
        scope: ``"subject"`` or ``"body"``.
        regex: Compiled pattern, matched against ``normalize_text()`` output.
        high_precision: True when a match alone is strong evidence. Only high-precision
            *subject* patterns earn the subject weight in ``confidence.py``.
    """

    rule_id: str
    event_type: EventType
    scope: PatternScope
    regex: re.Pattern[str]
    high_precision: bool = False


def _pattern(
    rule_id: str,
    event_type: EventType,
    scope: PatternScope,
    pattern: str,
    *,
    high_precision: bool = False,
    verbose: bool = False,
) -> EventPattern:
    """Compile one table row. Called only at module import, so every regex is compiled once.

    ``verbose`` turns on ``re.VERBOSE`` for the multi-line patterns that carry inline
    comments; those patterns must spell every literal space as ``\\s``.
    """
    flags = re.VERBOSE if verbose else 0
    return EventPattern(rule_id, event_type, scope, re.compile(pattern, flags), high_precision)


_WITHDRAWN: Final[tuple[EventPattern, ...]] = (
    _pattern(
        "wdr.subject.application_withdrawn",
        EventType.WITHDRAWN,
        "subject",
        r"\b(?:application\s+withdrawn|withdraw(?:n|al)\s+(?:of\s+)?(?:your\s+)?application)\b",
        high_precision=True,
    ),
    _pattern(
        "wdr.body.you_withdrew",
        EventType.WITHDRAWN,
        "body",
        r"\byou(?:'ve|\s+have)?\s+withdrew\b|\byou\s+have\s+withdrawn\b",
    ),
    _pattern(
        "wdr.body.application_withdrawn",
        EventType.WITHDRAWN,
        "body",
        r"\b(?:your\s+)?application\s+(?:has\s+been|was|is)\s+withdrawn\b",
    ),
    _pattern(
        "wdr.body.at_your_request",
        EventType.WITHDRAWN,
        "body",
        r"\bat\s+your\s+request\b",
    ),
)

_REJECTION: Final[tuple[EventPattern, ...]] = (
    _pattern(
        "rej.subject.not_moving_forward",
        EventType.REJECTION,
        "subject",
        r"\bnot\s+(?:be\s+)?mov(?:e|ing)\s+forward\b",
        high_precision=True,
    ),
    _pattern(
        "rej.subject.unsuccessful",
        EventType.REJECTION,
        "subject",
        r"\bunsuccessful\b",
        high_precision=True,
    ),
    _pattern(
        "rej.subject.position_filled",
        EventType.REJECTION,
        "subject",
        r"\b(?:position|role|req(?:uisition)?)\s+(?:has\s+been\s+|is\s+|was\s+)?(?:now\s+)?filled\b",
        high_precision=True,
    ),
    _pattern(
        "rej.body.not_moving_forward",
        EventType.REJECTION,
        "body",
        r"""
        \bnot\s+                                  # the negation …
        (?:be\s+)?
        (?:mov(?:e|ing)|proceed(?:ing)?|          # … attached to a progress verb
           continu(?:e|ing)|progress(?:ing)?|
           advanc(?:e|ing))\s+
        (?:forward|ahead|with\s+your|to\s+the\s+next)
        """,
        verbose=True,
    ),
    _pattern(
        "rej.body.decided_not_to",
        EventType.REJECTION,
        "body",
        r"\bdecided\s+not\s+to\s+(?:move|proceed|continue|advance|pursue)\b",
    ),
    _pattern(
        "rej.body.other_candidates",
        EventType.REJECTION,
        "body",
        r"\b(?:other|another)\s+candidates?\b",
    ),
    _pattern(
        "rej.body.pursue_other_candidates",
        EventType.REJECTION,
        "body",
        r"\b(?:pursu(?:e|ing)|proceed(?:ing)?\s+with|mov(?:e|ing)\s+ahead\s+with)"
        r"\s+(?:other|another)\s+(?:candidates?|applicants?)\b",
    ),
    _pattern(
        "rej.body.unfortunately",
        EventType.REJECTION,
        "body",
        r"""
        \bunfortunately\b          # the softener, which alone means nothing …
        [^.]{0,80}?
        \b(?:not|other|another|no\s+longer|unsuccessful|unable\s+to)\b
        """,
        verbose=True,
    ),
    _pattern(
        "rej.body.not_selected",
        EventType.REJECTION,
        "body",
        r"\b(?:were|was|have|has|are)\s+not\s+(?:been\s+)?(?:selected|chosen|shortlisted)\b",
    ),
    _pattern(
        "rej.body.no_longer_under_consideration",
        EventType.REJECTION,
        "body",
        r"\bno\s+longer\s+(?:be\s+)?(?:under\s+consideration|being\s+considered)\b",
    ),
    _pattern(
        "rej.body.not_a_match",
        EventType.REJECTION,
        "body",
        r"\bnot\s+(?:be\s+)?(?:a|the)\s+(?:right|best|strong(?:est)?|ideal)\s+(?:match|fit)\b",
    ),
    _pattern(
        "rej.body.position_filled",
        EventType.REJECTION,
        "body",
        r"\b(?:position|role|opening)\s+(?:has\s+been\s+|is\s+|was\s+)?(?:now\s+)?"
        r"(?:filled|closed)\b",
    ),
    _pattern(
        "rej.body.keep_resume_on_file",
        EventType.REJECTION,
        "body",
        r"\bkeep\s+your\s+(?:resume|cv|application|profile|details)\s+on\s+file\b",
    ),
    _pattern(
        "rej.body.wish_you_the_best",
        EventType.REJECTION,
        "body",
        r"\bwish\s+you\s+(?:all\s+)?the\s+(?:very\s+)?best\b",
    ),
)

_OFFER: Final[tuple[EventPattern, ...]] = (
    _pattern(
        "off.subject.offer",
        EventType.OFFER,
        "subject",
        r"\b(?:job\s+offer|offer\s+of\s+employment|offer\s+letter|your\s+offer)\b",
        high_precision=True,
    ),
    _pattern(
        "off.body.pleased_to_offer",
        EventType.OFFER,
        "body",
        r"\b(?:pleased|delighted|thrilled|excited|happy)\s+to\s+(?:formally\s+)?"
        r"(?:offer|extend)\b",
    ),
    _pattern(
        "off.body.extend_an_offer",
        EventType.OFFER,
        "body",
        r"\bextend(?:ing)?\s+(?:you\s+)?an\s+offer\b",
    ),
    _pattern(
        "off.body.offer_letter",
        EventType.OFFER,
        "body",
        r"\b(?:offer\s+letter|offer\s+package|compensation\s+package)\b",
    ),
    _pattern(
        "off.body.start_date",
        EventType.OFFER,
        "body",
        r"\b(?:proposed\s+|tentative\s+|your\s+)start\s+date\b",
    ),
)

_INTERVIEW: Final[tuple[EventPattern, ...]] = (
    _pattern(
        "itv.subject.interview_invitation",
        EventType.INTERVIEW,
        "subject",
        r"\b(?:interview\s+invitation|invitation\s+to\s+interview|"
        r"invit(?:e|ation)\s+(?:you\s+)?(?:to|for)\s+(?:an?\s+)?interview)\b",
        high_precision=True,
    ),
    _pattern(
        "itv.subject.schedule_interview",
        EventType.INTERVIEW,
        "subject",
        r"\b(?:schedul(?:e|ing)|book(?:ing)?)\s+(?:your\s+|an?\s+|the\s+)?"
        r"(?:phone\s+|video\s+|onsite\s+|on-site\s+|final\s+|technical\s+)?"
        r"(?:interview|screen)\b",
        high_precision=True,
    ),
    _pattern(
        "itv.subject.phone_screen",
        EventType.INTERVIEW,
        "subject",
        r"\b(?:phone\s+screen|recruiter\s+screen|initial\s+screen)\b",
        high_precision=True,
    ),
    _pattern(
        "itv.subject.onsite",
        EventType.INTERVIEW,
        "subject",
        r"\b(?:on-?site\s+interview|final\s+round|virtual\s+onsite)\b",
        high_precision=True,
    ),
    _pattern(
        "itv.subject.next_steps",
        EventType.INTERVIEW,
        "subject",
        r"\bnext\s+steps\b",
    ),
    _pattern(
        "itv.body.invite_to_interview",
        EventType.INTERVIEW,
        "body",
        r"\b(?:like\s+to\s+)?invite\s+you\s+(?:to|for)\s+(?:an?\s+)?"
        r"(?:interview|phone\s+screen|conversation|chat|call)\b",
    ),
    _pattern(
        "itv.body.select_a_time",
        EventType.INTERVIEW,
        "body",
        r"\b(?:select|choose|pick|book|grab|reserve)\s+(?:a\s+|some\s+)?time\b",
    ),
    _pattern(
        "itv.body.schedule_link",
        EventType.INTERVIEW,
        "body",
        r"""
        (?:click|use|follow|via)\s+
        (?:the\s+)?(?:link|calendar|scheduler)   # a concrete scheduling affordance …
        [^.]{0,40}?
        \bto\s+(?:schedule|book)                 # … whose purpose is booking
        """,
        verbose=True,
    ),
    _pattern(
        "itv.body.availability",
        EventType.INTERVIEW,
        "body",
        r"\b(?:share|send|let\s+(?:us|me)\s+know|provide|reply\s+with)\s+"
        r"(?:your|some|a\s+few)\s+(?:availability|available\s+times?|times?\s+that)\b",
    ),
    _pattern(
        "itv.body.interview_scheduled",
        EventType.INTERVIEW,
        "body",
        r"\b(?:your\s+)?interview\s+(?:is|has\s+been)\s+(?:confirmed|scheduled|booked|set)\b",
    ),
    _pattern(
        "itv.body.speak_with",
        EventType.INTERVIEW,
        "body",
        r"\b(?:speak|chat|meet)\s+with\s+(?:the\s+)?(?:hiring\s+manager|our\s+(?:team|engineers))\b",
    ),
)

_ASSESSMENT: Final[tuple[EventPattern, ...]] = (
    _pattern(
        "asm.subject.online_assessment",
        EventType.ASSESSMENT,
        "subject",
        r"\b(?:online|coding|technical|skills?)\s+(?:assessment|challenge|exercise|test)\b",
        high_precision=True,
    ),
    _pattern(
        "asm.subject.take_home",
        EventType.ASSESSMENT,
        "subject",
        r"\btake[\s\-]?home\b",
        high_precision=True,
    ),
    _pattern(
        "asm.body.complete_the_assessment",
        EventType.ASSESSMENT,
        "body",
        r"\bcomplete\s+(?:the|this|your)\s+(?:online\s+|coding\s+|technical\s+|short\s+)?"
        r"(?:assessment|challenge|exercise|test)\b",
    ),
    _pattern(
        "asm.body.assessment_vendor",
        EventType.ASSESSMENT,
        "body",
        r"\b(?:hackerrank|codesignal|codility|karat|coderpad|woven)\b",
    ),
    _pattern(
        "asm.body.time_limit",
        EventType.ASSESSMENT,
        "body",
        r"\byou\s+(?:will\s+have|have)\s+\d+\s+(?:minutes|hours|days)\s+to\s+complete\b",
    ),
)

_APPLICATION_RECEIVED: Final[tuple[EventPattern, ...]] = (
    _pattern(
        "ack.subject.thanks_for_applying",
        EventType.APPLICATION_RECEIVED,
        "subject",
        r"\bthank(?:s|\s+you)\s+for\s+(?:applying|your\s+application)\b",
        high_precision=True,
    ),
    _pattern(
        "ack.subject.application_received",
        EventType.APPLICATION_RECEIVED,
        "subject",
        r"\b(?:application\s+(?:received|submitted|confirmation)|"
        r"we\s+received\s+your\s+application)\b",
        high_precision=True,
    ),
    _pattern(
        "ack.subject.we_got_your_application",
        EventType.APPLICATION_RECEIVED,
        "subject",
        r"\b(?:we(?:'ve|\s+have)?\s+got|received)\s+your\s+application\b",
        high_precision=True,
    ),
    _pattern(
        "ack.body.thanks_for_applying",
        EventType.APPLICATION_RECEIVED,
        "body",
        r"\bthank(?:s|\s+you)\s+for\s+(?:applying|your\s+application|your\s+interest)\b",
    ),
    _pattern(
        "ack.body.application_received",
        EventType.APPLICATION_RECEIVED,
        "body",
        r"\b(?:we(?:'ve|\s+have)?\s+received\s+your\s+application|"
        r"your\s+application\s+(?:has\s+been|was)\s+received)\b",
    ),
    _pattern(
        "ack.body.reviewing_application",
        EventType.APPLICATION_RECEIVED,
        "body",
        r"\b(?:is|are|team\s+is)\s+(?:currently\s+)?review(?:ing)?\s+(?:your\s+)?"
        r"(?:application|it|resume)\b",
    ),
    _pattern(
        "ack.body.application_submitted",
        EventType.APPLICATION_RECEIVED,
        "body",
        r"\byour\s+application\s+(?:has\s+been|was)\s+(?:successfully\s+)?submitted\b",
    ),
)

_RECRUITER_OUTREACH: Final[tuple[EventPattern, ...]] = (
    _pattern(
        "rec.subject.opportunity",
        EventType.RECRUITER_OUTREACH,
        "subject",
        r"\b(?:exciting|new|great|job|career)\s+opportunit(?:y|ies)\b",
        high_precision=True,
    ),
    _pattern(
        "rec.subject.reaching_out",
        EventType.RECRUITER_OUTREACH,
        "subject",
        r"\breaching\s+out\b",
    ),
    _pattern(
        "rec.body.came_across_your_profile",
        EventType.RECRUITER_OUTREACH,
        "body",
        r"\b(?:came\s+across|stumbled\s+(?:up)?on|found)\s+your\s+"
        r"(?:profile|resume|cv|linkedin|github|background)\b",
    ),
    _pattern(
        "rec.body.i_am_a_recruiter",
        EventType.RECRUITER_OUTREACH,
        "body",
        r"\bi(?:'m|\s+am)\s+(?:a|an|the)\s+(?:technical\s+|senior\s+)?"
        r"(?:recruiter|talent\s+partner|sourcer)\b",
    ),
    _pattern(
        "rec.body.would_you_be_interested",
        EventType.RECRUITER_OUTREACH,
        "body",
        r"\b(?:would\s+you\s+be\s+(?:interested|open)|are\s+you\s+open\s+to)\b",
    ),
    _pattern(
        "rec.body.we_are_hiring",
        EventType.RECRUITER_OUTREACH,
        "body",
        r"\bwe(?:'re|\s+are)\s+(?:currently\s+)?hiring\s+(?:for|a|an)\b",
    ),
)

EVENT_PATTERNS: Final[tuple[EventPattern, ...]] = (
    _WITHDRAWN
    + _REJECTION
    + _OFFER
    + _INTERVIEW
    + _ASSESSMENT
    + _APPLICATION_RECEIVED
    + _RECRUITER_OUTREACH
)
"""Every event-typing pattern, in a fixed order. Table order fixes evidence order, so it is
part of the deterministic output (I2). NOT precedence order — see EVENT_PRECEDENCE."""

RULE_INDEX: Final[dict[str, EventPattern]] = {p.rule_id: p for p in EVENT_PATTERNS}
"""rule_id → pattern, so ``confidence.py`` can ask what a fired rule was without re-matching."""

# --------------------------------------------------------------------------------------
# Stage 3 — field extraction
# --------------------------------------------------------------------------------------

ExtractionScope = Literal["subject", "body"]
"""Which part of the message an extraction pattern reads. Both read the ORIGINAL text —
extraction has to preserve case for the display string, unlike event typing."""


@dataclass(frozen=True)
class ExtractionPattern:
    """One capture-group extractor for company, role, or location.

    Attributes:
        rule_id: Stable identifier recorded in ``Classification.evidence``.
        scope: ``"subject"`` or ``"body"``.
        regex: Compiled pattern with a single named group (``company``/``role``/``location``).
        group: Name of that group.
    """

    rule_id: str
    scope: ExtractionScope
    regex: re.Pattern[str]
    group: str


def _extractor(
    rule_id: str, scope: ExtractionScope, pattern: str, group: str, *, verbose: bool = False
) -> ExtractionPattern:
    """Compile one extraction row at import time."""
    flags = re.IGNORECASE | (re.VERBOSE if verbose else 0)
    return ExtractionPattern(rule_id, scope, re.compile(pattern, flags), group)


_COMPANY_STOP = r"[^,!?.|\n\-\u2013\u2014]"
"""A company name runs until punctuation that reliably ends it. Kept as one constant so
every company extractor agrees on where a name stops."""

COMPANY_PATTERNS: Final[tuple[ExtractionPattern, ...]] = (
    _extractor(
        "co.subject.applying_to",
        "subject",
        rf"thank(?:s|\s+you)\s+for\s+(?:applying|your\s+application)\s+"
        rf"(?:to|at|with)\s+(?P<company>{_COMPANY_STOP}+)",
        "company",
    ),
    _extractor(
        "co.subject.application_at",
        "subject",
        rf"\bapplication\s+(?:to|at|with)\s+(?P<company>{_COMPANY_STOP}+)",
        "company",
    ),
    _extractor(
        "co.subject.interview_with",
        "subject",
        rf"\b(?:interview|conversation|call|chat)\s+with\s+(?P<company>{_COMPANY_STOP}+)",
        "company",
    ),
    _extractor(
        "co.subject.role_at_company",
        "subject",
        rf"\bat\s+(?P<company>{_COMPANY_STOP}+)\s*$",
        "company",
    ),
    _extractor(
        "co.body.applying_to",
        "body",
        rf"thank(?:s|\s+you)\s+for\s+(?:applying|your\s+application)\s+"
        rf"(?:to|at|with)\s+(?P<company>{_COMPANY_STOP}+)",
        "company",
    ),
    _extractor(
        "co.body.application_to",
        "body",
        rf"\byour\s+application\s+(?:to|at|with)\s+(?P<company>{_COMPANY_STOP}+)",
        "company",
    ),
    _extractor(
        "co.body.signature",
        "body",
        r"""
        (?:^|\n)\s*(?:the\s+)?               # signature sits at the start of a line
        (?P<company>[A-Z][A-Za-z0-9&.'\-\ ]{1,60}?)\s+
        (?:recruiting|talent|hiring|people|university)\s+
        (?:team|partner|crew)
        """,
        "company",
        verbose=True,
    ),
    _extractor(
        "co.body.team_signature",
        "body",
        r"""
        (?:^|\n)\s*(?:the\s+)?
        (?P<company>[A-Z][A-Za-z0-9&.'\-\ ]{1,60}?)\s+team\s*(?:$|\n)
        """,
        "company",
        verbose=True,
    ),
    _extractor(
        "co.body.here_at",
        "body",
        # NB: "hiring for" is deliberately absent — it introduces a ROLE ("we are hiring
        # for a Site Reliability Engineer"), not a company. Only "hiring at" names one.
        r"\b(?:here\s+at|team\s+at|role\s+at|position\s+at|opportunity\s+at|opening\s+at|"
        r"recruiter\s+(?:at|with)|hiring\s+at)\s+"
        r"(?P<company>[A-Z][A-Za-z0-9&.'\-\ ]{1,60}?)"
        r"(?=[,.!?]|\s+(?:and|where|who|that|we)\b|$)",
        "company",
    ),
)
"""Ordered company chain. The engine also tries the sender before/after these; see
``rules.extract_company`` for the exact order, which is fixed by CONTRACTS.md §5."""

HIGH_PRECISION_COMPANY_RULES: Final[frozenset[str]] = frozenset(
    {
        "co.ats.sender_name",
        "co.subject.applying_to",
        "co.subject.application_at",
        "co.body.applying_to",
        "co.body.application_to",
        "co.body.signature",
    }
)
"""Company extractors precise enough to earn the ``company_extracted`` confidence weight."""

COMPANY_TRAILER_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    \s+(?:
        for\s+(?:the|our|your|a|an)\b   # "Acme Robotics for the Senior Engineer role"
      | regarding\b | about\b
      | position\b | role\b | opening\b
    ).*$
    """,
    re.IGNORECASE | re.VERBOSE,
)
"""Trim the clause that follows a company name in a subject line. Deliberately requires a
determiner after 'for' so a name like 'Bank for International Settlements' survives."""

COMPANY_TRAILING_PUNCT: Final[str] = " \t.,;:!?-\u2013\u2014'\"()"
"""Characters stripped from both ends of an extracted company or role."""

GENERIC_COMPANY_WORDS: Final[frozenset[str]] = frozenset(
    {
        "a",
        "an",
        "application",
        "job",
        "our",
        "position",
        "role",
        "the",
        "this",
        "us",
        "you",
        "your",
    }
)
"""A capture group that reduces to one of these caught filler, not a company."""

CORPORATE_NAME_WORDS: Final[frozenset[str]] = frozenset(
    {
        "ai",
        "analytics",
        "bank",
        "capital",
        "cloud",
        "data",
        "digital",
        "dynamics",
        "health",
        "industries",
        "labs",
        "media",
        "networks",
        "robotics",
        "security",
        "software",
        "solutions",
        "studio",
        "studios",
        "systems",
        "technologies",
        "technology",
        "ventures",
        "works",
    }
)
"""Words that mark a display name as a company rather than a person, so
'Jane Chen' falls through to the sender domain but 'Acme Robotics' does not."""

ROLE_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "application",
        "applying",
        "interview",
        "it",
        "job",
        "opening",
        "opportunity",
        "position",
        "role",
        "the",
        "this",
        "us",
        "you",
        "your",
    }
)
"""A capture group that reduces to one of these is filler, not a job title."""

ROLE_TRAILING_WORDS: Final[re.Pattern[str]] = re.compile(
    r"\s+(?:role|position|opening|job|req(?:uisition)?(?:\s+\S+)?)\s*$", re.IGNORECASE
)
"""'Senior Software Engineer role' → 'Senior Software Engineer'. Applied until stable."""

_ROLE_TAIL = r"(?:\s+(?:role|position|opening|job|req(?:uisition)?))?"
"""Job titles are routinely followed by the word 'role'/'position'; strip it in the pattern
so the capture group stops there rather than swallowing it."""

_ROLE_BODY_STOP = r"[^\n.;|]"
"""A title inside a sentence runs to the end of the clause. Commas must stay INSIDE the
class: real titles carry them ('Senior Software Engineer, Platform'), and excluding them
truncated every such title to nothing."""

ROLE_PATTERNS: Final[tuple[ExtractionPattern, ...]] = (
    _extractor(
        "role.subject.application_for",
        "subject",
        rf"\b(?:applic(?:ation|ant)|applying)\s+for\s+(?:the\s+)?"
        rf"(?P<role>[^|\n\u2013\u2014]+?){_ROLE_TAIL}\s*$",
        "role",
    ),
    _extractor(
        "role.subject.interview_for",
        "subject",
        rf"\b(?:interview|assessment|screen)\s+for\s+(?:the\s+)?"
        rf"(?P<role>[^|\n\u2013\u2014]+?){_ROLE_TAIL}\s*$",
        "role",
    ),
    _extractor(
        "role.subject.role_at_company",
        "subject",
        r"""
        ^(?!(?:                                  # the "<Title> at <Company>" job-board form.
            your|our|my|the|a|an|re|fwd|update|thanks|thank|offer|interview|invitation|
            congratulations|welcome|following|application|applying|exciting|new|great|
            we|i|it|this|that|here|hi|hello|quick|final|next|position|role
        )\b)
        (?P<role>[A-Za-z][^|\n\u2013\u2014]{2,60}?)\s+(?:at|@)\s+\S
        """,
        "role",
        verbose=True,
    ),
    _extractor(
        "role.subject.delimited",
        "subject",
        r"[|\u2013\u2014]\s*(?P<role>[A-Za-z][^|\n\u2013\u2014]{2,60}?)\s*$",
        "role",
    ),
    _extractor(
        "role.body.application_for_role",
        "body",
        rf"\bapplication\s+for\s+(?:the\s+)?(?P<role>{_ROLE_BODY_STOP}{{2,80}}?)\s+"
        rf"(?:role|position|opening|job)\b",
        "role",
    ),
    _extractor(
        "role.body.interest_in_role",
        "body",
        rf"\binterest\s+in\s+(?:the\s+)?(?P<role>{_ROLE_BODY_STOP}{{2,80}}?)\s+"
        rf"(?:role|position|opening|job)\b",
        "role",
    ),
    _extractor(
        "role.body.position_label",
        "body",
        r"\b(?:position|role|job\s+title)\s*:\s*(?P<role>[^\n|]{2,80})",
        "role",
    ),
)
"""Ordered role chain. Mostly subject capture groups, per PLAN.md §7."""

LOCATION_PATTERNS: Final[tuple[ExtractionPattern, ...]] = (
    _extractor(
        "loc.body.label",
        "body",
        r"\blocation\s*:\s*(?P<location>[^\n|,]{2,60})",
        "location",
    ),
    _extractor(
        "loc.subject.parenthetical",
        "subject",
        r"\((?P<location>remote|hybrid|on-?site|"
        r"[A-Z][A-Za-z .'\-]+,\s*[A-Z]{2}(?:,\s*[A-Z]{2,3})?)\)",
        "location",
    ),
    _extractor(
        "loc.body.based_in",
        "body",
        r"\b(?:based\s+in|located\s+in|office\s+in)\s+"
        r"(?P<location>[A-Z][A-Za-z .'\-]{2,40}(?:,\s*[A-Z]{2})?)",
        "location",
    ),
)
"""Location is best-effort: absent from most job mail, so it never affects confidence."""

# --------------------------------------------------------------------------------------
# Sender-name cleanup
# --------------------------------------------------------------------------------------

GENERIC_SENDER_NAMES: Final[frozenset[str]] = frozenset(
    {
        "careers",
        "do not reply",
        "donotreply",
        "hiring",
        "hiring team",
        "hr",
        "jobs",
        "no reply",
        "no-reply",
        "noreply",
        "notification",
        "notifications",
        "people team",
        "recruiting",
        "recruiting team",
        "recruitment",
        "talent",
        "talent acquisition",
        "team",
    }
)
"""Display names that carry no company. Compared against the casefolded, stripped name."""

SENDER_NAME_SUFFIX_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    \s*(?:
        \((?:via|through)\s[^)]*\)              # "Acme (via Greenhouse)"
      | \bvia\s+\w+                             # "Acme via Lever"
      | \b(?:recruiting|recruitment|talent\s+acquisition|talent|careers?|
            hiring(?:\s+team)?|jobs|people(?:\s+team)?|hr)
        (?:\s+team)?
    )\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)
"""Trailing department noise on a sender display name: 'Acme Robotics Recruiting' → 'Acme
Robotics'. Applied repeatedly until stable, so 'Acme Careers Team' collapses too."""

SENDER_NAME_PERSON_AT_RE: Final[re.Pattern[str]] = re.compile(
    r"^.{1,40}?\s+(?:at|from|@)\s+(?P<company>.{2,60})$", re.IGNORECASE
)
"""'Jane from Acme Robotics' → 'Acme Robotics'."""

__all__ = [
    "ATS_DOMAINS",
    "ATS_DOMAIN_ORDER",
    "ATS_HEADER_SOURCES",
    "COMPANY_PATTERNS",
    "COMPANY_TRAILER_RE",
    "COMPANY_TRAILING_PUNCT",
    "CORPORATE_NAME_WORDS",
    "DOMAIN_LABEL_STOPWORDS",
    "DOMAIN_RE",
    "EVENT_PATTERNS",
    "FREE_MAIL_DOMAINS",
    "GENERIC_COMPANY_WORDS",
    "GENERIC_SENDER_NAMES",
    "HIGH_PRECISION_COMPANY_RULES",
    "HOST_RE",
    "LOCATION_PATTERNS",
    "ROLE_PATTERNS",
    "ROLE_STOPWORDS",
    "ROLE_TRAILING_WORDS",
    "RULE_INDEX",
    "SENDER_NAME_PERSON_AT_RE",
    "SENDER_NAME_SUFFIX_RE",
    "EventPattern",
    "ExtractionPattern",
    "normalize_text",
]
