"""The deterministic rules engine.

Pipeline, in order: ``detect_ats`` → ``score_event_types`` → ``resolve_event_type`` →
``extract_company`` / ``extract_role`` / ``extract_location`` → ``score_confidence``.

The single most important property here is **I3: precedence, not first-match**. Every event
type is scored against the whole message before anything is decided, and the winner is picked
by ``EVENT_PRECEDENCE``. Confirmation and rejection emails are lexically near-identical —
both open "Thank you for applying to ___" — and only a later clause separates them. A
first-match-wins loop would label every rejection a confirmation.

The engine is pure (I2): no network, no database, no clock, no randomness, and no reliance on
dict iteration order. Everything it knows lives in ``patterns.py``; every number it uses lives
in ``confidence.py``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Final

from jobtrack.classify import confidence as conf
from jobtrack.classify import patterns as pat
from jobtrack.classify.normalize import normalize_company
from jobtrack.constants import EVENT_PRECEDENCE
from jobtrack.models import Classification, EventType, RawMessage

logger = logging.getLogger(__name__)

_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")
"""Collapse the newlines a capture group drags in from a wrapped body line."""

_MIN_DOMAIN_LABELS: Final[int] = 2
"""A host needs at least a name and a TLD before a company can be read out of it."""

MIN_EXTRACTED_LENGTH: Final[int] = 2
MAX_EXTRACTED_LENGTH: Final[int] = 80
"""Sanity bounds on a capture group. A one-character "company" is punctuation noise; an
eighty-plus-character one is a runaway match that swallowed half a sentence."""

PERSON_NAME_MIN_TOKENS: Final[int] = 2
PERSON_NAME_MAX_TOKENS: Final[int] = 3
"""A display name of this many purely alphabetic tokens, with no corporate word and no
overlap with the sender domain, reads as a human rather than a company."""

ATS_SENDER_COMPANY_RULE: Final[str] = "co.ats.sender_name"
SENDER_DISPLAY_COMPANY_RULE: Final[str] = "co.sender.display_name"
SENDER_DOMAIN_COMPANY_RULE: Final[str] = "co.sender.domain"
"""Rule ids for the sender-based links in the company chain. They are not regex table rows —
the sender is structured data, not prose — but they are recorded in evidence just the same."""


# --------------------------------------------------------------------------------------
# Stage 1 — ATS detection
# --------------------------------------------------------------------------------------


def _hosts(value: str) -> list[str]:
    """Every host mentioned in a header value, lower-cased, in order of appearance."""
    return [host.casefold().rstrip(".") for host in pat.HOST_RE.findall(value)]


def _match_ats_host(host: str) -> str | None:
    """Map one host to an ATS slug, longest known domain first."""
    for domain, slug in pat.ATS_DOMAIN_ORDER:
        if host == domain or host.endswith(f".{domain}"):
            return slug
    return None


def detect_ats(message: RawMessage) -> tuple[str | None, list[str]]:
    """Identify the applicant-tracking system from sender, Reply-To, and List-Unsubscribe.

    The From address decides; the headers are consulted only when it is inconclusive, because
    ATS mail is routinely relayed through a customer's own domain while the unsubscribe link
    still points at the vendor. Once a slug wins, every source that agrees with it contributes
    its own rule id, so the evidence shows how much corroboration there was.

    Args:
        message: The normalized email.

    Returns:
        (ats_slug or None, rule_ids that fired) e.g. ("greenhouse", ["ats.sender.greenhouse"]).
    """
    found: list[tuple[str, str]] = []

    for host in _hosts(f"@{message.from_email}"):
        slug = _match_ats_host(host)
        if slug is not None:
            found.append((slug, f"ats.sender.{slug}"))

    for header, token in pat.ATS_HEADER_SOURCES:
        value = message.headers.get(header)
        if not value:
            continue
        for host in _hosts(value):
            slug = _match_ats_host(host)
            if slug is not None:
                found.append((slug, f"ats.{token}.{slug}"))

    if not found:
        return None, []

    winner = found[0][0]
    rule_ids: list[str] = []
    for slug, rule_id in found:
        if slug == winner and rule_id not in rule_ids:
            rule_ids.append(rule_id)
    return winner, rule_ids


# --------------------------------------------------------------------------------------
# Stage 2 — event typing
# --------------------------------------------------------------------------------------


def score_event_types(message: RawMessage) -> dict[EventType, list[str]]:
    """Score the message against EVERY event type — never stop at the first hit (I3).

    Args:
        message: The normalized email.

    Returns:
        Mapping of each matched EventType to the rule ids that fired for it, keyed in
        EVENT_PRECEDENCE order and with rule ids in pattern-table order so the output is
        byte-identical across runs. Empty dict means nothing matched.
    """
    subject = pat.normalize_text(message.subject)
    body = pat.normalize_text(message.body_text)

    matched: dict[EventType, list[str]] = {}
    for pattern in pat.EVENT_PATTERNS:
        haystack = subject if pattern.scope == "subject" else body
        if pattern.regex.search(haystack):
            matched.setdefault(pattern.event_type, []).append(pattern.rule_id)

    return {event_type: matched[event_type] for event_type in EVENT_PRECEDENCE if event_type in matched}


def resolve_event_type(scores: dict[EventType, list[str]]) -> tuple[EventType, list[str]]:
    """Pick the winner by EVENT_PRECEDENCE.

    Args:
        scores: The per-type score map from ``score_event_types``.

    Returns:
        (winning type, its rule ids). (UNKNOWN, []) when scores is empty.
    """
    for event_type in EVENT_PRECEDENCE:
        rule_ids = scores.get(event_type)
        if rule_ids:
            return event_type, list(rule_ids)
    return EventType.UNKNOWN, []


# --------------------------------------------------------------------------------------
# Stage 3 — field extraction
# --------------------------------------------------------------------------------------


def _clean_capture(raw: str) -> str | None:
    """Trim a capture group to a plausible name, or None if nothing usable is left."""
    value = _WHITESPACE_RE.sub(" ", raw).strip(pat.COMPANY_TRAILING_PUNCT).strip()
    if not (MIN_EXTRACTED_LENGTH <= len(value) <= MAX_EXTRACTED_LENGTH):
        return None
    return value


def _clean_company(raw: str) -> str | None:
    """Trim a captured company: drop the trailing clause, punctuation, and filler words."""
    value = _clean_capture(pat.COMPANY_TRAILER_RE.sub("", raw))
    if value is None:
        return None
    if value.casefold() in pat.GENERIC_COMPANY_WORDS:
        return None
    return value


def _clean_role(raw: str) -> str | None:
    """Trim a captured role: drop trailing 'role'/'position'/req ids, punctuation, filler."""
    value = raw
    while True:
        trimmed = pat.ROLE_TRAILING_WORDS.sub("", value)
        if trimmed == value:
            break
        value = trimmed
    cleaned = _clean_capture(value)
    if cleaned is None:
        return None
    if cleaned.casefold() in pat.ROLE_STOPWORDS:
        return None
    return cleaned


def _clean_sender_name(from_name: str | None) -> str | None:
    """Reduce a sender display name to a company, or None when it carries none."""
    if from_name is None:
        return None
    value = _WHITESPACE_RE.sub(" ", from_name).strip(pat.COMPANY_TRAILING_PUNCT).strip()
    if not value:
        return None

    person_at = pat.SENDER_NAME_PERSON_AT_RE.match(value)
    if person_at is not None:
        value = person_at.group("company").strip()

    while True:
        trimmed = pat.SENDER_NAME_SUFFIX_RE.sub("", value).strip()
        if trimmed == value:
            break
        value = trimmed

    folded = value.casefold()
    if not value or folded in pat.GENERIC_SENDER_NAMES or folded in pat.ATS_DOMAINS:
        return None
    return _clean_capture(value)


def _looks_like_person(name: str, from_email: str) -> bool:
    """True when a display name reads as a human rather than a company."""
    tokens = name.split()
    if not PERSON_NAME_MIN_TOKENS <= len(tokens) <= PERSON_NAME_MAX_TOKENS:
        return False
    if not all(token.isalpha() for token in tokens):
        return False
    if any(token.casefold() in pat.CORPORATE_NAME_WORDS for token in tokens):
        return False
    label = _sender_domain_label(from_email)
    if label is not None:
        key = (normalize_company(name) or "").replace(" ", "")
        if label in key or key in label:
            return False
    return True


def _sender_domain_label(from_email: str) -> str | None:
    """The company-bearing label of a sender domain, or None for free mail and ATS hosts."""
    _, _, host = from_email.partition("@")
    host = host.casefold().strip().rstrip(".")
    if not host or host in pat.FREE_MAIL_DOMAINS or _match_ats_host(host) is not None:
        return None
    labels = host.split(".")
    if len(labels) < _MIN_DOMAIN_LABELS:
        return None
    for label in labels[:-1]:
        if label not in pat.DOMAIN_LABEL_STOPWORDS and len(label) >= MIN_EXTRACTED_LENGTH:
            return label
    return None


def extract_company(message: RawMessage, ats: str | None) -> tuple[str | None, list[str]]:
    """Extract the company, in the order fixed by CONTRACTS.md §5.

    ATS-specific sender → subject capture group → body signature → sender display name, with
    the sender domain as a last resort. The ATS link comes first because an ATS relay puts the
    customer's name in the display name and nothing else in the envelope; the domain link
    comes last because it recovers a name from ``jane@acme.com`` when nothing else does.

    Args:
        message: The normalized email.
        ats: The slug from ``detect_ats``, or None.

    Returns:
        (display name verbatim from the email, rule ids that fired). (None, []) when no link
        in the chain produced a usable name.
    """
    if ats is not None:
        ats_name = _clean_sender_name(message.from_name)
        if ats_name is not None:
            return ats_name, [ATS_SENDER_COMPANY_RULE]

    for pattern in pat.COMPANY_PATTERNS:
        haystack = message.subject if pattern.scope == "subject" else message.body_text
        match = pattern.regex.search(haystack)
        if match is None:
            continue
        company = _clean_company(match.group(pattern.group))
        if company is not None:
            return company, [pattern.rule_id]

    display_name = _clean_sender_name(message.from_name)
    if display_name is not None and not _looks_like_person(display_name, message.from_email):
        return display_name, [SENDER_DISPLAY_COMPANY_RULE]

    label = _sender_domain_label(message.from_email)
    if label is not None:
        return label.replace("-", " ").title(), [SENDER_DOMAIN_COMPANY_RULE]

    if display_name is not None:
        return display_name, [SENDER_DISPLAY_COMPANY_RULE]
    return None, []


def extract_role(message: RawMessage, ats: str | None) -> tuple[str | None, list[str]]:
    """Extract the job title, mostly from subject capture groups.

    When the message came through an ATS the body chain is tried first: ATS subject lines are
    templated boilerplate naming the company ("Thanks for applying to Acme"), while the body
    is the part that names the requisition.

    Args:
        message: The normalized email.
        ats: The slug from ``detect_ats``, or None.

    Returns:
        (role verbatim from the email, rule ids that fired), or (None, []).
    """
    ordered = pat.ROLE_PATTERNS
    if ats is not None:
        body_first = [p for p in pat.ROLE_PATTERNS if p.scope == "body"]
        body_first.extend(p for p in pat.ROLE_PATTERNS if p.scope == "subject")
        ordered = tuple(body_first)

    for pattern in ordered:
        haystack = message.subject if pattern.scope == "subject" else message.body_text
        match = pattern.regex.search(haystack)
        if match is None:
            continue
        role = _clean_role(match.group(pattern.group))
        if role is not None:
            return role, [pattern.rule_id]
    return None, []


def extract_location(message: RawMessage) -> str | None:
    """Extract a work location when the email states one outright.

    Best-effort by design: most job mail omits location entirely, so this never contributes to
    confidence and a None result is not a defect.

    Args:
        message: The normalized email.

    Returns:
        The location string, or None.
    """
    for pattern in pat.LOCATION_PATTERNS:
        haystack = message.subject if pattern.scope == "subject" else message.body_text
        match = pattern.regex.search(haystack)
        if match is None:
            continue
        location = _clean_capture(match.group(pattern.group))
        if location is not None:
            return location
    return None


# --------------------------------------------------------------------------------------
# The classifier
# --------------------------------------------------------------------------------------


class RulesClassifier:
    """Deterministic pattern-based classifier. No I/O, no clock, no randomness.

    Pipeline: detect_ats → score every EventType → resolve by EVENT_PRECEDENCE →
    extract company/role/location → score_confidence.
    """

    name = "rules"
    version = "1.0.0"  # bump on any pattern change

    def __init__(self, *, min_confidence: float = conf.DEFAULT_MIN_CONFIDENCE) -> None:
        """Construct the classifier.

        Args:
            min_confidence: Threshold below which a result is flagged ``needs_review``.
                Defaults to the same value as ``ClassifyConfig.min_confidence``; ``cli.py``
                passes the configured one.
        """
        self._min_confidence = min_confidence

    def classify(self, message: RawMessage) -> Classification:
        """Classify one message. Pure: same input always yields byte-identical output (I2).

        Field extraction is skipped for UNKNOWN. A message the rules do not recognize is not
        job mail, and running the low-precision tail of the company chain over it would invent
        a company out of a newsletter's sender name.

        Args:
            message: The normalized email to classify.

        Returns:
            The Classification. Never raises for ordinary input — an unrecognized message is
            UNKNOWN with confidence 0.0.
        """
        ats, ats_rules = detect_ats(message)
        scores = score_event_types(message)
        event_type, event_rules = resolve_event_type(scores)

        company: str | None = None
        role: str | None = None
        location: str | None = None
        field_rules: list[str] = []

        if event_type is not EventType.UNKNOWN:
            company, company_rules = extract_company(message, ats)
            role, role_rules = extract_role(message, ats)
            location = extract_location(message)
            field_rules = company_rules + role_rules

        evidence = ats_rules + event_rules + field_rules
        score = conf.score_confidence(
            ats=ats,
            winning_type=event_type,
            evidence=evidence,
            company=company,
            all_scores=scores,
        )

        return Classification(
            message_id=message.message_id,
            event_type=event_type,
            company=company,
            company_key=normalize_company(company),
            role=role,
            location=location,
            ats=ats,
            confidence=score,
            needs_review=conf.needs_review(score, company, threshold=self._min_confidence),
            evidence=evidence,
            classifier_name=self.name,
            classifier_version=self.version,
        )

    def classify_batch(self, messages: Sequence[RawMessage]) -> list[Classification]:
        """Classify a sequence, preserving input order.

        Args:
            messages: Messages to classify.

        Returns:
            One Classification per input message, in the same order.
        """
        return [self.classify(message) for message in messages]


__all__ = [
    "RulesClassifier",
    "detect_ats",
    "extract_company",
    "extract_location",
    "extract_role",
    "resolve_event_type",
    "score_event_types",
]
