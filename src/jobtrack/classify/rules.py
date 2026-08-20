"""The deterministic, pattern-based classifier.

Pipeline (PLAN.md §7): detect_ats -> score every EventType -> resolve by EVENT_PRECEDENCE ->
extract company/role/location -> score_confidence. Pure: no I/O, no clock, no randomness (I2).
"""

from __future__ import annotations

from collections.abc import Sequence

from jobtrack.classify.confidence import DEFAULT_MIN_CONFIDENCE, needs_review, score_confidence
from jobtrack.classify.normalize import normalize_company
from jobtrack.classify.patterns import (
    ATS_BRAND_NAMES,
    ATS_DOMAINS,
    BODY_PATTERNS,
    COMPANY_BODY_PATTERNS,
    COMPANY_SUBJECT_PATTERNS,
    LOCATION_ARRANGEMENT,
    LOCATION_BASED_IN,
    LOCATION_LABEL,
    ROLE_BODY_PATTERNS,
    ROLE_SUBJECT_PATTERNS,
    SUBJECT_PATTERNS,
)
from jobtrack.constants import EVENT_PRECEDENCE
from jobtrack.models import Classification, EventType, RawMessage


def _contains_any_domain(text: str, domains: tuple[str, ...]) -> bool:
    """True if any of `domains` appears as a substring of the casefolded `text`."""
    folded = text.casefold()
    return any(domain in folded for domain in domains)


def detect_ats(message: RawMessage) -> tuple[str | None, list[str]]:
    """Identify the applicant-tracking system from sender, Reply-To, and List-Unsubscribe.

    Args:
        message: The email to inspect.

    Returns:
        (ats_slug or None, rule_ids that fired) e.g. ("greenhouse", ["ats.sender.greenhouse"]).
    """
    reply_to = message.headers.get("reply-to", "")
    list_unsubscribe = message.headers.get("list-unsubscribe", "")
    for slug, domains in ATS_DOMAINS.items():
        rule_ids: list[str] = []
        if _contains_any_domain(message.from_email, domains):
            rule_ids.append(f"ats.sender.{slug}")
        if _contains_any_domain(reply_to, domains):
            rule_ids.append(f"ats.reply_to.{slug}")
        if _contains_any_domain(list_unsubscribe, domains):
            rule_ids.append(f"ats.list_unsubscribe.{slug}")
        if rule_ids:
            return slug, rule_ids
    return None, []


def score_event_types(message: RawMessage) -> dict[EventType, list[str]]:
    """Score the message against EVERY event type — never stop at the first hit (I3).

    Args:
        message: The email to score.

    Returns:
        Mapping of each matched EventType to the rule ids that fired for it.
        Empty dict means nothing matched.
    """
    scores: dict[EventType, list[str]] = {}
    for event_type in EVENT_PRECEDENCE:
        if event_type is EventType.UNKNOWN:
            continue
        rule_ids: list[str] = []
        for rule in SUBJECT_PATTERNS.get(event_type, ()):
            if rule.regex.search(message.subject):
                rule_ids.append(rule.rule_id)
        for rule in BODY_PATTERNS.get(event_type, ()):
            if rule.regex.search(message.body_text):
                rule_ids.append(rule.rule_id)
        if rule_ids:
            scores[event_type] = rule_ids
    return scores


def resolve_event_type(scores: dict[EventType, list[str]]) -> tuple[EventType, list[str]]:
    """Pick the winner by EVENT_PRECEDENCE.

    Args:
        scores: Every matched event type's rule ids, as returned by score_event_types.

    Returns:
        (winning type, its rule ids). (UNKNOWN, []) when scores is empty.
    """
    for event_type in EVENT_PRECEDENCE:
        rule_ids = scores.get(event_type)
        if rule_ids:
            return event_type, rule_ids
    return EventType.UNKNOWN, []


def _is_ats_brand_name(name: str) -> bool:
    """True if `name` names the ATS platform itself rather than an employer."""
    return name.strip().casefold() in ATS_BRAND_NAMES


def _clean_capture(raw: str) -> str | None:
    """Trim and collapse whitespace in a regex capture group; None if empty afterward."""
    cleaned = " ".join(raw.split()).strip(" ,.-")
    return cleaned or None


def extract_company(message: RawMessage, ats: str | None) -> tuple[str | None, list[str]]:
    """Ordered extraction chain for the display-form company name.

    Chain: ATS-specific sender display name -> subject capture group -> body signature ->
    sender display name.

    Args:
        message: The email to extract from.
        ats: The ATS slug detected by detect_ats, or None.

    Returns:
        (display name, rule ids).
    """
    if ats is not None and message.from_name and not _is_ats_brand_name(message.from_name):
        return message.from_name.strip(), [f"company.ats_sender.{ats}"]

    for rule in COMPANY_SUBJECT_PATTERNS:
        match = rule.regex.search(message.subject)
        if match:
            company = _clean_capture(match.group("company"))
            if company:
                return company, [rule.rule_id]

    for rule in COMPANY_BODY_PATTERNS:
        match = rule.regex.search(message.body_text)
        if match:
            company = _clean_capture(match.group("company"))
            if company:
                return company, [rule.rule_id]

    if message.from_name and not _is_ats_brand_name(message.from_name):
        return message.from_name.strip(), ["company.sender_display_name"]

    return None, []


def extract_role(message: RawMessage, ats: str | None) -> tuple[str | None, list[str]]:
    """Ordered extraction chain for the job title: subject capture group, then body.

    Args:
        message: The email to extract from.
        ats: The ATS slug detected by detect_ats. Unused by the current chain, kept for
            interface parity with extract_company and future ATS-specific role extractors.

    Returns:
        (role title, rule ids).
    """
    del ats  # not yet used by any role extractor; kept for signature parity
    for rule in ROLE_SUBJECT_PATTERNS:
        match = rule.regex.search(message.subject)
        if match:
            role = _clean_capture(match.group("role"))
            if role:
                return role, [rule.rule_id]

    for rule in ROLE_BODY_PATTERNS:
        match = rule.regex.search(message.body_text)
        if match:
            role = _clean_capture(match.group("role"))
            if role:
                return role, [rule.rule_id]

    return None, []


def extract_location(message: RawMessage) -> str | None:
    """Best-effort location extraction from a labeled line, prose, or a parenthetical.

    Args:
        message: The email to extract from.

    Returns:
        The extracted location string, or None.
    """
    for pattern in (LOCATION_LABEL, LOCATION_BASED_IN):
        match = pattern.search(message.body_text)
        if match:
            location = _clean_capture(match.group("location"))
            if location:
                return location
    match = LOCATION_ARRANGEMENT.search(f"{message.subject} {message.body_text}")
    if match:
        return match.group("location").strip().title()
    return None


class RulesClassifier:
    """Deterministic pattern-based classifier. No I/O, no clock, no randomness.

    Pipeline: detect_ats -> score every EventType -> resolve by EVENT_PRECEDENCE ->
    extract company/role/location -> score_confidence.
    """

    name = "rules"
    version = "1.0.0"  # bump on any pattern change

    def classify(self, message: RawMessage) -> Classification:
        """Classify one message.

        Args:
            message: The normalized email to classify.

        Returns:
            The resulting Classification. Never raises for ordinary input — an unparseable
            message classifies as EventType.UNKNOWN with confidence 0.0.
        """
        ats, ats_evidence = detect_ats(message)
        scores = score_event_types(message)
        winning_type, type_evidence = resolve_event_type(scores)

        if winning_type is EventType.UNKNOWN:
            company: str | None = None
            company_evidence: list[str] = []
            role: str | None = None
            role_evidence: list[str] = []
            location = None
        else:
            company, company_evidence = extract_company(message, ats)
            role, role_evidence = extract_role(message, ats)
            location = extract_location(message)

        confidence = score_confidence(
            ats=ats,
            winning_type=winning_type,
            evidence=type_evidence,
            company=company,
            all_scores=scores,
        )
        review = needs_review(confidence, company, threshold=DEFAULT_MIN_CONFIDENCE)

        evidence = [*ats_evidence, *type_evidence, *company_evidence, *role_evidence]

        return Classification(
            message_id=message.message_id,
            event_type=winning_type,
            company=company,
            company_key=normalize_company(company),
            role=role,
            location=location,
            ats=ats,
            confidence=confidence,
            needs_review=review,
            evidence=evidence,
            classifier_name=self.name,
            classifier_version=self.version,
        )

    def classify_batch(self, messages: Sequence[RawMessage]) -> list[Classification]:
        """Classify many messages, preserving order.

        Args:
            messages: Messages to classify, in order.

        Returns:
            One Classification per input message, same order.
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
