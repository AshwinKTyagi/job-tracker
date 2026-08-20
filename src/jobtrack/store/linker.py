"""Pure message-to-application matching and status derivation.

Nothing in this module touches SQLite. ``match_application`` receives its candidates
already fetched (see ``Store.match_candidates``), which is what lets M3 be built and
tested in parallel with M2 (PLAN.md §5) and keeps the matching rules unit-testable with
no database at all.

Role comparison note. ``CONTRACTS.md`` §5 assigns ``role_similarity`` to M2
(``classify/normalize.py``). M3 must not import a module that does not exist yet, so the
local ``_role_similarity`` below is used instead: it is deterministic, dependency-free,
and behaves like the contracted function (0.0-1.0, 1.0 for equal titles). When M2 lands,
swapping it out is a one-line change at its single call site.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Final

from jobtrack.constants import EVENT_PRECEDENCE, TERMINAL_EVENTS
from jobtrack.models import (
    ApplicationMatchCandidate,
    ApplicationStatus,
    Classification,
    EventRow,
    EventType,
)

logger = logging.getLogger(__name__)

LINK_WINDOW_DAYS: int = 180
"""How far back a company_key match may reach before it is treated as a new application."""

_TERMINAL_STATUS: Final[dict[EventType, ApplicationStatus]] = {
    EventType.REJECTION: ApplicationStatus.REJECTED,
    EventType.OFFER: ApplicationStatus.OFFER,
    EventType.WITHDRAWN: ApplicationStatus.WITHDRAWN,
}
"""Terminal event -> the status it pins the application to. Keys mirror TERMINAL_EVENTS."""

_STAGE_STATUS: Final[dict[EventType, ApplicationStatus]] = {
    EventType.INTERVIEW: ApplicationStatus.INTERVIEWING,
    EventType.ASSESSMENT: ApplicationStatus.ASSESSMENT,
    EventType.APPLICATION_RECEIVED: ApplicationStatus.APPLIED,
    EventType.RECRUITER_OUTREACH: ApplicationStatus.APPLIED,
    EventType.UNKNOWN: ApplicationStatus.APPLIED,
}
"""Non-terminal event -> the stage it represents. The furthest stage reached wins."""

_ROLE_NOISE_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    \b(
        senior | sr | junior | jr | staff | principal | lead | mid | entry | level
      | i{1,3} | iv | v          # roman numerals used as levels
      | [0-9]{1,2}               # numeric levels and stray req digits
      | intern(ship)?
      | full[- ]?time | part[- ]?time | contract | remote | hybrid | onsite
    )\b
    """,
    re.VERBOSE,
)
"""Seniority / employment-mode tokens that must not drive a role match either way."""

_ROLE_PUNCT_RE: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")
"""Everything that is not an alphanumeric collapses to a single space."""

_ROLE_ABBREVIATIONS: Final[dict[str, str]] = {
    "swe": "software engineer",
    "sde": "software engineer",
    "mle": "machine learning engineer",
    "ml": "machine learning",
    "eng": "engineer",
    "engr": "engineer",
    "dev": "developer",
    "mgr": "manager",
    "pm": "product manager",
    "sr": "senior",
}
"""Expanded before noise stripping so "Sr. SWE" and "Software Engineer" converge."""


def _normalize_role_local(title: str | None) -> str:
    """Casefold a job title down to a bag of comparable words."""
    if title is None:
        return ""
    lowered = _ROLE_PUNCT_RE.sub(" ", title.casefold())
    expanded = " ".join(_ROLE_ABBREVIATIONS.get(word, word) for word in lowered.split())
    stripped = _ROLE_NOISE_RE.sub(" ", expanded)
    return " ".join(stripped.split())


def _role_similarity(a: str | None, b: str | None) -> float:
    """Deterministic token-overlap similarity in [0, 1] between two job titles."""
    left = set(_normalize_role_local(a).split())
    right = set(_normalize_role_local(b).split())
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    if overlap == 0:
        return 0.0
    # Dice coefficient: symmetric, and forgiving of one title carrying extra words.
    return (2.0 * overlap) / (len(left) + len(right))


def _sort_key(candidate: ApplicationMatchCandidate) -> tuple[datetime, str]:
    """Most-recent-first ordering with a stable tiebreak on application_id."""
    return (candidate.last_event_at, candidate.application_id)


def _within_window(
    candidate: ApplicationMatchCandidate, *, now: datetime, window_days: int
) -> bool:
    """True when the candidate's last event is no older than the linking window."""
    return now - candidate.last_event_at <= timedelta(days=window_days)


def match_application(
    classification: Classification,
    candidates: Sequence[ApplicationMatchCandidate],
    message_thread_id: str,
    *,
    now: datetime,
    window_days: int = LINK_WINDOW_DAYS,
    role_threshold: float = 0.75,
) -> str | None:
    """Decide which existing application a message belongs to.

    PURE — candidates are pre-fetched by the caller, so this is unit-testable with no DB.

    Ordered rules:
        1. thread_id already belongs to an application -> that application.
        2. same company_key, role similarity >= role_threshold, within window_days -> that
           one (most recent last_event_at wins on ties).
        3. same company_key and either role is None, within window_days -> that one.
        4. otherwise -> None (the caller creates a new application).

    Args:
        classification: The classifier's output for the message being linked.
        candidates: Applications that share the thread or the company_key.
        message_thread_id: Thread the message arrived on.
        now: Injected tz-aware UTC clock, used only for the window check.
        window_days: How far back a company_key match may reach.
        role_threshold: Minimum role similarity for rule 2.

    Returns:
        The matching application_id, or None to create a new application.
    """
    thread_hits = [c for c in candidates if message_thread_id in c.thread_ids]
    if thread_hits:
        return max(thread_hits, key=_sort_key).application_id

    company_key = classification.company_key
    if company_key is None:
        return None

    in_window = [
        c
        for c in candidates
        if c.company_key == company_key and _within_window(c, now=now, window_days=window_days)
    ]
    if not in_window:
        return None

    scored = [
        (_role_similarity(classification.role, c.role), c)
        for c in in_window
        if c.role is not None and classification.role is not None
    ]
    qualifying = [(score, c) for score, c in scored if score >= role_threshold]
    if qualifying:
        best_score = max(score for score, _ in qualifying)
        tied = [c for score, c in qualifying if score == best_score]
        return max(tied, key=_sort_key).application_id

    loose = [c for c in in_window if c.role is None or classification.role is None]
    if loose:
        return max(loose, key=_sort_key).application_id

    return None


def derive_status(
    events: Sequence[EventRow], *, now: datetime, ghost_after_days: int
) -> ApplicationStatus:
    """Compute an application's status from its event history (I4).

    Terminal events (REJECTION/OFFER/WITHDRAWN) win regardless of recency — the most
    recent terminal event decides. Otherwise the furthest stage reached decides,
    downgraded to GHOSTED when the last event is older than ghost_after_days.

    Args:
        events: The application's events, overrides already applied. Any order.
        now: Injected tz-aware UTC clock.
        ghost_after_days: Silence, in days, after which a live application is GHOSTED.

    Returns:
        The derived status. An application with no events reads as APPLIED.
    """
    if not events:
        return ApplicationStatus.APPLIED

    terminal = [e for e in events if e.event_type in TERMINAL_EVENTS]
    if terminal:
        latest = max(terminal, key=lambda e: (e.occurred_at, e.event_id))
        return _TERMINAL_STATUS[latest.event_type]

    furthest = min(events, key=lambda e: EVENT_PRECEDENCE.index(e.event_type))
    last_event_at = max(e.occurred_at for e in events)
    if now - last_event_at > timedelta(days=ghost_after_days):
        return ApplicationStatus.GHOSTED
    return _STAGE_STATUS[furthest.event_type]


__all__ = ["LINK_WINDOW_DAYS", "derive_status", "match_application"]
