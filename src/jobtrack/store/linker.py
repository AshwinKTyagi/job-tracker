"""Pure message-to-application linking and status derivation.

No DB access here (I5 test note in CONTRACTS.md, and PLAN.md's brief for M3): candidates are
pre-fetched by ``Store.match_candidates`` and handed in, so both functions in this module are
unit-testable with plain constructed values and no SQLite involved.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Final

from jobtrack.classify.normalize import role_similarity
from jobtrack.constants import TERMINAL_EVENTS
from jobtrack.models import (
    ApplicationMatchCandidate,
    ApplicationStatus,
    Classification,
    EventRow,
    EventType,
)

LINK_WINDOW_DAYS: Final[int] = 180

_STAGE_ORDER: Final[tuple[EventType, ...]] = (
    EventType.RECRUITER_OUTREACH,
    EventType.APPLICATION_RECEIVED,
    EventType.ASSESSMENT,
    EventType.INTERVIEW,
    EventType.OFFER,
)
"""Non-terminal pipeline stages, least to most advanced. OFFER also appears in
TERMINAL_EVENTS; it is included here too so an application whose only OFFER event has since
been superseded by a later non-terminal event (a re-opened process) still ranks sensibly."""

_STAGE_TO_STATUS: Final[dict[EventType, ApplicationStatus]] = {
    EventType.RECRUITER_OUTREACH: ApplicationStatus.APPLIED,
    EventType.APPLICATION_RECEIVED: ApplicationStatus.APPLIED,
    EventType.ASSESSMENT: ApplicationStatus.ASSESSMENT,
    EventType.INTERVIEW: ApplicationStatus.INTERVIEWING,
    EventType.OFFER: ApplicationStatus.OFFER,
}

_TERMINAL_TO_STATUS: Final[dict[EventType, ApplicationStatus]] = {
    EventType.REJECTION: ApplicationStatus.REJECTED,
    EventType.OFFER: ApplicationStatus.OFFER,
    EventType.WITHDRAWN: ApplicationStatus.WITHDRAWN,
}


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

    PURE — candidates are pre-fetched so this is unit-testable with no DB.

    Ordered rules:
        1. ``message_thread_id`` already belongs to an application -> that application.
        2. Same ``company_key`` AND ``role_similarity >= role_threshold``, within
           ``window_days`` -> that one (most recent ``last_event_at`` wins ties).
        3. Same ``company_key`` and either role is ``None``, within ``window_days`` -> that
           one.
        4. Otherwise -> ``None`` (caller creates a new application).

    Args:
        classification: The incoming message's classification.
        candidates: Pre-fetched applications that might match, from ``Store.match_candidates``.
        message_thread_id: The Gmail thread id of the incoming message.
        now: Current instant, for the recency window.
        window_days: How far back a company-key match may reach.
        role_threshold: Minimum ``role_similarity`` to count as the same role.

    Returns:
        The matching ``application_id``, or ``None`` to create a new one.
    """
    for candidate in candidates:
        if message_thread_id in candidate.thread_ids:
            return candidate.application_id

    window_start = now - timedelta(days=window_days)
    same_company = [c for c in candidates if c.company_key == classification.company_key]
    in_window = [c for c in same_company if c.last_event_at >= window_start]

    both_roles_known = [
        c for c in in_window if classification.role is not None and c.role is not None
    ]
    role_matches = [
        c
        for c in both_roles_known
        if role_similarity(classification.role, c.role) >= role_threshold
    ]
    if role_matches:
        role_matches.sort(key=lambda c: c.last_event_at, reverse=True)
        return role_matches[0].application_id

    either_role_none = [c for c in in_window if classification.role is None or c.role is None]
    if either_role_none:
        either_role_none.sort(key=lambda c: c.last_event_at, reverse=True)
        return either_role_none[0].application_id

    return None


def derive_status(
    events: Sequence[EventRow], *, now: datetime, ghost_after_days: int
) -> ApplicationStatus:
    """Compute status from event history (I4).

    Terminal events (REJECTION/OFFER/WITHDRAWN) win regardless of recency — the most recent
    terminal event decides. Otherwise the furthest stage reached decides, downgraded to
    GHOSTED when the last event is older than ``ghost_after_days``.

    Args:
        events: This application's events, in any order (overrides already applied).
        now: Current instant, for the ghost check.
        ghost_after_days: Days of silence after which a non-terminal application is GHOSTED.

    Returns:
        The derived status. ``APPLIED`` for an application with no events at all.
    """
    if not events:
        return ApplicationStatus.APPLIED

    terminal_events = [e for e in events if e.event_type in TERMINAL_EVENTS]
    if terminal_events:
        most_recent_terminal = max(terminal_events, key=lambda e: e.occurred_at)
        return _TERMINAL_TO_STATUS[most_recent_terminal.event_type]

    furthest = max(
        events,
        key=lambda e: _STAGE_ORDER.index(e.event_type) if e.event_type in _STAGE_ORDER else -1,
    )
    status = _STAGE_TO_STATUS.get(furthest.event_type, ApplicationStatus.APPLIED)

    last_event = max(events, key=lambda e: e.occurred_at)
    if (now - last_event.occurred_at).days > ghost_after_days:
        return ApplicationStatus.GHOSTED
    return status


__all__ = ["LINK_WINDOW_DAYS", "derive_status", "match_application"]
