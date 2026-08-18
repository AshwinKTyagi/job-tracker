"""Frozen constants shared across modules.

``EVENT_PRECEDENCE`` and ``EXPORT_COLUMNS`` are load-bearing: they are what let Phase 1
modules be built in parallel without coordinating. Do not reorder either one.
"""

from __future__ import annotations

from typing import Final

from jobtrack.models import EventType

EVENT_PRECEDENCE: Final[tuple[EventType, ...]] = (
    EventType.WITHDRAWN,
    EventType.REJECTION,
    EventType.OFFER,
    EventType.INTERVIEW,
    EventType.ASSESSMENT,
    EventType.APPLICATION_RECEIVED,
    EventType.RECRUITER_OUTREACH,
    EventType.UNKNOWN,
)
"""Highest-precedence matched type wins (I3). Must cover every EventType.

WITHDRAWN is first: "you have withdrawn your application" is explicit, terminal, and
candidate-initiated, so nothing should override it.

REJECTION outranks APPLICATION_RECEIVED because rejection emails restate the application
language ("Thanks for applying to X ... unfortunately we are not moving forward"), and
outranks INTERVIEW because post-interview rejections restate the interview. Scoring must
evaluate every type and then resolve here — never stop at the first match.
"""

TERMINAL_EVENTS: Final[frozenset[EventType]] = frozenset(
    {EventType.REJECTION, EventType.OFFER, EventType.WITHDRAWN}
)
"""Events that close an application. The most recent one decides status regardless of recency."""

EXPORT_COLUMNS: Final[tuple[str, ...]] = (
    "application_id",
    "company",
    "role",
    "location",
    "ats",
    "status",
    "applied_at",
    "last_event_at",
    "last_event_type",
    "event_count",
    "days_to_first_response",
    "days_since_last_event",
    "needs_review",
)
"""FROZEN. M4 emits exactly these, in this order. M5 reads exactly these (I10)."""

EVENT_COLUMNS: Final[tuple[str, ...]] = (
    "application_id",
    "message_id",
    "event_type",
    "occurred_at",
    "confidence",
    "needs_review",
    "subject",
)
"""Long-format event frame shared by M4 and M5."""

DEFAULT_GMAIL_QUERY: Final[str] = (
    "-in:chats ("
    '"thank you for applying" OR "thanks for applying" OR "application received" OR '
    '"we received your application" OR "your application" OR "application status" OR '
    '"not moving forward" OR "other candidates" OR "unfortunately" OR '
    '"interview" OR "next steps" OR "coding challenge" OR "take-home" OR "assessment" OR '
    "from:greenhouse.io OR from:lever.co OR from:hire.lever.co OR from:myworkday.com OR "
    "from:ashbyhq.com OR from:smartrecruiters.com OR from:icims.com OR from:taleo.net OR "
    "from:jobvite.com OR from:workable.com OR from:breezy.hr OR from:bamboohr.com"
    ")"
)
"""Tuned for RECALL, not precision — the classifier is the real filter. Overridable in config."""

GMAIL_SCOPES: Final[list[str]] = ["https://www.googleapis.com/auth/gmail.readonly"]
"""Read-only, and that is the entire scope budget (I11). Never widen this list."""

__all__ = [
    "DEFAULT_GMAIL_QUERY",
    "EVENT_COLUMNS",
    "EVENT_PRECEDENCE",
    "EXPORT_COLUMNS",
    "GMAIL_SCOPES",
    "TERMINAL_EVENTS",
]
