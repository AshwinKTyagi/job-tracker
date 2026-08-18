"""Shared boundary types.

Every value that crosses a module boundary is defined here. These models are FROZEN by
CONTRACTS.md — downstream agents implement against them and must not edit this file.

All datetimes are timezone-aware UTC (invariant I7).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class EventType(StrEnum):
    """A single observed step in an application's lifecycle.

    Declaration order is NOT precedence order; see ``constants.EVENT_PRECEDENCE``.

    StrEnum, not (str, Enum): ``str(EventType.REJECTION)`` must yield "rejection", not
    "EventType.REJECTION". Store and export both interpolate these directly.
    """

    APPLICATION_RECEIVED = "application_received"
    ASSESSMENT = "assessment"  # OA, take-home, coding challenge
    INTERVIEW = "interview"  # screen, onsite, scheduling
    OFFER = "offer"
    REJECTION = "rejection"
    WITHDRAWN = "withdrawn"  # candidate-initiated
    RECRUITER_OUTREACH = "recruiter_outreach"  # inbound, no application yet
    UNKNOWN = "unknown"  # not job-related, or unparseable


class ApplicationStatus(StrEnum):
    """DERIVED from an application's event history. Never persisted as a column (I4)."""

    APPLIED = "applied"
    ASSESSMENT = "assessment"
    INTERVIEWING = "interviewing"
    OFFER = "offer"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"
    GHOSTED = "ghosted"  # non-terminal, no event for config.store.ghost_after_days


class RawMessage(BaseModel):
    """A normalized email: the only thing M1 produces and the only thing M2 consumes.

    ``body_text`` is plain text with HTML already stripped and whitespace collapsed.
    ``headers`` keys are lowercased.
    """

    model_config = ConfigDict(frozen=True)

    message_id: str  # Gmail message id — the global dedupe key (I1)
    thread_id: str
    received_at: datetime  # tz-aware UTC
    from_email: str  # lowercased
    from_name: str | None = None
    to_email: str | None = None
    subject: str = ""
    body_text: str = ""
    snippet: str = ""
    labels: list[str] = Field(default_factory=list)
    headers: dict[str, str] = Field(default_factory=dict)


class Classification(BaseModel):
    """M2's output. Fully determined by the RawMessage — no I/O, no clock, no randomness (I2)."""

    model_config = ConfigDict(frozen=True)

    message_id: str
    event_type: EventType
    company: str | None = None  # display form, verbatim from the email
    company_key: str | None = None  # normalize_company(company); the matching key (I8)
    role: str | None = None
    location: str | None = None
    ats: str | None = None  # "greenhouse" | "lever" | ... | None
    confidence: float = Field(ge=0.0, le=1.0)
    needs_review: bool = False
    evidence: list[str] = Field(default_factory=list)  # stable rule ids that fired
    classifier_name: str  # "rules" | "ollama" | "composite"
    classifier_version: str  # semver for rules; prompt SHA-256 for LLM


class Override(BaseModel):
    """A human correction. Wins over any classifier output, survives reclassify (I6)."""

    message_id: str
    event_type: EventType | None = None
    company: str | None = None
    role: str | None = None
    corrected_at: datetime
    note: str | None = None


class EventRow(BaseModel):
    """One stored event, after overrides are applied."""

    event_id: int
    application_id: str | None  # None only for UNKNOWN / unlinked
    message_id: str
    event_type: EventType
    occurred_at: datetime
    confidence: float
    needs_review: bool
    is_overridden: bool
    subject: str
    from_email: str


class ApplicationRow(BaseModel):
    """One application with its derived status: the unit of the spreadsheet and the charts."""

    application_id: str
    company: str
    company_key: str
    role: str | None
    location: str | None
    ats: str | None
    status: ApplicationStatus  # derived (I4)
    applied_at: datetime
    last_event_at: datetime
    last_event_type: EventType
    event_count: int
    days_to_first_response: int | None  # applied -> first non-APPLICATION_RECEIVED event
    days_since_last_event: int
    needs_review: bool
    source_thread_ids: list[str]


class ReviewItem(BaseModel):
    """One entry in the low-confidence queue."""

    message: RawMessage
    classification: Classification
    suggested_application_id: str | None


class ApplicationMatchCandidate(BaseModel):
    """Pre-fetched by the store, handed to the pure linker. Keeps matching testable."""

    application_id: str
    company_key: str
    role: str | None
    thread_ids: list[str]
    applied_at: datetime
    last_event_at: datetime


class StageFlow(BaseModel):
    """One Sankey link: ``count`` applications moved from ``source`` to ``target``."""

    source: str
    target: str
    count: int


class SyncReport(BaseModel):
    """Returned by the sync orchestration so the CLI can print a summary."""

    fetched: int
    new_messages: int
    events_created: int
    applications_created: int
    needs_review: int
    unknown: int
    errors: list[str] = Field(default_factory=list)
    started_at: datetime
    finished_at: datetime


__all__ = [
    "ApplicationMatchCandidate",
    "ApplicationRow",
    "ApplicationStatus",
    "Classification",
    "EventRow",
    "EventType",
    "Override",
    "RawMessage",
    "ReviewItem",
    "StageFlow",
    "SyncReport",
]
