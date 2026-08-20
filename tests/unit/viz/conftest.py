"""Frame builders for the viz tests.

M5 consumes DataFrames, not rows, so these helpers assemble frames with exactly
``EXPORT_COLUMNS`` / ``EVENT_COLUMNS`` in exactly that order (invariant I10) — the same wire
format M4 emits. Building them here rather than importing M4 keeps the two modules
independently testable, which is the whole point of freezing the column tuples.

Nothing here touches SQLite, the network, or the clock: every timestamp is derived from the
frozen ``BASE_TIME`` below.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import pytest

from jobtrack.constants import EVENT_COLUMNS, EXPORT_COLUMNS
from jobtrack.models import ApplicationRow, ApplicationStatus, EventRow, EventType

BASE_TIME = datetime(2026, 1, 5, 9, 0, 0, tzinfo=UTC)
"""Fixed, tz-aware UTC origin for every generated timestamp (I7)."""


def make_application(**overrides: Any) -> ApplicationRow:
    """Build one ApplicationRow, naming only the fields a test cares about."""
    defaults: dict[str, Any] = {
        "application_id": "app-0001",
        "company": "Acme Robotics",
        "company_key": "acme robotics",
        "role": "Software Engineer",
        "location": "Remote",
        "ats": "greenhouse",
        "status": ApplicationStatus.APPLIED,
        "applied_at": BASE_TIME,
        "last_event_at": BASE_TIME,
        "last_event_type": EventType.APPLICATION_RECEIVED,
        "event_count": 1,
        "days_to_first_response": None,
        "days_since_last_event": 0,
        "needs_review": False,
        "source_thread_ids": ["thread-0001"],
    }
    defaults.update(overrides)
    return ApplicationRow.model_validate(defaults)


def make_event(**overrides: Any) -> EventRow:
    """Build one EventRow, naming only the fields a test cares about."""
    defaults: dict[str, Any] = {
        "event_id": 1,
        "application_id": "app-0001",
        "message_id": "msg-0001",
        "event_type": EventType.APPLICATION_RECEIVED,
        "occurred_at": BASE_TIME,
        "confidence": 0.9,
        "needs_review": False,
        "is_overridden": False,
        "subject": "Thanks for applying",
        "from_email": "no-reply@greenhouse.io",
    }
    defaults.update(overrides)
    return EventRow.model_validate(defaults)


def applications_frame(rows: Sequence[ApplicationRow]) -> pd.DataFrame:
    """Applications frame with exactly EXPORT_COLUMNS, in order (I10)."""
    records = [{column: _cell(getattr(row, column)) for column in EXPORT_COLUMNS} for row in rows]
    df = pd.DataFrame.from_records(records, columns=list(EXPORT_COLUMNS))
    for column in ("applied_at", "last_event_at"):
        df[column] = pd.to_datetime(df[column], utc=True)
    df["days_to_first_response"] = df["days_to_first_response"].astype("Int64")
    df["event_count"] = df["event_count"].astype("int64")
    df["days_since_last_event"] = df["days_since_last_event"].astype("int64")
    df["needs_review"] = df["needs_review"].astype("bool")
    return df


def events_frame(rows: Sequence[EventRow]) -> pd.DataFrame:
    """Long-format events frame with exactly EVENT_COLUMNS, in order (I10)."""
    records = [{column: _cell(getattr(row, column)) for column in EVENT_COLUMNS} for row in rows]
    df = pd.DataFrame.from_records(records, columns=list(EVENT_COLUMNS))
    df["occurred_at"] = pd.to_datetime(df["occurred_at"], utc=True)
    df["confidence"] = df["confidence"].astype("float64")
    df["needs_review"] = df["needs_review"].astype("bool")
    return df


def _cell(value: object) -> object:
    """Enums land in the frame as their string values, exactly as StrEnum serializes them."""
    return value.value if isinstance(value, EventType | ApplicationStatus) else value


def linear_pipeline(
    application_id: str,
    event_types: Sequence[EventType],
    *,
    start: datetime = BASE_TIME,
) -> list[EventRow]:
    """Events for one application, one day apart, in the order given."""
    return [
        make_event(
            event_id=index + 1,
            application_id=application_id,
            message_id=f"{application_id}-msg-{index}",
            event_type=event_type,
            occurred_at=start + timedelta(days=index),
        )
        for index, event_type in enumerate(event_types)
    ]


@pytest.fixture
def empty_applications() -> pd.DataFrame:
    """An empty applications frame that still carries every EXPORT_COLUMN."""
    return applications_frame([])


@pytest.fixture
def empty_events() -> pd.DataFrame:
    """An empty events frame that still carries every EVENT_COLUMN."""
    return events_frame([])


@pytest.fixture
def sample_applications() -> pd.DataFrame:
    """Five applications spanning several statuses, companies, and response times."""
    return applications_frame(
        [
            make_application(
                application_id="app-1",
                company="Acme Robotics",
                company_key="acme robotics",
                status=ApplicationStatus.REJECTED,
                applied_at=BASE_TIME,
                last_event_at=BASE_TIME + timedelta(days=10),
                last_event_type=EventType.REJECTION,
                event_count=3,
                days_to_first_response=4,
                days_since_last_event=2,
            ),
            make_application(
                application_id="app-2",
                company="Acme Robotics",
                company_key="acme robotics",
                status=ApplicationStatus.INTERVIEWING,
                applied_at=BASE_TIME + timedelta(days=1),
                last_event_at=BASE_TIME + timedelta(days=9),
                last_event_type=EventType.INTERVIEW,
                event_count=2,
                days_to_first_response=8,
                days_since_last_event=3,
                needs_review=True,
            ),
            make_application(
                application_id="app-3",
                company="Globex",
                company_key="globex",
                status=ApplicationStatus.OFFER,
                applied_at=BASE_TIME + timedelta(days=20),
                last_event_at=BASE_TIME + timedelta(days=40),
                last_event_type=EventType.OFFER,
                event_count=4,
                days_to_first_response=6,
                days_since_last_event=1,
            ),
            make_application(
                application_id="app-4",
                company="Initech",
                company_key="initech",
                status=ApplicationStatus.GHOSTED,
                applied_at=BASE_TIME + timedelta(days=30),
                last_event_at=BASE_TIME + timedelta(days=30),
                last_event_type=EventType.APPLICATION_RECEIVED,
                event_count=1,
                days_to_first_response=None,
                days_since_last_event=45,
            ),
            make_application(
                application_id="app-5",
                company="Initech",
                company_key="initech",
                status=ApplicationStatus.APPLIED,
                applied_at=BASE_TIME + timedelta(days=31),
                last_event_at=BASE_TIME + timedelta(days=31),
                last_event_type=EventType.APPLICATION_RECEIVED,
                event_count=1,
                days_to_first_response=None,
                days_since_last_event=4,
            ),
        ]
    )


@pytest.fixture
def sample_events() -> pd.DataFrame:
    """Event histories matching `sample_applications`, plus one unlinked UNKNOWN."""
    rows = [
        *linear_pipeline(
            "app-1",
            [EventType.APPLICATION_RECEIVED, EventType.INTERVIEW, EventType.REJECTION],
        ),
        *linear_pipeline("app-2", [EventType.APPLICATION_RECEIVED, EventType.INTERVIEW]),
        *linear_pipeline(
            "app-3",
            [
                EventType.APPLICATION_RECEIVED,
                EventType.ASSESSMENT,
                EventType.INTERVIEW,
                EventType.OFFER,
            ],
        ),
        *linear_pipeline("app-4", [EventType.APPLICATION_RECEIVED]),
        *linear_pipeline("app-5", [EventType.APPLICATION_RECEIVED]),
        make_event(
            event_id=99,
            application_id=None,
            message_id="msg-unlinked",
            event_type=EventType.UNKNOWN,
            subject="Weekly newsletter",
        ),
    ]
    return events_frame(rows)
