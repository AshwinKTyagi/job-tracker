"""SQL text, row mapping, and read-model assembly for the store.

``db.py`` owns the connection, the transactions, and the public :class:`Store` API;
this module owns everything that translates between SQLite rows and the frozen models in
``jobtrack.models``. Splitting it this way keeps the assembly of an
:class:`~jobtrack.models.ApplicationRow` — the only genuinely intricate read in M3 — a
pure function that tests can drive without a database.

Rules that apply to every statement here:

* values are bound with ``?`` placeholders, never interpolated;
* every column is named in the projection, and every row is read by name;
* timestamps are ISO-8601 UTC text on disk and tz-aware ``datetime`` in memory (I7).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

from jobtrack.errors import StoreError
from jobtrack.models import (
    ApplicationRow,
    Classification,
    EventRow,
    EventType,
    Override,
    RawMessage,
)
from jobtrack.store.linker import derive_status

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Timestamps
# --------------------------------------------------------------------------------------


def to_iso(value: datetime) -> str:
    """Render a tz-aware datetime as ISO-8601 UTC text for storage.

    Args:
        value: A timezone-aware datetime in any zone.

    Returns:
        The UTC ISO-8601 rendering, e.g. ``"2026-08-18T12:00:00+00:00"``.

    Raises:
        StoreError: the datetime is naive, which violates invariant I7.
    """
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise StoreError(f"naive datetime crossed the store boundary: {value!r}")
    return value.astimezone(UTC).isoformat()


def from_iso(value: str) -> datetime:
    """Parse a stored timestamp back into a tz-aware UTC datetime.

    Args:
        value: ISO-8601 text as written by :func:`to_iso`.

    Returns:
        The parsed datetime, normalized to UTC.

    Raises:
        StoreError: the text is not a parseable timestamp.
    """
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise StoreError(f"unparseable timestamp in database: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


# --------------------------------------------------------------------------------------
# Statements
# --------------------------------------------------------------------------------------

INSERT_MESSAGE: Final[str] = """
INSERT INTO messages (
    message_id, thread_id, received_at, from_email, from_name, to_email,
    subject, body_text, snippet, labels_json, headers_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (message_id) DO NOTHING
"""

SELECT_MESSAGE_EXISTS: Final[str] = """
SELECT 1 AS present FROM messages WHERE message_id = ?
"""

SELECT_MESSAGE: Final[str] = """
SELECT
    message_id   AS message_id,
    thread_id    AS thread_id,
    received_at  AS received_at,
    from_email   AS from_email,
    from_name    AS from_name,
    to_email     AS to_email,
    subject      AS subject,
    body_text    AS body_text,
    snippet      AS snippet,
    labels_json  AS labels_json,
    headers_json AS headers_json
FROM messages
WHERE message_id = ?
"""

UPSERT_CLASSIFICATION: Final[str] = """
INSERT INTO classifications (
    message_id, event_type, company, company_key, role, location, ats,
    confidence, needs_review, evidence_json, classifier_name, classifier_version
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (message_id) DO UPDATE SET
    event_type         = excluded.event_type,
    company            = excluded.company,
    company_key        = excluded.company_key,
    role               = excluded.role,
    location           = excluded.location,
    ats                = excluded.ats,
    confidence         = excluded.confidence,
    needs_review       = excluded.needs_review,
    evidence_json      = excluded.evidence_json,
    classifier_name    = excluded.classifier_name,
    classifier_version = excluded.classifier_version
"""

SELECT_CLASSIFICATION: Final[str] = """
SELECT
    message_id         AS message_id,
    event_type         AS event_type,
    company            AS company,
    company_key        AS company_key,
    role               AS role,
    location           AS location,
    ats                AS ats,
    confidence         AS confidence,
    needs_review       AS needs_review,
    evidence_json      AS evidence_json,
    classifier_name    AS classifier_name,
    classifier_version AS classifier_version
FROM classifications
WHERE message_id = ?
"""

CLEAR_CLASSIFICATIONS_ALL: Final[str] = """
DELETE FROM classifications
"""

CLEAR_CLASSIFICATIONS_UNREVIEWED: Final[str] = """
DELETE FROM classifications
WHERE message_id NOT IN (SELECT message_id FROM overrides)
"""

MARK_CLASSIFICATION_REVIEWED: Final[str] = """
UPDATE classifications SET needs_review = 0 WHERE message_id = ?
"""

INSERT_APPLICATION: Final[str] = """
INSERT INTO applications (
    application_id, company, company_key, role, location, ats, applied_at, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

ENRICH_APPLICATION: Final[str] = """
UPDATE applications
SET role       = COALESCE(role, ?),
    location   = COALESCE(location, ?),
    ats        = COALESCE(ats, ?),
    applied_at = MIN(applied_at, ?)
WHERE application_id = ?
"""

SELECT_APPLICATION_CORES: Final[str] = """
SELECT
    application_id AS application_id,
    company        AS company,
    company_key    AS company_key,
    role           AS role,
    location       AS location,
    ats            AS ats
FROM applications
ORDER BY application_id
"""

SELECT_APPLICATION_CORE: Final[str] = """
SELECT
    application_id AS application_id,
    company        AS company,
    company_key    AS company_key,
    role           AS role,
    location       AS location,
    ats            AS ats
FROM applications
WHERE application_id = ?
"""

DELETE_ORPHAN_APPLICATIONS: Final[str] = """
DELETE FROM applications
WHERE application_id NOT IN (
    SELECT application_id FROM events WHERE application_id IS NOT NULL
)
"""

INSERT_EVENT: Final[str] = """
INSERT INTO events (application_id, message_id, event_type, occurred_at, created_at)
VALUES (?, ?, ?, ?, ?)
"""

RELINK_EVENT: Final[str] = """
UPDATE events SET application_id = ? WHERE message_id = ?
"""

SELECT_EVENT_RECORDS: Final[str] = """
SELECT
    e.event_id       AS event_id,
    e.application_id AS application_id,
    e.message_id     AS message_id,
    e.event_type     AS event_type,
    e.occurred_at    AS occurred_at,
    m.thread_id      AS thread_id,
    m.subject        AS subject,
    m.from_email     AS from_email,
    c.confidence     AS confidence,
    c.needs_review   AS needs_review,
    o.event_type     AS override_event_type,
    o.company        AS override_company,
    o.role           AS override_role,
    o.corrected_at   AS override_corrected_at
FROM events AS e
JOIN messages AS m ON m.message_id = e.message_id
LEFT JOIN classifications AS c ON c.message_id = e.message_id
LEFT JOIN overrides AS o ON o.message_id = e.message_id
"""

WHERE_EVENTS_FOR_APPLICATION: Final[str] = " WHERE e.application_id = ?"
WHERE_EVENTS_LINKED: Final[str] = " WHERE e.application_id IS NOT NULL"
WHERE_EVENTS_FOR_MESSAGE: Final[str] = " WHERE e.message_id = ?"
ORDER_EVENTS: Final[str] = " ORDER BY e.occurred_at ASC, e.event_id ASC"

SELECT_CANDIDATE_APPLICATIONS: Final[str] = """
SELECT
    a.application_id   AS application_id,
    a.company_key      AS company_key,
    a.role             AS role,
    a.applied_at       AS applied_at,
    MAX(e.occurred_at) AS last_event_at
FROM applications AS a
JOIN events AS e ON e.application_id = a.application_id
WHERE a.company_key = ?
   OR a.application_id IN (
        SELECT e2.application_id
        FROM events AS e2
        JOIN messages AS m2 ON m2.message_id = e2.message_id
        WHERE m2.thread_id = ? AND e2.application_id IS NOT NULL
   )
GROUP BY a.application_id, a.company_key, a.role, a.applied_at
ORDER BY a.application_id
"""

SELECT_APPLICATION_THREADS: Final[str] = """
SELECT DISTINCT
    e.application_id AS application_id,
    m.thread_id      AS thread_id
FROM events AS e
JOIN messages AS m ON m.message_id = e.message_id
WHERE e.application_id IS NOT NULL
ORDER BY e.application_id, m.thread_id
"""

SELECT_NEWEST_EVENT_AT: Final[str] = """
SELECT MAX(occurred_at) AS newest FROM events
"""

UPSERT_OVERRIDE: Final[str] = """
INSERT INTO overrides (message_id, event_type, company, role, corrected_at, note)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT (message_id) DO UPDATE SET
    event_type   = excluded.event_type,
    company      = excluded.company,
    role         = excluded.role,
    corrected_at = excluded.corrected_at,
    note         = excluded.note
"""

INSERT_ACCEPTANCE: Final[str] = """
INSERT INTO overrides (message_id, event_type, company, role, corrected_at, note)
VALUES (?, NULL, NULL, NULL, ?, ?)
ON CONFLICT (message_id) DO NOTHING
"""

SELECT_PENDING_REVIEW: Final[str] = """
SELECT
    m.message_id     AS message_id,
    e.application_id AS application_id
FROM messages AS m
JOIN classifications AS c ON c.message_id = m.message_id
LEFT JOIN overrides AS o ON o.message_id = m.message_id
LEFT JOIN events AS e ON e.message_id = m.message_id
WHERE c.needs_review = 1 AND o.message_id IS NULL
ORDER BY m.received_at ASC, m.message_id ASC
"""

LIMIT_CLAUSE: Final[str] = " LIMIT ?"

SELECT_CURSOR: Final[str] = """
SELECT cursor AS cursor FROM sync_state WHERE source = ?
"""

UPSERT_CURSOR: Final[str] = """
INSERT INTO sync_state (source, cursor, last_synced_at)
VALUES (?, ?, ?)
ON CONFLICT (source) DO UPDATE SET
    cursor         = excluded.cursor,
    last_synced_at = excluded.last_synced_at
"""

SELECT_SCHEMA_VERSION: Final[str] = """
SELECT MAX(version) AS version FROM schema_version
"""

INSERT_SCHEMA_VERSION: Final[str] = """
INSERT INTO schema_version (version, applied_at)
VALUES (?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
"""
"""``applied_at`` comes from the database clock: ``migrate`` has no injected ``now`` to
use, and nothing derives from this column — it is provenance only."""

ACCEPTED_NOTE: Final[str] = "accepted: classifier confirmed by reviewer"
"""Note written on the override row that records an accepted classification."""


# --------------------------------------------------------------------------------------
# Module-internal value objects
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ApplicationCore:
    """The stored, non-derived half of an application row."""

    application_id: str
    company: str
    company_key: str
    role: str | None
    location: str | None
    ats: str | None


@dataclass(frozen=True)
class EventRecord:
    """An :class:`EventRow` plus the extra columns needed to assemble an application.

    ``row`` already has the override folded into ``event_type``; ``override_company`` and
    ``override_role`` are carried separately because they correct the *application*, not
    the event.
    """

    row: EventRow
    thread_id: str
    override_company: str | None
    override_role: str | None
    corrected_at: datetime | None


# --------------------------------------------------------------------------------------
# Row mapping
# --------------------------------------------------------------------------------------


def message_params(message: RawMessage) -> tuple[object, ...]:
    """Bind a RawMessage to the placeholders of :data:`INSERT_MESSAGE`.

    Args:
        message: The message to persist.

    Returns:
        The positional parameters, in statement order.

    Raises:
        StoreError: ``received_at`` is naive (I7).
    """
    return (
        message.message_id,
        message.thread_id,
        to_iso(message.received_at),
        message.from_email,
        message.from_name,
        message.to_email,
        message.subject,
        message.body_text,
        message.snippet,
        json.dumps(message.labels),
        json.dumps(message.headers, sort_keys=True),
    )


def classification_params(classification: Classification) -> tuple[object, ...]:
    """Bind a Classification to the placeholders of :data:`UPSERT_CLASSIFICATION`.

    Args:
        classification: The classifier output to persist.

    Returns:
        The positional parameters, in statement order.
    """
    return (
        classification.message_id,
        str(classification.event_type),
        classification.company,
        classification.company_key,
        classification.role,
        classification.location,
        classification.ats,
        classification.confidence,
        int(classification.needs_review),
        json.dumps(classification.evidence),
        classification.classifier_name,
        classification.classifier_version,
    )


def override_params(override: Override) -> tuple[object, ...]:
    """Bind an Override to the placeholders of :data:`UPSERT_OVERRIDE`.

    Args:
        override: The human correction to persist.

    Returns:
        The positional parameters, in statement order.

    Raises:
        StoreError: ``corrected_at`` is naive (I7).
    """
    return (
        override.message_id,
        None if override.event_type is None else str(override.event_type),
        override.company,
        override.role,
        to_iso(override.corrected_at),
        override.note,
    )


def row_to_message(row: sqlite3.Row) -> RawMessage:
    """Rebuild a RawMessage from a :data:`SELECT_MESSAGE` row.

    Args:
        row: A row with the columns named by :data:`SELECT_MESSAGE`.

    Returns:
        The reconstructed message.

    Raises:
        StoreError: a JSON column is corrupt or a timestamp is unparseable.
    """
    return RawMessage(
        message_id=row["message_id"],
        thread_id=row["thread_id"],
        received_at=from_iso(row["received_at"]),
        from_email=row["from_email"],
        from_name=row["from_name"],
        to_email=row["to_email"],
        subject=row["subject"],
        body_text=row["body_text"],
        snippet=row["snippet"],
        labels=_json_list(row["labels_json"], column="labels_json"),
        headers=_json_dict(row["headers_json"], column="headers_json"),
    )


def row_to_classification(row: sqlite3.Row) -> Classification:
    """Rebuild a Classification from a :data:`SELECT_CLASSIFICATION` row.

    Args:
        row: A row with the columns named by :data:`SELECT_CLASSIFICATION`.

    Returns:
        The reconstructed classification.

    Raises:
        StoreError: the stored event_type is unknown, or evidence_json is corrupt.
    """
    return Classification(
        message_id=row["message_id"],
        event_type=to_event_type(row["event_type"]),
        company=row["company"],
        company_key=row["company_key"],
        role=row["role"],
        location=row["location"],
        ats=row["ats"],
        confidence=row["confidence"],
        needs_review=bool(row["needs_review"]),
        evidence=_json_list(row["evidence_json"], column="evidence_json"),
        classifier_name=row["classifier_name"],
        classifier_version=row["classifier_version"],
    )


def row_to_event_record(row: sqlite3.Row) -> EventRecord:
    """Build an EventRecord from a :data:`SELECT_EVENT_RECORDS` row, applying the override.

    A missing classification — one dropped by ``clear_classifications`` — reads as
    confidence 0.0 and ``needs_review`` True: it has not been classified yet.

    Args:
        row: A row with the columns named by :data:`SELECT_EVENT_RECORDS`.

    Returns:
        The event with its override folded into ``event_type`` (I6).

    Raises:
        StoreError: a stored event_type is unknown, or a timestamp is unparseable.
    """
    override_event_type = row["override_event_type"]
    override_company = row["override_company"]
    override_role = row["override_role"]
    corrected_at = row["override_corrected_at"]
    stored_type = to_event_type(row["event_type"])
    effective_type = (
        stored_type if override_event_type is None else to_event_type(override_event_type)
    )
    event = EventRow(
        event_id=int(row["event_id"]),
        application_id=row["application_id"],
        message_id=row["message_id"],
        event_type=effective_type,
        occurred_at=from_iso(row["occurred_at"]),
        confidence=0.0 if row["confidence"] is None else float(row["confidence"]),
        needs_review=True if row["needs_review"] is None else bool(row["needs_review"]),
        is_overridden=any(
            value is not None for value in (override_event_type, override_company, override_role)
        ),
        subject=row["subject"],
        from_email=row["from_email"],
    )
    return EventRecord(
        row=event,
        thread_id=row["thread_id"],
        override_company=override_company,
        override_role=override_role,
        corrected_at=None if corrected_at is None else from_iso(corrected_at),
    )


def row_to_application_core(row: sqlite3.Row) -> ApplicationCore:
    """Build an ApplicationCore from a :data:`SELECT_APPLICATION_CORE` row.

    Args:
        row: A row with the columns named by :data:`SELECT_APPLICATION_CORE`.

    Returns:
        The stored half of the application.
    """
    return ApplicationCore(
        application_id=row["application_id"],
        company=row["company"],
        company_key=row["company_key"],
        role=row["role"],
        location=row["location"],
        ats=row["ats"],
    )


def to_event_type(value: str) -> EventType:
    """Convert stored text back into an EventType.

    Args:
        value: The stored column value.

    Returns:
        The matching EventType.

    Raises:
        StoreError: the value is not a member of EventType.
    """
    try:
        return EventType(value)
    except ValueError as exc:
        raise StoreError(f"unknown event_type in database: {value!r}") from exc


def _json_list(raw: str, *, column: str) -> list[Any]:
    """Decode a JSON array column."""
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StoreError(f"corrupt JSON in {column}: {raw!r}") from exc
    if not isinstance(decoded, list):
        raise StoreError(f"expected a JSON array in {column}, got {type(decoded).__name__}")
    return decoded


def _json_dict(raw: str, *, column: str) -> dict[str, Any]:
    """Decode a JSON object column."""
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StoreError(f"corrupt JSON in {column}: {raw!r}") from exc
    if not isinstance(decoded, dict):
        raise StoreError(f"expected a JSON object in {column}, got {type(decoded).__name__}")
    return decoded


# --------------------------------------------------------------------------------------
# Read-model assembly
# --------------------------------------------------------------------------------------


def _latest_company_correction(records: Sequence[EventRecord]) -> str | None:
    """Most recently corrected company across an application's events, if any."""
    corrected = [
        r for r in records if r.override_company is not None and r.corrected_at is not None
    ]
    if not corrected:
        return None
    return max(corrected, key=_correction_key).override_company


def _latest_role_correction(records: Sequence[EventRecord]) -> str | None:
    """Most recently corrected role across an application's events, if any."""
    corrected = [r for r in records if r.override_role is not None and r.corrected_at is not None]
    if not corrected:
        return None
    return max(corrected, key=_correction_key).override_role


def _correction_key(record: EventRecord) -> tuple[datetime, int]:
    """Newest-correction-wins ordering, with a stable tiebreak on event_id."""
    if record.corrected_at is None:  # pragma: no cover - callers filter these out
        raise StoreError(f"correction on {record.row.message_id} has no corrected_at")
    return (record.corrected_at, record.row.event_id)


def build_application_row(
    core: ApplicationCore,
    records: Sequence[EventRecord],
    *,
    now: datetime,
    ghost_after_days: int,
) -> ApplicationRow:
    """Assemble one ApplicationRow, deriving status and applying overrides.

    Status is computed here on every read and is never stored (I4). Company and role
    corrections beat the stored display values (I6); when one application collected
    several corrections, the most recent wins.

    Args:
        core: The stored columns of the application.
        records: Its events, in any order. Must be non-empty.
        now: Injected tz-aware UTC clock.
        ghost_after_days: Silence, in days, after which the application reads as GHOSTED.

    Returns:
        The fully derived application row.

    Raises:
        StoreError: the application has no events, which the writer never produces.
    """
    if not records:
        raise StoreError(f"application {core.application_id} has no events")

    ordered = sorted(records, key=lambda record: (record.row.occurred_at, record.row.event_id))
    events = [record.row for record in ordered]
    applied_at = events[0].occurred_at
    last = events[-1]

    first_response = next(
        (e for e in events if e.event_type is not EventType.APPLICATION_RECEIVED), None
    )
    days_to_first_response = (
        None if first_response is None else (first_response.occurred_at - applied_at).days
    )

    return ApplicationRow(
        application_id=core.application_id,
        company=_latest_company_correction(ordered) or core.company,
        company_key=core.company_key,
        role=_latest_role_correction(ordered) or core.role,
        location=core.location,
        ats=core.ats,
        status=derive_status(events, now=now, ghost_after_days=ghost_after_days),
        applied_at=applied_at,
        last_event_at=last.occurred_at,
        last_event_type=last.event_type,
        event_count=len(events),
        days_to_first_response=days_to_first_response,
        days_since_last_event=(now - last.occurred_at).days,
        needs_review=any(e.needs_review and not e.is_overridden for e in events),
        source_thread_ids=sorted({record.thread_id for record in ordered}),
    )


__all__ = [
    "ACCEPTED_NOTE",
    "ApplicationCore",
    "EventRecord",
    "build_application_row",
    "classification_params",
    "from_iso",
    "message_params",
    "override_params",
    "row_to_application_core",
    "row_to_classification",
    "row_to_event_record",
    "row_to_message",
    "to_event_type",
    "to_iso",
]
