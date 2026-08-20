"""The repository/query surface of :class:`Store`.

Defines the public ``Store`` class (``_ConnectionMixin`` plus every method CONTRACTS.md §6
lists) and message<->application linking via the pure functions in ``linker.py``. Every query
here names its columns explicitly and binds with ``?`` placeholders — no ``SELECT *``, no
string interpolation into SQL (CLAUDE.md).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Final

from jobtrack.errors import StoreError
from jobtrack.models import (
    ApplicationMatchCandidate,
    ApplicationRow,
    ApplicationStatus,
    Classification,
    EventRow,
    EventType,
    Override,
    RawMessage,
    ReviewItem,
)
from jobtrack.store.db import _bookkeeping_now_iso, _ConnectionMixin
from jobtrack.store.linker import LINK_WINDOW_DAYS, derive_status, match_application

logger = logging.getLogger(__name__)

DEFAULT_GHOST_AFTER_DAYS: Final[int] = 30
"""Fallback used by ``get_application``/``list_applications``.

CONTRACTS.md's ``derive_status`` requires ``ghost_after_days``, but neither ``Store.open``
nor these read methods have a way to receive ``StoreConfig`` (see the final report — this is
a flagged contract gap). This constant mirrors ``StoreConfig.ghost_after_days``'s own default
of 30 so behavior matches an unconfigured install; a caller running with a non-default
``ghost_after_days`` will see it applied by ``jobtrack list``/``stats`` recomputing from
``list_events`` and ``config.store.ghost_after_days`` directly once M6 wires it, but the
``Store`` methods themselves cannot honor a custom value until CONTRACTS.md is amended.
"""

_EVENT_ROW_QUERY: Final[str] = """
    SELECT
        e.event_id AS event_id,
        e.application_id AS application_id,
        e.message_id AS message_id,
        e.event_type AS event_type,
        e.occurred_at AS occurred_at,
        c.confidence AS confidence,
        c.needs_review AS needs_review,
        o.message_id AS override_message_id,
        o.event_type AS override_event_type,
        m.subject AS subject,
        m.from_email AS from_email
    FROM events e
    JOIN messages m ON m.message_id = e.message_id
    LEFT JOIN classifications c ON c.message_id = e.message_id
    LEFT JOIN overrides o ON o.message_id = e.message_id
"""


def _dt_to_iso(value: datetime) -> str:
    """Serialize a tz-aware UTC datetime for storage (I7).

    Raises:
        StoreError: ``value`` is naive.
    """
    if value.tzinfo is None:
        raise StoreError(f"naive datetime {value!r} cannot cross the store boundary (I7)")
    return value.astimezone(UTC).isoformat()


def _iso_to_dt(value: str) -> datetime:
    """Parse a stored ISO-8601 timestamp back into a tz-aware UTC datetime (I7)."""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _dumps(value: list[str] | dict[str, str]) -> str:
    """Serialize a JSON-shaped field (labels, headers, evidence) for storage."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads_list(value: str) -> list[str]:
    """Parse a JSON array column into a list of strings."""
    data: list[Any] = json.loads(value)
    return [str(item) for item in data]


def _loads_dict(value: str) -> dict[str, str]:
    """Parse a JSON object column into a str-to-str mapping."""
    data: dict[Any, Any] = json.loads(value)
    return {str(k): str(v) for k, v in data.items()}


def _row_to_event(row: sqlite3.Row) -> EventRow:
    """Build an ``EventRow`` from one row of ``_EVENT_ROW_QUERY``, applying overrides (I6)."""
    is_overridden = row["override_message_id"] is not None
    override_event_type = row["override_event_type"]
    has_override_type = is_overridden and override_event_type is not None
    event_type_raw = override_event_type if has_override_type else row["event_type"]
    confidence = float(row["confidence"]) if row["confidence"] is not None else 0.0
    needs_review = bool(row["needs_review"]) if row["needs_review"] is not None else False
    if is_overridden:
        # A human has already looked at this message; it no longer needs review.
        needs_review = False
    return EventRow(
        event_id=int(row["event_id"]),
        application_id=row["application_id"],
        message_id=row["message_id"],
        event_type=EventType(event_type_raw),
        occurred_at=_iso_to_dt(row["occurred_at"]),
        confidence=confidence,
        needs_review=needs_review,
        is_overridden=is_overridden,
        subject=row["subject"],
        from_email=row["from_email"],
    )


class Store(_ConnectionMixin):
    """The only writer of SQLite. The only module that knows SQL.

    Tables: messages · classifications · applications · events · overrides · sync_state ·
    schema_version. Parameterized queries only; every column named explicitly. Connection
    lifecycle (``open``/``migrate``/``close``/context-manager) is inherited from
    ``_ConnectionMixin`` in ``db.py``; this class adds the repository methods.
    """

    # --- ingest side ---------------------------------------------------------------------

    def has_message(self, message_id: str) -> bool:
        """Dedupe check (I1). Cheap — indexed primary key lookup.

        Args:
            message_id: Gmail message id.

        Returns:
            True if a message with this id is already stored.

        Raises:
            StoreError: the query failed.
        """
        try:
            row = self._connection.execute(
                "SELECT message_id FROM messages WHERE message_id = ?", (message_id,)
            ).fetchone()
        except sqlite3.Error as exc:
            raise StoreError(f"could not check message {message_id}: {exc}") from exc
        return row is not None

    def record_message(self, message: RawMessage) -> None:
        """Persist raw metadata. Idempotent: a second call for the same id is a no-op.

        Args:
            message: The normalized email to persist.

        Raises:
            StoreError: the write failed.
        """
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO messages (
                    message_id, thread_id, received_at, from_email, from_name, to_email,
                    subject, body_text, snippet, labels_json, headers_json, ingested_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(message_id) DO NOTHING
                """,
                (
                    message.message_id,
                    message.thread_id,
                    _dt_to_iso(message.received_at),
                    message.from_email,
                    message.from_name,
                    message.to_email,
                    message.subject,
                    message.body_text,
                    message.snippet,
                    _dumps(message.labels),
                    _dumps(message.headers),
                    _bookkeeping_now_iso(),
                ),
            )

    def record_classification(self, classification: Classification) -> None:
        """Upsert on message_id. Replaces a prior classification (reclassify); never touches
        overrides (I6).

        A fresh classification clears any prior ``reviewed_at`` marker — see
        ``clear_classifications`` and ``accept_classification``.

        Args:
            classification: The classifier's output for one message.

        Raises:
            StoreError: the write failed.
        """
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO classifications (
                    message_id, event_type, company, company_key, role, location, ats,
                    confidence, needs_review, evidence_json, classifier_name,
                    classifier_version, classified_at, reviewed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                ON CONFLICT(message_id) DO UPDATE SET
                    event_type = excluded.event_type,
                    company = excluded.company,
                    company_key = excluded.company_key,
                    role = excluded.role,
                    location = excluded.location,
                    ats = excluded.ats,
                    confidence = excluded.confidence,
                    needs_review = excluded.needs_review,
                    evidence_json = excluded.evidence_json,
                    classifier_name = excluded.classifier_name,
                    classifier_version = excluded.classifier_version,
                    classified_at = excluded.classified_at,
                    reviewed_at = NULL
                """,
                (
                    classification.message_id,
                    str(classification.event_type),
                    classification.company,
                    classification.company_key,
                    classification.role,
                    classification.location,
                    classification.ats,
                    classification.confidence,
                    int(classification.needs_review),
                    _dumps(classification.evidence),
                    classification.classifier_name,
                    classification.classifier_version,
                    _bookkeeping_now_iso(),
                ),
            )

    def link_and_record_event(
        self, message: RawMessage, classification: Classification, *, now: datetime
    ) -> EventRow:
        """Link the message to an application (creating one if needed) and append its event.

        Fetches candidates, delegates the decision to ``linker.match_application``, and
        creates a new application when there is no match. UNKNOWN classifications are
        recorded with ``application_id=None``. Idempotent on message_id (I1). Append-only
        (I5).

        Args:
            message: The message this event is about.
            classification: Its classification.
            now: Current instant, used for the link window and any new application's
                ``created_at``.

        Returns:
            The recorded (or, if already present, the existing) event row.

        Raises:
            StoreError: the write failed; the transaction is rolled back.
        """
        existing = self._event_row_for_message(message.message_id)
        if existing is not None:
            return existing

        application_id: str | None = None
        if classification.event_type != EventType.UNKNOWN:
            candidates = self.match_candidates(
                classification.company_key or "",
                message.thread_id,
                within_days=LINK_WINDOW_DAYS,
            )
            application_id = match_application(
                classification, candidates, message.thread_id, now=now
            )
            if application_id is None:
                application_id = self._create_application(message, classification, now=now)

        try:
            cursor = self._connection.execute(
                """
                INSERT INTO events (application_id, message_id, event_type, occurred_at, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    application_id,
                    message.message_id,
                    str(classification.event_type),
                    _dt_to_iso(message.received_at),
                    _dt_to_iso(now),
                ),
            )
            self._connection.commit()
        except sqlite3.IntegrityError:
            # Another call already recorded this message's event (I1 race, e.g. a retried
            # sync); the UNIQUE(message_id) constraint is the safety net under the
            # pre-check above.
            self._connection.rollback()
            existing_again = self._event_row_for_message(message.message_id)
            if existing_again is not None:
                return existing_again
            raise StoreError(
                f"could not record event for {message.message_id}: unique constraint violated"
            ) from None
        except sqlite3.Error as exc:
            self._connection.rollback()
            raise StoreError(f"could not record event for {message.message_id}: {exc}") from exc

        event_id = cursor.lastrowid
        if event_id is None:
            raise StoreError(f"event insert for {message.message_id} did not return a row id")

        return EventRow(
            event_id=event_id,
            application_id=application_id,
            message_id=message.message_id,
            event_type=classification.event_type,
            occurred_at=message.received_at,
            confidence=classification.confidence,
            needs_review=classification.needs_review,
            is_overridden=False,
            subject=message.subject,
            from_email=message.from_email,
        )

    def _create_application(
        self, message: RawMessage, classification: Classification, *, now: datetime
    ) -> str:
        """Insert a brand-new application and return its generated id."""
        application_id = uuid.uuid4().hex
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO applications (
                    application_id, company, company_key, role, location, ats,
                    applied_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    application_id,
                    classification.company or "Unknown",
                    classification.company_key or "",
                    classification.role,
                    classification.location,
                    classification.ats,
                    _dt_to_iso(message.received_at),
                    _dt_to_iso(now),
                ),
            )
        return application_id

    # --- read side -------------------------------------------------------------------------

    def get_application(self, application_id: str, *, now: datetime) -> ApplicationRow | None:
        """Fetch one application with its derived status.

        Args:
            application_id: The application to fetch.
            now: Current instant, for status derivation and day counts.

        Returns:
            The application, or ``None`` if no such id exists.

        Raises:
            StoreError: the query failed.
        """
        try:
            row = self._connection.execute(
                """
                SELECT application_id, company, company_key, role, location, ats, applied_at
                FROM applications WHERE application_id = ?
                """,
                (application_id,),
            ).fetchone()
        except sqlite3.Error as exc:
            raise StoreError(f"could not fetch application {application_id}: {exc}") from exc
        if row is None:
            return None
        events = self.list_events(application_id)
        return self._build_application_row(row, events, now=now)

    def list_applications(
        self,
        *,
        now: datetime,
        status: ApplicationStatus | None = None,
        company: str | None = None,
        needs_review: bool | None = None,
    ) -> list[ApplicationRow]:
        """All applications with derived status (I4), overrides applied (I6).

        Args:
            now: Current instant, for status derivation and day counts.
            status: If given, only applications whose derived status matches.
            company: If given, matches against ``company_key`` (normalization-insensitive).
            needs_review: If given, only applications whose derived ``needs_review`` matches.

        Returns:
            Matching applications, ordered by ``applied_at`` ascending.

        Raises:
            StoreError: the query failed.
        """
        try:
            rows = self._connection.execute(
                """
                SELECT application_id, company, company_key, role, location, ats, applied_at
                FROM applications
                ORDER BY applied_at ASC
                """
            ).fetchall()
        except sqlite3.Error as exc:
            raise StoreError(f"could not list applications: {exc}") from exc

        results: list[ApplicationRow] = []
        for row in rows:
            if company is not None and row["company_key"] != company:
                continue
            events = self.list_events(row["application_id"])
            app_row = self._build_application_row(row, events, now=now)
            if status is not None and app_row.status != status:
                continue
            if needs_review is not None and app_row.needs_review != needs_review:
                continue
            results.append(app_row)
        return results

    def _build_application_row(
        self, row: sqlite3.Row, events: Sequence[EventRow], *, now: datetime
    ) -> ApplicationRow:
        """Assemble an ``ApplicationRow`` from an ``applications`` row and its events."""
        applied_at = _iso_to_dt(row["applied_at"])
        status = derive_status(events, now=now, ghost_after_days=DEFAULT_GHOST_AFTER_DAYS)

        if events:
            ordered = sorted(events, key=lambda e: e.occurred_at)
            last_event = ordered[-1]
            last_event_at = last_event.occurred_at
            last_event_type = last_event.event_type
            first_response = next(
                (e for e in ordered if e.event_type != EventType.APPLICATION_RECEIVED), None
            )
            days_to_first_response = (
                (first_response.occurred_at - applied_at).days
                if first_response is not None
                else None
            )
        else:
            last_event_at = applied_at
            last_event_type = EventType.UNKNOWN
            days_to_first_response = None

        return ApplicationRow(
            application_id=row["application_id"],
            company=row["company"],
            company_key=row["company_key"],
            role=row["role"],
            location=row["location"],
            ats=row["ats"],
            status=status,
            applied_at=applied_at,
            last_event_at=last_event_at,
            last_event_type=last_event_type,
            event_count=len(events),
            days_to_first_response=days_to_first_response,
            days_since_last_event=(now - last_event_at).days,
            needs_review=any(e.needs_review for e in events),
            source_thread_ids=self._thread_ids_for_application(row["application_id"]),
        )

    def list_events(self, application_id: str | None = None) -> list[EventRow]:
        """Events, oldest first. None returns all, including unlinked UNKNOWNs.

        Args:
            application_id: If given, only events linked to this application.

        Returns:
            Matching events ordered by ``occurred_at`` (then ``event_id`` as a tiebreak).

        Raises:
            StoreError: the query failed.
        """
        query = _EVENT_ROW_QUERY
        params: tuple[Any, ...] = ()
        if application_id is not None:
            query += " WHERE e.application_id = ?"
            params = (application_id,)
        query += " ORDER BY e.occurred_at ASC, e.event_id ASC"
        try:
            rows = self._connection.execute(query, params).fetchall()
        except sqlite3.Error as exc:
            raise StoreError(f"could not list events: {exc}") from exc
        return [_row_to_event(row) for row in rows]

    def _event_row_for_message(self, message_id: str) -> EventRow | None:
        """Fetch the (at most one) event for a message, or None."""
        query = _EVENT_ROW_QUERY + " WHERE e.message_id = ?"
        try:
            row = self._connection.execute(query, (message_id,)).fetchone()
        except sqlite3.Error as exc:
            raise StoreError(f"could not look up event for {message_id}: {exc}") from exc
        return _row_to_event(row) if row is not None else None

    def _thread_ids_for_application(self, application_id: str) -> list[str]:
        """Distinct thread ids of every message linked to this application."""
        try:
            rows = self._connection.execute(
                """
                SELECT DISTINCT m.thread_id AS thread_id
                FROM events e JOIN messages m ON m.message_id = e.message_id
                WHERE e.application_id = ?
                """,
                (application_id,),
            ).fetchall()
        except sqlite3.Error as exc:
            raise StoreError(f"could not fetch thread ids for {application_id}: {exc}") from exc
        return sorted({row["thread_id"] for row in rows})

    def match_candidates(
        self, company_key: str, thread_id: str, *, within_days: int
    ) -> list[ApplicationMatchCandidate]:
        """Fetch linking candidates: same thread_id, or same company_key within the window.

        ``within_days`` is accepted per CONTRACTS.md but this method has no injected clock —
        its signature is frozen without a ``now`` parameter — so it cannot itself evaluate
        "within the window" against the current instant. It returns the safe superset (same
        thread, or same company_key, with no time filter), and the authoritative window check
        happens in ``linker.match_application``, which does receive ``now``. See the final
        report for this flagged contract gap.

        Args:
            company_key: The incoming message's normalized company key (may be ``""``).
            thread_id: The incoming message's Gmail thread id.
            within_days: Accepted for contract compatibility; not applied as a SQL filter here.

        Returns:
            Candidate applications, each with its distinct thread ids.

        Raises:
            StoreError: the query failed.
        """
        del within_days  # see docstring: no clock available to apply it here.
        try:
            rows = self._connection.execute(
                """
                SELECT
                    a.application_id AS application_id,
                    a.company_key AS company_key,
                    a.role AS role,
                    a.applied_at AS applied_at,
                    (
                        SELECT MAX(e2.occurred_at) FROM events e2
                        WHERE e2.application_id = a.application_id
                    ) AS last_event_at
                FROM applications a
                WHERE a.company_key = ?
                   OR EXISTS (
                        SELECT 1 FROM events e
                        JOIN messages m ON m.message_id = e.message_id
                        WHERE e.application_id = a.application_id AND m.thread_id = ?
                   )
                """,
                (company_key, thread_id),
            ).fetchall()
        except sqlite3.Error as exc:
            raise StoreError(f"could not fetch match candidates: {exc}") from exc

        candidates: list[ApplicationMatchCandidate] = []
        for row in rows:
            applied_at = _iso_to_dt(row["applied_at"])
            last_event_at_raw = row["last_event_at"]
            last_event_at = (
                _iso_to_dt(last_event_at_raw) if last_event_at_raw is not None else applied_at
            )
            candidates.append(
                ApplicationMatchCandidate(
                    application_id=row["application_id"],
                    company_key=row["company_key"],
                    role=row["role"],
                    thread_ids=self._thread_ids_for_application(row["application_id"]),
                    applied_at=applied_at,
                    last_event_at=last_event_at,
                )
            )
        return candidates

    # --- review / overrides ------------------------------------------------------------------

    def pending_review(self, limit: int | None = None) -> list[ReviewItem]:
        """Messages flagged needs_review that have no override yet, oldest first.

        Args:
            limit: Maximum number of items to return.

        Returns:
            Review items, oldest message first.

        Raises:
            StoreError: the query failed.
        """
        query = """
            SELECT
                m.message_id AS message_id, m.thread_id AS thread_id,
                m.received_at AS received_at, m.from_email AS from_email,
                m.from_name AS from_name, m.to_email AS to_email, m.subject AS subject,
                m.body_text AS body_text, m.snippet AS snippet,
                m.labels_json AS labels_json, m.headers_json AS headers_json,
                c.event_type AS event_type, c.company AS company,
                c.company_key AS company_key, c.role AS role, c.location AS location,
                c.ats AS ats, c.confidence AS confidence, c.needs_review AS needs_review,
                c.evidence_json AS evidence_json, c.classifier_name AS classifier_name,
                c.classifier_version AS classifier_version, e.application_id AS application_id
            FROM classifications c
            JOIN messages m ON m.message_id = c.message_id
            LEFT JOIN overrides o ON o.message_id = c.message_id
            LEFT JOIN events e ON e.message_id = c.message_id
            WHERE c.needs_review = 1 AND o.message_id IS NULL
            ORDER BY m.received_at ASC
        """
        params: tuple[Any, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (limit,)
        try:
            rows = self._connection.execute(query, params).fetchall()
        except sqlite3.Error as exc:
            raise StoreError(f"could not list pending review: {exc}") from exc

        items: list[ReviewItem] = []
        for row in rows:
            message = RawMessage(
                message_id=row["message_id"],
                thread_id=row["thread_id"],
                received_at=_iso_to_dt(row["received_at"]),
                from_email=row["from_email"],
                from_name=row["from_name"],
                to_email=row["to_email"],
                subject=row["subject"],
                body_text=row["body_text"],
                snippet=row["snippet"],
                labels=_loads_list(row["labels_json"]),
                headers=_loads_dict(row["headers_json"]),
            )
            classification = Classification(
                message_id=row["message_id"],
                event_type=EventType(row["event_type"]),
                company=row["company"],
                company_key=row["company_key"],
                role=row["role"],
                location=row["location"],
                ats=row["ats"],
                confidence=float(row["confidence"]),
                needs_review=bool(row["needs_review"]),
                evidence=_loads_list(row["evidence_json"]),
                classifier_name=row["classifier_name"],
                classifier_version=row["classifier_version"],
            )
            items.append(
                ReviewItem(
                    message=message,
                    classification=classification,
                    suggested_application_id=row["application_id"],
                )
            )
        return items

    def set_override(self, override: Override) -> None:
        """Record a human correction and re-derive affected applications.

        Upsert on message_id. Survives reclassify (I6). "Re-derive" happens implicitly: since
        status is never stored (I4), the next read of the affected application already
        reflects this override — no extra write is needed here.

        Args:
            override: The human correction.

        Raises:
            StoreError: the write failed.
        """
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO overrides (message_id, event_type, company, role, corrected_at, note)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(message_id) DO UPDATE SET
                    event_type = excluded.event_type,
                    company = excluded.company,
                    role = excluded.role,
                    corrected_at = excluded.corrected_at,
                    note = excluded.note
                """,
                (
                    override.message_id,
                    str(override.event_type) if override.event_type is not None else None,
                    override.company,
                    override.role,
                    _dt_to_iso(override.corrected_at),
                    override.note,
                ),
            )

    def accept_classification(self, message_id: str, *, now: datetime) -> None:
        """Confirm the classifier was right: clears needs_review without altering fields.

        Still recorded as labeled data for future eval: sets ``reviewed_at`` so
        ``clear_classifications(only_unreviewed=True)`` will preserve this row across a
        reclassify.

        Args:
            message_id: The message whose classification is confirmed.
            now: Current instant, recorded as the review timestamp.

        Raises:
            StoreError: the write failed, or no classification exists for this message.
        """
        with self._transaction() as conn:
            cursor = conn.execute(
                "UPDATE classifications SET needs_review = 0, reviewed_at = ? WHERE message_id = ?",
                (_dt_to_iso(now), message_id),
            )
            changed = cursor.rowcount
        if changed == 0:
            raise StoreError(f"no classification found for message {message_id}")

    def clear_classifications(self, *, only_unreviewed: bool = True) -> int:
        """Drop stored classifications ahead of a reclassify. Returns rows cleared.

        Never deletes messages, applications, or overrides.

        Args:
            only_unreviewed: If True (default), only clear classifications nobody has
                confirmed via ``accept_classification`` yet (``reviewed_at IS NULL``) — data
                a human has already labeled is preserved for future eval. If False, clear
                every classification.

        Returns:
            The number of classification rows deleted.

        Raises:
            StoreError: the write failed.
        """
        query = "DELETE FROM classifications"
        if only_unreviewed:
            query += " WHERE reviewed_at IS NULL"
        with self._transaction() as conn:
            cursor = conn.execute(query)
            cleared = cursor.rowcount
        return cleared

    # --- sync state ----------------------------------------------------------------------

    def get_cursor(self, source: str) -> str | None:
        """Fetch the persisted sync cursor for a source.

        Args:
            source: The ``EmailSource.name`` this cursor belongs to.

        Returns:
            The opaque cursor, or ``None`` if never synced.

        Raises:
            StoreError: the query failed.
        """
        try:
            row = self._connection.execute(
                "SELECT cursor FROM sync_state WHERE source = ?", (source,)
            ).fetchone()
        except sqlite3.Error as exc:
            raise StoreError(f"could not read cursor for {source}: {exc}") from exc
        return str(row["cursor"]) if row is not None and row["cursor"] is not None else None

    def set_cursor(self, source: str, cursor: str | None, *, synced_at: datetime) -> None:
        """Persist the cursor. Callers must only call this AFTER the batch commits (I9).

        Args:
            source: The ``EmailSource.name`` this cursor belongs to.
            cursor: The opaque cursor value, or ``None`` to clear it.
            synced_at: When this sync completed.

        Raises:
            StoreError: the write failed.
        """
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO sync_state (source, cursor, last_synced_at)
                VALUES (?, ?, ?)
                ON CONFLICT(source) DO UPDATE SET
                    cursor = excluded.cursor,
                    last_synced_at = excluded.last_synced_at
                """,
                (source, cursor, _dt_to_iso(synced_at)),
            )


__all__ = ["DEFAULT_GHOST_AFTER_DAYS", "Store"]
