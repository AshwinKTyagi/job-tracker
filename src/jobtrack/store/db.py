"""Connection handling, migrations, and the repository API.

:class:`Store` is the only writer of SQLite and the only object in the project that knows
SQL. Nothing below leaks a ``sqlite3`` exception: every driver error is wrapped in a
:class:`~jobtrack.errors.StoreError` (or :class:`~jobtrack.errors.MigrationError`) at the
method boundary.

The statements themselves and the row-to-model mapping live in :mod:`jobtrack.store.repo`;
the matching and status rules live in :mod:`jobtrack.store.linker`. This module owns the
connection, the transaction boundaries, and the public surface frozen in ``CONTRACTS.md``
§6.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

from jobtrack.config import Config
from jobtrack.errors import MigrationError, StoreError
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
from jobtrack.store import repo
from jobtrack.store.linker import LINK_WINDOW_DAYS, match_application

logger = logging.getLogger(__name__)

SCHEMA_VERSION: int = 1
"""The schema version this build expects. Bump it when a migration is added."""

DEFAULT_GHOST_AFTER_DAYS: Final[int] = 30
"""Mirrors ``StoreConfig.ghost_after_days``.

``Store.open`` takes only a path (CONTRACTS.md §6), so a store opened that way uses this
default. ``Store.open_from_config`` — or assigning ``store.ghost_after_days`` — threads the
configured value through instead.
"""

MIGRATIONS_DIR: Final[Path] = Path(__file__).resolve().parent / "migrations"
"""Directory holding ``NNNN_name.sql`` migration files, applied in numeric order."""

MIGRATION_FILENAME_RE: Final[re.Pattern[str]] = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")
"""A migration filename: four-digit zero-padded version, underscore, lowercase slug."""

_TABLE_EXISTS_SQL: Final[str] = "SELECT name AS name FROM sqlite_master WHERE type = ? AND name = ?"
_PRAGMA_FOREIGN_KEYS: Final[str] = "PRAGMA foreign_keys = ON"
_PRAGMA_WAL: Final[str] = "PRAGMA journal_mode = WAL"
_PROBE_SQL: Final[str] = "SELECT count(*) AS objects FROM sqlite_master"
_APPLICATION_ID_PREFIX: Final[str] = "app_"
_APPLICATION_ID_LENGTH: Final[int] = 16
_ID_FIELD_SEPARATOR: Final[str] = "\x1f"
_NON_ALNUM_RE: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")


def migration_files() -> list[tuple[int, Path]]:
    """List the migration files, ascending by version.

    Returns:
        ``(version, path)`` pairs sorted by version.

    Raises:
        MigrationError: the directory is missing, or two files claim one version.
    """
    if not MIGRATIONS_DIR.is_dir():
        raise MigrationError(f"migrations directory is missing: {MIGRATIONS_DIR}")
    found: dict[int, Path] = {}
    for path in sorted(MIGRATIONS_DIR.iterdir()):
        if path.suffix != ".sql":
            continue
        matched = MIGRATION_FILENAME_RE.match(path.name)
        if matched is None:
            raise MigrationError(f"migration filename is not NNNN_name.sql: {path.name}")
        version = int(matched.group(1))
        if version in found:
            raise MigrationError(f"duplicate migration version {version}: {path.name}")
        found[version] = path
    return sorted(found.items())


def _application_id(company_key: str, role: str | None, seed_message_id: str) -> str:
    """Derive a stable application id from its company, role, and seeding message."""
    material = _ID_FIELD_SEPARATOR.join((company_key, role or "", seed_message_id))
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return f"{_APPLICATION_ID_PREFIX}{digest[:_APPLICATION_ID_LENGTH]}"


def _company_filter_matches(company_key: str, needle: str) -> bool:
    """True when a user-supplied company string selects this company_key."""
    folded = _NON_ALNUM_RE.sub("", needle.casefold())
    target = _NON_ALNUM_RE.sub("", company_key.casefold())
    if not folded:
        return True
    return folded in target


class Store:
    """The only writer of SQLite. The only module that knows SQL.

    Tables: messages, classifications, applications, events, overrides, sync_state,
    schema_version. Parameterized queries only; every column named explicitly.
    """

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        ghost_after_days: int = DEFAULT_GHOST_AFTER_DAYS,
    ) -> None:
        """Wrap an already-configured connection.

        Prefer :meth:`open` or :meth:`open_from_config`; this constructor exists so tests
        can hand in an in-tmp_path connection they built themselves.

        Args:
            connection: A connection with ``row_factory`` set to ``sqlite3.Row``.
            ghost_after_days: Silence, in days, after which an application reads as GHOSTED.
        """
        self._conn = connection
        self.ghost_after_days = ghost_after_days

    # --- lifecycle ----------------------------------------------------------

    @classmethod
    def open(cls, path: Path) -> Store:
        """Open (creating parent dirs as needed) with foreign_keys=ON and WAL enabled.

        Args:
            path: Location of the SQLite file. Parent directories are created.

        Returns:
            An open store. Call :meth:`migrate` before using it.

        Raises:
            StoreError: the file exists but is not a readable jobtrack database, or the
                parent directory could not be created.
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StoreError(f"could not create {path.parent}: {exc}") from exc
        try:
            connection = sqlite3.connect(path)
            connection.row_factory = sqlite3.Row
            connection.execute(_PRAGMA_FOREIGN_KEYS)
            connection.execute(_PRAGMA_WAL)
            connection.execute(_PROBE_SQL).fetchone()
        except sqlite3.Error as exc:
            raise StoreError(f"{path} is not a readable jobtrack database: {exc}") from exc
        return cls(connection)

    @classmethod
    def open_from_config(cls, config: Config) -> Store:
        """Open the database named by a Config, carrying its ``ghost_after_days`` through.

        Args:
            config: The resolved runtime configuration.

        Returns:
            An open store. Call :meth:`migrate` before using it.

        Raises:
            StoreError: the database could not be opened.
        """
        store = cls.open(config.db_path)
        store.ghost_after_days = config.store.ghost_after_days
        return store

    def migrate(self) -> None:
        """Apply pending migrations in order, each in its own transaction.

        Raises:
            MigrationError: a migration failed; the database is left at its prior version.
        """
        current = self.schema_version()
        for version, path in migration_files():
            if version <= current:
                continue
            try:
                script = path.read_text(encoding="utf-8")
            except OSError as exc:
                raise MigrationError(f"could not read migration {path.name}: {exc}") from exc
            try:
                # executescript() commits first, so the BEGIN has to live inside the
                # script for the DDL and the version bump to land atomically.
                self._conn.executescript(f"BEGIN;\n{script}")
                self._conn.execute(repo.INSERT_SCHEMA_VERSION, (version,))
                self._conn.commit()
            except sqlite3.Error as exc:
                self._conn.rollback()
                raise MigrationError(f"migration {path.name} failed: {exc}") from exc
            logger.info("applied migration %s", path.name)

    def schema_version(self) -> int:
        """Report the highest applied migration version.

        Returns:
            The applied version, or 0 for a database with no schema yet.

        Raises:
            StoreError: the version could not be read.
        """
        try:
            present = self._conn.execute(_TABLE_EXISTS_SQL, ("table", "schema_version")).fetchone()
            if present is None:
                return 0
            row = self._conn.execute(repo.SELECT_SCHEMA_VERSION).fetchone()
        except sqlite3.Error as exc:
            raise StoreError(f"could not read schema version: {exc}") from exc
        if row is None or row["version"] is None:
            return 0
        return int(row["version"])

    def close(self) -> None:
        """Close the underlying connection. Safe to call more than once."""
        try:
            self._conn.close()
        except sqlite3.Error as exc:  # pragma: no cover - close rarely fails
            raise StoreError(f"could not close the database: {exc}") from exc

    def __enter__(self) -> Store:
        """Return self so ``with Store.open(path) as store`` works."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the connection on the way out of the ``with`` block."""
        self.close()

    # --- ingest side -------------------------------------------------------

    def has_message(self, message_id: str) -> bool:
        """Dedupe check (I1). Cheap — indexed primary key lookup.

        Args:
            message_id: The Gmail message id.

        Returns:
            True when the message has already been recorded.

        Raises:
            StoreError: the lookup failed.
        """
        return self._query_one(repo.SELECT_MESSAGE_EXISTS, (message_id,)) is not None

    def record_message(self, message: RawMessage) -> None:
        """Persist raw metadata. Idempotent: a second call for the same id is a no-op.

        Args:
            message: The normalized email to persist.

        Raises:
            StoreError: the write failed, or ``received_at`` is naive (I7).
        """
        self._write(repo.INSERT_MESSAGE, repo.message_params(message))

    def record_classification(self, classification: Classification) -> None:
        """Upsert on message_id.

        Replaces a prior classification (reclassify); never touches overrides (I6).

        Args:
            classification: The classifier output to persist.

        Raises:
            StoreError: the write failed, e.g. the message has not been recorded.
        """
        self._write(repo.UPSERT_CLASSIFICATION, repo.classification_params(classification))

    def link_and_record_event(
        self, message: RawMessage, classification: Classification, *, now: datetime
    ) -> EventRow:
        """Link the message to an application (creating one if needed) and append its event.

        Fetches candidates, delegates the decision to ``linker.match_application``, and
        creates a new application when there is no match. UNKNOWN classifications — and
        classifications with no ``company_key`` to match on — are recorded with
        ``application_id=None``. Idempotent on message_id (I1): a replay returns the
        existing event and writes nothing. Append-only (I5).

        Args:
            message: The message the event came from.
            classification: Its classification.
            now: Injected tz-aware UTC clock.

        Returns:
            The stored event, with any override already applied.

        Raises:
            StoreError: the write failed; the transaction is rolled back.
        """
        existing = self._event_record_for_message(message.message_id)
        if existing is not None:
            logger.debug("message %s already has an event; replay is a no-op", message.message_id)
            return existing.row

        try:
            self._conn.execute(repo.INSERT_MESSAGE, repo.message_params(message))
            self._conn.execute(
                repo.UPSERT_CLASSIFICATION, repo.classification_params(classification)
            )
            application_id = self._resolve_application(message, classification, now=now)
            self._conn.execute(
                repo.INSERT_EVENT,
                (
                    application_id,
                    message.message_id,
                    str(classification.event_type),
                    repo.to_iso(message.received_at),
                    repo.to_iso(now),
                ),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            self._conn.rollback()
            raise StoreError(f"could not record event for {message.message_id}: {exc}") from exc
        except StoreError:
            self._conn.rollback()
            raise

        recorded = self._event_record_for_message(message.message_id)
        if recorded is None:  # pragma: no cover - the insert above guarantees a row
            raise StoreError(f"event for {message.message_id} vanished after commit")
        return recorded.row

    # --- read side ---------------------------------------------------------

    def get_application(self, application_id: str, *, now: datetime) -> ApplicationRow | None:
        """Read one application with its derived status.

        Args:
            application_id: The application to read.
            now: Injected tz-aware UTC clock.

        Returns:
            The derived row, or None when no such application exists.

        Raises:
            StoreError: the read failed.
        """
        row = self._query_one(repo.SELECT_APPLICATION_CORE, (application_id,))
        if row is None:
            return None
        records = self._event_records(
            repo.WHERE_EVENTS_FOR_APPLICATION, (application_id,)
        )
        if not records:
            return None
        return repo.build_application_row(
            repo.row_to_application_core(row),
            records,
            now=now,
            ghost_after_days=self.ghost_after_days,
        )

    def list_applications(
        self,
        *,
        now: datetime,
        status: ApplicationStatus | None = None,
        company: str | None = None,
        needs_review: bool | None = None,
    ) -> list[ApplicationRow]:
        """All applications with derived status (I4), overrides applied (I6).

        ``company`` matches against ``company_key``, so it is normalization-insensitive:
        the needle is casefolded, stripped of punctuation, and matched as a substring.

        Args:
            now: Injected tz-aware UTC clock.
            status: Keep only applications with this derived status.
            company: Keep only applications whose company_key contains this needle.
            needs_review: Keep only applications with (or without) a review flag.

        Returns:
            Matching applications, ordered by application_id.

        Raises:
            StoreError: the read failed.
        """
        cores = [
            repo.row_to_application_core(row)
            for row in self._query_all(repo.SELECT_APPLICATION_CORES, ())
        ]
        grouped: dict[str, list[repo.EventRecord]] = defaultdict(list)
        for record in self._event_records(repo.WHERE_EVENTS_LINKED, ()):
            if record.row.application_id is not None:
                grouped[record.row.application_id].append(record)

        rows: list[ApplicationRow] = []
        for core in cores:
            records = grouped.get(core.application_id)
            if not records:
                logger.debug("skipping application %s: no events", core.application_id)
                continue
            if company is not None and not _company_filter_matches(core.company_key, company):
                continue
            row = repo.build_application_row(
                core, records, now=now, ghost_after_days=self.ghost_after_days
            )
            if status is not None and row.status is not status:
                continue
            if needs_review is not None and row.needs_review is not needs_review:
                continue
            rows.append(row)
        return rows

    def list_events(self, application_id: str | None = None) -> list[EventRow]:
        """Events, oldest first. None returns all, including unlinked UNKNOWNs.

        Args:
            application_id: Restrict to one application, or None for everything.

        Returns:
            The events, with overrides applied (I6).

        Raises:
            StoreError: the read failed.
        """
        if application_id is None:
            return [record.row for record in self._event_records("", ())]
        return [
            record.row
            for record in self._event_records(
                repo.WHERE_EVENTS_FOR_APPLICATION, (application_id,)
            )
        ]

    def match_candidates(
        self, company_key: str, thread_id: str, *, within_days: int
    ) -> list[ApplicationMatchCandidate]:
        """Fetch linking candidates: same thread_id, or same company_key within the window.

        The store has no clock of its own, so ``within_days`` is applied as a coarse
        pre-filter anchored on the newest event in the database. That is always a superset
        of the window the pure linker enforces against the injected ``now``, so no valid
        match is dropped. Thread matches bypass the window entirely.

        Args:
            company_key: The normalized company key to match on (I8).
            thread_id: The thread the message arrived on.
            within_days: How far back a company_key match may reach.

        Returns:
            Candidates ordered by application_id, ready for ``linker.match_application``.

        Raises:
            StoreError: the read failed.
        """
        rows = self._query_all(repo.SELECT_CANDIDATE_APPLICATIONS, (company_key, thread_id))
        if not rows:
            return []
        wanted = {str(row["application_id"]) for row in rows}

        threads: dict[str, list[str]] = defaultdict(list)
        for link in self._query_all(repo.SELECT_APPLICATION_THREADS, ()):
            application_id = str(link["application_id"])
            if application_id in wanted:
                threads[application_id].append(str(link["thread_id"]))

        newest_row = self._query_one(repo.SELECT_NEWEST_EVENT_AT, ())
        anchor = (
            repo.from_iso(newest_row["newest"])
            if newest_row is not None and newest_row["newest"] is not None
            else None
        )

        candidates: list[ApplicationMatchCandidate] = []
        for row in rows:
            application_id = str(row["application_id"])
            thread_ids = threads.get(application_id, [])
            last_event_at = repo.from_iso(row["last_event_at"])
            stale = anchor is not None and anchor - last_event_at > timedelta(days=within_days)
            if stale and thread_id not in thread_ids:
                continue
            candidates.append(
                ApplicationMatchCandidate(
                    application_id=application_id,
                    company_key=row["company_key"],
                    role=row["role"],
                    thread_ids=thread_ids,
                    applied_at=repo.from_iso(row["applied_at"]),
                    last_event_at=last_event_at,
                )
            )
        return candidates

    # --- review / overrides -------------------------------------------------

    def pending_review(self, limit: int | None = None) -> list[ReviewItem]:
        """Messages flagged needs_review that have no override yet, oldest first.

        Args:
            limit: Maximum number of items to return, or None for all of them.

        Returns:
            The review queue, oldest message first.

        Raises:
            StoreError: the read failed, or a queued message lost its classification.
        """
        sql = repo.SELECT_PENDING_REVIEW
        params: tuple[object, ...] = ()
        if limit is not None:
            sql = f"{sql}{repo.LIMIT_CLAUSE}"
            params = (limit,)

        items: list[ReviewItem] = []
        for row in self._query_all(sql, params):
            message_id = str(row["message_id"])
            message_row = self._query_one(repo.SELECT_MESSAGE, (message_id,))
            classification_row = self._query_one(repo.SELECT_CLASSIFICATION, (message_id,))
            if message_row is None or classification_row is None:  # pragma: no cover
                raise StoreError(f"review queue references incomplete message {message_id}")
            suggested = row["application_id"]
            items.append(
                ReviewItem(
                    message=repo.row_to_message(message_row),
                    classification=repo.row_to_classification(classification_row),
                    suggested_application_id=None if suggested is None else str(suggested),
                )
            )
        return items

    def set_override(self, override: Override) -> None:
        """Record a human correction and re-derive the affected application.

        Upsert on message_id. Survives reclassify (I6). The event row itself is never
        mutated (I5) — only its application link is revisited, because a correction can
        turn an UNKNOWN into a real event or vice versa.

        Args:
            override: The correction to record.

        Raises:
            StoreError: the write failed; the transaction is rolled back.
        """
        try:
            self._conn.execute(repo.UPSERT_OVERRIDE, repo.override_params(override))
            self._relink_after_override(override)
            self._conn.execute(repo.DELETE_ORPHAN_APPLICATIONS)
            self._conn.commit()
        except sqlite3.Error as exc:
            self._conn.rollback()
            raise StoreError(f"could not record override for {override.message_id}: {exc}") from exc
        except StoreError:
            self._conn.rollback()
            raise

    def accept_classification(self, message_id: str, *, now: datetime) -> None:
        """Confirm the classifier was right: clears needs_review without altering fields.

        The acceptance is retained as labeled data — an override row whose correction
        columns are all NULL, which changes nothing on read but records that a human
        looked at this message.

        Args:
            message_id: The message being accepted.
            now: Injected tz-aware UTC clock, stored as the correction time.

        Raises:
            StoreError: no classification is recorded for that message, or the write failed.
        """
        try:
            cursor = self._conn.execute(repo.MARK_CLASSIFICATION_REVIEWED, (message_id,))
            if cursor.rowcount == 0:
                raise StoreError(f"no classification recorded for message {message_id!r}")
            self._conn.execute(
                repo.INSERT_ACCEPTANCE, (message_id, repo.to_iso(now), repo.ACCEPTED_NOTE)
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            self._conn.rollback()
            raise StoreError(f"could not accept classification {message_id}: {exc}") from exc
        except StoreError:
            self._conn.rollback()
            raise

    def clear_classifications(self, *, only_unreviewed: bool = True) -> int:
        """Drop stored classifications ahead of a reclassify.

        Never deletes messages, applications, or overrides. With ``only_unreviewed`` the
        rows a human already touched — corrected or accepted — are left alone, so their
        labels survive (I6).

        Args:
            only_unreviewed: Keep classifications that have an override row.

        Returns:
            The number of rows cleared.

        Raises:
            StoreError: the delete failed.
        """
        sql = (
            repo.CLEAR_CLASSIFICATIONS_UNREVIEWED
            if only_unreviewed
            else repo.CLEAR_CLASSIFICATIONS_ALL
        )
        try:
            cursor = self._conn.execute(sql)
            cleared = cursor.rowcount
            self._conn.commit()
        except sqlite3.Error as exc:
            self._conn.rollback()
            raise StoreError(f"could not clear classifications: {exc}") from exc
        logger.info("cleared %d classification(s)", cleared)
        return cleared

    # --- sync state ---------------------------------------------------------

    def get_cursor(self, source: str) -> str | None:
        """Read the stored sync cursor for a source.

        Args:
            source: The ``EmailSource.name`` the cursor belongs to.

        Returns:
            The opaque cursor, or None when the source has never synced.

        Raises:
            StoreError: the read failed.
        """
        row = self._query_one(repo.SELECT_CURSOR, (source,))
        if row is None or row["cursor"] is None:
            return None
        return str(row["cursor"])

    def set_cursor(self, source: str, cursor: str | None, *, synced_at: datetime) -> None:
        """Persist the cursor. Callers must only call this AFTER the batch commits (I9).

        Args:
            source: The ``EmailSource.name`` the cursor belongs to.
            cursor: The opaque provider cursor, or None to clear it.
            synced_at: Injected tz-aware UTC clock.

        Raises:
            StoreError: the write failed, or ``synced_at`` is naive (I7).
        """
        self._write(repo.UPSERT_CURSOR, (source, cursor, repo.to_iso(synced_at)))

    # --- internals ----------------------------------------------------------

    def _resolve_application(
        self, message: RawMessage, classification: Classification, *, now: datetime
    ) -> str | None:
        """Find or create the application this classified message belongs to."""
        if classification.event_type is EventType.UNKNOWN:
            return None
        company_key = classification.company_key
        if company_key is None:
            logger.debug("message %s has no company_key; left unlinked", message.message_id)
            return None

        candidates = self.match_candidates(
            company_key, message.thread_id, within_days=LINK_WINDOW_DAYS
        )
        matched = match_application(classification, candidates, message.thread_id, now=now)
        if matched is not None:
            self._conn.execute(
                repo.ENRICH_APPLICATION,
                (
                    classification.role,
                    classification.location,
                    classification.ats,
                    repo.to_iso(message.received_at),
                    matched,
                ),
            )
            return matched

        created = _application_id(company_key, classification.role, message.message_id)
        self._conn.execute(
            repo.INSERT_APPLICATION,
            (
                created,
                classification.company or company_key,
                company_key,
                classification.role,
                classification.location,
                classification.ats,
                repo.to_iso(message.received_at),
                repo.to_iso(now),
            ),
        )
        logger.debug("created application %s for %s", created, company_key)
        return created

    def _relink_after_override(self, override: Override) -> None:
        """Revisit one event's application link after a correction, without mutating it."""
        record = self._event_record_for_message(override.message_id)
        if record is None:
            return
        if record.row.event_type is EventType.UNKNOWN:
            self._conn.execute(repo.RELINK_EVENT, (None, override.message_id))
            return
        if record.row.application_id is not None:
            return

        message_row = self._query_one(repo.SELECT_MESSAGE, (override.message_id,))
        classification_row = self._query_one(repo.SELECT_CLASSIFICATION, (override.message_id,))
        if message_row is None or classification_row is None:
            return
        message = repo.row_to_message(message_row)
        update: dict[str, object] = {"event_type": record.row.event_type}
        if override.role is not None:
            update["role"] = override.role
        if override.company is not None:
            update["company"] = override.company
        corrected = repo.row_to_classification(classification_row).model_copy(update=update)
        application_id = self._resolve_application(message, corrected, now=override.corrected_at)
        self._conn.execute(repo.RELINK_EVENT, (application_id, override.message_id))

    def _event_records(
        self, where: str, params: tuple[object, ...]
    ) -> list[repo.EventRecord]:
        """Run the event projection with a fixed WHERE fragment and map every row."""
        sql = f"{repo.SELECT_EVENT_RECORDS}{where}{repo.ORDER_EVENTS}"
        return [repo.row_to_event_record(row) for row in self._query_all(sql, params)]

    def _event_record_for_message(self, message_id: str) -> repo.EventRecord | None:
        """The single event recorded for a message, or None if it has none."""
        records = self._event_records(repo.WHERE_EVENTS_FOR_MESSAGE, (message_id,))
        return records[0] if records else None

    def _query_all(self, sql: str, params: tuple[object, ...]) -> list[sqlite3.Row]:
        """Run a SELECT and return every row, wrapping driver errors."""
        try:
            return self._conn.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            raise StoreError(f"query failed: {exc}") from exc

    def _query_one(self, sql: str, params: tuple[object, ...]) -> sqlite3.Row | None:
        """Run a SELECT and return the first row, wrapping driver errors."""
        try:
            return self._conn.execute(sql, params).fetchone()
        except sqlite3.Error as exc:
            raise StoreError(f"query failed: {exc}") from exc

    def _write(self, sql: str, params: tuple[object, ...]) -> None:
        """Run one statement in its own transaction, wrapping driver errors."""
        try:
            self._conn.execute(sql, params)
            self._conn.commit()
        except sqlite3.Error as exc:
            self._conn.rollback()
            raise StoreError(f"write failed: {exc}") from exc


__all__ = ["DEFAULT_GHOST_AFTER_DAYS", "MIGRATIONS_DIR", "SCHEMA_VERSION", "Store"]
