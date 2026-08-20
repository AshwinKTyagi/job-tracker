"""Connection handling for the store: opening, migrating, and closing the SQLite database.

Owns the low-level plumbing (foreign_keys/WAL pragmas, transaction wrapping) that
``repo.py``'s ``Store`` methods build on. Split out per PLAN.md's db.py/repo.py division:
this module defines ``_ConnectionMixin``, which ``repo.py``'s ``Store`` class inherits so
that ``Store.open`` / ``Store.migrate`` / ``Store.close`` / ``__enter__`` / ``__exit__``
live here while the query surface lives in ``repo.py``.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Self

from jobtrack.errors import MigrationError, StoreError

logger = logging.getLogger(__name__)

SCHEMA_VERSION: Final[int] = 1

MIGRATIONS_DIR: Final[Path] = Path(__file__).resolve().parent / "migrations"

_MIGRATION_NAME_RE: Final[re.Pattern[str]] = re.compile(r"^(?P<version>\d{4})_.+\.sql$")

_KNOWN_TABLES: Final[frozenset[str]] = frozenset(
    {
        "messages",
        "classifications",
        "applications",
        "events",
        "overrides",
        "sync_state",
        "schema_version",
    }
)


def _bookkeeping_now_iso() -> str:
    """Return the current UTC instant as ISO-8601, for internal audit columns only.

    CONTRACTS.md freezes ``Store.migrate()`` (and, in repo.py, ``record_message`` and
    ``record_classification``) without an injected ``now: datetime`` parameter, even though
    CLAUDE.md forbids calling the wall clock inside store logic. This is the one narrow,
    documented exception: it is used ONLY for bookkeeping columns
    (``schema_version.applied_at``, ``messages.ingested_at``, ``classifications.classified_at``)
    that no method in the frozen ``Store`` API ever reads back. Every column that feeds a
    contract-visible model (``EventRow``, ``ApplicationRow``, ...) is populated from an
    explicitly injected ``now`` instead. See the final report for the full rationale.
    """
    return datetime.now(UTC).isoformat()


def _pending_migrations(current_version: int) -> list[tuple[int, Path]]:
    """List migration files newer than ``current_version``, sorted ascending by version."""
    pending: list[tuple[int, Path]] = []
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        match = _MIGRATION_NAME_RE.match(path.name)
        if match is None:
            continue
        version = int(match.group("version"))
        if version > current_version:
            pending.append((version, path))
    pending.sort(key=lambda item: item[0])
    return pending


class _ConnectionMixin:
    """Connection lifecycle and low-level SQLite plumbing shared by :class:`Store`.

    ``repo.py`` defines the public ``Store`` class as ``class Store(_ConnectionMixin)``,
    adding the repository/query methods. This mixin owns everything CONTRACTS.md assigns to
    db.py: opening the connection, running migrations, closing it, and the context-manager
    protocol, plus a small ``_transaction`` helper repo.py uses to keep every multi-statement
    write atomic and to translate ``sqlite3.Error`` into ``StoreError`` at the boundary.
    """

    _connection: sqlite3.Connection
    _path: Path

    def __init__(self, connection: sqlite3.Connection, path: Path) -> None:
        """Wrap an already-configured SQLite connection.

        Args:
            connection: An open sqlite3 connection with foreign_keys and WAL already set.
            path: The database file path, kept for diagnostics and error messages.
        """
        self._connection = connection
        self._path = path

    @classmethod
    def open(cls, path: Path) -> Self:
        """Open (creating parent dirs as needed) with foreign_keys=ON and WAL enabled.

        Args:
            path: Filesystem path to the SQLite database file.

        Returns:
            An opened store, ready for ``migrate()``.

        Raises:
            StoreError: the file exists but is not a readable jobtrack database, or the
                connection could not be established.
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StoreError(f"could not create directory for {path}: {exc}") from exc

        try:
            connection = sqlite3.connect(str(path))
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
        except sqlite3.Error as exc:
            raise StoreError(f"could not open database at {path}: {exc}") from exc

        store = cls(connection, path)
        store._verify_readable()
        return store

    def _verify_readable(self) -> None:
        """Confirm the file is either empty or an actual jobtrack database.

        Raises:
            StoreError: the file has content that is not a jobtrack schema, or is corrupt.
        """
        try:
            rows = self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            self._connection.close()
            raise StoreError(f"{self._path} is not a readable jobtrack database: {exc}") from exc
        table_names = {row["name"] for row in rows}
        if table_names and not (table_names & _KNOWN_TABLES):
            self._connection.close()
            raise StoreError(f"{self._path} exists but is not a jobtrack database")

    def _current_schema_version(self) -> int:
        """Return the highest applied schema version, or 0 if unmigrated.

        Raises:
            StoreError: the version could not be read.
        """
        try:
            exists = self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'schema_version'"
            ).fetchone()
            if exists is None:
                return 0
            row = self._connection.execute(
                "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
            ).fetchone()
            return int(row["version"]) if row is not None else 0
        except sqlite3.Error as exc:
            raise StoreError(f"could not read schema version of {self._path}: {exc}") from exc

    def migrate(self) -> None:
        """Apply pending migrations in order, in a transaction.

        Each migration file is applied and recorded as its own commit, so a failure leaves
        the database at the version of the last migration that committed successfully.

        Raises:
            MigrationError: a migration failed; the DB is left at its prior version.
        """
        current = self._current_schema_version()
        for version, path in _pending_migrations(current):
            sql_text = path.read_text(encoding="utf-8")
            try:
                self._connection.executescript(sql_text)
                self._connection.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                    (version, _bookkeeping_now_iso()),
                )
                self._connection.commit()
            except sqlite3.Error as exc:
                self._connection.rollback()
                raise MigrationError(f"migration {path.name} failed: {exc}") from exc
            logger.info("applied migration %s (schema version %d)", path.name, version)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a block of writes as one commit/rollback unit.

        Wraps any ``sqlite3.Error`` raised inside the block in a ``StoreError`` so no
        third-party exception escapes ``store/``.

        Yields:
            The underlying connection, for issuing statements.

        Raises:
            StoreError: the block raised a ``sqlite3.Error``; the transaction was rolled back.
        """
        try:
            yield self._connection
            self._connection.commit()
        except sqlite3.Error as exc:
            self._connection.rollback()
            raise StoreError(str(exc)) from exc

    def close(self) -> None:
        """Close the underlying connection."""
        self._connection.close()

    def __enter__(self) -> Self:
        """Enter a context manager, returning this store."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Exit the context manager, closing the connection."""
        self.close()


__all__ = ["MIGRATIONS_DIR", "SCHEMA_VERSION"]
