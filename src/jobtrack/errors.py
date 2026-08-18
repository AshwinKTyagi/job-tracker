"""Exception hierarchy for jobtrack.

Every module raises from this tree. ``cli.py`` is the only layer that catches these and
maps them to process exit codes; library modules must never call ``sys.exit()``.

Third-party exceptions are wrapped at module boundaries: no ``googleapiclient.errors.HttpError``
escapes ``ingest/``, no ``sqlite3.Error`` escapes ``store/``.
"""

from __future__ import annotations


class JobTrackError(Exception):
    """Base for every error jobtrack raises deliberately."""


class ConfigError(JobTrackError):
    """Missing or malformed config, or an unusable JOBTRACK_HOME."""


class AuthError(JobTrackError):
    """No credentials, or a token that is expired or revoked. Maps to exit code 3."""


class IngestError(JobTrackError):
    """Base for mailbox-ingestion failures (M1)."""


class TransientIngestError(IngestError):
    """Rate limit, 5xx, or socket timeout. Retry with backoff. Maps to exit code 4."""


class PermanentIngestError(IngestError):
    """A 4xx or malformed payload that retrying will not fix."""


class ClassificationError(JobTrackError):
    """A classifier could not produce a Classification at all.

    Note: an *unparseable* message is not an error — it is ``EventType.UNKNOWN`` with
    confidence 0.0. This is for genuine backend failures.
    """


class StoreError(JobTrackError):
    """Base for persistence failures (M3)."""


class MigrationError(StoreError):
    """A schema migration failed; the database is left at its prior version."""


class ExportError(JobTrackError):
    """Spreadsheet or dashboard output could not be written (M4/M5)."""


__all__ = [
    "AuthError",
    "ClassificationError",
    "ConfigError",
    "ExportError",
    "IngestError",
    "JobTrackError",
    "MigrationError",
    "PermanentIngestError",
    "StoreError",
    "TransientIngestError",
]
