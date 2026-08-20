"""The mailbox abstraction.

``EmailSource`` is the seam between "where mail comes from" and everything downstream.
M2 through M6 depend on this Protocol and on ``FetchResult``; none of them import
``GmailSource``. Adding a provider (IMAP, an .mbox archive, a test double) means writing
a class that satisfies this Protocol and nothing else.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, field_validator

from jobtrack.models import RawMessage


class FetchResult(BaseModel):
    """One batch of messages, plus the cursor that resumes after it.

    ``next_cursor`` is opaque to every caller: only the source that produced it may
    interpret it. For ``GmailSource`` it is a Gmail ``historyId``. Per invariant I9 the
    caller persists it only after the batch commits, so a crash mid-sync re-fetches and
    ``message_id`` dedupe (I1) makes the replay a no-op.
    """

    messages: list[RawMessage]
    next_cursor: str | None  # opaque; Gmail historyId for GmailSource
    fetched_at: datetime
    truncated: bool = False  # True if capped by limit — more remains

    @field_validator("fetched_at")
    @classmethod
    def _require_utc(cls, value: datetime) -> datetime:
        """Enforce invariant I7: tz-aware UTC, never naive."""
        if value.tzinfo is None:
            raise ValueError("fetched_at must be timezone-aware UTC (invariant I7)")
        return value.astimezone(UTC)


@runtime_checkable
class EmailSource(Protocol):
    """Any mailbox that can yield RawMessages. Implement to add a provider."""

    name: str

    def fetch(
        self,
        *,
        query: str,
        since: datetime | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> FetchResult:
        """Fetch messages matching `query`.

        Prefers an incremental delta from `cursor` when the provider supports it and the
        cursor is still valid; otherwise falls back to a dated query from `since`.

        Args:
            query: Provider-native search expression, e.g. ``constants.DEFAULT_GMAIL_QUERY``.
            since: Lower bound on receipt time for the dated fallback. Tz-aware UTC.
            cursor: Opaque resume token from a previous ``FetchResult.next_cursor``.
            limit: Maximum number of messages to return. None means no cap beyond the
                provider's own.

        Returns:
            The batch, the next cursor, and whether ``limit`` truncated the results.

        Raises:
            TransientIngestError: rate limited, 5xx, or timed out — retry with backoff.
            PermanentIngestError: malformed query or unrecoverable 4xx.
            AuthError: credentials missing, expired, or revoked.
        """
        ...


__all__ = ["EmailSource", "FetchResult"]
