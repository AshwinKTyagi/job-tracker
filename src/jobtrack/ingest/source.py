"""The mailbox-agnostic ingest interface.

``EmailSource`` is the seam M1 exposes to the rest of the system: anything that can yield
``RawMessage``s. ``GmailSource`` (in ``gmail.py``) is the only implementation in Phase 1.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from pydantic import BaseModel

from jobtrack.models import RawMessage


class FetchResult(BaseModel):
    """One page (or one complete pull) of messages from an ``EmailSource``."""

    messages: list[RawMessage]
    next_cursor: str | None  # opaque; Gmail historyId for GmailSource
    fetched_at: datetime
    truncated: bool = False  # True if capped by limit — more remains


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
            query: Provider-specific search query.
            since: Lower bound for a dated query, used when `cursor` is absent or stale.
            cursor: Opaque resume token from a prior fetch's `next_cursor`.
            limit: Maximum number of messages to return.

        Returns:
            The matching messages plus a cursor for the next incremental fetch.

        Raises:
            TransientIngestError: rate limited, 5xx, or timed out — retry with backoff.
            PermanentIngestError: malformed query or unrecoverable 4xx.
            AuthError: credentials missing, expired, or revoked.
        """
        ...


__all__ = ["EmailSource", "FetchResult"]
