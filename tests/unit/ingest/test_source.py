"""Tests for ``ingest.source`` — the mailbox abstraction M2-M6 compile against."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from jobtrack.ingest.gmail import GmailSource
from jobtrack.ingest.source import EmailSource, FetchResult
from jobtrack.models import RawMessage

from .conftest import FakeGmailService

NOW = datetime(2026, 8, 18, 12, 0, 0, tzinfo=UTC)


class StubSource:
    """A minimal EmailSource, standing in for the providers M6 may inject."""

    name = "stub"

    def fetch(
        self,
        *,
        query: str,
        since: datetime | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> FetchResult:
        """Return an empty batch."""
        return FetchResult(messages=[], next_cursor=None, fetched_at=NOW)


def test_fetch_result_defaults_to_not_truncated() -> None:
    result = FetchResult(messages=[], next_cursor=None, fetched_at=NOW)
    assert result.truncated is False
    assert result.messages == []
    assert result.next_cursor is None


def test_fetch_result_carries_messages(make_message: Any) -> None:
    message: RawMessage = make_message(subject="Thanks for applying")
    result = FetchResult(messages=[message], next_cursor="12345", fetched_at=NOW, truncated=True)
    assert result.messages[0].subject == "Thanks for applying"
    assert result.next_cursor == "12345"
    assert result.truncated is True


def test_naive_fetched_at_is_rejected() -> None:
    """Invariant I7: a naive datetime crossing a module boundary is a bug."""
    with pytest.raises(ValidationError, match="timezone-aware"):
        FetchResult(messages=[], next_cursor=None, fetched_at=datetime(2026, 8, 18, 12, 0, 0))


def test_aware_non_utc_fetched_at_is_converted() -> None:
    eastern = timezone(timedelta(hours=-4))
    result = FetchResult(
        messages=[],
        next_cursor=None,
        fetched_at=datetime(2026, 8, 18, 8, 0, 0, tzinfo=eastern),
    )
    assert result.fetched_at == NOW
    assert result.fetched_at.tzinfo is UTC


def test_stub_satisfies_the_protocol() -> None:
    assert isinstance(StubSource(), EmailSource)


def test_gmail_source_satisfies_the_protocol() -> None:
    """The contract that matters: M6 depends on EmailSource, never on GmailSource."""
    source = GmailSource(credentials=None, service=FakeGmailService())  # type: ignore[arg-type]
    # credentials are unused when a service is injected; None keeps the test offline.
    assert isinstance(source, EmailSource)
    assert source.name == "gmail"


def test_protocol_rejects_an_object_without_fetch() -> None:
    class NotASource:
        name = "nope"

    assert not isinstance(NotASource(), EmailSource)
