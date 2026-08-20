"""Tests for GmailSource.fetch: pagination, backoff, delta/fallback, and error mapping.

All network is faked via `FakeGmailService` (see conftest.py) — GmailSource is exercised
purely through the injected `service=` parameter, per CONTRACTS.md §4.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import pytest
from google.oauth2.credentials import Credentials

from jobtrack.errors import AuthError, PermanentIngestError, TransientIngestError
from jobtrack.ingest.gmail import GmailSource
from tests.unit.ingest.conftest import FakeGmailService, make_http_error, make_message_payload

_FAKE_CREDENTIALS = Credentials(token="fake-access-token")  # never used: service is injected


def _make_source(service: FakeGmailService) -> GmailSource:
    return GmailSource(_FAKE_CREDENTIALS, service=service)


def test_name_attribute_matches_protocol() -> None:
    source = _make_source(FakeGmailService())
    assert source.name == "gmail"


def test_fetch_dated_paginates_across_pages() -> None:
    service = FakeGmailService(
        list_pages=[
            {"messages": [{"id": "m1"}, {"id": "m2"}], "nextPageToken": "page-2"},
            {"messages": [{"id": "m3"}], "nextPageToken": None},
        ],
        messages={
            "m1": make_message_payload("m1"),
            "m2": make_message_payload("m2"),
            "m3": make_message_payload("m3"),
        },
        profile={"historyId": "555"},
    )
    result = _make_source(service).fetch(query="thanks for applying", limit=10)

    assert [m.message_id for m in result.messages] == ["m1", "m2", "m3"]
    assert result.next_cursor == "555"
    assert result.truncated is False
    assert len(service.list_calls) == 2
    assert service.list_calls[1]["pageToken"] == "page-2"


def test_fetch_dated_appends_since_as_after_operator() -> None:
    service = FakeGmailService(list_pages=[{"messages": [], "nextPageToken": None}])
    since = datetime(2026, 1, 15, tzinfo=UTC)

    _make_source(service).fetch(query="base query", since=since, limit=5)

    assert service.list_calls[0]["q"] == "base query after:2026/01/15"


def test_fetch_dated_omits_after_operator_when_since_is_none() -> None:
    service = FakeGmailService(list_pages=[{"messages": [], "nextPageToken": None}])

    _make_source(service).fetch(query="base query", limit=5)

    assert service.list_calls[0]["q"] == "base query"


def test_fetch_default_limit_requests_full_page_size() -> None:
    service = FakeGmailService(list_pages=[{"messages": [], "nextPageToken": None}])

    _make_source(service).fetch(query="q")  # limit omitted -> defaults to 500

    assert service.list_calls[0]["maxResults"] == 100  # capped at the 100-page batch size


def test_fetch_dated_caps_at_limit_and_marks_truncated() -> None:
    service = FakeGmailService(
        list_pages=[
            {
                "messages": [{"id": f"m{i}"} for i in range(5)],
                "nextPageToken": "more",
            }
        ],
        messages={f"m{i}": make_message_payload(f"m{i}") for i in range(5)},
    )

    result = _make_source(service).fetch(query="q", limit=3)

    assert [m.message_id for m in result.messages] == ["m0", "m1", "m2"]
    assert result.truncated is True
    assert service.list_calls[0]["maxResults"] == 3
    # Only messages within the cap are fetched.
    assert len(service.get_calls) == 3


def test_fetch_delta_returns_history_id_as_next_cursor() -> None:
    service = FakeGmailService(
        history_pages=[
            {
                "history": [{"messagesAdded": [{"message": {"id": "m1"}}]}],
                "historyId": "1001",
            }
        ],
        messages={"m1": make_message_payload("m1")},
        profile=RuntimeError("getProfile must not be called on a successful delta fetch"),
    )

    result = _make_source(service).fetch(query="q", cursor="900", limit=10)

    assert [m.message_id for m in result.messages] == ["m1"]
    assert result.next_cursor == "1001"
    assert result.truncated is False
    assert service.history_calls[0]["startHistoryId"] == "900"


def test_fetch_delta_paginates_and_dedupes_message_ids() -> None:
    service = FakeGmailService(
        history_pages=[
            {
                "history": [
                    {
                        "messagesAdded": [
                            {"message": {"id": "m1"}},
                            {"message": {"id": "m2"}},
                        ]
                    }
                ],
                "historyId": "1001",
                "nextPageToken": "h2",
            },
            {
                "history": [{"messagesAdded": [{"message": {"id": "m2"}}]}],
                "historyId": "1002",
            },
        ],
        messages={"m1": make_message_payload("m1"), "m2": make_message_payload("m2")},
    )

    result = _make_source(service).fetch(query="q", cursor="900", limit=10)

    assert sorted(m.message_id for m in result.messages) == ["m1", "m2"]
    assert result.next_cursor == "1002"


def test_fetch_delta_expired_falls_back_to_dated_query(caplog: pytest.LogCaptureFixture) -> None:
    service = FakeGmailService(
        history_pages=[make_http_error(404, "historyId is too old")],
        list_pages=[{"messages": [{"id": "m1"}], "nextPageToken": None}],
        messages={"m1": make_message_payload("m1")},
        profile={"historyId": "2000"},
    )

    with caplog.at_level(logging.WARNING):
        result = _make_source(service).fetch(
            query="q", since=datetime(2026, 1, 1, tzinfo=UTC), cursor="stale-cursor", limit=5
        )

    assert [m.message_id for m in result.messages] == ["m1"]
    assert result.next_cursor == "2000"
    assert any("expired" in record.message.lower() for record in caplog.records)
    assert service.list_calls[0]["q"] == "q after:2026/01/01"


def test_fetch_raises_transient_on_429_after_retries_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("jobtrack.ingest.gmail.time.sleep", lambda _seconds: None)
    service = FakeGmailService(list_pages=[make_http_error(429, "rateLimitExceeded")] * 6)

    with pytest.raises(TransientIngestError):
        _make_source(service).fetch(query="q", limit=5)

    assert len(service.list_calls) == 1  # one logical call; retries happen inside it
    assert service._list_pages == []  # every queued error was consumed by a retry attempt


def test_fetch_retries_then_succeeds_on_transient_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("jobtrack.ingest.gmail.time.sleep", lambda _seconds: None)
    service = FakeGmailService(
        list_pages=[
            make_http_error(503, "backend overloaded"),
            make_http_error(500, "internal error"),
            {"messages": [{"id": "m1"}], "nextPageToken": None},
        ],
        messages={"m1": make_message_payload("m1")},
    )

    result = _make_source(service).fetch(query="q", limit=5)

    assert [m.message_id for m in result.messages] == ["m1"]


def test_fetch_raises_permanent_on_400() -> None:
    service = FakeGmailService(list_pages=[make_http_error(400, "invalid query syntax")])

    with pytest.raises(PermanentIngestError):
        _make_source(service).fetch(query="((( bad query", limit=5)


def test_fetch_raises_autherror_on_401() -> None:
    service = FakeGmailService(list_pages=[make_http_error(401, "invalid_grant: token revoked")])

    with pytest.raises(AuthError):
        _make_source(service).fetch(query="q", limit=5)


def test_fetch_403_with_rate_reason_is_transient_and_not_retried() -> None:
    service = FakeGmailService(
        list_pages=[make_http_error(403, "quotaExceeded: User Rate Limit Exceeded")]
    )

    with pytest.raises(TransientIngestError):
        _make_source(service).fetch(query="q", limit=5)

    assert len(service.list_calls) == 1
    assert service._list_pages == []  # the single queued error was consumed, none left to retry


def test_fetch_403_without_rate_reason_is_permanent() -> None:
    service = FakeGmailService(
        list_pages=[make_http_error(403, "Forbidden: insufficient permission")]
    )

    with pytest.raises(PermanentIngestError):
        _make_source(service).fetch(query="q", limit=5)


def test_current_history_id_degrades_gracefully_on_transient_profile_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr("jobtrack.ingest.gmail.time.sleep", lambda _seconds: None)
    service = FakeGmailService(
        list_pages=[{"messages": [{"id": "m1"}], "nextPageToken": None}],
        messages={"m1": make_message_payload("m1")},
        profile=make_http_error(500, "backend overloaded"),
    )

    with caplog.at_level(logging.WARNING):
        result = _make_source(service).fetch(query="q", limit=5)

    assert [m.message_id for m in result.messages] == ["m1"]  # already-fetched data survives
    assert result.next_cursor is None
    assert any("historyid" in record.message.lower() for record in caplog.records)


def test_current_history_id_autherror_propagates() -> None:
    service = FakeGmailService(
        list_pages=[{"messages": [{"id": "m1"}], "nextPageToken": None}],
        messages={"m1": make_message_payload("m1")},
        profile=make_http_error(401, "credentials revoked"),
    )

    with pytest.raises(AuthError):
        _make_source(service).fetch(query="q", limit=5)


def test_fetch_is_deterministic_given_identical_payloads() -> None:
    def _service() -> FakeGmailService:
        return FakeGmailService(
            list_pages=[{"messages": [{"id": "m1"}], "nextPageToken": None}],
            messages={"m1": make_message_payload("m1", subject="Thanks for applying")},
            profile={"historyId": "1"},
        )

    first = _make_source(_service()).fetch(query="q", limit=5)
    second = _make_source(_service()).fetch(query="q", limit=5)

    assert [m.model_dump() for m in first.messages] == [m.model_dump() for m in second.messages]


def test_fetch_result_fetched_at_is_utc_aware() -> None:
    service = FakeGmailService(list_pages=[{"messages": [], "nextPageToken": None}])

    result = _make_source(service).fetch(query="q", limit=5)

    assert result.fetched_at.tzinfo is not None


def test_fetch_raises_transient_on_timeout_after_retries_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("jobtrack.ingest.gmail.time.sleep", lambda _seconds: None)
    service = FakeGmailService(list_pages=[TimeoutError("timed out")] * 6)

    with pytest.raises(TransientIngestError):
        _make_source(service).fetch(query="q", limit=5)


def test_fetch_delta_non_404_http_error_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("jobtrack.ingest.gmail.time.sleep", lambda _seconds: None)
    service = FakeGmailService(history_pages=[make_http_error(500, "backend error")] * 6)

    with pytest.raises(TransientIngestError):
        _make_source(service).fetch(query="q", cursor="900", limit=5)


def test_fetch_delta_permanent_http_error_is_wrapped() -> None:
    service = FakeGmailService(history_pages=[make_http_error(400, "bad startHistoryId")])

    with pytest.raises(PermanentIngestError):
        _make_source(service).fetch(query="q", cursor="900", limit=5)


def test_fetch_delta_timeout_raises_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("jobtrack.ingest.gmail.time.sleep", lambda _seconds: None)
    service = FakeGmailService(history_pages=[TimeoutError("timed out")] * 6)

    with pytest.raises(TransientIngestError):
        _make_source(service).fetch(query="q", cursor="900", limit=5)


def test_service_is_built_from_credentials_when_not_injected(monkeypatch: pytest.MonkeyPatch) -> None:
    """The `service=None` branch delegates to googleapiclient's `build()`."""
    sentinel = object()
    calls: dict[str, Any] = {}

    def _fake_build(*args: Any, **kwargs: Any) -> Any:
        calls["args"] = args
        calls["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr("jobtrack.ingest.gmail.build", _fake_build)
    source = GmailSource(_FAKE_CREDENTIALS)

    assert calls["args"][:2] == ("gmail", "v1")
    assert calls["kwargs"]["credentials"] is _FAKE_CREDENTIALS
    assert source._service is sentinel
