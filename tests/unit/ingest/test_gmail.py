"""Tests for ``ingest.gmail``.

No socket is opened anywhere in this file: every ``GmailSource`` is constructed with an
injected ``FakeGmailService``, which is the entire reason the ``service`` parameter is in
the contract. Backoff is exercised with ``gmail._sleep`` replaced, so the retry tests are
instant rather than a minute long.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from googleapiclient.errors import HttpError

from jobtrack.errors import AuthError, JobTrackError, PermanentIngestError, TransientIngestError
from jobtrack.ingest import gmail
from jobtrack.ingest.gmail import MAX_RETRIES, PAGE_SIZE, GmailSource, parse_gmail_message

from .conftest import READ_ONLY_ENDPOINTS, FakeGmailService, encode_body, make_http_error
from .conftest import make_payload as payload_for

QUERY = "subject:application"


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace the backoff sleep with a recorder, returning the delays it was asked for."""
    delays: list[float] = []

    def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(gmail, "_sleep", fake_sleep)
    return delays


def make_source(service: FakeGmailService) -> GmailSource:
    """A GmailSource wired to `service`. Credentials are unused when a service is injected."""
    return GmailSource(credentials=None, service=service)  # type: ignore[arg-type]


# =====================================================================================
# parse_gmail_message
# =====================================================================================


def test_parses_a_plain_text_message() -> None:
    message = parse_gmail_message(payload_for("m1", body="Hello there."))

    assert message.message_id == "m1"
    assert message.thread_id == "t1"
    assert message.body_text == "Hello there."
    assert message.subject == "Thanks for applying"
    assert message.labels == ["INBOX", "CATEGORY_UPDATES"]


def test_internal_date_becomes_tz_aware_utc() -> None:
    """I7: Gmail hands over epoch milliseconds; nothing naive may leave the module."""
    message = parse_gmail_message(payload_for(internal_date="1753349462000"))
    assert message.received_at == datetime(2025, 7, 24, 9, 31, 2, tzinfo=UTC)
    assert message.received_at.tzinfo is UTC


def test_internal_date_keeps_sub_second_precision() -> None:
    message = parse_gmail_message(payload_for(internal_date="1753349462123"))
    assert message.received_at.microsecond == 123_000


def test_sender_address_is_lowercased_and_name_preserved() -> None:
    message = parse_gmail_message(payload_for(sender="Acme Robotics <No-Reply@Greenhouse.IO>"))
    assert message.from_email == "no-reply@greenhouse.io"
    assert message.from_name == "Acme Robotics"


def test_bare_sender_address_has_no_display_name() -> None:
    message = parse_gmail_message(payload_for(sender="jobs@acme.com"))
    assert message.from_email == "jobs@acme.com"
    assert message.from_name is None


def test_header_keys_are_lowercased() -> None:
    message = parse_gmail_message(
        payload_for(extra_headers=[("List-Unsubscribe", "<https://x/unsub>")])
    )
    assert message.headers["list-unsubscribe"] == "<https://x/unsub>"
    assert message.headers["subject"] == "Thanks for applying"
    assert all(key == key.lower() for key in message.headers)


def test_duplicate_headers_keep_the_first_occurrence() -> None:
    """Gmail stacks ``received``; a stable choice keeps parsing deterministic."""
    message = parse_gmail_message(
        payload_for(extra_headers=[("Received", "first"), ("Received", "second")])
    )
    assert message.headers["received"] == "first"


def test_rfc2047_encoded_headers_are_decoded() -> None:
    encoded_subject = "=?utf-8?B?" + base64.b64encode("Café interview".encode()).decode() + "?="
    message = parse_gmail_message(payload_for(subject=encoded_subject))
    assert message.subject == "Café interview"


def test_undecodable_encoded_word_falls_back_to_the_raw_header() -> None:
    message = parse_gmail_message(payload_for(subject="=?bogus-charset?B?zzzz?="))
    assert "=?bogus-charset?" in message.subject


def test_snippet_entities_are_unescaped() -> None:
    message = parse_gmail_message(payload_for(snippet="We&#39;re moving  forward &amp; up"))
    assert message.snippet == "We're moving forward & up"


def test_html_only_message_is_converted_to_text() -> None:
    message = parse_gmail_message(
        payload_for(body=None, html_body="<p>Thanks for <b>applying</b>.</p><script>x()</script>")
    )
    assert message.body_text == "Thanks for applying."


def test_multipart_alternative_prefers_text_plain() -> None:
    message = parse_gmail_message(
        payload_for(body="The plain version.", html_body="<p>The HTML version.</p>")
    )
    assert message.body_text == "The plain version."


def test_body_whitespace_is_collapsed() -> None:
    message = parse_gmail_message(payload_for(body="Hi Alex,\r\n\r\n\r\n   Thanks.   \r\n"))
    assert message.body_text == "Hi Alex,\n\nThanks."


def test_attachments_are_skipped() -> None:
    payload = payload_for(body="Body text.")
    payload["payload"] = {
        "mimeType": "multipart/mixed",
        "headers": payload["payload"]["headers"],
        "parts": [
            {
                "mimeType": "text/plain",
                "filename": "",
                "body": {"data": encode_body("Body text.")},
            },
            {
                "mimeType": "text/plain",
                "filename": "resume.txt",
                "body": {"data": encode_body("SHOULD NOT APPEAR")},
            },
        ],
    }
    message = parse_gmail_message(payload)
    assert message.body_text == "Body text."


def test_nested_multipart_is_walked() -> None:
    payload = payload_for(body="ignored")
    payload["payload"] = {
        "mimeType": "multipart/mixed",
        "headers": [{"name": "Subject", "value": "s"}],
        "parts": [
            {
                "mimeType": "multipart/alternative",
                "filename": "",
                "parts": [
                    {
                        "mimeType": "text/plain",
                        "filename": "",
                        "body": {"data": encode_body("Deeply nested prose.")},
                    }
                ],
            }
        ],
    }
    assert parse_gmail_message(payload).body_text == "Deeply nested prose."


def test_pathologically_deep_mime_tree_does_not_recurse_forever() -> None:
    leaf: dict[str, Any] = {
        "mimeType": "text/plain",
        "filename": "",
        "body": {"data": encode_body("too deep")},
    }
    node = leaf
    for _ in range(gmail.MAX_MIME_DEPTH + 5):
        node = {"mimeType": "multipart/mixed", "filename": "", "parts": [node]}
    payload = payload_for()
    payload["payload"] = node
    assert parse_gmail_message(payload).body_text == ""


def test_declared_charset_is_honoured() -> None:
    payload = payload_for()
    payload["payload"] = {
        "mimeType": "text/plain",
        "filename": "",
        "headers": [{"name": "Content-Type", "value": 'text/plain; charset="iso-8859-1"'}],
        "body": {"data": base64.urlsafe_b64encode("café".encode("iso-8859-1")).decode()},
    }
    assert parse_gmail_message(payload).body_text == "café"


def test_unknown_charset_falls_back_to_utf8() -> None:
    payload = payload_for()
    payload["payload"] = {
        "mimeType": "text/plain",
        "filename": "",
        "headers": [{"name": "Content-Type", "value": "text/plain; charset=x-not-a-codec"}],
        "body": {"data": encode_body("plain ascii")},
    }
    assert parse_gmail_message(payload).body_text == "plain ascii"


def test_undecodable_base64_body_yields_empty_text() -> None:
    payload = payload_for()
    payload["payload"] = {
        "mimeType": "text/plain",
        "filename": "",
        "body": {"data": "!!!not base64!!!"},
    }
    assert parse_gmail_message(payload).body_text == ""


def test_message_without_a_mime_payload_still_parses() -> None:
    payload = {"id": "m9", "threadId": "t9", "internalDate": "1753349462000"}
    message = parse_gmail_message(payload)
    assert message.body_text == ""
    assert message.headers == {}
    assert message.from_email == ""
    assert message.to_email is None


@pytest.mark.parametrize("missing", ["id", "threadId", "internalDate"])
def test_missing_required_field_raises_permanent(missing: str) -> None:
    payload = payload_for()
    del payload[missing]
    with pytest.raises(PermanentIngestError, match=missing):
        parse_gmail_message(payload)


def test_non_numeric_internal_date_raises_permanent() -> None:
    with pytest.raises(PermanentIngestError, match="epoch milliseconds"):
        parse_gmail_message(payload_for(internal_date="yesterday"))


def test_parsing_is_deterministic() -> None:
    """M2's purity (I2) starts here: the same payload must parse identically every time."""
    payload = payload_for(body="Plain", html_body="<p>HTML</p>")
    first = parse_gmail_message(payload)
    assert all(parse_gmail_message(payload) == first for _ in range(5))


# =====================================================================================
# error mapping — no HttpError may escape ingest/
# =====================================================================================


@pytest.mark.parametrize(
    ("status", "reason", "expected"),
    [
        (401, "authError", AuthError),
        (403, "insufficientPermissions", AuthError),
        (403, "rateLimitExceeded", TransientIngestError),
        (403, "userRateLimitExceeded", TransientIngestError),
        (403, "quotaExceeded", TransientIngestError),
        (429, "rateLimitExceeded", TransientIngestError),
        (500, "backendError", TransientIngestError),
        (503, None, TransientIngestError),
        (400, "invalidArgument", PermanentIngestError),
        (404, "notFound", PermanentIngestError),
    ],
)
def test_http_status_maps_to_the_right_error(
    status: int, reason: str | None, expected: type[Exception], no_sleep: list[float]
) -> None:
    service = FakeGmailService(list_responses=[make_http_error(status, reason)])
    with pytest.raises(expected):
        make_source(service).fetch(query=QUERY)


def test_http_error_never_escapes_ingest(no_sleep: list[float]) -> None:
    """CLAUDE.md: a googleapiclient exception must be wrapped at the module boundary."""
    service = FakeGmailService(list_responses=[make_http_error(418, "teapot")])
    with pytest.raises(JobTrackError) as caught:
        make_source(service).fetch(query=QUERY)
    assert not isinstance(caught.value, HttpError)
    assert isinstance(caught.value.__cause__, HttpError)


def test_transient_failure_is_retried_with_exponential_backoff(no_sleep: list[float]) -> None:
    service = FakeGmailService(
        list_responses=[
            make_http_error(429, "rateLimitExceeded"),
            make_http_error(503, None),
            {"messages": []},
        ]
    )
    result = make_source(service).fetch(query=QUERY)

    assert result.messages == []
    assert len(service.calls_to("users.messages.list")) == 3
    assert no_sleep == [1.0, 2.0]


def test_backoff_gives_up_after_max_retries(no_sleep: list[float]) -> None:
    service = FakeGmailService(list_responses=[make_http_error(500, "backendError")])
    with pytest.raises(TransientIngestError, match=r"attempts|retryable"):
        make_source(service).fetch(query=QUERY)
    assert len(service.calls_to("users.messages.list")) == MAX_RETRIES


def test_permanent_failure_is_not_retried(no_sleep: list[float]) -> None:
    service = FakeGmailService(list_responses=[make_http_error(400, "invalidArgument")])
    with pytest.raises(PermanentIngestError):
        make_source(service).fetch(query=QUERY)
    assert len(service.calls_to("users.messages.list")) == 1
    assert no_sleep == []


def test_socket_timeout_is_retried_then_surfaces_as_transient(no_sleep: list[float]) -> None:
    service = FakeGmailService(list_responses=[TimeoutError("timed out")])
    with pytest.raises(TransientIngestError, match="attempts"):
        make_source(service).fetch(query=QUERY)
    assert len(service.calls_to("users.messages.list")) == MAX_RETRIES


def test_transport_error_that_clears_is_recovered(no_sleep: list[float]) -> None:
    service = FakeGmailService(
        list_responses=[ConnectionResetError("reset by peer"), {"messages": []}]
    )
    assert make_source(service).fetch(query=QUERY).messages == []
    assert no_sleep == [1.0]


def test_unparseable_message_surfaces_as_permanent(no_sleep: list[float]) -> None:
    """A payload Gmail should never send is loud, not silently dropped."""
    broken = {"id": "m1", "threadId": "t1"}  # no internalDate
    service = FakeGmailService(
        payloads={"m1": broken}, list_responses=[{"messages": [{"id": "m1"}]}]
    )
    with pytest.raises(PermanentIngestError, match="internalDate"):
        make_source(service).fetch(query=QUERY)


# =====================================================================================
# GmailSource.fetch — the search path
# =====================================================================================


def test_name_is_gmail() -> None:
    assert GmailSource.name == "gmail"


def test_fetch_returns_parsed_messages_and_the_profile_cursor(no_sleep: list[float]) -> None:
    service = FakeGmailService(
        payloads={"m1": payload_for("m1"), "m2": payload_for("m2", thread_id="t2")},
        list_responses=[{"messages": [{"id": "m1"}, {"id": "m2"}]}],
        profile_responses=[{"historyId": "9000"}],
    )
    result = make_source(service).fetch(query=QUERY)

    assert [m.message_id for m in result.messages] == ["m1", "m2"]
    assert result.next_cursor == "9000"
    assert result.truncated is False
    assert result.fetched_at.tzinfo is UTC


def test_profile_is_read_before_any_message(no_sleep: list[float]) -> None:
    """The cursor must predate the batch, or mail arriving mid-fetch is skipped forever."""
    service = FakeGmailService(
        payloads={"m1": payload_for("m1")},
        list_responses=[{"messages": [{"id": "m1"}]}],
    )
    make_source(service).fetch(query=QUERY)
    assert service.endpoints()[0] == "users.getProfile"


def test_pagination_walks_every_page(no_sleep: list[float]) -> None:
    service = FakeGmailService(
        payloads={f"m{i}": payload_for(f"m{i}") for i in range(1, 4)},
        list_responses=[
            {"messages": [{"id": "m1"}, {"id": "m2"}], "nextPageToken": "page-2"},
            {"messages": [{"id": "m3"}]},
        ],
    )
    result = make_source(service).fetch(query=QUERY)

    assert [m.message_id for m in result.messages] == ["m1", "m2", "m3"]
    calls = service.calls_to("users.messages.list")
    assert len(calls) == 2
    assert calls[0]["pageToken"] is None
    assert calls[1]["pageToken"] == "page-2"
    assert result.truncated is False


def test_pages_are_requested_one_hundred_at_a_time(no_sleep: list[float]) -> None:
    service = FakeGmailService(list_responses=[{"messages": []}])
    make_source(service).fetch(query=QUERY)
    assert service.calls_to("users.messages.list")[0]["maxResults"] == PAGE_SIZE
    assert PAGE_SIZE == 100


def test_limit_truncates_and_reports_it(no_sleep: list[float]) -> None:
    service = FakeGmailService(
        payloads={"m1": payload_for("m1"), "m2": payload_for("m2")},
        list_responses=[{"messages": [{"id": "m1"}, {"id": "m2"}]}],
    )
    result = make_source(service).fetch(query=QUERY, limit=1)

    assert [m.message_id for m in result.messages] == ["m1"]
    assert result.truncated is True


def test_limit_that_exactly_consumes_the_mailbox_is_not_truncated(no_sleep: list[float]) -> None:
    service = FakeGmailService(
        payloads={"m1": payload_for("m1")},
        list_responses=[{"messages": [{"id": "m1"}]}],
    )
    result = make_source(service).fetch(query=QUERY, limit=1)
    assert result.truncated is False


def test_limit_caps_the_requested_page_size(no_sleep: list[float]) -> None:
    service = FakeGmailService(
        payloads={"m1": payload_for("m1")},
        list_responses=[{"messages": [{"id": "m1"}]}],
    )
    make_source(service).fetch(query=QUERY, limit=1)
    assert service.calls_to("users.messages.list")[0]["maxResults"] == 1


def test_zero_limit_fetches_nothing(no_sleep: list[float]) -> None:
    service = FakeGmailService(list_responses=[{"messages": [{"id": "m1"}]}])
    result = make_source(service).fetch(query=QUERY, limit=0)
    assert result.messages == []
    assert service.calls_to("users.messages.list") == []


def test_since_becomes_a_gmail_after_operator(no_sleep: list[float]) -> None:
    service = FakeGmailService(list_responses=[{"messages": []}])
    make_source(service).fetch(query=QUERY, since=datetime(2026, 7, 1, tzinfo=UTC))
    assert service.calls_to("users.messages.list")[0]["q"] == f"{QUERY} after:2026/07/01"


def test_since_is_converted_to_utc_before_formatting(no_sleep: list[float]) -> None:
    """A local-midnight `since` must not silently shift the date boundary (I7)."""
    service = FakeGmailService(list_responses=[{"messages": []}])
    eastern = timezone(timedelta(hours=-5))
    make_source(service).fetch(query=QUERY, since=datetime(2026, 7, 1, 23, 0, tzinfo=eastern))
    assert service.calls_to("users.messages.list")[0]["q"].endswith("after:2026/07/02")


def test_query_is_passed_through_unchanged_without_since(no_sleep: list[float]) -> None:
    service = FakeGmailService(list_responses=[{"messages": []}])
    make_source(service).fetch(query=QUERY)
    assert service.calls_to("users.messages.list")[0]["q"] == QUERY


def test_chat_and_spam_messages_are_dropped(no_sleep: list[float]) -> None:
    service = FakeGmailService(
        payloads={
            "m1": payload_for("m1"),
            "m2": payload_for("m2", labels=["SPAM"]),
            "m3": payload_for("m3", labels=["CHAT"]),
        },
        list_responses=[{"messages": [{"id": "m1"}, {"id": "m2"}, {"id": "m3"}]}],
    )
    result = make_source(service).fetch(query=QUERY)
    assert [m.message_id for m in result.messages] == ["m1"]


def test_malformed_list_entries_are_ignored(no_sleep: list[float]) -> None:
    service = FakeGmailService(
        payloads={"m1": payload_for("m1")},
        list_responses=[{"messages": [{"id": "m1"}, {"noId": True}, "garbage"]}],
    )
    assert len(make_source(service).fetch(query=QUERY).messages) == 1


# =====================================================================================
# GmailSource.fetch — the history delta path
# =====================================================================================


def history_page(
    *message_ids: str, labels: list[str] | None = None, **extra: Any
) -> dict[str, Any]:
    """Build a ``history.list`` response adding `message_ids`."""
    page: dict[str, Any] = {
        "history": [
            {
                "id": "1",
                "messagesAdded": [
                    {"message": {"id": mid, "labelIds": labels or ["INBOX"]}} for mid in message_ids
                ],
            }
        ]
    }
    page.update(extra)
    return page


def test_cursor_uses_the_history_delta_not_a_search(no_sleep: list[float]) -> None:
    service = FakeGmailService(
        payloads={"m1": payload_for("m1")},
        history_responses=[history_page("m1")],
    )
    result = make_source(service).fetch(query=QUERY, cursor="8000")

    assert [m.message_id for m in result.messages] == ["m1"]
    assert service.calls_to("users.messages.list") == []
    assert service.calls_to("users.history.list")[0]["startHistoryId"] == "8000"


def test_history_requests_only_message_additions(no_sleep: list[float]) -> None:
    service = FakeGmailService(history_responses=[{"history": []}])
    make_source(service).fetch(query=QUERY, cursor="8000")
    call = service.calls_to("users.history.list")[0]
    assert call["historyTypes"] == ["messageAdded"]
    assert call["maxResults"] == PAGE_SIZE


def test_expired_history_cursor_downgrades_to_a_dated_query(
    no_sleep: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    """Gmail drops history after roughly a week; a 404 is a downgrade, not a failure."""
    service = FakeGmailService(
        payloads={"m1": payload_for("m1")},
        history_responses=[make_http_error(404, "notFound")],
        list_responses=[{"messages": [{"id": "m1"}]}],
    )
    with caplog.at_level("WARNING", logger="jobtrack.ingest.gmail"):
        result = make_source(service).fetch(
            query=QUERY, since=datetime(2026, 7, 1, tzinfo=UTC), cursor="1"
        )

    assert [m.message_id for m in result.messages] == ["m1"]
    assert service.calls_to("users.messages.list")[0]["q"].endswith("after:2026/07/01")
    assert "expired" in caplog.text.lower()


def test_expired_cursor_is_not_retried_as_a_transient_failure(no_sleep: list[float]) -> None:
    service = FakeGmailService(
        history_responses=[make_http_error(404, "notFound")],
        list_responses=[{"messages": []}],
    )
    make_source(service).fetch(query=QUERY, cursor="1")
    assert len(service.calls_to("users.history.list")) == 1
    assert no_sleep == []


def test_history_pagination_walks_every_page(no_sleep: list[float]) -> None:
    service = FakeGmailService(
        payloads={"m1": payload_for("m1"), "m2": payload_for("m2")},
        history_responses=[
            history_page("m1", nextPageToken="h2"),
            history_page("m2"),
        ],
    )
    result = make_source(service).fetch(query=QUERY, cursor="8000")
    assert [m.message_id for m in result.messages] == ["m1", "m2"]
    assert service.calls_to("users.history.list")[1]["pageToken"] == "h2"


def test_history_deduplicates_repeated_ids(no_sleep: list[float]) -> None:
    """One message can appear in several history records; it is still one message (I1)."""
    service = FakeGmailService(
        payloads={"m1": payload_for("m1")},
        history_responses=[history_page("m1", "m1")],
    )
    result = make_source(service).fetch(query=QUERY, cursor="8000")
    assert [m.message_id for m in result.messages] == ["m1"]
    assert len(service.calls_to("users.messages.get")) == 1


def test_history_skips_chat_drafts_and_trash(no_sleep: list[float]) -> None:
    service = FakeGmailService(
        payloads={"m1": payload_for("m1")},
        history_responses=[history_page("m1", labels=["CHAT"])],
    )
    result = make_source(service).fetch(query=QUERY, cursor="8000")
    assert result.messages == []
    assert service.calls_to("users.messages.get") == []


def test_history_respects_the_limit(no_sleep: list[float]) -> None:
    service = FakeGmailService(
        payloads={"m1": payload_for("m1"), "m2": payload_for("m2")},
        history_responses=[history_page("m1", "m2")],
    )
    result = make_source(service).fetch(query=QUERY, cursor="8000", limit=1)
    assert len(result.messages) == 1
    assert result.truncated is True


def test_history_with_a_zero_limit_fetches_nothing(no_sleep: list[float]) -> None:
    service = FakeGmailService(history_responses=[history_page("m1")])
    result = make_source(service).fetch(query=QUERY, cursor="8000", limit=0)
    assert result.messages == []
    assert service.calls_to("users.history.list") == []


def test_empty_delta_still_advances_the_cursor(no_sleep: list[float]) -> None:
    """I9: an uneventful sync must still move the cursor, or every run re-scans."""
    service = FakeGmailService(
        history_responses=[{"history": []}], profile_responses=[{"historyId": "9500"}]
    )
    result = make_source(service).fetch(query=QUERY, cursor="8000")
    assert result.messages == []
    assert result.next_cursor == "9500"


def test_malformed_history_records_are_ignored(no_sleep: list[float]) -> None:
    service = FakeGmailService(
        payloads={"m1": payload_for("m1")},
        history_responses=[
            {
                "history": [
                    "not-a-record",
                    {"messagesAdded": ["nope", {"message": "also-nope"}, {"message": {}}]},
                    {"messagesAdded": [{"message": {"id": "m1", "labelIds": ["INBOX"]}}]},
                ]
            }
        ],
    )
    assert [m.message_id for m in make_source(service).fetch(query=QUERY, cursor="1").messages] == [
        "m1"
    ]


# =====================================================================================
# cursor derivation
# =====================================================================================


def test_cursor_falls_back_to_the_highest_message_history_id(no_sleep: list[float]) -> None:
    """A refused profile call must not cost the sync its cursor."""
    service = FakeGmailService(
        payloads={
            "m1": payload_for("m1", history_id="900"),
            "m2": payload_for("m2", history_id="1100"),
        },
        list_responses=[{"messages": [{"id": "m1"}, {"id": "m2"}]}],
        profile_responses=[make_http_error(400, "invalidArgument")],
    )
    result = make_source(service).fetch(query=QUERY)
    # Numeric, not lexicographic: "1100" > "900" only when compared as integers.
    assert result.next_cursor == "1100"


def test_cursor_falls_back_to_the_incoming_cursor_when_nothing_else_is_known(
    no_sleep: list[float],
) -> None:
    service = FakeGmailService(
        history_responses=[{"history": []}],
        profile_responses=[make_http_error(400, "invalidArgument")],
    )
    result = make_source(service).fetch(query=QUERY, cursor="8000")
    assert result.next_cursor == "8000"


def test_profile_auth_failure_still_propagates(no_sleep: list[float]) -> None:
    """A 401 on getProfile is a credential problem, not a cursor problem."""
    service = FakeGmailService(profile_responses=[make_http_error(401, "authError")])
    with pytest.raises(AuthError):
        make_source(service).fetch(query=QUERY)


def test_numeric_profile_history_id_is_stringified(no_sleep: list[float]) -> None:
    service = FakeGmailService(
        list_responses=[{"messages": []}], profile_responses=[{"historyId": 9700}]
    )
    assert make_source(service).fetch(query=QUERY).next_cursor == "9700"


# =====================================================================================
# invariant I11 — read-only Gmail
# =====================================================================================


def test_only_read_only_endpoints_are_reached(no_sleep: list[float]) -> None:
    """I11: no code path may call a mutating Gmail method."""
    service = FakeGmailService(
        payloads={"m1": payload_for("m1")},
        list_responses=[{"messages": [{"id": "m1"}]}],
        history_responses=[history_page("m1")],
    )
    source = make_source(service)
    source.fetch(query=QUERY)
    source.fetch(query=QUERY, cursor="8000")
    source.fetch(query=QUERY, since=datetime(2026, 7, 1, tzinfo=UTC), limit=5)

    touched = set(service.endpoints())
    assert touched <= READ_ONLY_ENDPOINTS, f"unexpected endpoints: {touched - READ_ONLY_ENDPOINTS}"


def test_messages_are_fetched_in_full_format(no_sleep: list[float]) -> None:
    service = FakeGmailService(
        payloads={"m1": payload_for("m1")}, list_responses=[{"messages": [{"id": "m1"}]}]
    )
    make_source(service).fetch(query=QUERY)
    call = service.calls_to("users.messages.get")[0]
    assert call["format"] == "full"
    assert call["userId"] == "me"
