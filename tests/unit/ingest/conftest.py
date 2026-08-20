"""Fakes shared by ingest tests.

`FakeGmailService` mimics the shape of the googleapiclient Gmail resource
(`service.users().messages().list(...)`, `.get(...)`, `.history().list(...)`,
`.getProfile(...)`) closely enough for `GmailSource` to drive it, without ever touching a
socket. This is the injected `service=` from CONTRACTS.md §4 — the reason ingest is
testable without a network.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from googleapiclient.errors import HttpError


class _FakeHttpResponse:
    """Stand-in for the `httplib2.Response` googleapiclient attaches to `HttpError.resp`."""

    def __init__(self, status: int, reason: str) -> None:
        self.status = status
        self.reason = reason


def make_http_error(status: int, reason: str = "error") -> HttpError:
    """Build an `HttpError` with a given status and reason, as googleapiclient would raise."""
    body = f'{{"error": {{"message": "{reason}"}}}}'.encode()
    return HttpError(
        _FakeHttpResponse(status, reason), body, uri="https://gmail.googleapis.com/fake"
    )


class _FakeApiCall:
    """Mimics an unexecuted googleapiclient request: `.execute()` invokes `fn`."""

    def __init__(self, fn: Callable[[], dict[str, Any]]) -> None:
        self._fn = fn

    def execute(self) -> dict[str, Any]:
        return self._fn()


class _FakeMessagesResource:
    def __init__(self, service: FakeGmailService) -> None:
        self._service = service

    def list(self, **kwargs: Any) -> _FakeApiCall:
        self._service.list_calls.append(kwargs)
        return _FakeApiCall(self._service._next_list_page)

    def get(self, **kwargs: Any) -> _FakeApiCall:
        self._service.get_calls.append(kwargs)
        message_id = kwargs["id"]
        return _FakeApiCall(lambda: self._service._get_message(message_id))


class _FakeHistoryResource:
    def __init__(self, service: FakeGmailService) -> None:
        self._service = service

    def list(self, **kwargs: Any) -> _FakeApiCall:
        self._service.history_calls.append(kwargs)
        return _FakeApiCall(self._service._next_history_page)


class _FakeUsersResource:
    def __init__(self, service: FakeGmailService) -> None:
        self._messages = _FakeMessagesResource(service)
        self._history = _FakeHistoryResource(service)
        self._service = service

    def messages(self) -> _FakeMessagesResource:
        return self._messages

    def history(self) -> _FakeHistoryResource:
        return self._history

    def getProfile(self, **kwargs: Any) -> _FakeApiCall:  # noqa: N802 - googleapiclient's name
        return _FakeApiCall(self._service._get_profile)


class FakeGmailService:
    """Configurable stand-in for the injected googleapiclient Gmail resource.

    `list_pages` and `history_pages` are queues: each call to `messages().list()` /
    `history().list()` pops and returns the next entry, or raises it if it is an
    `Exception`. `profile` is returned (or raised) on every `getProfile()` call — it is
    not a queue, since production code calls it at most once per fetch.
    """

    def __init__(
        self,
        *,
        list_pages: list[dict[str, Any] | Exception] | None = None,
        history_pages: list[dict[str, Any] | Exception] | None = None,
        messages: dict[str, dict[str, Any]] | None = None,
        profile: dict[str, Any] | Exception | None = None,
    ) -> None:
        self._list_pages: list[dict[str, Any] | Exception] = list(list_pages or [])
        self._history_pages: list[dict[str, Any] | Exception] = list(history_pages or [])
        self._messages: dict[str, dict[str, Any]] = dict(messages or {})
        self._profile: dict[str, Any] | Exception = (
            profile if profile is not None else {"historyId": "999"}
        )
        self.list_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []
        self.history_calls: list[dict[str, Any]] = []
        self._users = _FakeUsersResource(self)

    def users(self) -> _FakeUsersResource:
        return self._users

    def _next_list_page(self) -> dict[str, Any]:
        item = self._list_pages.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def _next_history_page(self) -> dict[str, Any]:
        item = self._history_pages.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def _get_message(self, message_id: str) -> dict[str, Any]:
        return self._messages[message_id]

    def _get_profile(self) -> dict[str, Any]:
        if isinstance(self._profile, Exception):
            raise self._profile
        return self._profile


def make_message_payload(
    message_id: str, *, thread_id: str | None = None, subject: str = "Test subject"
) -> dict[str, Any]:
    """A minimal, valid `users.messages.get(format='full')` payload for `message_id`."""
    return {
        "id": message_id,
        "threadId": thread_id or f"thread-{message_id}",
        "internalDate": "1751472251000",
        "labelIds": ["INBOX"],
        "snippet": subject,
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "From", "value": "Careers <careers@example.com>"},
                {"name": "To", "value": "candidate@example.com"},
                {"name": "Subject", "value": subject},
            ],
            "body": {"data": ""},
        },
    }
