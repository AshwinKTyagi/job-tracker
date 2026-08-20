"""Test doubles for M1.

Nothing here touches the network — pytest runs with ``--disable-socket``, and the whole
point of ``GmailSource(credentials, service=...)`` is that the transport is injected.
``FakeGmailService`` mimics the shape of a googleapiclient discovery resource closely
enough that ``GmailSource`` cannot tell the difference: ``users().messages().list(...)``
returns an object with ``.execute()``.

It also records every endpoint it is asked for, which is how ``test_gmail.py`` proves
invariant I11 — that no mutating Gmail method is ever reached.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable, Sequence
from typing import Any

from googleapiclient.errors import HttpError

READ_ONLY_ENDPOINTS: frozenset[str] = frozenset(
    {
        "users.getProfile",
        "users.messages.list",
        "users.messages.get",
        "users.history.list",
    }
)
"""Every Gmail endpoint M1 is allowed to touch. All four are read-only (I11)."""


def encode_body(text: str, charset: str = "utf-8") -> str:
    """Base64url-encode `text` the way Gmail encodes a message part body."""
    return base64.urlsafe_b64encode(text.encode(charset)).decode("ascii")


class _FakeResponse:
    """The minimal ``httplib2.Response`` surface that HttpError actually reads.

    Standing this up locally rather than importing httplib2 keeps the test suite free of
    a stub-less transitive dependency — types-httplib2 is not in PLAN.md's dependency
    list and M1 is not entitled to add one.
    """

    def __init__(self, status: int) -> None:
        self.status = status
        self.reason = "synthetic failure"


def make_http_error(status: int, reason: str | None = None) -> HttpError:
    """Build a googleapiclient HttpError with the given status and Google error reason.

    Args:
        status: HTTP status code.
        reason: Google's machine-readable reason, e.g. "userRateLimitExceeded".

    Returns:
        An HttpError shaped like the real thing, for driving the error-mapping paths.
    """
    response = _FakeResponse(status)
    payload: dict[str, Any] = {"error": {"code": status, "message": "synthetic failure"}}
    if reason is not None:
        payload["error"]["errors"] = [{"reason": reason, "message": "synthetic failure"}]
    return HttpError(response, json.dumps(payload).encode("utf-8"))


def make_payload(
    message_id: str = "m1",
    *,
    thread_id: str = "t1",
    internal_date: str = "1753349462000",  # 2025-07-24T09:31:02Z
    subject: str = "Thanks for applying",
    sender: str = "Acme Robotics <No-Reply@Greenhouse.IO>",
    recipient: str = "candidate@example.com",
    body: str | None = "Hello there.",
    html_body: str | None = None,
    labels: Sequence[str] = ("INBOX", "CATEGORY_UPDATES"),
    snippet: str = "Hello there.",
    history_id: str = "1000",
    extra_headers: Sequence[tuple[str, str]] = (),
) -> dict[str, Any]:
    """Build a ``users.messages.get(format='full')`` payload.

    With both `body` and `html_body` the result is multipart/alternative; with only one it
    is a single-part message of the matching type.

    Returns:
        A payload dict shaped like Gmail's response.
    """
    headers: list[dict[str, str]] = [
        {"name": "Subject", "value": subject},
        {"name": "From", "value": sender},
        {"name": "To", "value": recipient},
    ]
    headers += [{"name": name, "value": value} for name, value in extra_headers]

    payload: dict[str, Any] = {
        "id": message_id,
        "threadId": thread_id,
        "internalDate": internal_date,
        "historyId": history_id,
        "labelIds": list(labels),
        "snippet": snippet,
        "payload": {"headers": headers},
    }

    plain_part = {
        "mimeType": "text/plain",
        "filename": "",
        "headers": [{"name": "Content-Type", "value": "text/plain; charset=UTF-8"}],
        "body": {"data": encode_body(body or "")},
    }
    html_part = {
        "mimeType": "text/html",
        "filename": "",
        "headers": [{"name": "Content-Type", "value": "text/html; charset=UTF-8"}],
        "body": {"data": encode_body(html_body or "")},
    }

    if body is not None and html_body is not None:
        payload["payload"]["mimeType"] = "multipart/alternative"
        payload["payload"]["parts"] = [plain_part, html_part]
    elif body is not None or html_body is not None:
        # A single-part message has no sub-parts: Gmail puts the body and the content type
        # on the same node that carries the RFC822 headers. Merging the part's own
        # "headers" list here would wipe Subject/From/To, so only the scalar keys move.
        part = plain_part if body is not None else html_part
        payload["payload"]["mimeType"] = part["mimeType"]
        payload["payload"]["filename"] = part["filename"]
        payload["payload"]["body"] = part["body"]
        subtype = "plain" if body is not None else "html"
        headers.append({"name": "Content-Type", "value": f"text/{subtype}; charset=UTF-8"})
    else:
        payload["payload"]["mimeType"] = "text/plain"
        payload["payload"]["body"] = {}
    return payload


class _Script:
    """A queue of scripted outcomes for one endpoint.

    Dicts are returned from ``execute()``; exceptions are raised. The last entry repeats
    forever, so a one-element script answers every call.
    """

    def __init__(self, outcomes: Sequence[object] | None) -> None:
        self._outcomes: list[object] = list(outcomes) if outcomes else [{}]

    def next(self) -> object:
        """Pop the next outcome, holding on the final one."""
        if len(self._outcomes) > 1:
            return self._outcomes.pop(0)
        return self._outcomes[0]


class _FakeRequest:
    """The object googleapiclient returns from a resource method call.

    The outcome is resolved inside ``execute()``, not when the request is built. That
    matters: the retry loop re-executes the *same* request object, exactly as it does
    against the real client, so resolving early would make every retry see the first
    scripted outcome and never recover.
    """

    def __init__(
        self,
        service: FakeGmailService,
        endpoint: str,
        kwargs: dict[str, Any],
        resolve: Callable[[], object],
    ) -> None:
        self._service = service
        self._endpoint = endpoint
        self._kwargs = kwargs
        self._resolve = resolve

    def execute(self) -> Any:
        """Return the scripted response, or raise the scripted exception."""
        self._service.record(self._endpoint, self._kwargs)
        outcome = self._resolve()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _FakeMessages:
    def __init__(self, service: FakeGmailService) -> None:
        self._service = service

    def list(self, **kwargs: Any) -> _FakeRequest:
        """Stand in for ``users().messages().list``."""
        return _FakeRequest(
            self._service, "users.messages.list", kwargs, self._service.list_script.next
        )

    def get(self, **kwargs: Any) -> _FakeRequest:
        """Stand in for ``users().messages().get``."""
        service = self._service
        message_id = kwargs.get("id", "")

        def resolve() -> object:
            if message_id in service.get_errors:
                return service.get_errors[message_id]
            if message_id in service.payloads:
                return service.payloads[message_id]
            return make_http_error(404, "notFound")

        return _FakeRequest(service, "users.messages.get", kwargs, resolve)


class _FakeHistory:
    def __init__(self, service: FakeGmailService) -> None:
        self._service = service

    def list(self, **kwargs: Any) -> _FakeRequest:
        """Stand in for ``users().history().list``."""
        return _FakeRequest(
            self._service, "users.history.list", kwargs, self._service.history_script.next
        )


class _FakeUsers:
    def __init__(self, service: FakeGmailService) -> None:
        self._service = service

    def getProfile(self, **kwargs: Any) -> _FakeRequest:  # noqa: N802 — Gmail's own name
        """Stand in for ``users().getProfile``."""
        return _FakeRequest(
            self._service, "users.getProfile", kwargs, self._service.profile_script.next
        )

    def messages(self) -> _FakeMessages:
        """Stand in for ``users().messages()``."""
        return _FakeMessages(self._service)

    def history(self) -> _FakeHistory:
        """Stand in for ``users().history()``."""
        return _FakeHistory(self._service)


class FakeGmailService:
    """An injectable stand-in for a googleapiclient Gmail resource.

    Every endpoint is driven by a script: a sequence whose entries are either response
    dicts or exceptions to raise, with the final entry repeating. That is enough to
    exercise pagination, backoff, and the history-cursor downgrade without a socket.
    """

    def __init__(
        self,
        *,
        payloads: dict[str, dict[str, Any]] | None = None,
        list_responses: Sequence[object] | None = None,
        history_responses: Sequence[object] | None = None,
        profile_responses: Sequence[object] | None = None,
        get_errors: dict[str, BaseException] | None = None,
    ) -> None:
        """Args:
        payloads: message id to ``messages.get`` payload.
        list_responses: scripted ``messages.list`` outcomes.
        history_responses: scripted ``history.list`` outcomes.
        profile_responses: scripted ``getProfile`` outcomes.
        get_errors: message ids that should fail, and how.
        """
        self.payloads = payloads or {}
        self.list_script = _Script(list_responses)
        self.history_script = _Script(history_responses)
        self.profile_script = _Script(profile_responses or [{"historyId": "9000"}])
        self.get_errors = get_errors or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def record(self, endpoint: str, kwargs: dict[str, Any]) -> None:
        """Log one endpoint invocation for later assertions."""
        self.calls.append((endpoint, kwargs))

    def endpoints(self) -> list[str]:
        """Every endpoint touched, in call order."""
        return [name for name, _ in self.calls]

    def calls_to(self, endpoint: str) -> list[dict[str, Any]]:
        """The keyword arguments of every call to `endpoint`."""
        return [kwargs for name, kwargs in self.calls if name == endpoint]

    def users(self) -> _FakeUsers:
        """Stand in for ``service.users()``."""
        return _FakeUsers(self)
