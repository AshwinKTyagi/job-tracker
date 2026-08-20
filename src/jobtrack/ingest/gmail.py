"""The Gmail-backed `EmailSource`.

Owns pagination, exponential backoff on 429/5xx, historyId delta sync with a fallback to a
dated query when the delta has expired, and the Gmail-payload -> RawMessage transform.
"""

from __future__ import annotations

import base64
import binascii
import logging
import re
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from email.header import decode_header, make_header
from email.utils import parseaddr
from typing import Any, Final

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from jobtrack.errors import AuthError, JobTrackError, PermanentIngestError, TransientIngestError
from jobtrack.ingest.html import html_to_text
from jobtrack.ingest.source import FetchResult
from jobtrack.models import RawMessage

logger = logging.getLogger(__name__)

# Used when the caller passes limit=None. Matches GmailConfig.max_per_sync's default so a
# bare `source.fetch(query=...)` call behaves like a normal sync.
_DEFAULT_FETCH_LIMIT: Final[int] = 500

# PLAN.md §4: "Batch in pages of 100 with exponential backoff on 429/5xx."
_LIST_PAGE_SIZE: Final[int] = 100

_MAX_RETRIES: Final[int] = 5
_INITIAL_BACKOFF_SECONDS: Final[float] = 1.0
_BACKOFF_MULTIPLIER: Final[float] = 2.0
_MAX_BACKOFF_SECONDS: Final[float] = 30.0
_RETRYABLE_STATUS_CODES: Final[frozenset[int]] = frozenset({429, 500, 502, 503, 504})

_HTTP_UNAUTHORIZED: Final[int] = 401
_HTTP_FORBIDDEN: Final[int] = 403
_HTTP_NOT_FOUND: Final[int] = 404

_HISTORY_TYPE_MESSAGE_ADDED: Final[str] = "messageAdded"
_SINCE_QUERY_DATE_FORMAT: Final[str] = "%Y/%m/%d"

# googleapiclient folds Gmail's rate/quota errors into HTTP 403 alongside genuine
# permission failures; the reason text is the only way to tell them apart.
_RATE_LIMIT_REASON_RE: Final[re.Pattern[str]] = re.compile(r"rate|quota", re.IGNORECASE)


class _HistoryExpiredError(Exception):
    """Internal signal only: `history.list` 404'd, so the cursor is stale.

    Caught inside `GmailSource.fetch`; never escapes this module.
    """


def _http_status(exc: HttpError) -> int | None:
    """Best-effort extraction of the HTTP status code from an `HttpError`."""
    status = getattr(exc.resp, "status", None)
    if status is None:
        return None
    try:
        return int(status)
    except (TypeError, ValueError):
        return None


def _wrap_http_error(exc: HttpError) -> JobTrackError:
    """Map a googleapiclient `HttpError` to the jobtrack error it must never escape as.

    401 -> AuthError. 403 -> TransientIngestError if the reason text names a rate/quota
    limit, else PermanentIngestError (genuine permission denial). 429 and 5xx ->
    TransientIngestError. Every other 4xx -> PermanentIngestError.
    """
    status = _http_status(exc)
    reason = str(exc.reason) if exc.reason else str(exc)
    if status == _HTTP_UNAUTHORIZED:
        return AuthError(f"gmail credentials invalid or revoked: {reason}")
    if status == _HTTP_FORBIDDEN:
        if _RATE_LIMIT_REASON_RE.search(reason):
            return TransientIngestError(f"gmail rate limited: {reason}")
        return PermanentIngestError(f"gmail permission denied: {reason}")
    if status is not None and (status == 429 or status >= 500):
        return TransientIngestError(f"gmail transient error {status}: {reason}")
    return PermanentIngestError(f"gmail request failed (status={status}): {reason}")


def _execute_with_retry(request: Any) -> dict[str, Any]:
    """Execute a googleapiclient request, retrying with exponential backoff on 429/5xx.

    Args:
        request: An unexecuted googleapiclient request, e.g.
            `service.users().messages().get(...)`.

    Returns:
        The decoded JSON response body.

    Raises:
        HttpError: a non-retryable status, or retries were exhausted — raised un-wrapped
            so callers (notably history.list's 404-means-stale-cursor case) can inspect
            the status before deciding how to wrap it.
        TimeoutError: the request timed out and retries were exhausted.
    """
    attempt = 0
    while True:
        try:
            result: dict[str, Any] = request.execute()
            return result
        except HttpError as exc:
            status = _http_status(exc)
            if status in _RETRYABLE_STATUS_CODES and attempt < _MAX_RETRIES:
                _backoff_and_log(attempt, f"gmail api returned {status}")
                attempt += 1
                continue
            raise
        except TimeoutError:
            if attempt < _MAX_RETRIES:
                _backoff_and_log(attempt, "gmail api timed out")
                attempt += 1
                continue
            raise


def _backoff_and_log(attempt: int, reason: str) -> None:
    """Sleep the exponential backoff delay for `attempt` and log the retry."""
    delay = min(_INITIAL_BACKOFF_SECONDS * (_BACKOFF_MULTIPLIER**attempt), _MAX_BACKOFF_SECONDS)
    logger.warning(
        "%s; retrying in %.1fs (attempt %d/%d)", reason, delay, attempt + 1, _MAX_RETRIES
    )
    time.sleep(delay)


def _execute(request: Any) -> dict[str, Any]:
    """`_execute_with_retry`, wrapping any escaping HttpError/timeout into a JobTrackError.

    Raises:
        AuthError: credentials invalid or revoked (401).
        TransientIngestError: rate limited, 5xx, or timed out even after retries.
        PermanentIngestError: an unrecoverable 4xx.
    """
    try:
        return _execute_with_retry(request)
    except HttpError as exc:
        raise _wrap_http_error(exc) from exc
    except TimeoutError as exc:
        raise TransientIngestError(f"gmail request timed out: {exc}") from exc


def _augment_query_with_since(query: str, since: datetime | None) -> str:
    """Append a Gmail `after:` operator derived from `since`, when given."""
    if since is None:
        return query
    return f"{query} after:{since.strftime(_SINCE_QUERY_DATE_FORMAT)}"


def _decode_mime_header(value: str) -> str:
    """Decode an RFC 2047 MIME encoded-word header value to plain text.

    Falls back to the raw value if it is not validly encoded.
    """
    if not value:
        return value
    try:
        return str(make_header(decode_header(value)))
    except (UnicodeDecodeError, ValueError, LookupError):
        return value


def _header_map(headers: list[dict[str, Any]]) -> dict[str, str]:
    """Build a lowercased header dict from Gmail's `[{"name": ..., "value": ...}]` list."""
    result: dict[str, str] = {}
    for entry in headers:
        name = entry.get("name")
        value = entry.get("value")
        if name is None or value is None:
            continue
        result[str(name).lower()] = str(value)
    return result


def _decode_body_data(data: str) -> str:
    """Decode Gmail's URL-safe, unpadded base64 body payload to text.

    Returns "" for undecodable data rather than raising — a single malformed MIME part
    should not fail the whole message.
    """
    padded = data + "=" * (-len(data) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
    except (binascii.Error, ValueError, UnicodeEncodeError):
        return ""
    return raw.decode("utf-8", errors="replace")


def _iter_mime_parts(part: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Depth-first walk of a Gmail MIME part tree."""
    yield part
    for child in part.get("parts") or []:
        yield from _iter_mime_parts(child)


def _extract_body_text(root_part: dict[str, Any]) -> str:
    """Prefer the first `text/plain` part; fall back to `html_to_text` on `text/html`.

    Both branches are routed through `html_to_text` so whitespace collapse is uniform
    regardless of source MIME type — RawMessage.body_text must be deterministic (I2).
    """
    plain: str | None = None
    html_body: str | None = None
    for part in _iter_mime_parts(root_part):
        mime_type = part.get("mimeType", "")
        data = (part.get("body") or {}).get("data")
        if not data:
            continue
        if mime_type == "text/plain" and plain is None:
            plain = _decode_body_data(data)
        elif mime_type == "text/html" and html_body is None:
            html_body = _decode_body_data(data)
    if plain is not None:
        return html_to_text(plain)
    if html_body is not None:
        return html_to_text(html_body)
    return ""


def parse_gmail_message(payload: dict[str, Any]) -> RawMessage:
    """Convert a raw `users.messages.get(format='full')` payload into a RawMessage.

    Walks the MIME tree preferring text/plain, falling back to html_to_text(text/html).
    Lowercases header keys and the sender address; normalizes `internalDate` to UTC.

    Args:
        payload: The decoded JSON body of a `users.messages.get` call.

    Returns:
        The normalized message.

    Raises:
        PermanentIngestError: payload is missing id, threadId, or internalDate.
    """
    message_id = payload.get("id")
    thread_id = payload.get("threadId")
    internal_date = payload.get("internalDate")
    if not message_id or not thread_id or not internal_date:
        raise PermanentIngestError(
            "gmail message payload is missing id, threadId, or internalDate "
            f"(present keys: {sorted(payload.keys())})"
        )
    try:
        received_at = datetime.fromtimestamp(int(internal_date) / 1000, tz=UTC)
    except (TypeError, ValueError) as exc:
        raise PermanentIngestError(
            f"gmail message {message_id} has a malformed internalDate: {internal_date!r}"
        ) from exc

    root_part = payload.get("payload") or {}
    headers = _header_map(root_part.get("headers") or [])

    from_name, from_email = parseaddr(headers.get("from", ""))
    from_email = from_email.lower()

    _, to_email_raw = parseaddr(headers.get("to", ""))
    to_email = to_email_raw.lower() or None

    subject = _decode_mime_header(headers.get("subject", ""))
    body_text = _extract_body_text(root_part)

    return RawMessage(
        message_id=str(message_id),
        thread_id=str(thread_id),
        received_at=received_at,
        from_email=from_email,
        from_name=_decode_mime_header(from_name) or None,
        to_email=to_email,
        subject=subject,
        body_text=body_text,
        snippet=str(payload.get("snippet", "")),
        labels=list(payload.get("labelIds") or []),
        headers=headers,
    )


class GmailSource:
    """EmailSource backed by the Gmail API.

    Owns: pagination, exponential backoff on 429/5xx, historyId delta sync with a
    documented fallback to a dated query when history.list 404s (deltas expire after
    ~1 week), and the Gmail-payload -> RawMessage transform.
    """

    name = "gmail"

    def __init__(self, credentials: Credentials, *, service: Any | None = None) -> None:
        """Construct a Gmail source.

        Args:
            credentials: From `auth.load_credentials`.
            service: Injected googleapiclient resource. Tests pass a fake here — this
                parameter is the reason ingest is testable without a network. When
                omitted, a real `gmail` v1 service is built from `credentials`.
        """
        self._credentials = credentials
        self._service: Any = (
            service
            if service is not None
            else build("gmail", "v1", credentials=credentials, cache_discovery=False)
        )

    def fetch(
        self,
        *,
        query: str,
        since: datetime | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> FetchResult:
        """Fetch messages matching `query`.

        Prefers a `history.list` delta from `cursor` when given; if the delta has
        expired (history.list 404s), the downgrade is logged and this falls back to a
        dated `messages.list` query built from `since`.

        Args:
            query: Gmail search query, e.g. `constants.DEFAULT_GMAIL_QUERY`.
            since: Lower bound for the dated-query fallback.
            cursor: A prior `FetchResult.next_cursor` (a Gmail historyId).
            limit: Maximum number of messages to return; defaults to 500.

        Returns:
            The matching messages plus a cursor for the next incremental fetch.

        Raises:
            TransientIngestError: rate limited, 5xx, or timed out — retry with backoff.
            PermanentIngestError: malformed query or unrecoverable 4xx.
            AuthError: credentials missing, expired, or revoked.
        """
        fetched_at = datetime.now(UTC)
        effective_limit = limit if limit is not None else _DEFAULT_FETCH_LIMIT
        if cursor:
            try:
                return self._fetch_delta(
                    cursor=cursor, limit=effective_limit, fetched_at=fetched_at
                )
            except _HistoryExpiredError:
                logger.warning(
                    "gmail history cursor %r expired (history.list 404); "
                    "falling back to a dated query",
                    cursor,
                )
        return self._fetch_dated(
            query=query, since=since, limit=effective_limit, fetched_at=fetched_at
        )

    def _fetch_delta(self, *, cursor: str, limit: int, fetched_at: datetime) -> FetchResult:
        """Fetch messages added since `cursor` via `history.list`, paginated.

        Raises:
            _HistoryExpiredError: the cursor is stale; caller falls back to a dated query.
            AuthError: credentials invalid or revoked.
            TransientIngestError: rate limited, 5xx, or timed out.
            PermanentIngestError: an unrecoverable, non-404 4xx.
        """
        message_ids: list[str] = []
        page_token: str | None = None
        latest_history_id = cursor
        while True:
            request = (
                self._service.users()
                .history()
                .list(
                    userId="me",
                    startHistoryId=cursor,
                    historyTypes=[_HISTORY_TYPE_MESSAGE_ADDED],
                    pageToken=page_token,
                )
            )
            try:
                payload = _execute_with_retry(request)
            except HttpError as exc:
                if _http_status(exc) == _HTTP_NOT_FOUND:
                    raise _HistoryExpiredError from exc
                raise _wrap_http_error(exc) from exc
            except TimeoutError as exc:
                raise TransientIngestError(f"gmail history.list timed out: {exc}") from exc

            for record in payload.get("history") or []:
                for added in record.get("messagesAdded") or []:
                    msg_id = (added.get("message") or {}).get("id")
                    if msg_id and msg_id not in message_ids:
                        message_ids.append(msg_id)

            latest_history_id = str(payload.get("historyId", latest_history_id))
            page_token = payload.get("nextPageToken")
            if page_token is None or len(message_ids) >= limit:
                break

        truncated = page_token is not None or len(message_ids) > limit
        messages = [self._fetch_full_message(mid) for mid in message_ids[:limit]]
        return FetchResult(
            messages=messages,
            next_cursor=latest_history_id,
            fetched_at=fetched_at,
            truncated=truncated,
        )

    def _fetch_dated(
        self, *, query: str, since: datetime | None, limit: int, fetched_at: datetime
    ) -> FetchResult:
        """Fetch messages matching `query`/`since` via `messages.list`, paginated.

        Raises:
            AuthError: credentials invalid or revoked.
            TransientIngestError: rate limited, 5xx, or timed out.
            PermanentIngestError: a malformed query or unrecoverable 4xx.
        """
        effective_query = _augment_query_with_since(query, since)
        message_ids: list[str] = []
        page_token: str | None = None
        while len(message_ids) < limit:
            page_size = min(_LIST_PAGE_SIZE, limit - len(message_ids))
            request = (
                self._service.users()
                .messages()
                .list(
                    userId="me",
                    q=effective_query,
                    maxResults=page_size,
                    pageToken=page_token,
                )
            )
            payload = _execute(request)
            for stub in payload.get("messages") or []:
                msg_id = stub.get("id")
                if msg_id:
                    message_ids.append(msg_id)
            page_token = payload.get("nextPageToken")
            if page_token is None:
                break

        truncated = page_token is not None
        messages = [self._fetch_full_message(mid) for mid in message_ids[:limit]]
        return FetchResult(
            messages=messages,
            next_cursor=self._current_history_id(),
            fetched_at=fetched_at,
            truncated=truncated,
        )

    def _fetch_full_message(self, message_id: str) -> RawMessage:
        """Fetch and parse one message by id."""
        request = self._service.users().messages().get(userId="me", id=message_id, format="full")
        payload = _execute(request)
        return parse_gmail_message(payload)

    def _current_history_id(self) -> str | None:
        """Best-effort lookup of the mailbox's current historyId for the next delta sync.

        A failure here degrades to `None` rather than failing the whole fetch: the
        messages already fetched are still good, and the next sync simply falls back to
        a dated query again (safe by I1/I9). Only degrades on Transient/Permanent
        errors — an AuthError still propagates, since by this point earlier calls on the
        same credentials will already have failed the same way.
        """
        request = self._service.users().getProfile(userId="me")
        try:
            payload = _execute(request)
        except (TransientIngestError, PermanentIngestError) as exc:
            logger.warning("could not read the current gmail historyId: %s", exc)
            return None
        history_id = payload.get("historyId")
        return str(history_id) if history_id is not None else None


__all__ = ["GmailSource", "parse_gmail_message"]
