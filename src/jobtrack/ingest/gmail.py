"""The Gmail implementation of ``EmailSource``.

Owns four things the rest of the system should never have to think about:

* **Pagination** — 100 ids per page, capped by ``limit``.
* **Backoff** — 429/403-quota/5xx/socket timeouts retry with exponential delay; everything
  else fails fast.
* **Delta sync** — ``history.list`` from the stored cursor, with a documented downgrade to
  a dated query when the cursor has aged out (Gmail expires history after roughly a week
  and answers 404).
* **The payload transform** — Gmail's MIME tree to a flat ``RawMessage``.

Read-only, always (invariant I11): the only Gmail methods reached from here are
``users.getProfile``, ``users.messages.list``, ``users.messages.get`` and
``users.history.list``. No mutating call exists in this module and none may be added.

No ``googleapiclient`` exception escapes: every API call goes through ``_execute``, which
maps ``HttpError`` onto the ``JobTrackError`` tree.
"""

from __future__ import annotations

import base64
import binascii
import codecs
import json
import logging
import re
import time
from datetime import UTC, datetime
from email.header import decode_header, make_header
from email.utils import parseaddr
from html import unescape
from typing import Any, Final, cast

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from jobtrack.errors import AuthError, IngestError, PermanentIngestError, TransientIngestError
from jobtrack.ingest.html import collapse_whitespace, html_to_text
from jobtrack.ingest.source import FetchResult
from jobtrack.models import RawMessage

logger = logging.getLogger(__name__)

USER_ID: Final[str] = "me"
"""Gmail's alias for the authenticated user. This tool never reads another mailbox."""

PAGE_SIZE: Final[int] = 100
"""Ids per list/history page. Gmail's own maximum for messages.list is 500, but 100 keeps
each round trip small and the cap arithmetic honest."""

MAX_RETRIES: Final[int] = 5
INITIAL_BACKOFF_SECONDS: Final[float] = 1.0
BACKOFF_MULTIPLIER: Final[float] = 2.0
MAX_BACKOFF_SECONDS: Final[float] = 32.0

TRANSIENT_STATUSES: Final[frozenset[int]] = frozenset({408, 429, 500, 502, 503, 504})
"""Statuses worth retrying. 403 is decided by its reason — see ``_wrap_http_error``."""

RATE_LIMIT_REASONS: Final[frozenset[str]] = frozenset(
    {
        "ratelimitexceeded",
        "userratelimitexceeded",
        "quotaexceeded",
        "dailylimitexceeded",
        "backenderror",
        "internalerror",
    }
)
"""Gmail returns quota exhaustion as 403, not 429. Without this the sync would give up on
a condition that clears in seconds."""

AUTH_STATUSES: Final[frozenset[int]] = frozenset({401})
"""Unambiguously a credential problem. A 403 is only an auth problem once its reason has
been ruled out as a quota message."""

HISTORY_EXPIRED_STATUS: Final[int] = 404
"""``history.list`` answers 404 once ``startHistoryId`` is older than Gmail's retention
window (about a week). That is a cursor downgrade, not a failure."""

HISTORY_TYPES: Final[list[str]] = ["messageAdded"]
"""Deletions and label changes do not create job-application events."""

EXCLUDED_LABELS: Final[frozenset[str]] = frozenset({"CHAT", "DRAFT", "SPAM", "TRASH"})
"""The delta path sees everything that lands in the mailbox, including things the search
query would have excluded. Filtering these mirrors the query's ``-in:chats``."""

MAX_MIME_DEPTH: Final[int] = 20
"""Recursion guard. Real mail nests three or four levels; anything deeper is malformed."""

GMAIL_DATE_FORMAT: Final[str] = "%Y/%m/%d"
"""Gmail's ``after:`` operator takes YYYY/MM/DD and nothing finer."""

_CHARSET_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    charset \s* = \s*      # the Content-Type parameter
    "? ([\w.:+-]+) "?      # value, optionally quoted
    """,
    re.IGNORECASE | re.VERBOSE,
)

DEFAULT_CHARSET: Final[str] = "utf-8"


class _CursorExpiredError(Exception):
    """Internal signal that ``history.list`` 404'd. Never leaves this module."""


def _sleep(seconds: float) -> None:
    """Block for `seconds`. A seam so backoff tests do not actually wait."""
    time.sleep(seconds)


def _http_status(exc: HttpError) -> int | None:
    """Best-effort HTTP status from an HttpError, or None if it carries none."""
    response = getattr(exc, "resp", None)
    status = getattr(response, "status", None)
    if isinstance(status, int):
        return status
    if isinstance(status, str) and status.isdigit():
        return int(status)
    code = getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def _http_reason(exc: HttpError) -> str:
    """Lowercased Google error reason (e.g. "userRateLimitExceeded"), or "" if absent."""
    content = getattr(exc, "content", None)
    if not content:
        return ""
    try:
        text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content
        body = json.loads(text)
    except (ValueError, AttributeError):
        return ""
    if not isinstance(body, dict):
        return ""
    error = body.get("error")
    if not isinstance(error, dict):
        return ""
    errors = error.get("errors")
    if isinstance(errors, list) and errors and isinstance(errors[0], dict):
        reason = errors[0].get("reason")
        if isinstance(reason, str):
            return reason.lower()
    status = error.get("status")
    return status.lower() if isinstance(status, str) else ""


def _wrap_http_error(exc: HttpError, context: str) -> IngestError | AuthError:
    """Map an HttpError onto the JobTrackError tree.

    Args:
        exc: The googleapiclient error. It must not escape ingest/.
        context: The API call that failed, for the message.

    Returns:
        The error to raise: transient for 429/5xx/quota-403, AuthError for 401 and a
        non-quota 403, permanent for everything else.
    """
    status = _http_status(exc)
    reason = _http_reason(exc)
    detail = f"{context} failed with HTTP {status}" + (f" ({reason})" if reason else "")

    if status in AUTH_STATUSES:
        return AuthError(f"{detail}: the Gmail token was rejected. Run `jobtrack auth login`.")
    if status == 403:
        if reason in RATE_LIMIT_REASONS:
            return TransientIngestError(f"{detail}: quota exhausted, retry later")
        return AuthError(f"{detail}: the token lacks permission for this mailbox")
    if status is None or status in TRANSIENT_STATUSES or status >= 500:
        return TransientIngestError(f"{detail}: retryable")
    return PermanentIngestError(f"{detail}: not retryable")


def _execute(
    request: Any, context: str, *, expired_cursor_status: int | None = None
) -> dict[str, Any]:
    """Run one Gmail API request, retrying transient failures with exponential backoff.

    Args:
        request: A googleapiclient request object exposing ``execute()``.
        context: Human-readable name of the call, used in error messages and logs.
        expired_cursor_status: If set, this status raises ``_CursorExpiredError`` instead
            of being wrapped — the caller treats it as a downgrade, not a failure.

    Returns:
        The decoded JSON response.

    Raises:
        TransientIngestError: still failing after MAX_RETRIES.
        PermanentIngestError: unrecoverable 4xx.
        AuthError: 401, or a 403 that is not a quota message.
        _CursorExpiredError: the status matched ``expired_cursor_status``.
    """
    delay = INITIAL_BACKOFF_SECONDS
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return cast("dict[str, Any]", request.execute())
        except HttpError as exc:
            if expired_cursor_status is not None and _http_status(exc) == expired_cursor_status:
                raise _CursorExpiredError(context) from exc
            error = _wrap_http_error(exc, context)
            if not isinstance(error, TransientIngestError) or attempt == MAX_RETRIES:
                raise error from exc
            logger.warning(
                "%s: %s — retry %d/%d in %.1fs", context, error, attempt, MAX_RETRIES, delay
            )
        except OSError as exc:
            # Socket timeouts, connection resets, DNS blips: transport-level and retryable.
            if attempt == MAX_RETRIES:
                raise TransientIngestError(
                    f"{context} failed after {attempt} attempts: {exc}"
                ) from exc
            logger.warning(
                "%s: %s — retry %d/%d in %.1fs", context, exc, attempt, MAX_RETRIES, delay
            )
        _sleep(delay)
        delay = min(delay * BACKOFF_MULTIPLIER, MAX_BACKOFF_SECONDS)
    raise TransientIngestError(f"{context} exhausted {MAX_RETRIES} attempts")


def _require_field(payload: dict[str, Any], key: str) -> str:
    """Read a required scalar field from a Gmail payload as a string."""
    value = payload.get(key)
    if isinstance(value, str) and value:
        return value
    if isinstance(value, int):
        return str(value)
    raise PermanentIngestError(f"Gmail payload is missing required field {key!r}")


def _decode_header_value(raw: str) -> str:
    """Decode RFC 2047 encoded words (``=?utf-8?q?...?=``) to plain text."""
    if "=?" not in raw:
        return raw
    try:
        return str(make_header(decode_header(raw)))
    except (UnicodeDecodeError, LookupError, ValueError) as exc:
        logger.debug("undecodable header %r: %s", raw, exc)
        return raw


def _extract_headers(part: dict[str, Any]) -> dict[str, str]:
    """Flatten a Gmail header list into a dict with lowercased keys.

    The first occurrence of a repeated header wins, which keeps the result stable for
    headers Gmail stacks (``received``, ``dkim-signature``).
    """
    headers: dict[str, str] = {}
    entries = part.get("headers")
    if not isinstance(entries, list):
        return headers
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        value = entry.get("value")
        if not isinstance(name, str) or not isinstance(value, str):
            continue
        key = name.lower()
        if key not in headers:
            headers[key] = _decode_header_value(value)
    return headers


def _parse_address(raw: str) -> tuple[str | None, str]:
    """Split an address header into (display name or None, lowercased address)."""
    if not raw:
        return None, ""
    name, address = parseaddr(raw)
    address = address.strip().lower()
    if not address:
        address = raw.strip().lower()
    display = name.strip()
    return (display or None), address


def _charset_of(part: dict[str, Any], headers: dict[str, str]) -> str:
    """Codec named by the part's Content-Type, or utf-8 when absent or unknown."""
    content_type = headers.get("content-type") or part.get("mimeType") or ""
    match = _CHARSET_RE.search(content_type)
    if match is None:
        return DEFAULT_CHARSET
    candidate = match.group(1)
    try:
        codecs.lookup(candidate)
    except LookupError:
        logger.debug("unknown charset %r; falling back to %s", candidate, DEFAULT_CHARSET)
        return DEFAULT_CHARSET
    return candidate


def _decode_part_body(part: dict[str, Any]) -> str:
    """Base64url-decode one leaf part's body, honouring its declared charset."""
    body = part.get("body")
    if not isinstance(body, dict):
        return ""
    data = body.get("data")
    if not isinstance(data, str) or not data:
        return ""
    padded = data + "=" * (-len(data) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded)
    except (binascii.Error, ValueError) as exc:
        logger.warning("undecodable message part body: %s", exc)
        return ""
    charset = _charset_of(part, _extract_headers(part))
    return raw.decode(charset, errors="replace")


def _walk_parts(
    part: dict[str, Any], plain: list[str], html_parts: list[str], depth: int = 0
) -> None:
    """Depth-first MIME walk collecting text/plain and text/html leaves.

    Attachments (any part with a filename) are skipped: their bytes are not prose, and
    downloading them would need a second API call per part.
    """
    if depth > MAX_MIME_DEPTH:
        logger.warning("MIME tree deeper than %d levels; truncating", MAX_MIME_DEPTH)
        return
    if part.get("filename"):
        return

    mime_type = part.get("mimeType")
    mime = mime_type.lower() if isinstance(mime_type, str) else ""

    children = part.get("parts")
    if isinstance(children, list) and children:
        for child in children:
            if isinstance(child, dict):
                _walk_parts(child, plain, html_parts, depth + 1)
        return

    if mime == "text/plain":
        text = _decode_part_body(part)
        if text:
            plain.append(text)
    elif mime == "text/html":
        text = _decode_part_body(part)
        if text:
            html_parts.append(text)


def _body_text(payload: dict[str, Any]) -> str:
    """Extract the readable body, preferring text/plain over converted text/html."""
    plain: list[str] = []
    html_parts: list[str] = []
    _walk_parts(payload, plain, html_parts)
    if plain:
        return collapse_whitespace("\n".join(plain))
    if html_parts:
        return html_to_text("\n".join(html_parts))
    return ""


def _internal_date_to_utc(raw: str) -> datetime:
    """Convert Gmail's ``internalDate`` (epoch milliseconds) to a tz-aware UTC datetime."""
    try:
        milliseconds = int(raw)
    except ValueError as exc:
        raise PermanentIngestError(f"internalDate {raw!r} is not epoch milliseconds") from exc
    seconds, remainder = divmod(milliseconds, 1000)
    try:
        return datetime.fromtimestamp(seconds, tz=UTC).replace(microsecond=remainder * 1000)
    except (OverflowError, OSError, ValueError) as exc:
        raise PermanentIngestError(f"internalDate {raw!r} is out of range") from exc


def parse_gmail_message(payload: dict[str, Any]) -> RawMessage:
    """Convert a raw ``users.messages.get(format='full')`` payload into a RawMessage.

    Walks the MIME tree preferring text/plain, falling back to ``html_to_text(text/html)``.
    Lowercases header keys and the sender address; normalizes ``internalDate`` to UTC.

    Args:
        payload: The decoded JSON body of a ``users.messages.get`` response.

    Returns:
        The normalized message.

    Raises:
        PermanentIngestError: payload is missing id, threadId, or internalDate.
    """
    message_id = _require_field(payload, "id")
    thread_id = _require_field(payload, "threadId")
    received_at = _internal_date_to_utc(_require_field(payload, "internalDate"))

    mime_root = payload.get("payload")
    root: dict[str, Any] = mime_root if isinstance(mime_root, dict) else {}
    headers = _extract_headers(root)

    from_name, from_email = _parse_address(headers.get("from", ""))
    _, to_address = _parse_address(headers.get("to", ""))

    raw_labels = payload.get("labelIds")
    labels = (
        [label for label in raw_labels if isinstance(label, str)]
        if (isinstance(raw_labels, list))
        else []
    )

    raw_snippet = payload.get("snippet")
    snippet = collapse_whitespace(unescape(raw_snippet)) if isinstance(raw_snippet, str) else ""

    return RawMessage(
        message_id=message_id,
        thread_id=thread_id,
        received_at=received_at,
        from_email=from_email,
        from_name=from_name,
        to_email=to_address or None,
        subject=headers.get("subject", ""),
        body_text=_body_text(root),
        snippet=snippet,
        labels=labels,
        headers=headers,
    )


def _is_excluded(label_ids: Any) -> bool:
    """True if a message carries a label the mailbox query would have excluded."""
    if not isinstance(label_ids, list):
        return False
    return any(label in EXCLUDED_LABELS for label in label_ids if isinstance(label, str))


def _max_history_id(values: list[str]) -> str | None:
    """Largest history id, compared numerically. Gmail ids are monotonic decimal strings."""
    numeric = [value for value in values if value.isdigit()]
    if not numeric:
        return None
    return max(numeric, key=int)


class GmailSource:
    """EmailSource backed by the Gmail API.

    Owns: pagination, exponential backoff on 429/5xx, historyId delta sync with a
    documented fallback to a dated query when history.list 404s (deltas expire after
    ~1 week), and the Gmail-payload to RawMessage transform.
    """

    name = "gmail"

    def __init__(self, credentials: Credentials, *, service: Any | None = None) -> None:
        """Args:
        credentials: from auth.load_credentials.
        service: injected googleapiclient resource. Tests pass a fake here — this
            parameter is the reason ingest is testable without a network.
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

        Prefers a ``history.list`` delta from `cursor`; when Gmail has expired that cursor
        (404, after roughly a week) the downgrade to a dated query from `since` is logged
        at WARNING and the fetch continues rather than failing.

        The returned cursor is the mailbox's historyId read *before* any message is
        fetched, so mail arriving mid-fetch is picked up next run instead of being skipped.

        Args:
            query: A Gmail search expression; ``after:`` is appended when `since` is given.
            since: Lower bound for the dated path. Tz-aware UTC.
            cursor: A Gmail historyId from a previous ``FetchResult.next_cursor``.
            limit: Maximum number of message ids to pull. The batch can still come back
                smaller, since chat/draft/spam/trash messages are dropped after listing.

        Returns:
            The batch, the next cursor, and whether `limit` truncated the results.

        Raises:
            TransientIngestError: rate limited, 5xx, or timed out — retry with backoff.
            PermanentIngestError: malformed query, unrecoverable 4xx, or an unparseable
                message payload.
            AuthError: credentials missing, expired, or revoked.
        """
        fetched_at = datetime.now(UTC)
        start_history_id = self._current_history_id()

        message_ids: list[str]
        truncated: bool
        if cursor:
            try:
                message_ids, truncated = self._delta_message_ids(cursor, limit)
            except _CursorExpiredError:
                logger.warning(
                    "Gmail history cursor %s has expired (deltas live ~1 week); "
                    "downgrading to a dated query from %s",
                    cursor,
                    since.isoformat() if since else "the beginning",
                )
                message_ids, truncated = self._search_message_ids(query, since, limit)
        else:
            message_ids, truncated = self._search_message_ids(query, since, limit)

        messages: list[RawMessage] = []
        history_ids: list[str] = []
        for message_id in message_ids:
            payload = self._get_payload(message_id)
            if _is_excluded(payload.get("labelIds")):
                logger.debug("skipping %s: chat/draft/spam/trash", message_id)
                continue
            history_id = payload.get("historyId")
            if isinstance(history_id, str):
                history_ids.append(history_id)
            messages.append(parse_gmail_message(payload))

        next_cursor = start_history_id or _max_history_id(history_ids) or cursor
        logger.info(
            "fetched %d message(s) from Gmail; next cursor %s%s",
            len(messages),
            next_cursor,
            " (truncated)" if truncated else "",
        )
        return FetchResult(
            messages=messages,
            next_cursor=next_cursor,
            fetched_at=fetched_at,
            truncated=truncated,
        )

    def _current_history_id(self) -> str | None:
        """Read the mailbox's current historyId, or None if the profile call is refused.

        A permanent failure here is not worth aborting a sync for: the cursor falls back
        to the highest historyId among the fetched messages.
        """
        request = self._service.users().getProfile(userId=USER_ID)
        try:
            profile = _execute(request, "users.getProfile")
        except PermanentIngestError as exc:
            logger.warning("could not read the mailbox profile (%s); deriving cursor instead", exc)
            return None
        history_id = profile.get("historyId")
        if isinstance(history_id, str):
            return history_id
        if isinstance(history_id, int):
            return str(history_id)
        return None

    def _search_message_ids(
        self, query: str, since: datetime | None, limit: int | None
    ) -> tuple[list[str], bool]:
        """Page through ``messages.list`` collecting ids.

        Returns:
            (ids in Gmail's newest-first order, whether `limit` cut the results short).
        """
        if limit is not None and limit <= 0:
            return [], False

        search = query
        if since is not None:
            search = f"{query} after:{since.astimezone(UTC).strftime(GMAIL_DATE_FORMAT)}"

        ids: list[str] = []
        page_token: str | None = None
        while True:
            page_size = PAGE_SIZE if limit is None else min(PAGE_SIZE, limit - len(ids))
            request = (
                self._service.users()
                .messages()
                .list(userId=USER_ID, q=search, maxResults=page_size, pageToken=page_token)
            )
            response = _execute(request, "users.messages.list")
            entries = response.get("messages")
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, dict) and isinstance(entry.get("id"), str):
                        ids.append(entry["id"])
            next_token = response.get("nextPageToken")
            page_token = next_token if isinstance(next_token, str) and next_token else None

            if limit is not None and len(ids) >= limit:
                return ids[:limit], page_token is not None or len(ids) > limit
            if page_token is None:
                return ids, False

    def _delta_message_ids(self, cursor: str, limit: int | None) -> tuple[list[str], bool]:
        """Page through ``history.list`` collecting ids added since `cursor`.

        History is not searchable, so this path cannot apply the mailbox query; it returns
        every new message and lets the classifier filter, which is what the query is tuned
        for anyway (``DEFAULT_GMAIL_QUERY`` is recall-first). Chat/draft/spam/trash are
        dropped here because history reports their labels for free.

        Raises:
            _CursorExpiredError: Gmail no longer retains history from `cursor`.
        """
        if limit is not None and limit <= 0:
            return [], False

        ids: list[str] = []
        seen: set[str] = set()
        page_token: str | None = None
        while True:
            request = (
                self._service.users()
                .history()
                .list(
                    userId=USER_ID,
                    startHistoryId=cursor,
                    historyTypes=HISTORY_TYPES,
                    maxResults=PAGE_SIZE,
                    pageToken=page_token,
                )
            )
            response = _execute(
                request, "users.history.list", expired_cursor_status=HISTORY_EXPIRED_STATUS
            )
            for record in _as_dict_list(response.get("history")):
                for added in _as_dict_list(record.get("messagesAdded")):
                    message = added.get("message")
                    if not isinstance(message, dict):
                        continue
                    message_id = message.get("id")
                    if not isinstance(message_id, str) or message_id in seen:
                        continue
                    if _is_excluded(message.get("labelIds")):
                        continue
                    seen.add(message_id)
                    ids.append(message_id)

            next_token = response.get("nextPageToken")
            page_token = next_token if isinstance(next_token, str) and next_token else None

            if limit is not None and len(ids) >= limit:
                return ids[:limit], page_token is not None or len(ids) > limit
            if page_token is None:
                return ids, False

    def _get_payload(self, message_id: str) -> dict[str, Any]:
        """Fetch one full message payload."""
        request = self._service.users().messages().get(userId=USER_ID, id=message_id, format="full")
        return _execute(request, f"users.messages.get({message_id})")


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    """Coerce an untrusted JSON field to a list of dicts, dropping anything else."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


__all__ = [
    "EXCLUDED_LABELS",
    "MAX_RETRIES",
    "PAGE_SIZE",
    "USER_ID",
    "GmailSource",
    "parse_gmail_message",
]
