"""Gmail OAuth: the desktop consent flow, token persistence, and silent refresh.

Nothing here is allowed to widen the scope. ``gmail.readonly`` is the entire budget
(invariant I11) and CI greps ``src/`` for any other Gmail scope, so a widened list fails
the build rather than shipping.

Secrets live under ``JOBTRACK_HOME``, never in the repo: ``credentials.json`` is the OAuth
client downloaded from Google Cloud, ``token.json`` is the grant this tool persists, mode
0600.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, cast

from google.auth.exceptions import GoogleAuthError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from jobtrack.config import Config
from jobtrack.errors import AuthError

logger = logging.getLogger(__name__)

GMAIL_SCOPES: list[str] = ["https://www.googleapis.com/auth/gmail.readonly"]
"""Read-only. Never widen this list. (I11)"""

TOKEN_FILE_MODE: Final[int] = 0o600
"""Owner read/write only. The token is a bearer credential for the user's mailbox."""

OAUTH_LOCAL_PORT: Final[int] = 0
"""0 = let the OS pick a free loopback port for the desktop consent redirect."""


# google-auth ships py.typed but leaves several Credentials methods unannotated, so
# --strict rejects calling them from typed code. Every untyped call is confined to one of
# the three shims below, each presenting a fully typed surface to the rest of this module,
# rather than scattering ignores across the call sites.


def _credentials_to_json(credentials: Credentials) -> str:
    """JSON serialization of `credentials`. Typed shim over an unannotated method."""
    # google-auth 2.x: to_json() has no annotations but returns a JSON string.
    return str(credentials.to_json())  # type: ignore[no-untyped-call]


def _credentials_from_file(path: Path) -> Credentials:
    """Parse an authorized-user JSON file. Typed shim over an unannotated classmethod.

    The `scopes` argument is deliberately left as None. Passing a scope list does not
    *validate* the file against it — google-auth simply overwrites ``credentials.scopes``
    with whatever was passed, so supplying GMAIL_SCOPES here would make a token granted
    ``gmail.modify`` report itself as read-only. Reading with None surfaces the scopes the
    user actually consented to, which is the only version worth reporting (I11).
    """
    # google-auth 2.x: from_authorized_user_file() has no annotations but returns Credentials.
    loaded = Credentials.from_authorized_user_file(str(path), None)  # type: ignore[no-untyped-call]
    return cast(Credentials, loaded)


def _refresh_in_place(credentials: Credentials) -> None:
    """Exchange the refresh token for a fresh access token, mutating `credentials`.

    Typed shim over an unannotated method. Performs network I/O.
    """
    # google-auth 2.x: refresh() has no annotations and returns None.
    credentials.refresh(Request())  # type: ignore[no-untyped-call]


def _write_token(path: Path, credentials: Credentials) -> None:
    """Persist credentials to `path` with mode 0600, creating the parent directory."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Open with the restrictive mode rather than chmod-ing afterwards: the latter
        # leaves a window in which the token is world-readable.
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, TOKEN_FILE_MODE)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(_credentials_to_json(credentials))
        os.chmod(path, TOKEN_FILE_MODE)  # in case the file already existed
    except OSError as exc:
        raise AuthError(f"could not write {path}: {exc}") from exc


def _read_token(path: Path) -> Credentials:
    """Parse token.json into Credentials, or raise AuthError if it is unusable."""
    try:
        credentials = _credentials_from_file(path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise AuthError(
            f"{path} is not a usable OAuth token ({exc}). Run `jobtrack auth login`."
        ) from exc
    return credentials


def _granted_scopes(credentials: Credentials) -> list[str]:
    """Scopes recorded on the token, sorted for deterministic reporting."""
    scopes = getattr(credentials, "scopes", None)
    return sorted(scopes) if scopes else []


def load_credentials(config: Config) -> Credentials:
    """Load and silently refresh the stored OAuth token.

    Args:
        config: Resolved configuration; ``config.token_path`` is read and, after a
            successful refresh, rewritten.

    Returns:
        Valid credentials for the Gmail API.

    Raises:
        AuthError: no token.json, or refresh failed (revoked/expired).
    """
    path = config.token_path
    if not path.is_file():
        raise AuthError(f"no OAuth token at {path}. Run `jobtrack auth login` first.")

    credentials = _read_token(path)

    granted = _granted_scopes(credentials)
    if granted and granted != sorted(GMAIL_SCOPES):
        # Not fatal — the API call itself is the real authority — but a token carrying
        # anything other than gmail.readonly means a wider grant is sitting on disk (I11).
        logger.warning(
            "the token at %s was granted %s, not %s; re-run `jobtrack auth login`",
            path,
            ", ".join(granted),
            ", ".join(GMAIL_SCOPES),
        )

    if not credentials.valid:
        if not credentials.refresh_token:
            raise AuthError(
                f"the token at {path} is expired and carries no refresh token. "
                "Run `jobtrack auth login` again."
            )
        logger.info("access token expired; refreshing silently")
        try:
            _refresh_in_place(credentials)
        except GoogleAuthError as exc:
            raise AuthError(
                f"could not refresh the token at {path} ({exc}). It was probably revoked; "
                "run `jobtrack auth login` again."
            ) from exc
        except OSError as exc:  # transport failure during refresh
            raise AuthError(f"could not reach Google to refresh the token: {exc}") from exc
        _write_token(path, credentials)

    return credentials


def run_oauth_flow(config: Config) -> Credentials:
    """Run the interactive desktop OAuth consent flow and persist token.json (mode 0600).

    Args:
        config: Resolved configuration; ``config.credentials_path`` supplies the OAuth
            client and ``config.token_path`` receives the grant.

    Returns:
        The freshly granted credentials.

    Raises:
        AuthError: credentials.json missing or the user declined consent.
    """
    client_secrets = config.credentials_path
    if not client_secrets.is_file():
        raise AuthError(
            f"no OAuth client at {client_secrets}. Download the desktop credentials from "
            "Google Cloud Console and save them there."
        )

    try:
        flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets), GMAIL_SCOPES)
    except (OSError, ValueError, KeyError) as exc:
        raise AuthError(f"{client_secrets} is not a valid OAuth client file: {exc}") from exc

    try:
        credentials = flow.run_local_server(port=OAUTH_LOCAL_PORT)
    except GoogleAuthError as exc:
        raise AuthError(f"consent was not granted: {exc}") from exc
    except OSError as exc:
        raise AuthError(f"could not run the local consent server: {exc}") from exc

    if credentials is None:
        raise AuthError("the consent flow returned no credentials")

    granted = cast(Credentials, credentials)
    _write_token(config.token_path, granted)
    logger.info("stored OAuth token at %s", config.token_path)
    return granted


def credential_status(config: Config) -> dict[str, Any]:
    """Report token presence, expiry, and granted scopes for `jobtrack auth status`.

    Never raises for a missing or malformed token — an unusable token is a *reported*
    state, since reporting it is the whole point of the command.

    Args:
        config: Resolved configuration.

    Returns:
        A dict with these keys, stable for M6 to render:

        * ``token_path`` (str), ``credentials_path`` (str)
        * ``has_client_secrets`` (bool): credentials.json exists
        * ``has_token`` (bool): token.json exists
        * ``valid`` (bool): usable right now, without a refresh
        * ``expired`` (bool): present but past its expiry
        * ``expiry`` (str | None): ISO-8601 UTC, or None if the token records none
        * ``has_refresh_token`` (bool): a silent refresh is possible
        * ``scopes`` (list[str]): granted scopes, sorted
        * ``scopes_ok`` (bool): granted scopes cover GMAIL_SCOPES and nothing wider
        * ``error`` (str | None): why the token is unreadable, if it is
    """
    status: dict[str, Any] = {
        "token_path": str(config.token_path),
        "credentials_path": str(config.credentials_path),
        "has_client_secrets": config.credentials_path.is_file(),
        "has_token": config.token_path.is_file(),
        "valid": False,
        "expired": False,
        "expiry": None,
        "has_refresh_token": False,
        "scopes": [],
        "scopes_ok": False,
        "error": None,
    }

    if not status["has_token"]:
        status["error"] = "no token.json — run `jobtrack auth login`"
        return status

    try:
        credentials = _read_token(config.token_path)
    except AuthError as exc:
        status["error"] = str(exc)
        return status

    expiry: datetime | None = getattr(credentials, "expiry", None)
    if expiry is not None:
        # google-auth stores a naive UTC expiry; I7 says nothing naive leaves this module.
        aware = expiry.replace(tzinfo=UTC) if expiry.tzinfo is None else expiry.astimezone(UTC)
        status["expiry"] = aware.isoformat()

    scopes = _granted_scopes(credentials)
    status["scopes"] = scopes
    status["scopes_ok"] = scopes == sorted(GMAIL_SCOPES)
    status["valid"] = bool(credentials.valid)
    status["expired"] = bool(credentials.expired)
    status["has_refresh_token"] = credentials.refresh_token is not None
    return status


__all__ = [
    "GMAIL_SCOPES",
    "OAUTH_LOCAL_PORT",
    "TOKEN_FILE_MODE",
    "credential_status",
    "load_credentials",
    "run_oauth_flow",
]
