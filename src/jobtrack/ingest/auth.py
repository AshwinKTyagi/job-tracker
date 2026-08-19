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
from typing import Any, Final

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


def _write_token(path: Path, credentials: Credentials) -> None:
    """Persist credentials to `path` with mode 0600, creating the parent directory."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Open with the restrictive mode rather than chmod-ing afterwards: the latter
        # leaves a window in which the token is world-readable.
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, TOKEN_FILE_MODE)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(credentials.to_json())
        os.chmod(path, TOKEN_FILE_MODE)  # in case the file already existed
    except OSError as exc:
        raise AuthError(f"could not write {path}: {exc}") from exc


def _read_token(path: Path) -> Credentials:
    """Parse token.json into Credentials, or raise AuthError if it is unusable."""
    try:
        credentials = Credentials.from_authorized_user_file(str(path), GMAIL_SCOPES)
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

    if not credentials.valid:
        if not credentials.refresh_token:
            raise AuthError(
                f"the token at {path} is expired and carries no refresh token. "
                "Run `jobtrack auth login` again."
            )
        logger.info("access token expired; refreshing silently")
        try:
            credentials.refresh(Request())
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

    _write_token(config.token_path, credentials)
    logger.info("stored OAuth token at %s", config.token_path)
    return credentials


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
