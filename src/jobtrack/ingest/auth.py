"""Gmail OAuth: token load/refresh, the interactive consent flow, and status reporting.

Scope is read-only and that is the whole budget (I11) — see ``GMAIL_SCOPES``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Final

from google.auth.exceptions import GoogleAuthError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from jobtrack.config import Config, ensure_home
from jobtrack.constants import GMAIL_SCOPES as _GMAIL_SCOPES
from jobtrack.errors import AuthError

logger = logging.getLogger(__name__)

# Re-exported (not redefined) from constants.py, which M0 already froze as the single
# source of truth — duplicating the literal here would risk the two lists drifting apart.
GMAIL_SCOPES: list[str] = list(_GMAIL_SCOPES)
"""Read-only. Never widen this list. (I11)"""

_TOKEN_FILE_MODE: Final[int] = 0o600


def _persist_token(config: Config, credentials: Credentials) -> None:
    """Write `credentials` to `config.token_path` as JSON, mode 0600.

    Raises:
        AuthError: the token file could not be written.
    """
    ensure_home(config)
    try:
        config.token_path.write_text(credentials.to_json())
        os.chmod(config.token_path, _TOKEN_FILE_MODE)
    except OSError as exc:
        raise AuthError(f"could not write {config.token_path}: {exc}") from exc


def load_credentials(config: Config) -> Credentials:
    """Load and silently refresh the stored OAuth token.

    A near-expiry token with a refresh token is refreshed in place and the renewed token
    is written back to `config.token_path`.

    Args:
        config: Resolved runtime configuration; `config.token_path` is read.

    Returns:
        Valid, unexpired credentials.

    Raises:
        AuthError: no token.json, or refresh failed (revoked/expired).
    """
    if not config.token_path.is_file():
        raise AuthError(
            f"no token at {config.token_path}; run `jobtrack auth login` first"
        )
    try:
        credentials = Credentials.from_authorized_user_file(
            str(config.token_path), GMAIL_SCOPES
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise AuthError(f"could not read {config.token_path}: {exc}") from exc

    if credentials.valid:
        return credentials

    if credentials.expired and credentials.refresh_token:
        try:
            credentials.refresh(Request())
        except GoogleAuthError as exc:
            raise AuthError(f"gmail token refresh failed: {exc}") from exc
        _persist_token(config, credentials)
        return credentials

    raise AuthError(
        f"token at {config.token_path} is invalid and cannot be refreshed; "
        "run `jobtrack auth login` again"
    )


def run_oauth_flow(config: Config) -> Credentials:
    """Run the interactive desktop OAuth consent flow and persist token.json (mode 0600).

    Args:
        config: Resolved runtime configuration; `config.credentials_path` is read.

    Returns:
        The freshly granted credentials.

    Raises:
        AuthError: credentials.json missing or the user declined consent.
    """
    if not config.credentials_path.is_file():
        raise AuthError(
            f"no OAuth client secrets at {config.credentials_path}; download the Desktop "
            "app credentials from Google Cloud Console and save them there"
        )
    try:
        flow = InstalledAppFlow.from_client_secrets_file(
            str(config.credentials_path), scopes=GMAIL_SCOPES
        )
        credentials = flow.run_local_server(port=0)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise AuthError(f"could not read {config.credentials_path}: {exc}") from exc
    except GoogleAuthError as exc:
        raise AuthError(f"gmail oauth consent flow failed: {exc}") from exc

    _persist_token(config, credentials)
    return credentials


def credential_status(config: Config) -> dict[str, Any]:
    """Report token presence, expiry, and granted scopes for `jobtrack auth status`.

    Never refreshes or performs network I/O — this is a local, read-only inspection of
    `config.token_path`.

    Args:
        config: Resolved runtime configuration.

    Returns:
        A dict with keys ``authenticated`` (bool), ``valid`` (bool), ``expired`` (bool),
        ``expiry`` (ISO-8601 string or None), ``scopes`` (list[str]), ``token_path`` (str),
        and ``error`` (str, present only when the token file could not be parsed).
    """
    status: dict[str, Any] = {
        "authenticated": False,
        "valid": False,
        "expired": True,
        "expiry": None,
        "scopes": [],
        "token_path": str(config.token_path),
    }
    if not config.token_path.is_file():
        return status

    try:
        credentials = Credentials.from_authorized_user_file(
            str(config.token_path), GMAIL_SCOPES
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        status["error"] = str(exc)
        return status

    status["authenticated"] = True
    status["valid"] = credentials.valid
    status["expired"] = bool(credentials.expired)
    status["expiry"] = credentials.expiry.isoformat() if credentials.expiry else None
    status["scopes"] = list(credentials.scopes) if credentials.scopes else []
    return status


__all__ = [
    "GMAIL_SCOPES",
    "credential_status",
    "load_credentials",
    "run_oauth_flow",
]
