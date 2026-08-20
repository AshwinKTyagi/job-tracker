"""Tests for ``ingest.auth``.

Every token here is synthetic and every credential path is exercised on a real file in
``tmp_path``. The one operation that genuinely needs a socket — refreshing an access token
— is replaced at the module's own shim (``_refresh_in_place``), which is why that shim
exists as a separate function.
"""

from __future__ import annotations

import json
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

import pytest
from google.auth.exceptions import RefreshError

from jobtrack.config import Config
from jobtrack.errors import AuthError
from jobtrack.ingest import auth
from jobtrack.ingest.auth import (
    GMAIL_SCOPES,
    TOKEN_FILE_MODE,
    credential_status,
    load_credentials,
    run_oauth_flow,
)

READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"


def write_token(
    path: Path,
    *,
    expires_in_days: float = 1.0,
    refresh_token: str | None = "refresh-token",
    scopes: list[str] | None = None,
) -> None:
    """Write a synthetic authorized-user token file at `path`."""
    expiry = (datetime.now(UTC) + timedelta(days=expires_in_days)).replace(tzinfo=None)
    info: dict[str, Any] = {
        "token": "access-token",
        "refresh_token": refresh_token,
        "token_uri": "https://oauth2.googleapis.com/token",
        "client_id": "client-id",
        "client_secret": "client-secret",
        "scopes": scopes if scopes is not None else [READONLY_SCOPE],
        "expiry": expiry.isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info), encoding="utf-8")


class FakeFlow:
    """Stand-in for ``InstalledAppFlow``; records the scopes it was constructed with."""

    last_scopes: ClassVar[list[str]] = []
    outcome: BaseException | None = None

    def __init__(self, credentials: Any) -> None:
        self._credentials = credentials

    @classmethod
    def from_client_secrets_file(cls, path: str, scopes: list[str]) -> FakeFlow:
        """Mirror the real classmethod, capturing the requested scopes."""
        cls.last_scopes = list(scopes)
        return cls(credentials=None)

    def run_local_server(self, port: int = 0) -> Any:
        """Mirror the real consent step."""
        outcome = type(self).outcome
        if outcome is not None:
            raise outcome
        return self._credentials


# --- the scope budget (I11) ------------------------------------------------------------


def test_scope_list_is_exactly_readonly() -> None:
    """Invariant I11: gmail.readonly is the entire budget. CI greps for this too."""
    assert GMAIL_SCOPES == [READONLY_SCOPE]


# --- load_credentials ------------------------------------------------------------------


def test_missing_token_raises_auth_error(tmp_config: Config) -> None:
    with pytest.raises(AuthError, match="no OAuth token"):
        load_credentials(tmp_config)


def test_malformed_token_raises_auth_error(tmp_config: Config) -> None:
    tmp_config.token_path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(AuthError, match="not a usable OAuth token"):
        load_credentials(tmp_config)


def test_token_missing_required_fields_raises_auth_error(tmp_config: Config) -> None:
    tmp_config.token_path.write_text(json.dumps({"token": "only"}), encoding="utf-8")
    with pytest.raises(AuthError, match="not a usable OAuth token"):
        load_credentials(tmp_config)


def test_valid_token_loads_without_refreshing(tmp_config: Config) -> None:
    write_token(tmp_config.token_path)
    credentials = load_credentials(tmp_config)
    assert credentials.valid
    assert credentials.token == "access-token"


def test_expired_token_without_refresh_token_raises(tmp_config: Config) -> None:
    write_token(tmp_config.token_path, expires_in_days=-1, refresh_token=None)
    with pytest.raises(AuthError, match="no refresh token"):
        load_credentials(tmp_config)


def test_expired_token_is_refreshed_and_rewritten(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_token(tmp_config.token_path, expires_in_days=-1)
    before = tmp_config.token_path.read_text(encoding="utf-8")
    refreshed: list[Any] = []

    def fake_refresh(credentials: Any) -> None:
        refreshed.append(credentials)
        credentials.token = "fresh-access-token"
        credentials.expiry = datetime.now(UTC).replace(tzinfo=None) + timedelta(days=1)

    monkeypatch.setattr(auth, "_refresh_in_place", fake_refresh)
    credentials = load_credentials(tmp_config)

    assert len(refreshed) == 1
    assert credentials.token == "fresh-access-token"
    # I9-adjacent: the refreshed token has to survive the process, not just this call.
    assert tmp_config.token_path.read_text(encoding="utf-8") != before
    assert json.loads(tmp_config.token_path.read_text(encoding="utf-8"))["token"] == (
        "fresh-access-token"
    )


def test_refresh_failure_becomes_auth_error(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A revoked grant surfaces as AuthError (exit 3), never as a google-auth exception."""
    write_token(tmp_config.token_path, expires_in_days=-1)

    def fake_refresh(credentials: Any) -> None:
        # google-auth 2.x: RefreshError.__init__ is unannotated.
        raise RefreshError("token has been revoked")  # type: ignore[no-untyped-call]

    monkeypatch.setattr(auth, "_refresh_in_place", fake_refresh)
    with pytest.raises(AuthError, match="revoked"):
        load_credentials(tmp_config)


def test_refresh_transport_failure_becomes_auth_error(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_token(tmp_config.token_path, expires_in_days=-1)

    def fake_refresh(credentials: Any) -> None:
        raise OSError("connection reset")

    monkeypatch.setattr(auth, "_refresh_in_place", fake_refresh)
    with pytest.raises(AuthError, match="could not reach Google"):
        load_credentials(tmp_config)


def test_wider_scope_on_disk_is_reported(
    tmp_config: Config, caplog: pytest.LogCaptureFixture
) -> None:
    """A token granting more than gmail.readonly must not pass unremarked (I11)."""
    write_token(tmp_config.token_path, scopes=["https://www.googleapis.com/auth/gmail.modify"])
    with caplog.at_level("WARNING", logger="jobtrack.ingest.auth"):
        load_credentials(tmp_config)
    assert "gmail.modify" in caplog.text


# --- run_oauth_flow --------------------------------------------------------------------


def test_oauth_flow_without_client_secrets_raises(tmp_config: Config) -> None:
    with pytest.raises(AuthError, match="no OAuth client"):
        run_oauth_flow(tmp_config)


def test_oauth_flow_persists_token_with_owner_only_permissions(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_config.credentials_path.write_text(json.dumps({"installed": {}}), encoding="utf-8")
    granted = _granted_credentials()

    class Flow(FakeFlow):
        outcome = None

        def run_local_server(self, port: int = 0) -> Any:
            return granted

    monkeypatch.setattr(auth, "InstalledAppFlow", Flow)
    result = run_oauth_flow(tmp_config)

    assert result is granted
    assert tmp_config.token_path.is_file()
    mode = stat.S_IMODE(tmp_config.token_path.stat().st_mode)
    assert mode == TOKEN_FILE_MODE, f"token.json is mode {oct(mode)}, expected 0600"
    assert Flow.last_scopes == GMAIL_SCOPES


def test_oauth_flow_overwrites_an_existing_world_readable_token(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-running login must tighten a loose file, not inherit its permissions."""
    tmp_config.credentials_path.write_text(json.dumps({"installed": {}}), encoding="utf-8")
    tmp_config.token_path.write_text("stale", encoding="utf-8")
    tmp_config.token_path.chmod(0o644)
    granted = _granted_credentials()

    class Flow(FakeFlow):
        def run_local_server(self, port: int = 0) -> Any:
            return granted

    monkeypatch.setattr(auth, "InstalledAppFlow", Flow)
    run_oauth_flow(tmp_config)
    assert stat.S_IMODE(tmp_config.token_path.stat().st_mode) == TOKEN_FILE_MODE


def test_declined_consent_becomes_auth_error(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_config.credentials_path.write_text(json.dumps({"installed": {}}), encoding="utf-8")

    class Flow(FakeFlow):
        # google-auth 2.x: RefreshError.__init__ is unannotated.
        outcome = RefreshError("access_denied")  # type: ignore[no-untyped-call]

    monkeypatch.setattr(auth, "InstalledAppFlow", Flow)
    with pytest.raises(AuthError, match="consent was not granted"):
        run_oauth_flow(tmp_config)


def test_unusable_client_secrets_file_becomes_auth_error(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_config.credentials_path.write_text("{}", encoding="utf-8")

    class Flow(FakeFlow):
        @classmethod
        def from_client_secrets_file(cls, path: str, scopes: list[str]) -> FakeFlow:
            raise ValueError("Client secrets must be for a web or installed app")

    monkeypatch.setattr(auth, "InstalledAppFlow", Flow)
    with pytest.raises(AuthError, match="not a valid OAuth client file"):
        run_oauth_flow(tmp_config)


def test_local_server_failure_becomes_auth_error(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_config.credentials_path.write_text(json.dumps({"installed": {}}), encoding="utf-8")

    class Flow(FakeFlow):
        def run_local_server(self, port: int = 0) -> Any:
            raise OSError("address already in use")

    monkeypatch.setattr(auth, "InstalledAppFlow", Flow)
    with pytest.raises(AuthError, match="local consent server"):
        run_oauth_flow(tmp_config)


def test_flow_returning_nothing_becomes_auth_error(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_config.credentials_path.write_text(json.dumps({"installed": {}}), encoding="utf-8")

    class Flow(FakeFlow):
        def run_local_server(self, port: int = 0) -> Any:
            return None

    monkeypatch.setattr(auth, "InstalledAppFlow", Flow)
    with pytest.raises(AuthError, match="no credentials"):
        run_oauth_flow(tmp_config)


def test_unwritable_token_destination_becomes_auth_error(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tmp_config.credentials_path.write_text(json.dumps({"installed": {}}), encoding="utf-8")
    granted = _granted_credentials()

    class Flow(FakeFlow):
        def run_local_server(self, port: int = 0) -> Any:
            return granted

    monkeypatch.setattr(auth, "InstalledAppFlow", Flow)
    # A directory where the file should go: os.open fails with EISDIR.
    tmp_config.token_path.mkdir(parents=True, exist_ok=True)
    with pytest.raises(AuthError, match="could not write"):
        run_oauth_flow(tmp_config)


# --- credential_status -----------------------------------------------------------------


def test_status_reports_a_missing_token_without_raising(tmp_config: Config) -> None:
    status = credential_status(tmp_config)
    assert status["has_token"] is False
    assert status["valid"] is False
    assert status["error"]
    assert status["token_path"] == str(tmp_config.token_path)


def test_status_reports_a_valid_token(tmp_config: Config) -> None:
    write_token(tmp_config.token_path)
    status = credential_status(tmp_config)
    assert status["has_token"] is True
    assert status["valid"] is True
    assert status["expired"] is False
    assert status["has_refresh_token"] is True
    assert status["scopes"] == [READONLY_SCOPE]
    assert status["scopes_ok"] is True
    assert status["error"] is None


def test_status_expiry_is_iso_utc(tmp_config: Config) -> None:
    """I7: nothing naive leaves the module, not even a rendered timestamp."""
    write_token(tmp_config.token_path)
    status = credential_status(tmp_config)
    expiry = datetime.fromisoformat(status["expiry"])
    assert expiry.tzinfo is not None
    assert expiry.utcoffset() == timedelta(0)


def test_status_reports_an_expired_token(tmp_config: Config) -> None:
    write_token(tmp_config.token_path, expires_in_days=-1)
    status = credential_status(tmp_config)
    assert status["expired"] is True
    assert status["valid"] is False


def test_status_reports_a_wider_scope_as_not_ok(tmp_config: Config) -> None:
    """Regression guard: passing GMAIL_SCOPES into google-auth would mask the real grant."""
    write_token(tmp_config.token_path, scopes=["https://www.googleapis.com/auth/gmail.modify"])
    status = credential_status(tmp_config)
    assert status["scopes"] == ["https://www.googleapis.com/auth/gmail.modify"]
    assert status["scopes_ok"] is False


def test_status_reports_a_malformed_token_without_raising(tmp_config: Config) -> None:
    tmp_config.token_path.write_text("{ not json", encoding="utf-8")
    status = credential_status(tmp_config)
    assert status["has_token"] is True
    assert status["valid"] is False
    assert "not a usable OAuth token" in status["error"]


def test_status_notices_the_client_secrets_file(tmp_config: Config) -> None:
    assert credential_status(tmp_config)["has_client_secrets"] is False
    tmp_config.credentials_path.write_text("{}", encoding="utf-8")
    assert credential_status(tmp_config)["has_client_secrets"] is True


def _granted_credentials() -> Any:
    """A Credentials object standing in for a completed consent flow."""
    from google.oauth2.credentials import Credentials

    # google-auth 2.x: Credentials.__init__ is unannotated.
    return Credentials(  # type: ignore[no-untyped-call]
        token="granted-access-token",
        refresh_token="granted-refresh-token",
        token_uri="https://oauth2.googleapis.com/token",
        client_id="client-id",
        client_secret="client-secret",
        scopes=list(GMAIL_SCOPES),
    )
