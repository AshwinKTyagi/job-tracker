"""Tests for auth.py: token load/refresh, the consent flow, and status reporting.

No real OAuth network calls happen here — `Credentials.refresh` and
`InstalledAppFlow.from_client_secrets_file` are monkeypatched at their call sites so the
module is exercised without a socket, per CLAUDE.md.
"""

from __future__ import annotations

import json
import os
import stat
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials

from jobtrack.config import Config
from jobtrack.errors import AuthError
from jobtrack.ingest import auth
from jobtrack.ingest.auth import GMAIL_SCOPES


def _write_token(config: Config, *, expired: bool = False, has_refresh_token: bool = True) -> None:
    config.home.mkdir(parents=True, exist_ok=True)
    expiry = datetime.now(UTC) + (timedelta(hours=-1) if expired else timedelta(hours=1))
    data: dict[str, Any] = {
        "token": "access-token-abc",
        "refresh_token": "refresh-token-abc" if has_refresh_token else None,
        "token_uri": "https://oauth2.googleapis.com/token",
        "client_id": "client-id",
        "client_secret": "client-secret",
        "scopes": GMAIL_SCOPES,
        "expiry": expiry.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
    }
    config.token_path.write_text(json.dumps(data))


class _FakeCredentials:
    """Minimal stand-in for the object InstalledAppFlow.run_local_server() returns."""

    def __init__(self, token: str) -> None:
        self.token = token

    def to_json(self) -> str:
        return json.dumps({"token": self.token})


def test_gmail_scopes_is_read_only_and_matches_constants() -> None:
    from jobtrack import constants

    assert list(constants.GMAIL_SCOPES) == auth.GMAIL_SCOPES
    assert auth.GMAIL_SCOPES == ["https://www.googleapis.com/auth/gmail.readonly"]


def test_credential_status_no_token(tmp_config: Config) -> None:
    status = auth.credential_status(tmp_config)

    assert status["authenticated"] is False
    assert status["valid"] is False
    assert status["scopes"] == []


def test_credential_status_valid_token(tmp_config: Config) -> None:
    _write_token(tmp_config, expired=False)

    status = auth.credential_status(tmp_config)

    assert status["authenticated"] is True
    assert status["expired"] is False
    assert status["scopes"] == GMAIL_SCOPES


def test_credential_status_expired_token(tmp_config: Config) -> None:
    _write_token(tmp_config, expired=True)

    status = auth.credential_status(tmp_config)

    assert status["authenticated"] is True
    assert status["expired"] is True


def test_credential_status_malformed_token_reports_error(tmp_config: Config) -> None:
    tmp_config.home.mkdir(parents=True, exist_ok=True)
    tmp_config.token_path.write_text("not valid json")

    status = auth.credential_status(tmp_config)

    assert status["authenticated"] is False
    assert "error" in status


def test_load_credentials_missing_token_raises(tmp_config: Config) -> None:
    with pytest.raises(AuthError):
        auth.load_credentials(tmp_config)


def test_load_credentials_valid_token_returns_without_refresh(tmp_config: Config) -> None:
    _write_token(tmp_config, expired=False)

    creds = auth.load_credentials(tmp_config)

    assert creds.token == "access-token-abc"


def test_load_credentials_refreshes_expired_token(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_token(tmp_config, expired=True, has_refresh_token=True)

    def _fake_refresh(self: Credentials, request: Any) -> None:
        self.token = "refreshed-access-token"
        self.expiry = datetime.now(UTC) + timedelta(hours=1)

    monkeypatch.setattr(Credentials, "refresh", _fake_refresh)

    creds = auth.load_credentials(tmp_config)

    assert creds.token == "refreshed-access-token"
    persisted = json.loads(tmp_config.token_path.read_text())
    assert persisted["token"] == "refreshed-access-token"
    mode = stat.S_IMODE(os.stat(tmp_config.token_path).st_mode)
    assert mode == 0o600


def test_load_credentials_refresh_failure_raises_autherror(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_token(tmp_config, expired=True, has_refresh_token=True)

    def _fake_refresh(self: Credentials, request: Any) -> None:
        raise RefreshError("token has been revoked")

    monkeypatch.setattr(Credentials, "refresh", _fake_refresh)

    with pytest.raises(AuthError):
        auth.load_credentials(tmp_config)


def test_load_credentials_malformed_token_raises(tmp_config: Config) -> None:
    tmp_config.home.mkdir(parents=True, exist_ok=True)
    tmp_config.token_path.write_text("not valid json")

    with pytest.raises(AuthError):
        auth.load_credentials(tmp_config)


def test_load_credentials_persist_failure_raises_autherror(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_token(tmp_config, expired=True, has_refresh_token=True)

    def _fake_refresh(self: Credentials, request: Any) -> None:
        self.token = "refreshed-access-token"
        self.expiry = datetime.now(UTC) + timedelta(hours=1)

    def _fail_chmod(path: Any, mode: int) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Credentials, "refresh", _fake_refresh)
    monkeypatch.setattr(auth.os, "chmod", _fail_chmod)

    with pytest.raises(AuthError):
        auth.load_credentials(tmp_config)


def test_load_credentials_expired_no_refresh_token_raises(tmp_config: Config) -> None:
    _write_token(tmp_config, expired=True, has_refresh_token=False)

    with pytest.raises(AuthError):
        auth.load_credentials(tmp_config)


def test_run_oauth_flow_missing_credentials_file_raises(tmp_config: Config) -> None:
    with pytest.raises(AuthError):
        auth.run_oauth_flow(tmp_config)


def test_run_oauth_flow_success_persists_token(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_config.home.mkdir(parents=True, exist_ok=True)
    tmp_config.credentials_path.write_text(
        json.dumps(
            {
                "installed": {
                    "client_id": "id",
                    "client_secret": "secret",
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": ["http://localhost"],
                }
            }
        )
    )
    fake_credentials = _FakeCredentials("granted-token")

    class _FakeFlow:
        def run_local_server(self, *, port: int) -> _FakeCredentials:
            return fake_credentials

    class _FakeInstalledAppFlow:
        @staticmethod
        def from_client_secrets_file(path: str, scopes: list[str]) -> _FakeFlow:
            assert scopes == GMAIL_SCOPES
            return _FakeFlow()

    monkeypatch.setattr(auth, "InstalledAppFlow", _FakeInstalledAppFlow)

    result = auth.run_oauth_flow(tmp_config)

    assert result is fake_credentials
    assert tmp_config.token_path.is_file()
    persisted = json.loads(tmp_config.token_path.read_text())
    assert persisted["token"] == "granted-token"
    mode = stat.S_IMODE(os.stat(tmp_config.token_path).st_mode)
    assert mode == 0o600


def test_run_oauth_flow_malformed_credentials_file_raises(tmp_config: Config) -> None:
    tmp_config.home.mkdir(parents=True, exist_ok=True)
    tmp_config.credentials_path.write_text("not valid json")

    with pytest.raises(AuthError):
        auth.run_oauth_flow(tmp_config)


def test_run_oauth_flow_declined_consent_raises_autherror(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_config.home.mkdir(parents=True, exist_ok=True)
    tmp_config.credentials_path.write_text(json.dumps({"installed": {}}))

    class _FakeFlow:
        def run_local_server(self, *, port: int) -> Credentials:
            raise RefreshError("access_denied: user declined consent")

    class _FakeInstalledAppFlow:
        @staticmethod
        def from_client_secrets_file(path: str, scopes: list[str]) -> _FakeFlow:
            return _FakeFlow()

    monkeypatch.setattr(auth, "InstalledAppFlow", _FakeInstalledAppFlow)

    with pytest.raises(AuthError):
        auth.run_oauth_flow(tmp_config)
