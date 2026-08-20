"""Tests for parse_gmail_message: raw Gmail payload -> RawMessage."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from jobtrack.errors import PermanentIngestError
from jobtrack.ingest.gmail import parse_gmail_message
from jobtrack.models import RawMessage

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text())


def test_parse_plain_text_message() -> None:
    payload = _load("gmail_message_full_plain.json")

    msg = parse_gmail_message(payload)

    assert isinstance(msg, RawMessage)
    assert msg.message_id == "msg-full-plain-001"
    assert msg.thread_id == "thread-full-plain-001"
    assert msg.from_email == "no-reply@us.greenhouse-mail.io"
    assert msg.to_email == "candidate@example.com"
    assert msg.subject == "Thanks for applying to Acme Robotics"
    assert "Thanks for applying to Acme Robotics" in msg.body_text
    assert msg.received_at.tzinfo is not None
    assert msg.received_at == datetime.fromtimestamp(1751472251, tz=UTC)
    assert msg.headers["from"] == "Acme Robotics <no-reply@us.greenhouse-mail.io>"
    assert msg.headers["reply-to"] == "no-reply@greenhouse.io"
    assert msg.labels == ["INBOX", "CATEGORY_UPDATES"]


def test_parse_multipart_prefers_plain_text_and_decodes_display_name() -> None:
    payload = _load("gmail_message_full_multipart.json")

    msg = parse_gmail_message(payload)

    assert msg.from_name == "Acme Robotics"  # RFC 2047 encoded-word decoded
    assert "Unfortunately" in msg.body_text
    # The text/html twin is present in the payload but text/plain must win.
    assert "<p>" not in msg.body_text
    assert "<html>" not in msg.body_text


def test_parse_html_only_message_converts_to_text() -> None:
    payload = _load("gmail_message_full_html_only.json")

    msg = parse_gmail_message(payload)

    assert msg.to_email == "candidate@example.com"  # lowercased
    assert "Congratulations" in msg.body_text
    assert "<div>" not in msg.body_text


@pytest.mark.parametrize("missing_key", ["id", "threadId", "internalDate"])
def test_parse_missing_required_field_raises(missing_key: str) -> None:
    payload = _load("gmail_message_full_plain.json")
    del payload[missing_key]

    with pytest.raises(PermanentIngestError):
        parse_gmail_message(payload)


def test_parse_missing_thread_id_fixture_raises() -> None:
    payload = _load("gmail_message_missing_thread_id.json")

    with pytest.raises(PermanentIngestError):
        parse_gmail_message(payload)


def test_parse_malformed_internal_date_raises() -> None:
    payload = _load("gmail_message_full_plain.json")
    payload["internalDate"] = "not-a-number"

    with pytest.raises(PermanentIngestError):
        parse_gmail_message(payload)


def test_parse_empty_payload_dict_raises() -> None:
    with pytest.raises(PermanentIngestError):
        parse_gmail_message({})


def test_parse_is_deterministic() -> None:
    payload = _load("gmail_message_full_multipart.json")

    first = parse_gmail_message(payload)
    second = parse_gmail_message(payload)

    assert first.model_dump() == second.model_dump()


def test_parse_missing_body_yields_empty_text_and_headers() -> None:
    payload: dict[str, Any] = {
        "id": "msg-empty-001",
        "threadId": "thread-empty-001",
        "internalDate": "1751472251000",
        "payload": {"mimeType": "text/plain", "headers": []},
    }

    msg = parse_gmail_message(payload)

    assert msg.body_text == ""
    assert msg.from_email == ""
    assert msg.subject == ""
    assert msg.headers == {}


def test_parse_falls_back_to_raw_value_for_undecodable_mime_header() -> None:
    payload: dict[str, Any] = {
        "id": "msg-badheader-001",
        "threadId": "thread-badheader-001",
        "internalDate": "1751472251000",
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "From", "value": "no-reply@example.com"},
                {"name": "Subject", "value": "=?UNKNOWN-CHARSET-XYZ?B?QQ==?="},
            ],
            "body": {"data": ""},
        },
    }

    msg = parse_gmail_message(payload)

    # Not validly decodable -> the raw encoded-word string passes through unchanged.
    assert msg.subject == "=?UNKNOWN-CHARSET-XYZ?B?QQ==?="


def test_parse_body_part_with_undecodable_base64_yields_empty_text() -> None:
    payload: dict[str, Any] = {
        "id": "msg-badbody-001",
        "threadId": "thread-badbody-001",
        "internalDate": "1751472251000",
        "payload": {
            "mimeType": "text/plain",
            "headers": [],
            "body": {"data": "!!!not-base64!!!"},
        },
    }

    msg = parse_gmail_message(payload)

    assert msg.body_text == ""


def test_parse_lowercases_header_keys() -> None:
    payload: dict[str, Any] = {
        "id": "msg-headers-001",
        "threadId": "thread-headers-001",
        "internalDate": "1751472251000",
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "X-Custom-Header", "value": "some-value"},
                {"name": "MESSAGE-ID", "value": "<abc@example.com>"},
            ],
            "body": {"data": ""},
        },
    }

    msg = parse_gmail_message(payload)

    assert msg.headers["x-custom-header"] == "some-value"
    assert msg.headers["message-id"] == "<abc@example.com>"
