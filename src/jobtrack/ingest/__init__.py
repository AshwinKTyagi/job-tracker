"""Mailbox ingestion: OAuth, the `EmailSource` protocol, and the Gmail implementation.

Produces `RawMessage`. Knows nothing about jobs, applications, or classification — see
CONTRACTS.md §4.
"""

from __future__ import annotations

from jobtrack.ingest.auth import (
    GMAIL_SCOPES,
    credential_status,
    load_credentials,
    run_oauth_flow,
)
from jobtrack.ingest.gmail import GmailSource, parse_gmail_message
from jobtrack.ingest.html import html_to_text
from jobtrack.ingest.source import EmailSource, FetchResult

__all__ = [
    "GMAIL_SCOPES",
    "EmailSource",
    "FetchResult",
    "GmailSource",
    "credential_status",
    "html_to_text",
    "load_credentials",
    "parse_gmail_message",
    "run_oauth_flow",
]
