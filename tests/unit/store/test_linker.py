"""Tests for the pure linker: match_application and derive_status.

No DB access anywhere in this file — candidates and events are constructed directly, per
CLAUDE.md's testing brief for M3.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from jobtrack.models import (
    ApplicationMatchCandidate,
    ApplicationStatus,
    Classification,
    EventRow,
    EventType,
)
from jobtrack.store.linker import LINK_WINDOW_DAYS, derive_status, match_application

NOW = datetime(2026, 8, 18, 12, 0, 0, tzinfo=UTC)


def _classification(**overrides: object) -> Classification:
    defaults: dict[str, object] = {
        "message_id": "m1",
        "event_type": EventType.APPLICATION_RECEIVED,
        "company": "Acme Robotics",
        "company_key": "acme robotics",
        "role": "Backend Engineer",
        "confidence": 0.9,
        "classifier_name": "rules",
        "classifier_version": "1.0.0",
    }
    defaults.update(overrides)
    return Classification.model_validate(defaults)


def _candidate(**overrides: object) -> ApplicationMatchCandidate:
    defaults: dict[str, object] = {
        "application_id": "app-1",
        "company_key": "acme robotics",
        "role": "Backend Engineer",
        "thread_ids": ["t-1"],
        "applied_at": NOW - timedelta(days=10),
        "last_event_at": NOW - timedelta(days=10),
    }
    defaults.update(overrides)
    return ApplicationMatchCandidate.model_validate(defaults)


def _event(**overrides: object) -> EventRow:
    defaults: dict[str, object] = {
        "event_id": 1,
        "application_id": "app-1",
        "message_id": "m1",
        "event_type": EventType.APPLICATION_RECEIVED,
        "occurred_at": NOW,
        "confidence": 0.9,
        "needs_review": False,
        "is_overridden": False,
        "subject": "Thanks for applying",
        "from_email": "no-reply@example.com",
    }
    defaults.update(overrides)
    return EventRow.model_validate(defaults)


# --- match_application: rule 1, thread already linked -----------------------------------


def test_rule1_thread_already_linked_wins_even_with_different_company() -> None:
    """Rule 1: an existing thread match wins regardless of company_key or role."""
    candidate = _candidate(
        application_id="app-thread",
        company_key="totally different co",
        role=None,
        thread_ids=["t-target"],
    )
    other = _candidate(application_id="app-other", company_key="acme robotics")
    classification = _classification(company_key="acme robotics", role="Backend Engineer")

    result = match_application(classification, [other, candidate], "t-target", now=NOW)

    assert result == "app-thread"


# --- rule 2: same company_key + role similarity ------------------------------------------


def test_rule2_matches_on_company_and_similar_role() -> None:
    """Rule 2: same company_key and role_similarity >= threshold, within window."""
    candidate = _candidate(
        application_id="app-role-match",
        company_key="acme robotics",
        role="Backend Engineer",
        thread_ids=["t-other"],
        last_event_at=NOW - timedelta(days=5),
    )
    classification = _classification(company_key="acme robotics", role="Backend Engineer")

    result = match_application(classification, [candidate], "t-new-thread", now=NOW)

    assert result == "app-role-match"


def test_rule2_rejects_dissimilar_role_below_threshold() -> None:
    """A same-company candidate whose role is unrelated must NOT match under rule 2/3."""
    candidate = _candidate(
        application_id="app-different-role",
        company_key="acme robotics",
        role="Director of Sales",
        thread_ids=["t-other"],
    )
    classification = _classification(company_key="acme robotics", role="Backend Engineer")

    result = match_application(classification, [candidate], "t-new-thread", now=NOW)

    assert result is None


def test_rule2_ties_broken_by_most_recent_last_event_at() -> None:
    """When two candidates both clear the role threshold, the more recently active one wins."""
    older = _candidate(
        application_id="app-older",
        company_key="acme robotics",
        role="Backend Engineer",
        thread_ids=["t-older"],
        last_event_at=NOW - timedelta(days=100),
    )
    newer = _candidate(
        application_id="app-newer",
        company_key="acme robotics",
        role="Backend Engineer",
        thread_ids=["t-newer"],
        last_event_at=NOW - timedelta(days=1),
    )
    classification = _classification(company_key="acme robotics", role="Backend Engineer")

    result = match_application(classification, [older, newer], "t-fresh", now=NOW)

    assert result == "app-newer"


def test_rule2_outside_window_is_not_a_candidate() -> None:
    """A same-company, same-role application outside window_days does not match (rule 2)."""
    stale = _candidate(
        application_id="app-stale",
        company_key="acme robotics",
        role="Backend Engineer",
        thread_ids=["t-stale"],
        last_event_at=NOW - timedelta(days=LINK_WINDOW_DAYS + 1),
    )
    classification = _classification(company_key="acme robotics", role="Backend Engineer")

    result = match_application(classification, [stale], "t-fresh", now=NOW)

    assert result is None


# --- rule 3: same company_key, either role missing ----------------------------------------


def test_rule3_matches_when_candidate_role_is_none() -> None:
    """Rule 3: same company_key, candidate has no role on file, within window."""
    candidate = _candidate(
        application_id="app-no-role",
        company_key="acme robotics",
        role=None,
        thread_ids=["t-other"],
    )
    classification = _classification(company_key="acme robotics", role="Backend Engineer")

    result = match_application(classification, [candidate], "t-new-thread", now=NOW)

    assert result == "app-no-role"


def test_rule3_matches_when_incoming_role_is_none() -> None:
    """Rule 3: same company_key, the incoming message itself has no extracted role."""
    candidate = _candidate(
        application_id="app-has-role",
        company_key="acme robotics",
        role="Backend Engineer",
        thread_ids=["t-other"],
    )
    classification = _classification(company_key="acme robotics", role=None)

    result = match_application(classification, [candidate], "t-new-thread", now=NOW)

    assert result == "app-has-role"


# --- rule 4: no match -----------------------------------------------------------------------


def test_rule4_different_company_key_yields_no_match() -> None:
    """Rule 4: nothing matches -> None, caller creates a new application. NEGATIVE case."""
    candidate = _candidate(company_key="wayne enterprises", role="Backend Engineer")
    classification = _classification(company_key="acme robotics", role="Backend Engineer")

    result = match_application(classification, [candidate], "t-new-thread", now=NOW)

    assert result is None


def test_rule4_no_candidates_at_all_yields_no_match() -> None:
    """An empty candidate list always yields None."""
    classification = _classification()

    result = match_application(classification, [], "t-any", now=NOW)

    assert result is None


# --- derive_status ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("event_type", "expected"),
    [
        (EventType.REJECTION, ApplicationStatus.REJECTED),
        (EventType.OFFER, ApplicationStatus.OFFER),
        (EventType.WITHDRAWN, ApplicationStatus.WITHDRAWN),
    ],
)
def test_derive_status_terminal_events_win(
    event_type: EventType, expected: ApplicationStatus
) -> None:
    """Terminal events decide status regardless of what came before."""
    events = [
        _event(event_id=1, event_type=EventType.APPLICATION_RECEIVED, occurred_at=NOW - timedelta(days=20)),
        _event(event_id=2, event_type=EventType.INTERVIEW, occurred_at=NOW - timedelta(days=10)),
        _event(event_id=3, event_type=event_type, occurred_at=NOW - timedelta(days=5)),
    ]
    assert derive_status(events, now=NOW, ghost_after_days=30) == expected


def test_derive_status_most_recent_terminal_wins_regardless_of_recency() -> None:
    """Two terminal events: the most recent ONE decides, even if it's a very old one overall."""
    events = [
        _event(event_id=1, event_type=EventType.WITHDRAWN, occurred_at=NOW - timedelta(days=200)),
        _event(event_id=2, event_type=EventType.REJECTION, occurred_at=NOW - timedelta(days=5)),
    ]
    assert derive_status(events, now=NOW, ghost_after_days=30) == ApplicationStatus.REJECTED


@pytest.mark.parametrize(
    ("event_type", "expected"),
    [
        (EventType.RECRUITER_OUTREACH, ApplicationStatus.APPLIED),
        (EventType.APPLICATION_RECEIVED, ApplicationStatus.APPLIED),
        (EventType.ASSESSMENT, ApplicationStatus.ASSESSMENT),
        (EventType.INTERVIEW, ApplicationStatus.INTERVIEWING),
    ],
)
def test_derive_status_furthest_non_terminal_stage(
    event_type: EventType, expected: ApplicationStatus
) -> None:
    """Non-terminal: the furthest pipeline stage reached decides, when still fresh."""
    events = [
        _event(event_id=1, event_type=EventType.APPLICATION_RECEIVED, occurred_at=NOW - timedelta(days=2)),
        _event(event_id=2, event_type=event_type, occurred_at=NOW - timedelta(days=1)),
    ]
    assert derive_status(events, now=NOW, ghost_after_days=30) == expected


def test_derive_status_ghosted_when_stale_and_non_terminal() -> None:
    """No terminal event and the last event is older than ghost_after_days -> GHOSTED."""
    events = [
        _event(event_id=1, event_type=EventType.INTERVIEW, occurred_at=NOW - timedelta(days=45)),
    ]
    assert derive_status(events, now=NOW, ghost_after_days=30) == ApplicationStatus.GHOSTED


def test_derive_status_not_ghosted_at_exactly_the_threshold() -> None:
    """"Older than" ghost_after_days is a strict inequality, not >=."""
    events = [
        _event(event_id=1, event_type=EventType.INTERVIEW, occurred_at=NOW - timedelta(days=30)),
    ]
    assert derive_status(events, now=NOW, ghost_after_days=30) == ApplicationStatus.INTERVIEWING


def test_derive_status_empty_events_is_applied() -> None:
    """An application with no events at all defaults to APPLIED rather than crashing."""
    assert derive_status([], now=NOW, ghost_after_days=30) == ApplicationStatus.APPLIED


def test_derive_status_is_deterministic() -> None:
    """Calling derive_status twice on the same input yields byte-identical output."""
    events = [
        _event(event_id=1, event_type=EventType.APPLICATION_RECEIVED, occurred_at=NOW - timedelta(days=3)),
        _event(event_id=2, event_type=EventType.INTERVIEW, occurred_at=NOW - timedelta(days=1)),
    ]
    first = derive_status(events, now=NOW, ghost_after_days=30)
    second = derive_status(events, now=NOW, ghost_after_days=30)
    assert first == second
