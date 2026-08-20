"""Tests for the pure linker: match_application and derive_status.

No database here at all — that is the point of pre-fetching candidates.
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
from jobtrack.store.linker import (
    LINK_WINDOW_DAYS,
    _normalize_role_local,
    _role_similarity,
    derive_status,
    match_application,
)

NOW = datetime(2026, 8, 18, 12, 0, 0, tzinfo=UTC)


def make_candidate(
    application_id: str,
    *,
    company_key: str = "acme",
    role: str | None = "Software Engineer",
    thread_ids: list[str] | None = None,
    days_ago: int = 1,
) -> ApplicationMatchCandidate:
    """Build a candidate whose last event is ``days_ago`` days before NOW."""
    last = NOW - timedelta(days=days_ago)
    return ApplicationMatchCandidate(
        application_id=application_id,
        company_key=company_key,
        role=role,
        thread_ids=thread_ids if thread_ids is not None else [],
        applied_at=last - timedelta(days=1),
        last_event_at=last,
    )


def make_classification(
    *,
    company_key: str | None = "acme",
    role: str | None = "Software Engineer",
    event_type: EventType = EventType.INTERVIEW,
) -> Classification:
    """Build a minimal Classification for linking."""
    return Classification(
        message_id="m1",
        event_type=event_type,
        company="Acme",
        company_key=company_key,
        role=role,
        confidence=0.9,
        classifier_name="rules",
        classifier_version="1.0.0",
    )


def make_event(
    event_type: EventType, *, days_ago: int = 0, event_id: int = 1, needs_review: bool = False
) -> EventRow:
    """Build an EventRow that occurred ``days_ago`` days before NOW."""
    return EventRow(
        event_id=event_id,
        application_id="app_1",
        message_id=f"m{event_id}",
        event_type=event_type,
        occurred_at=NOW - timedelta(days=days_ago),
        confidence=0.9,
        needs_review=needs_review,
        is_overridden=False,
        subject="subject",
        from_email="careers@example.com",
    )


# --- match_application ------------------------------------------------------


def test_thread_match_wins_over_everything_else() -> None:
    """Rule 1: a known thread links even when company and window disagree."""
    threaded = make_candidate(
        "app_thread", company_key="other", role=None, thread_ids=["t9"], days_ago=999
    )
    company = make_candidate("app_company")
    matched = match_application(make_classification(), [company, threaded], "t9", now=NOW)
    assert matched == "app_thread"


def test_thread_match_prefers_the_most_recent_on_a_tie() -> None:
    """Two applications on one thread: the most recently active one wins."""
    old = make_candidate("app_old", thread_ids=["t1"], days_ago=40)
    new = make_candidate("app_new", thread_ids=["t1"], days_ago=2)
    assert match_application(make_classification(), [old, new], "t1", now=NOW) == "app_new"


def test_company_and_similar_role_match() -> None:
    """Rule 2: same company_key with a near-identical title links."""
    candidate = make_candidate("app_1", role="Sr. Software Engineer II")
    matched = match_application(make_classification(), [candidate], "t-new", now=NOW)
    assert matched == "app_1"


def test_dissimilar_roles_do_not_match() -> None:
    """Rule 2 fails and rule 3 does not apply when both titles are present."""
    candidate = make_candidate("app_1", role="Data Scientist")
    assert match_application(make_classification(), [candidate], "t-new", now=NOW) is None


def test_null_role_on_the_candidate_falls_through_to_rule_three() -> None:
    """Rule 3: an application with no recorded role accepts any title."""
    candidate = make_candidate("app_1", role=None)
    assert match_application(make_classification(), [candidate], "t-new", now=NOW) == "app_1"


def test_null_role_on_the_message_falls_through_to_rule_three() -> None:
    """Rule 3 also applies when the classifier could not extract a title."""
    candidate = make_candidate("app_1", role="Software Engineer")
    classification = make_classification(role=None)
    assert match_application(classification, [candidate], "t-new", now=NOW) == "app_1"


def test_rule_three_prefers_the_most_recently_active_application() -> None:
    """Ties in rule 3 are broken by last_event_at, newest first."""
    old = make_candidate("app_old", role=None, days_ago=100)
    new = make_candidate("app_new", role=None, days_ago=3)
    classification = make_classification()
    assert match_application(classification, [old, new], "t-new", now=NOW) == "app_new"


def test_rule_two_prefers_the_most_recently_active_on_equal_scores() -> None:
    """Equal role similarity is broken by last_event_at."""
    old = make_candidate("app_old", days_ago=60)
    new = make_candidate("app_new", days_ago=6)
    assert match_application(make_classification(), [old, new], "t-new", now=NOW) == "app_new"


def test_candidate_outside_the_window_is_ignored() -> None:
    """Rule 2/3 only reach back window_days."""
    stale = make_candidate("app_stale", days_ago=LINK_WINDOW_DAYS + 1)
    assert match_application(make_classification(), [stale], "t-new", now=NOW) is None


def test_window_boundary_is_inclusive() -> None:
    """Exactly window_days old still links."""
    edge = make_candidate("app_edge", days_ago=LINK_WINDOW_DAYS)
    assert match_application(make_classification(), [edge], "t-new", now=NOW) == "app_edge"


def test_missing_company_key_never_matches() -> None:
    """Without a company_key (I8) there is nothing to match on."""
    candidate = make_candidate("app_1")
    classification = make_classification(company_key=None)
    assert match_application(classification, [candidate], "t-new", now=NOW) is None


def test_different_company_key_does_not_match() -> None:
    """A different employer is always a new application."""
    candidate = make_candidate("app_1", company_key="globex")
    assert match_application(make_classification(), [candidate], "t-new", now=NOW) is None


def test_no_candidates_returns_none() -> None:
    """An empty candidate list means the caller creates a new application."""
    assert match_application(make_classification(), [], "t-new", now=NOW) is None


def test_role_threshold_is_configurable() -> None:
    """Lowering role_threshold admits a looser title match."""
    candidate = make_candidate("app_1", role="Software Engineer, Backend Platform")
    strict = match_application(
        make_classification(), [candidate], "t-new", now=NOW, role_threshold=0.95
    )
    loose = match_application(
        make_classification(), [candidate], "t-new", now=NOW, role_threshold=0.5
    )
    assert strict is None
    assert loose == "app_1"


def test_abbreviations_are_expanded_before_comparison() -> None:
    """'PM' and 'Product Manager' are the same job."""
    candidate = make_candidate("app_1", role="PM")
    classification = make_classification(role="Product Manager")
    assert match_application(classification, [candidate], "t-new", now=NOW) == "app_1"


# --- derive_status ----------------------------------------------------------


def test_empty_history_reads_as_applied() -> None:
    """An application with no events is APPLIED, never a crash."""
    assert derive_status([], now=NOW, ghost_after_days=30) is ApplicationStatus.APPLIED


def test_terminal_event_wins_regardless_of_recency() -> None:
    """A rejection followed by a stray interview email is still REJECTED."""
    events = [
        make_event(EventType.REJECTION, days_ago=10, event_id=1),
        make_event(EventType.INTERVIEW, days_ago=1, event_id=2),
    ]
    assert derive_status(events, now=NOW, ghost_after_days=30) is ApplicationStatus.REJECTED


def test_most_recent_terminal_event_decides() -> None:
    """Rejected then offered: the newer terminal event is the answer."""
    events = [
        make_event(EventType.REJECTION, days_ago=20, event_id=1),
        make_event(EventType.OFFER, days_ago=2, event_id=2),
    ]
    assert derive_status(events, now=NOW, ghost_after_days=30) is ApplicationStatus.OFFER


def test_withdrawn_is_terminal() -> None:
    """WITHDRAWN pins the status even though the candidate initiated it."""
    events = [make_event(EventType.WITHDRAWN, days_ago=400)]
    assert derive_status(events, now=NOW, ghost_after_days=30) is ApplicationStatus.WITHDRAWN


def test_terminal_applications_never_ghost() -> None:
    """Silence after a rejection is expected, not a ghosting."""
    events = [make_event(EventType.REJECTION, days_ago=365)]
    assert derive_status(events, now=NOW, ghost_after_days=30) is ApplicationStatus.REJECTED


def test_furthest_stage_reached_decides() -> None:
    """An acknowledgement plus an interview reads as INTERVIEWING."""
    events = [
        make_event(EventType.APPLICATION_RECEIVED, days_ago=8, event_id=1),
        make_event(EventType.INTERVIEW, days_ago=3, event_id=2),
    ]
    assert derive_status(events, now=NOW, ghost_after_days=30) is ApplicationStatus.INTERVIEWING


def test_assessment_stage() -> None:
    """An online assessment is its own stage."""
    events = [
        make_event(EventType.APPLICATION_RECEIVED, days_ago=8, event_id=1),
        make_event(EventType.ASSESSMENT, days_ago=4, event_id=2),
    ]
    assert derive_status(events, now=NOW, ghost_after_days=30) is ApplicationStatus.ASSESSMENT


def test_recruiter_outreach_reads_as_applied() -> None:
    """Inbound outreach has no further stage of its own."""
    events = [make_event(EventType.RECRUITER_OUTREACH, days_ago=1)]
    assert derive_status(events, now=NOW, ghost_after_days=30) is ApplicationStatus.APPLIED


def test_unknown_only_history_reads_as_applied() -> None:
    """An UNKNOWN event still resolves to a status rather than raising."""
    events = [make_event(EventType.UNKNOWN, days_ago=1)]
    assert derive_status(events, now=NOW, ghost_after_days=30) is ApplicationStatus.APPLIED


def test_silence_past_the_threshold_ghosts() -> None:
    """No event for longer than ghost_after_days downgrades to GHOSTED."""
    events = [make_event(EventType.APPLICATION_RECEIVED, days_ago=31)]
    assert derive_status(events, now=NOW, ghost_after_days=30) is ApplicationStatus.GHOSTED


def test_ghost_threshold_is_inclusive() -> None:
    """Exactly ghost_after_days of silence is not yet ghosted."""
    events = [make_event(EventType.APPLICATION_RECEIVED, days_ago=30)]
    assert derive_status(events, now=NOW, ghost_after_days=30) is ApplicationStatus.APPLIED


def test_ghost_threshold_is_configurable() -> None:
    """A shorter patience ghosts sooner."""
    events = [make_event(EventType.INTERVIEW, days_ago=10)]
    assert derive_status(events, now=NOW, ghost_after_days=7) is ApplicationStatus.GHOSTED


@pytest.mark.parametrize(
    ("event_type", "expected"),
    [
        (EventType.REJECTION, ApplicationStatus.REJECTED),
        (EventType.OFFER, ApplicationStatus.OFFER),
        (EventType.WITHDRAWN, ApplicationStatus.WITHDRAWN),
        (EventType.INTERVIEW, ApplicationStatus.INTERVIEWING),
        (EventType.ASSESSMENT, ApplicationStatus.ASSESSMENT),
        (EventType.APPLICATION_RECEIVED, ApplicationStatus.APPLIED),
        (EventType.RECRUITER_OUTREACH, ApplicationStatus.APPLIED),
        (EventType.UNKNOWN, ApplicationStatus.APPLIED),
    ],
)
def test_every_event_type_maps_to_a_status(
    event_type: EventType, expected: ApplicationStatus
) -> None:
    """Every EventType member resolves, so a new enum member cannot slip through."""
    events = [make_event(event_type, days_ago=1)]
    assert derive_status(events, now=NOW, ghost_after_days=30) is expected


# --- the local role-similarity fallback -------------------------------------


def test_role_similarity_treats_missing_titles_as_no_evidence() -> None:
    """A missing title scores 0, so rule 2 never fires on it and rule 3 decides."""
    assert _role_similarity(None, "Software Engineer") == 0.0
    assert _role_similarity("Software Engineer", None) == 0.0
    assert _role_similarity(None, None) == 0.0


def test_role_similarity_is_symmetric_and_self_identical() -> None:
    """Equal titles score 1.0, and argument order does not matter."""
    assert _role_similarity("Software Engineer", "Software Engineer") == 1.0
    forward = _role_similarity("Staff Data Engineer", "Data Engineer, Analytics")
    backward = _role_similarity("Data Engineer, Analytics", "Staff Data Engineer")
    assert forward == backward


def test_titles_that_are_pure_seniority_noise_score_zero() -> None:
    """'Senior II' carries no job information, so it must not match anything."""
    assert _normalize_role_local("Senior II") == ""
    assert _role_similarity("Senior II", "Software Engineer") == 0.0


def test_a_noise_only_title_does_not_link() -> None:
    """Rule 2 cannot fire on an empty normalization, and rule 3 does not apply."""
    candidate = make_candidate("app_1", role="Senior II")
    assert match_application(make_classification(), [candidate], "t-new", now=NOW) is None
