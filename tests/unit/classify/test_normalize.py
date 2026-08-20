"""Tests for jobtrack.classify.normalize."""

from __future__ import annotations

import pytest

from jobtrack.classify.normalize import normalize_company, normalize_role, role_similarity


def test_normalize_company_none_is_none() -> None:
    assert normalize_company(None) is None


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("Acme Robotics, Inc.", "acme robotics"),
        ("acme robotics", "acme robotics"),
        ("ACME ROBOTICS INC", "acme robotics"),
        ("Acme Robotics LLC", "acme robotics"),
        ("Acme Robotics Ltd.", "acme robotics"),
        ("Acme Robotics Corp.", "acme robotics"),
        ("Acme GmbH", "acme"),
        ("Open Collective, PBC", "open collective"),
        ("Acme & Co.", "acme"),
        ("  Acme   Robotics  ", "acme robotics"),
    ],
)
def test_normalize_company_strips_suffixes_and_punctuation(raw: str, want: str) -> None:
    assert normalize_company(raw) == want


def test_normalize_company_matches_across_legal_suffix_variants() -> None:
    assert normalize_company("Acme Robotics, Inc.") == normalize_company("acme robotics")


def test_normalize_company_is_idempotent() -> None:
    for raw in ("Acme Robotics, Inc.", "Solstice Health", "BrightPath Analytics, LLC"):
        once = normalize_company(raw)
        twice = normalize_company(once)
        assert once == twice


def test_normalize_company_all_suffix_tokens_yields_none() -> None:
    assert normalize_company("Inc.") is None
    assert normalize_company("   ") is None
    assert normalize_company("") is None


def test_normalize_role_none_is_none() -> None:
    assert normalize_role(None) is None


def test_normalize_role_strips_seniority_and_expands_abbreviations() -> None:
    assert normalize_role("Senior SWE") == normalize_role("Software Engineer")


def test_normalize_role_strips_requisition_ids() -> None:
    assert normalize_role("Backend Engineer (Req-12345)") == normalize_role("Backend Engineer")


def test_normalize_role_empty_after_stripping_is_none() -> None:
    assert normalize_role("Senior") is None


def test_role_similarity_exact_match_after_normalization_is_one() -> None:
    assert role_similarity("Senior Software Engineer", "senior software engineer") == 1.0
    assert role_similarity("SWE", "Software Engineer") == 1.0


def test_role_similarity_none_operand_is_zero() -> None:
    assert role_similarity(None, "Software Engineer") == 0.0
    assert role_similarity("Software Engineer", None) == 0.0
    assert role_similarity(None, None) == 0.0


def test_role_similarity_is_between_zero_and_one_for_related_titles() -> None:
    score = role_similarity("Senior Software Engineer, Platform", "Software Engineer")
    assert 0.0 < score < 1.0


def test_role_similarity_unrelated_titles_is_low() -> None:
    score = role_similarity("Software Engineer", "Marketing Manager")
    assert score < 0.5


def test_role_similarity_is_deterministic() -> None:
    a, b = "Senior Backend Engineer", "Backend Engineer II"
    assert role_similarity(a, b) == role_similarity(a, b)
