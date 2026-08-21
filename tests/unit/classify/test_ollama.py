"""Tests for the M7 Ollama backend.

No daemon is contacted anywhere in this file: every classifier is constructed with an
injected ``transport``, the same seam ``GmailSource`` uses for the Gmail API. The suite runs
with sockets disabled, so a real request would fail loudly rather than silently pass.

The properties that matter are the reproducibility contract in CONTRACTS.md §10: never
raise, degrade to a zero-confidence sentinel so the composite prefers rules, honour the
triage gate, and cache on (prompt_sha, model fingerprint, message_id).
"""

from __future__ import annotations

import json
import urllib.error
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import jobtrack.classify.ollama as ollama_module
from jobtrack.classify.base import CompositeClassifier
from jobtrack.classify.ollama import OllamaClassifier
from jobtrack.classify.rules import RulesClassifier
from jobtrack.errors import ClassificationError
from jobtrack.models import EventType, RawMessage

MessageFactory = Callable[..., RawMessage]

TAGS = {
    "models": [
        {
            "name": "qwen2.5:7b",
            "digest": "845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e",
            "details": {"quantization_level": "Q4_K_M"},
        }
    ]
}


def verdict(**overrides: Any) -> dict[str, Any]:
    """Build a model response body, defaulting to a clean rejection."""
    payload: dict[str, Any] = {
        "is_job_application_email": True,
        "event_type": "rejection",
        "company": "2K",
        "role": "Engineering Graduate Program",
        "location": "US",
    }
    payload.update(overrides)
    return payload


def transport_for(*bodies: dict[str, Any]) -> tuple[Callable[..., dict[str, Any]], list[str]]:
    """A transport replaying `bodies` in order, plus the list of URLs it was called with."""
    calls: list[str] = []
    queue = list(bodies)

    def _transport(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(url)
        if url.endswith("/api/tags"):
            return TAGS
        body = queue.pop(0) if queue else verdict()
        return {"message": {"content": json.dumps(body)}}

    return _transport, calls


def build(transport: Callable[..., dict[str, Any]], cache: Path | None = None) -> OllamaClassifier:
    """Construct a classifier against an injected transport."""
    return OllamaClassifier("qwen2.5:7b", cache=cache, transport=transport)


# --- happy path -------------------------------------------------------------


def test_extracts_the_company_the_rules_engine_gets_wrong(make_message: MessageFactory) -> None:
    """The whole reason M7 exists: 'at this time' must not become a company name."""
    transport, _ = transport_for(verdict())
    message = make_message(
        subject="2K | Regarding your recent application",
        body_text="We have decided not to proceed with your application at this time.",
    )

    result = build(transport).classify(message)

    assert result.event_type is EventType.REJECTION
    assert result.company == "2K"
    assert result.role == "Engineering Graduate Program"


def test_company_key_comes_from_normalize_company(make_message: MessageFactory) -> None:
    """I8: normalize_company is the sole producer of company_key."""
    transport, _ = transport_for(verdict(company="Acme Robotics, Inc."))

    result = build(transport).classify(make_message())

    assert result.company == "Acme Robotics, Inc."
    assert result.company_key == "acme robotics"


def test_version_pins_prompt_digest_and_quantization(make_message: MessageFactory) -> None:
    """A prompt edit, a model change, or a requant must all change classifier_version."""
    transport, _ = transport_for(verdict())

    version = build(transport).version

    assert "845dbda0ea48" in version
    assert "Q4_K_M" in version


def test_attribution_names_the_backend(make_message: MessageFactory) -> None:
    """Stored rows stay attributable to the backend that wrote them, never 'composite'."""
    transport, _ = transport_for(verdict())

    result = build(transport).classify(make_message())

    assert result.classifier_name == "ollama"
    assert result.classifier_version.startswith("51b618142b53") or "+" in result.classifier_version


# --- the triage gate --------------------------------------------------------


def test_non_job_mail_becomes_unknown(make_message: MessageFactory) -> None:
    """A newsletter is UNKNOWN even if the model filled in the fields anyway."""
    transport, _ = transport_for(
        verdict(is_job_application_email=False, event_type="application_received", company="Medium")
    )

    result = build(transport).classify(make_message(subject="Why everything looks the same"))

    assert result.event_type is EventType.UNKNOWN
    assert result.company is None
    assert result.role is None
    assert "ollama.triage.not_job_mail" in result.evidence


def test_non_job_mail_scores_low_enough_to_reach_the_rules(make_message: MessageFactory) -> None:
    """UNKNOWN is deliberately low-confidence so the composite lets rules disagree."""
    transport, _ = transport_for(verdict(is_job_application_email=False))

    result = build(transport).classify(make_message())

    assert result.confidence < 0.60
    assert result.needs_review is True


# --- placeholder and nullish handling ---------------------------------------


@pytest.mark.parametrize("placeholder", ["null", "None", "N/A", "", "  ", "unknown", "-"])
def test_placeholder_strings_become_none(placeholder: str, make_message: MessageFactory) -> None:
    """Ollama's JSON mode returns the *string* 'null'; that must not become a company."""
    transport, _ = transport_for(verdict(company=placeholder))

    result = build(transport).classify(make_message())

    assert result.company is None


def test_whitespace_and_stray_commas_are_trimmed(make_message: MessageFactory) -> None:
    """Padding is noise, but a period is not: "Inc." must survive as a legal suffix."""
    transport, _ = transport_for(verdict(company="  Acme Robotics,  "))
    padded = build(transport).classify(make_message())
    assert padded.company == "Acme Robotics"

    transport2, _ = transport_for(verdict(company="Acme Robotics, Inc."))
    suffixed = build(transport2).classify(make_message())
    assert suffixed.company == "Acme Robotics, Inc."


# --- never raises -----------------------------------------------------------


def test_a_dead_daemon_degrades_to_a_sentinel(make_message: MessageFactory) -> None:
    """A stopped daemon must not break sync — it scores 0.0 so rules win."""

    def dead(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        raise ClassificationError("connection refused")

    result = OllamaClassifier("qwen2.5:7b", transport=dead).classify(make_message())

    assert result.confidence == 0.0
    assert result.event_type is EventType.UNKNOWN
    assert "ollama.unavailable.transport" in result.evidence


def test_an_invented_event_type_degrades_to_a_sentinel(make_message: MessageFactory) -> None:
    """The enum constrains the model; anything outside it is rejected, not stored."""
    transport, _ = transport_for(verdict(event_type="ghosted_probably"))

    result = build(transport).classify(make_message())

    assert result.confidence == 0.0
    assert "ollama.unavailable.schema" in result.evidence


def test_non_json_content_degrades_to_a_sentinel(make_message: MessageFactory) -> None:
    """Free-text prose from the model is a failure, never something to regex over."""

    def prose(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        if url.endswith("/api/tags"):
            return TAGS
        return {"message": {"content": "Sure! Here's the classification:"}}

    result = build(prose).classify(make_message())

    assert result.confidence == 0.0


def test_an_unreachable_daemon_at_construction_is_not_fatal(make_message: MessageFactory) -> None:
    """Construction must survive a dead daemon; the version records it as unresolved."""

    def dead(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        raise ClassificationError("connection refused")

    classifier = OllamaClassifier("qwen2.5:7b", transport=dead)

    assert "unresolved" in classifier.version


# --- determinism and caching ------------------------------------------------


def test_classifying_twice_is_byte_identical(make_message: MessageFactory) -> None:
    """I2: the same message in produces byte-identical output."""
    transport, _ = transport_for(verdict(), verdict())
    classifier = build(transport)
    message = make_message()

    first = classifier.classify(message)
    second = classifier.classify(message)

    assert first.model_dump_json() == second.model_dump_json()


def test_a_cached_response_skips_the_daemon(tmp_path: Path, make_message: MessageFactory) -> None:
    """Caching is what makes reclassify free and byte-identical."""
    transport, calls = transport_for(verdict())
    classifier = build(transport, cache=tmp_path / "cache")
    message = make_message()

    classifier.classify(message)
    chat_calls_after_first = sum(1 for url in calls if url.endswith("/api/chat"))
    classifier.classify(message)
    chat_calls_after_second = sum(1 for url in calls if url.endswith("/api/chat"))

    assert chat_calls_after_first == 1
    assert chat_calls_after_second == 1


def test_the_cache_key_includes_the_prompt(tmp_path: Path, make_message: MessageFactory) -> None:
    """Editing the prompt must miss the cache rather than return a stale attribution."""
    transport, _ = transport_for(verdict())
    classifier = build(transport, cache=tmp_path / "cache")
    message = make_message()
    classifier.classify(message)

    written = list((tmp_path / "cache").glob("*.json"))
    assert len(written) == 1
    # The filename is a hash of prompt_sha|fingerprint|message_id, so it is not just the id.
    assert message.message_id not in written[0].name


def test_a_corrupt_cache_entry_is_ignored(tmp_path: Path, make_message: MessageFactory) -> None:
    """A truncated cache file must not poison a classification."""
    transport, _ = transport_for(verdict())
    cache = tmp_path / "cache"
    classifier = build(transport, cache=cache)
    message = make_message()
    classifier.classify(message)
    for path in cache.glob("*.json"):
        path.write_text("{not json", encoding="utf-8")

    result = classifier.classify(message)

    assert result.event_type is EventType.REJECTION


def test_caching_can_be_disabled(make_message: MessageFactory) -> None:
    """cache=None means every call goes to the daemon."""
    transport, calls = transport_for(verdict(), verdict())
    classifier = build(transport, cache=None)
    message = make_message()

    classifier.classify(message)
    classifier.classify(message)

    assert sum(1 for url in calls if url.endswith("/api/chat")) == 2


# --- request shape ----------------------------------------------------------


def test_the_request_pins_determinism_options(make_message: MessageFactory) -> None:
    """temperature=0, top_p=1, a fixed seed, and thinking disabled are all mandatory."""
    seen: list[dict[str, Any]] = []

    def recorder(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        seen.append(payload)
        if url.endswith("/api/tags"):
            return TAGS
        return {"message": {"content": json.dumps(verdict())}}

    build(recorder).classify(make_message())

    chat = [p for p in seen if p.get("model")][-1]
    assert chat["options"]["temperature"] == 0
    assert chat["options"]["top_p"] == 1
    assert chat["options"]["seed"] == 42
    assert chat["think"] is False
    assert chat["stream"] is False


def test_the_request_constrains_the_event_type_enum(make_message: MessageFactory) -> None:
    """Grammar-constrained decoding is what stops the model inventing an EventType."""
    seen: list[dict[str, Any]] = []

    def recorder(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        seen.append(payload)
        if url.endswith("/api/tags"):
            return TAGS
        return {"message": {"content": json.dumps(verdict())}}

    build(recorder).classify(make_message())

    chat = [p for p in seen if p.get("model")][-1]
    enum = chat["format"]["properties"]["event_type"]["enum"]
    assert set(enum) == {member.value for member in EventType}


# --- composite wiring -------------------------------------------------------


def test_rules_take_over_when_ollama_is_down(make_message: MessageFactory) -> None:
    """The whole availability story: a dead daemon degrades quality, not uptime."""

    def dead(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        raise ClassificationError("connection refused")

    composite = CompositeClassifier(
        OllamaClassifier("qwen2.5:7b", transport=dead), RulesClassifier(), min_confidence=0.60
    )
    message = make_message(
        subject="Thank you for applying to Acme",
        body_text="We have received your application and will be in touch.",
        from_email="no-reply@greenhouse.io",
    )

    result = composite.classify(message)

    assert result.classifier_name == "rules"
    assert result.event_type is not EventType.UNKNOWN


def test_ollama_wins_when_it_is_confident(make_message: MessageFactory) -> None:
    """A confident model answer is taken without consulting the rules at all."""
    transport, _ = transport_for(verdict())
    composite = CompositeClassifier(build(transport), RulesClassifier(), min_confidence=0.60)

    result = composite.classify(make_message(subject="2K | Regarding your application"))

    assert result.classifier_name == "ollama"
    assert result.company == "2K"


def test_model_listing_is_fetched_with_a_get(make_message: MessageFactory) -> None:
    """Regression: /api/tags is GET-only. POSTing to it 405s, and the digest then silently
    fails to resolve — classification still works, so the broken pin goes unnoticed."""
    methods: list[tuple[str, bool]] = []

    def recorder(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        methods.append((url, bool(payload)))
        if url.endswith("/api/tags"):
            return TAGS
        return {"message": {"content": json.dumps(verdict())}}

    classifier = build(recorder)
    classifier.classify(make_message())

    tags_calls = [has_payload for url, has_payload in methods if url.endswith("/api/tags")]
    assert tags_calls == [False], "the tags listing must be sent with an empty payload (GET)"
    assert "unresolved" not in classifier.version


# --- the real HTTP transport ------------------------------------------------


class _FakeResponse:
    """Minimal stand-in for the object urlopen returns."""

    def __init__(self, body: bytes) -> None:
        """Args:
        body: The raw response bytes.
        """
        self._body = body

    def read(self) -> bytes:
        """Return the response body."""
        return self._body

    def __enter__(self) -> _FakeResponse:
        """Support the with-statement urlopen is used under."""
        return self

    def __exit__(self, *exc: object) -> None:
        """No cleanup needed."""
        return None


def test_transport_posts_json_when_there_is_a_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """A chat call carries a JSON body and the matching content type."""
    seen: list[Any] = []

    def fake_urlopen(request: Any, timeout: float = 0) -> _FakeResponse:
        seen.append(request)
        return _FakeResponse(b'{"ok": true}')

    monkeypatch.setattr(ollama_module.urllib.request, "urlopen", fake_urlopen)

    result = ollama_module._http_transport("http://h/api/chat", {"model": "m"}, timeout=1.0)

    assert result == {"ok": True}
    assert seen[0].data == b'{"model": "m"}'
    assert seen[0].get_header("Content-type") == "application/json"


def test_transport_gets_when_the_payload_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tags listing must go out as a GET with no body."""
    seen: list[Any] = []

    def fake_urlopen(request: Any, timeout: float = 0) -> _FakeResponse:
        seen.append(request)
        return _FakeResponse(b'{"models": []}')

    monkeypatch.setattr(ollama_module.urllib.request, "urlopen", fake_urlopen)

    ollama_module._http_transport("http://h/api/tags", {}, timeout=1.0)

    assert seen[0].data is None
    assert seen[0].get_method() == "GET"


def test_transport_wraps_a_connection_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dead daemon becomes a ClassificationError, never a raw URLError."""

    def boom(request: Any, timeout: float = 0) -> _FakeResponse:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(ollama_module.urllib.request, "urlopen", boom)

    with pytest.raises(ClassificationError, match="unreachable"):
        ollama_module._http_transport("http://h/api/chat", {"a": 1}, timeout=1.0)


def test_transport_wraps_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """A truncated response is a ClassificationError, not a JSONDecodeError."""

    def truncated(request: Any, timeout: float = 0) -> _FakeResponse:
        return _FakeResponse(b'{"partial"')

    monkeypatch.setattr(ollama_module.urllib.request, "urlopen", truncated)

    with pytest.raises(ClassificationError, match="invalid JSON"):
        ollama_module._http_transport("http://h/api/chat", {"a": 1}, timeout=1.0)


def test_a_missing_prompt_template_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """The prompt is the version; losing it must fail loudly, not silently."""
    monkeypatch.setattr(ollama_module, "PROMPT_PATH", Path("/nonexistent/prompt.txt"))

    with pytest.raises(ClassificationError, match="prompt template missing"):
        ollama_module._prompt_sha()


def test_a_model_absent_from_the_daemon_is_unresolved(make_message: MessageFactory) -> None:
    """Naming a model that was never pulled must not silently pin the wrong digest."""

    def empty_listing(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        if url.endswith("/api/tags"):
            return {"models": []}
        return {"message": {"content": json.dumps(verdict())}}

    classifier = OllamaClassifier("never-pulled:1b", transport=empty_listing)

    assert "unresolved" in classifier.version
