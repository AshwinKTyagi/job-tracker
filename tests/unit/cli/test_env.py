"""Tests for .env loading and the classifier-selection overlay.

``config.py`` deliberately never reads the environment beyond JOBTRACK_HOME, so the overlay
lives in the composition root. Precedence, lowest first: defaults, config.toml, .env, real
environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from jobtrack import cli
from jobtrack.classify.base import CompositeClassifier
from jobtrack.config import Config
from jobtrack.errors import ConfigError


@pytest.fixture(autouse=True)
def _clear_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test from a clean environment."""
    for key in (
        cli.ENV_BACKEND,
        cli.ENV_OLLAMA_MODEL,
        cli.ENV_OLLAMA_HOST,
        cli.ENV_MIN_CONFIDENCE,
    ):
        monkeypatch.delenv(key, raising=False)


# --- parsing ----------------------------------------------------------------


def test_parses_plain_assignments() -> None:
    """The common case: KEY=value, one per line."""
    parsed = cli._parse_dotenv("JOBTRACK_OLLAMA_MODEL=qwen2.5:7b\nOTHER=2\n")
    assert parsed == {"JOBTRACK_OLLAMA_MODEL": "qwen2.5:7b", "OTHER": "2"}


def test_ignores_comments_and_blank_lines() -> None:
    """A commented .env must not set anything."""
    parsed = cli._parse_dotenv("# a comment\n\n  # indented\nA=1\n")
    assert parsed == {"A": "1"}


def test_strips_export_prefix_and_quotes() -> None:
    """Shell-style .env files are common; both forms should work."""
    parsed = cli._parse_dotenv("export A=\"one\"\nB='two'\n")
    assert parsed == {"A": "one", "B": "two"}


def test_ignores_lines_without_an_equals() -> None:
    """A stray word is skipped rather than crashing the CLI at startup."""
    assert cli._parse_dotenv("nonsense\nA=1\n") == {"A": "1"}


def test_a_url_value_keeps_its_colons() -> None:
    """Only the first = splits, so http://host:11434 survives intact."""
    parsed = cli._parse_dotenv("JOBTRACK_OLLAMA_HOST=http://localhost:11434\n")
    assert parsed == {"JOBTRACK_OLLAMA_HOST": "http://localhost:11434"}


# --- loading ----------------------------------------------------------------


def test_loads_a_dotenv_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Values in .env reach the process environment."""
    env = tmp_path / ".env"
    env.write_text("JOBTRACK_OLLAMA_MODEL=qwen2.5:7b\n", encoding="utf-8")

    cli._load_dotenv((str(env),))

    assert os.environ[cli.ENV_OLLAMA_MODEL] == "qwen2.5:7b"


def test_a_real_environment_variable_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """.env is a default, not an override — the shell always wins."""
    monkeypatch.setenv(cli.ENV_OLLAMA_MODEL, "from-shell")
    env = tmp_path / ".env"
    env.write_text("JOBTRACK_OLLAMA_MODEL=from-file\n", encoding="utf-8")

    cli._load_dotenv((str(env),))

    assert os.environ[cli.ENV_OLLAMA_MODEL] == "from-shell"


def test_a_missing_dotenv_is_not_an_error() -> None:
    """Every value .env can set has a working default, so absence is normal."""
    cli._load_dotenv(("/nonexistent/.env",))


# --- overlay ----------------------------------------------------------------


def test_overlay_applies_backend_and_model(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Environment values land on ClassifyConfig."""
    monkeypatch.setenv(cli.ENV_BACKEND, "ollama")
    monkeypatch.setenv(cli.ENV_OLLAMA_MODEL, "qwen2.5:7b")

    merged = cli._apply_env_overrides(tmp_config)

    assert merged.classify.backend == "ollama"
    assert merged.classify.ollama_model == "qwen2.5:7b"


def test_overlay_leaves_config_untouched_when_unset(tmp_config: Config) -> None:
    """No overrides means the loaded Config is returned as-is."""
    assert cli._apply_env_overrides(tmp_config) is tmp_config


def test_overlay_parses_min_confidence(tmp_config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    """The review threshold is tunable without editing config.toml."""
    monkeypatch.setenv(cli.ENV_MIN_CONFIDENCE, "0.8")

    assert cli._apply_env_overrides(tmp_config).classify.min_confidence == 0.8


def test_a_non_numeric_min_confidence_is_a_config_error(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A typo in .env is reported, not silently ignored."""
    monkeypatch.setenv(cli.ENV_MIN_CONFIDENCE, "very")

    with pytest.raises(ConfigError):
        cli._apply_env_overrides(tmp_config)


def test_an_out_of_range_min_confidence_is_a_config_error(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ClassifyConfig bounds it to [0, 1]; the overlay must surface that as ConfigError."""
    monkeypatch.setenv(cli.ENV_MIN_CONFIDENCE, "5.0")

    with pytest.raises(ConfigError):
        cli._apply_env_overrides(tmp_config)


class _StubOllama:
    """Stands in for OllamaClassifier so wiring can be asserted without a daemon."""

    name = "ollama"
    version = "stub"

    def classify(self, message: object) -> object:  # pragma: no cover - never called
        raise AssertionError("the stub must not be asked to classify")

    def classify_batch(self, messages: object) -> object:  # pragma: no cover
        raise AssertionError("the stub must not be asked to classify")


def _stub_ollama(model: str, **kwargs: object) -> _StubOllama:
    """Factory matching OllamaClassifier's call signature."""
    return _StubOllama()


# --- classifier selection ---------------------------------------------------


def test_default_backend_is_rules_only(tmp_config: Config) -> None:
    """Without configuration the tool stays fully offline."""
    built = cli._build_classifier(tmp_config)

    assert isinstance(built, CompositeClassifier)
    assert built.classify.__self__._fallback is None  # type: ignore[attr-defined]


def test_ollama_backend_without_a_model_is_a_config_error(tmp_config: Config) -> None:
    """Naming a backend but no model is a configuration mistake worth reporting."""
    config = tmp_config.model_copy(
        update={"classify": tmp_config.classify.model_copy(update={"backend": "ollama"})}
    )

    with pytest.raises(ConfigError):
        cli._build_classifier(config)


def test_ollama_backend_builds_a_composite_with_a_rules_fallback(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wiring that matters: ollama leads, rules catch it when it fails."""
    # Inject a transport rather than letting construction reach for the daemon: resolving
    # the model digest is a real HTTP call, and no test may touch the network.
    monkeypatch.setattr(cli, "OllamaClassifier", _stub_ollama)
    config = tmp_config.model_copy(
        update={
            "classify": tmp_config.classify.model_copy(
                update={
                    "backend": "ollama",
                    "ollama_model": "qwen2.5:7b",
                    "ollama_host": "http://localhost:1",
                }
            )
        }
    )

    built = cli._build_classifier(config)

    assert isinstance(built, CompositeClassifier)
    assert built.classify.__self__._primary.name == "ollama"  # type: ignore[attr-defined]
    assert built.classify.__self__._fallback.name == "rules"  # type: ignore[attr-defined]
