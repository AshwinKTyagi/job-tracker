"""Runtime configuration.

Config is constructed once in ``cli.py`` and passed explicitly to every collaborator.
Library modules never read a global, never consult the environment, and never re-read
config.toml themselves.

Secrets and data live under JOBTRACK_HOME, never in the repo.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, Field, ValidationError

from jobtrack.constants import DEFAULT_GMAIL_QUERY
from jobtrack.errors import ConfigError

DEFAULT_JOBTRACK_HOME: Final[Path] = Path.home() / ".local" / "share" / "jobtrack"
"""Overridable by the JOBTRACK_HOME environment variable. Holds config.toml,
credentials.json, token.json, and jobtrack.db. NEVER inside the repo."""

CONFIG_FILENAME: Final[str] = "config.toml"
ENV_HOME: Final[str] = "JOBTRACK_HOME"


class GmailConfig(BaseModel):
    """Mailbox query parameters."""

    query: str = DEFAULT_GMAIL_QUERY
    lookback_days: int = Field(default=400, gt=0)
    max_per_sync: int = Field(default=500, gt=0)


class ClassifyConfig(BaseModel):
    """Classifier selection and the review threshold."""

    min_confidence: float = Field(default=0.60, ge=0.0, le=1.0)
    backend: str = "rules"  # Phase 3: "rules+ollama"
    ollama_model: str | None = None  # pinned tag; resolved to a digest at runtime
    ollama_host: str = "http://localhost:11434"


class StoreConfig(BaseModel):
    """Persistence tuning."""

    ghost_after_days: int = Field(default=30, gt=0)


class ExportConfig(BaseModel):
    """Spreadsheet defaults."""

    default_format: str = "xlsx"  # "xlsx" | "csv"


class Config(BaseModel):
    """Fully-resolved runtime configuration."""

    home: Path = DEFAULT_JOBTRACK_HOME
    gmail: GmailConfig = Field(default_factory=GmailConfig)
    classify: ClassifyConfig = Field(default_factory=ClassifyConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)

    @property
    def db_path(self) -> Path:
        """Path to the SQLite database."""
        return self.home / "jobtrack.db"

    @property
    def credentials_path(self) -> Path:
        """Path to the Google OAuth client secrets downloaded from GCP."""
        return self.home / "credentials.json"

    @property
    def token_path(self) -> Path:
        """Path to the persisted OAuth token."""
        return self.home / "token.json"

    @property
    def config_path(self) -> Path:
        """Path to config.toml (may not exist; every field has a default)."""
        return self.home / CONFIG_FILENAME


def resolve_home(home: Path | None = None) -> Path:
    """Resolve JOBTRACK_HOME.

    Precedence: explicit argument, then the JOBTRACK_HOME environment variable, then
    ``DEFAULT_JOBTRACK_HOME``.

    Args:
        home: Explicit override, usually from a CLI flag.

    Returns:
        The expanded, absolute home directory. Not created here.
    """
    if home is not None:
        return home.expanduser().resolve()
    env = os.environ.get(ENV_HOME)
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_JOBTRACK_HOME


def load_config(home: Path | None = None) -> Config:
    """Resolve JOBTRACK_HOME, read config.toml if present, and merge it over the defaults.

    A missing config.toml is NOT an error — every field has a usable default.

    Args:
        home: Explicit JOBTRACK_HOME override.

    Returns:
        The fully-resolved configuration.

    Raises:
        ConfigError: config.toml is malformed, or an unknown/invalid value was supplied.
    """
    resolved = resolve_home(home)
    path = resolved / CONFIG_FILENAME

    data: dict[str, Any] = {}
    if path.is_file():
        try:
            with path.open("rb") as handle:
                data = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{path} is not valid TOML: {exc}") from exc
        except OSError as exc:
            raise ConfigError(f"could not read {path}: {exc}") from exc

    data["home"] = resolved
    try:
        return Config.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration in {path}: {exc}") from exc


def ensure_home(config: Config) -> Path:
    """Create JOBTRACK_HOME if it does not exist and confirm it is writable.

    Args:
        config: The resolved configuration.

    Returns:
        The home directory.

    Raises:
        ConfigError: the directory could not be created, or is not writable.
    """
    try:
        config.home.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigError(f"could not create {config.home}: {exc}") from exc
    if not os.access(config.home, os.W_OK):
        raise ConfigError(f"{config.home} is not writable")
    return config.home


__all__ = [
    "CONFIG_FILENAME",
    "DEFAULT_JOBTRACK_HOME",
    "ClassifyConfig",
    "Config",
    "ExportConfig",
    "GmailConfig",
    "StoreConfig",
    "ensure_home",
    "load_config",
    "resolve_home",
]
