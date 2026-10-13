"""Environment configuration.

The model endpoint is always the local Ollama server. This module refuses the
OpenAI cloud host so a mis-set base URL cannot silently send tasks off-machine.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# Browser Use phones home unless this is set. Default it off for a local agent.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")

_CLOUD_HOSTS = {"api.openai.com", "openai.com"}


def assert_local_openai_endpoint(base_url: str) -> None:
    """Raise if the configured endpoint is the OpenAI cloud API."""

    host = (urlparse(base_url).hostname or "").lower()
    if host in _CLOUD_HOSTS:
        raise RuntimeError(
            "Refusing the OpenAI cloud API. Set OPENAI_BASE_URL to the local "
            "Ollama server, for example http://localhost:11434/v1."
        )


class Settings(BaseSettings):
    """Runtime settings loaded from the environment and the project .env file."""

    openai_api_key: str = Field(default="ollama")
    openai_base_url: str = Field(default="http://localhost:11434/v1")
    openai_model: str = Field(default="qwen3.6:35b")

    headless: bool = Field(default=False)
    browser_profile: str = Field(default="./browser-profile")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    database_path: str = Field(default="./database/agent.db")
    screenshots_dir: str = Field(default="./screenshots")
    logs_dir: str = Field(default="./logs")

    max_steps: int = Field(default=25)
    llm_timeout: int = Field(default=180)
    step_timeout: int = Field(default=240)
    use_vision: str = Field(default="false")
    use_thinking: bool = Field(default=False)
    shell_enabled: bool = Field(default=True)

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    def resolved_path(self, value: str) -> Path:
        """Resolve a config path against the project root."""

        path = Path(value).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    @property
    def profile_path(self) -> Path:
        return self.resolved_path(self.browser_profile)

    @property
    def database_file(self) -> Path:
        return self.resolved_path(self.database_path)

    @property
    def screenshots_path(self) -> Path:
        return self.resolved_path(self.screenshots_dir)

    @property
    def logs_path(self) -> Path:
        return self.resolved_path(self.logs_dir)

    @property
    def action_log_file(self) -> Path:
        return self.logs_path / "actions.jsonl"

    def vision_mode(self) -> bool | str:
        """Return the Browser Use vision flag: True, False, or 'auto'."""

        value = self.use_vision.strip().lower()
        if value == "auto":
            return "auto"
        return value in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings object."""

    settings = Settings()
    assert_local_openai_endpoint(settings.openai_base_url)
    return settings
