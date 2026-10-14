"""Chromium configuration for Browser Use.

The profile directory is persistent, so a login you complete in this Chromium
window is still there on the next task. The agent does not read or store
passwords. Cloud browsers are never used.
"""

from __future__ import annotations

from typing import Any

from app.config import Settings, get_settings


def build_browser(settings: Settings | None = None) -> Any:
    """Launch a local Chromium session through Playwright."""

    from browser_use import Browser

    cfg = settings or get_settings()
    cfg.profile_path.mkdir(parents=True, exist_ok=True)
    return Browser(
        headless=cfg.headless,
        user_data_dir=str(cfg.profile_path),
        window_size={"width": 1280, "height": 800},
        keep_alive=True,
        enable_default_extensions=True,
    )
