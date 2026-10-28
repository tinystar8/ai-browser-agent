"""Manual mode: a short command list, and questions answered from the screen."""

from __future__ import annotations

import re
from pathlib import Path

MANUAL_COMMANDS = """
Commands you can send:

click 420 310
right click 420 310
click index 5
right click index 5
press Escape
open https://example.com
scroll down
close
commands

Ask about what is on screen. Example: Tell me the coordinate of the first blue bottle.
""".strip()

_COMMANDS = re.compile(
    r"^(?:commands|command list|help|what can i (?:run|do|type)|show commands)[.!]?$",
    re.IGNORECASE,
)
_QUESTION = re.compile(
    r"^(?:tell me|what|where|which|who|how many|how much|how do|how to|"
    r"is there|are there|can you see|do you see)\b",
    re.IGNORECASE,
)
_ACTION = re.compile(
    r"^(?:click|right click|right-click|press|open|go|navigate|scroll|type|close)\b",
    re.IGNORECASE,
)


def is_command_request(message: str) -> bool:
    """True when the user is asking which manual commands exist."""

    return _COMMANDS.fullmatch(message.strip()) is not None


def is_screen_question(message: str) -> bool:
    """True when the user wants an answer about the screen, not a browser action."""

    text = message.strip()
    if not text or _ACTION.match(text):
        return False
    if _QUESTION.match(text):
        return True
    return text.endswith("?")


def latest_screenshot(directory: Path) -> Path | None:
    """Return the newest saved page screenshot, if one exists."""

    if not directory.is_dir():
        return None
    files = [path for path in directory.glob("task-*/step-*.png") if path.is_file()]
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_mtime)
