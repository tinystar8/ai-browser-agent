"""Task and action safety gate.

Dangerous browser work is refused before Chromium launches, and again after
each model step before Browser Use executes the proposed actions. A negated
phrase such as "do not delete files" is not treated as an attempt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# (pattern, category). Patterns are matched case-insensitively.
_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b(delete|remove|erase|shred|wipe)\b.{0,48}\b(file|files|folder|directory|database)\b",
            re.IGNORECASE,
        ),
        "deleting files",
    ),
    (
        re.compile(r"\brm\s+-[a-zA-Z]*[rfd]", re.IGNORECASE),
        "deleting files",
    ),
    (
        re.compile(
            r"\b(send|submit|post)\b.{0,40}\b(message|email|e-mail|dm|tweet|sms)\b",
            re.IGNORECASE,
        ),
        "sending messages",
    ),
    (
        re.compile(
            r"\b(buy|purchase|checkout|pay|payment|wire|transfer)\b.{0,48}\b"
            r"(money|funds|usd|\$|crypto|order|cart|invoice)\b",
            re.IGNORECASE,
        ),
        "financial transactions",
    ),
    (
        re.compile(
            r"\b(pay now|place order|complete purchase|confirm payment|submit payment)\b",
            re.IGNORECASE,
        ),
        "financial transactions",
    ),
    (
        re.compile(
            r"\b(change|reset|update|set|create)\b.{0,24}\bpassword\b",
            re.IGNORECASE,
        ),
        "changing passwords",
    ),
    (
        re.compile(
            r"\b(apt(-get)?|dnf|yum|pacman|pip|npm|brew|snap)\s+install\b",
            re.IGNORECASE,
        ),
        "installing unknown software",
    ),
    (
        re.compile(r"\b(curl|wget)\b.{0,80}\|\s*(sh|bash|zsh)\b", re.IGNORECASE),
        "installing unknown software",
    ),
    (
        re.compile(
            r"\b(install|download)\b.{0,40}\b(software|package|program|binary|installer)\b",
            re.IGNORECASE,
        ),
        "installing unknown software",
    ),
)

_NEGATION = re.compile(
    r"\b(do not|don't|dont|never|without|avoid)\b[\w\s,'\"]{0,40}$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SafetyDecision:
    """Outcome of a safety check."""

    blocked: bool
    category: str | None = None
    reason: str = ""


def _negated(text: str, start: int) -> bool:
    """Return True when the match is preceded by a negation."""

    window = text[max(0, start - 60) : start]
    return _NEGATION.search(window) is not None


def assess_text(text: str) -> SafetyDecision:
    """Block text that asks for a dangerous side effect."""

    for pattern, category in _RULES:
        match = pattern.search(text)
        if match is None or _negated(text, match.start()):
            continue
        reason = (
            f"Blocked automatically: {category}. "
            "The agent will not delete files, send messages, make payments, "
            "change passwords, or install software."
        )
        return SafetyDecision(blocked=True, category=category, reason=reason)
    return SafetyDecision(blocked=False)


def assess_task(task: str) -> SafetyDecision:
    """Check a user task before the browser starts."""

    return assess_text(task)


def assess_action(action_name: str, detail: str, goal: str = "") -> SafetyDecision:
    """Check one proposed browser action before it is executed."""

    # Password-field typing is blocked even when the task itself looked harmless.
    if action_name.lower() in {"input", "input_text", "type", "send_keys"} and re.search(
        r"(password.?field|type=.?password|name.password|autocomplete..current-password)",
        detail,
        re.IGNORECASE,
    ):
        return SafetyDecision(
            blocked=True,
            category="changing passwords",
            reason="Blocked automatically: the model tried to type into a password field.",
        )
    return assess_text(f"{goal}\n{action_name} {detail}")
