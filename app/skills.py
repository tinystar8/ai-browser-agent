"""Load prompt skills from the skills directory.

Add a skill by creating skills/<name>.md. The header names the skill and its
arguments. The body is the prompt, with {argument} placeholders.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from app.config import PROJECT_ROOT
from app.tools.solidity import SolidityError, save_verified_source

_ETH_ADDRESS = re.compile(r"^0x[0-9a-fA-F]{40}$")
_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class SkillError(Exception):
    """Raised when a skill file or its arguments cannot be used."""


@dataclass(frozen=True)
class Skill:
    """One reusable prompt loaded from skills/."""

    name: str
    description: str
    args: tuple[str, ...]
    checks: dict[str, str]
    fetch: str
    body: str
    path: Path


def skills_directory() -> Path:
    """Return the folder that holds skill files."""

    return PROJECT_ROOT / "skills"


def list_skills(directory: Path | None = None) -> list[Skill]:
    """Return every skill, sorted by name."""

    folder = directory or skills_directory()
    if not folder.is_dir():
        return []
    found = [load_skill(path) for path in sorted(folder.glob("*.md"))]
    return sorted(found, key=lambda skill: skill.name)


def get_skill(name: str, directory: Path | None = None) -> Skill:
    """Find a skill by its header name or its file name."""

    key = name.strip().lower()
    for skill in list_skills(directory):
        if skill.name.lower() == key or skill.path.stem.lower() == key:
            return skill
    known = ", ".join(skill.name for skill in list_skills(directory)) or "(none)"
    raise SkillError(f"Unknown skill '{name}'. Available skills: {known}")


def render_skill(name: str, arguments: dict[str, str], directory: Path | None = None) -> str:
    """Fill a skill's placeholders and return the prompt sent to the model."""

    skill = get_skill(name, directory)
    missing = [arg for arg in skill.args if not arguments.get(arg, "").strip()]
    if missing:
        joined = ", ".join(missing)
        raise SkillError(f"Skill '{skill.name}' needs: {joined}")

    for arg in skill.args:
        value = arguments[arg].strip()
        check = skill.checks.get(arg)
        if check == "ethereum" and _ETH_ADDRESS.fullmatch(value) is None:
            raise SkillError(f"{arg} must be 0x followed by 40 hex characters.")

    text = skill.body
    for arg in skill.args:
        text = text.replace("{" + arg + "}", arguments[arg].strip())
    text = text.strip()
    if skill.fetch == "solidity":
        address = arguments.get("address", "").strip()
        if not address:
            raise SkillError(f"Skill '{skill.name}' needs an address to save Solidity source.")
        try:
            saved = save_verified_source(address, PROJECT_ROOT)
        except SolidityError as exc:
            raise SkillError(str(exc)) from exc
        text = f"{text}\n\n{saved.note()}"
    return text


def load_skill(path: Path) -> Skill:
    """Read one skill file."""

    raw = path.read_text(encoding="utf-8")
    header, body = _split_header(raw)
    fields = _parse_header(header)
    name = fields.get("name", path.stem).strip()
    if not name:
        raise SkillError(f"{path.name} is missing a name.")
    description = fields.get("description", "").strip()
    args = tuple(part.strip() for part in fields.get("args", "").split(",") if part.strip())
    checks = _parse_checks(fields.get("validate", ""))
    fetch = fields.get("fetch", "").strip().lower()
    if fetch and fetch != "solidity":
        raise SkillError(f"{path.name} has unknown fetch '{fetch}'. Use fetch: solidity.")
    if not body.strip():
        raise SkillError(f"{path.name} has no prompt body.")
    return Skill(
        name=name,
        description=description,
        args=args,
        checks=checks,
        fetch=fetch,
        body=body.strip(),
        path=path,
    )


def _split_header(raw: str) -> tuple[str, str]:
    match = _FRONT_MATTER.match(raw)
    if match:
        return match.group(1), raw[match.end() :]
    if "\n\n" not in raw:
        raise SkillError("A skill file needs a header, a blank line, then the prompt.")
    header, body = raw.split("\n\n", 1)
    return header, body


def _parse_header(header: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in header.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        fields[key.strip().lower()] = value.strip()
    return fields


def _parse_checks(raw: str) -> dict[str, str]:
    checks: dict[str, str] = {}
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise SkillError(f"Invalid validate entry '{item}'. Use arg=ethereum.")
        arg, rule = item.split("=", 1)
        checks[arg.strip()] = rule.strip()
    return checks


def expand_prompt(text: str, directory: Path | None = None) -> str:
    """Turn a skill name plus arguments into a prompt. Other text stays as written."""

    words = text.split()
    if not words:
        raise SkillError("Type a prompt.")
    key = words[0].strip().lower()
    skill: Skill | None = None
    for item in list_skills(directory):
        if item.name.lower() == key or item.path.stem.lower() == key:
            skill = item
            break
    if skill is None:
        return " ".join(words)
    if not skill.args:
        body = render_skill(skill.name, {}, directory)
        extra = " ".join(words[1:]).strip()
        return f"{body}\n\n{extra}" if extra else body
    values: dict[str, str] = {}
    if len(skill.args) == 1 and len(words) >= 2:
        values[skill.args[0]] = " ".join(words[1:])
    else:
        if len(words) - 1 < len(skill.args):
            expected = " ".join(skill.args)
            raise SkillError(f"Usage: {skill.name} {expected}")
        for name, value in zip(skill.args, words[1:], strict=False):
            values[name] = value
    return render_skill(skill.name, values, directory)
