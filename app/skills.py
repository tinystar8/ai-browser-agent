"""Load prompt skills from the skills directory.

Add a skill by creating skills/<name>.md. The header names the skill and its
arguments. The body is the prompt, with {argument} placeholders.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from app.config import PROJECT_ROOT
from app.tools.solidity import SolidityError, save_verified_source

_ETH_ADDRESS = re.compile(r"^0x[0-9a-fA-F]{40}$")
_ANY_ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}")
_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_ADD_SKILL = re.compile(
    r"^add\s+skill(?:\s+([A-Za-z][A-Za-z0-9-]{0,40}))?[.!]?\s*$",
    re.IGNORECASE,
)
_PICTURE_NOTE = re.compile(
    r"\n*A picture is attached to this instruction\. Use that picture\.\s*\Z"
)
_SAVED_NOTE = re.compile(
    r"\n*Verified Solidity source is already saved on disk\..*\Z",
    re.DOTALL,
)
_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_NAME_SKIP = {
    "the",
    "a",
    "an",
    "to",
    "and",
    "for",
    "of",
    "on",
    "in",
    "open",
    "please",
    "you",
    "are",
}


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


def add_skill_request(message: str) -> str | None:
    """Return '' for 'Add skill', a name for 'Add skill some-name', or None."""

    match = _ADD_SKILL.fullmatch(message.strip())
    if match is None:
        return None
    return (match.group(1) or "").lower()


def save_task_as_skill(
    tasks: list[object],
    *,
    name: str | None = None,
    directory: Path | None = None,
) -> Skill:
    """Write the latest real task into skills/ and return the loaded skill.

    The prompt uses the steps the browser actually completed. A second save of
    the same work updates that skill instead of creating another copy.
    """

    instruction, prompt, fetched = _work_prompt(tasks)
    folder = directory or skills_directory()
    folder.mkdir(parents=True, exist_ok=True)
    if name:
        chosen = name.strip().lower()
        if _NAME.fullmatch(chosen) is None:
            raise SkillError("A skill name uses letters, numbers, and hyphens.")
        existing = folder / f"{chosen}.md"
        if existing.exists():
            raise SkillError(f"A skill named {chosen} already exists.")
    else:
        chosen = _name_from_task(instruction)
    written, args, checks, fetch = _parameterize(prompt, fetched)
    description = _description(instruction)
    path = folder / f"{chosen}.md"
    path.write_text(
        _skill_file(chosen, description, args, checks, fetch, written),
        encoding="utf-8",
    )
    return load_skill(path)


def replace_skill(
    current: str,
    *,
    name: str,
    description: str,
    args: list[str],
    validate: str,
    fetch: str,
    body: str,
    directory: Path | None = None,
) -> Skill:
    """Rewrite one skill file. A new name renames the file."""

    skill = get_skill(current, directory)
    chosen = name.strip().lower()
    if _NAME.fullmatch(chosen) is None:
        raise SkillError("A skill name uses letters, numbers, and hyphens.")
    prompt = body.strip()
    if not prompt:
        raise SkillError("A skill needs a prompt.")
    fetch_value = fetch.strip().lower()
    if fetch_value and fetch_value != "solidity":
        raise SkillError("fetch must be solidity or empty.")
    arg_tuple = tuple(part.strip() for part in args if part.strip())
    checks = _parse_checks(validate)
    unknown = [arg for arg in checks if arg not in arg_tuple]
    if unknown:
        raise SkillError(f"validate refers to unknown argument '{unknown[0]}'.")
    bad = [rule for rule in checks.values() if rule != "ethereum"]
    if bad:
        raise SkillError(f"Unknown validate rule '{bad[0]}'. Use ethereum.")
    folder = skill.path.parent
    new_path = folder / f"{chosen}.md"
    if new_path.exists() and new_path.resolve() != skill.path.resolve():
        raise SkillError(f"A skill named {chosen} already exists.")
    new_path.write_text(
        _skill_file(chosen, description.strip(), arg_tuple, checks, fetch_value, prompt),
        encoding="utf-8",
    )
    if new_path.resolve() != skill.path.resolve() and skill.path.exists():
        skill.path.unlink()
    return load_skill(new_path)


def _work_prompt(tasks: list[object]) -> tuple[str, str, bool]:
    """Return the user request, a reusable prompt, and whether Solidity was saved."""

    real = [item for item in tasks if _task_text(item) and add_skill_request(_task_text(item)) is None]
    if not real:
        raise SkillError("There is no recent task to save as a skill.")
    newest = real[0]
    source = newest
    if not _browser_steps(_actions_of(newest)) and _gave_up(newest):
        for earlier in real[1:]:
            if _browser_steps(_actions_of(earlier)) and _same_work(_task_text(newest), _task_text(earlier)):
                source = earlier
                break
    instruction, fetched = _clean_task(_task_text(newest))
    steps, blocked = _steps_from(_actions_of(source))
    if not steps:
        return instruction, instruction, fetched
    lines = [
        instruction,
        "",
        "Follow these steps from the last run. Click the named control. Open the named URL.",
        "Do not reuse old element indexes or screen coordinates.",
    ]
    for number, step in enumerate(steps, start=1):
        lines.append(f"{number}. {step}")
    if blocked:
        lines.append(f"{len(steps) + 1}. {_stop_line(blocked)}")
    return instruction, "\n".join(lines).strip(), fetched


def _task_text(item: object) -> str:
    if isinstance(item, tuple):
        return str(item[1] if len(item) > 1 else "").strip()
    return str(getattr(item, "task", "") or "").strip()


def _actions_of(item: object) -> list[object]:
    if isinstance(item, tuple):
        return []
    return list(getattr(item, "actions", ()) or ())


def _result_of(item: object) -> str:
    if isinstance(item, tuple):
        return ""
    return str(getattr(item, "result", "") or "")


def _gave_up(item: object) -> bool:
    result = _result_of(item).lower()
    return any(phrase in result for phrase in ("unable", "cannot", "can't", "blocked", "not sure"))


def _same_work(left: str, right: str) -> bool:
    """True when two requests are about the same job."""

    words = re.compile(r"[a-z0-9]{4,}")
    return len(set(words.findall(left.lower())) & set(words.findall(right.lower()))) >= 2


def _browser_steps(actions: list[object]) -> list[str]:
    steps, _blocked = _steps_from(actions)
    return steps


def _steps_from(actions: list[object]) -> tuple[list[str], str]:
    """Keep URLs and control names. Drop indexes, coordinates, and repeated goals."""

    lines: list[str] = []
    blocked = ""
    for action in actions:
        name = str(getattr(action, "action", "") or "")
        info = _action_info(str(getattr(action, "details", "") or ""))
        goal = " ".join(str(info.get("goal") or "").split())
        params = info.get("params") if isinstance(info.get("params"), dict) else {}
        if name == "blocked":
            blocked = str(info.get("reason") or goal)
            continue
        if name in {"start", "done"}:
            continue
        if isinstance(params, dict) and "coordinate_x" in params:
            continue
        if re.search(r"captcha|password|private key", goal, re.IGNORECASE):
            continue
        line = ""
        if name == "navigate" and isinstance(params, dict):
            url = str(params.get("url") or "")
            if url.startswith("http://") or url.startswith("https://"):
                line = f"Open {url}."
        elif name == "scroll":
            line = _sentence(goal or "Scroll the page until the next section is visible")
        elif name in {"click", "input", "send_keys", "wait"} and goal:
            line = _sentence(goal)
        elif goal:
            line = _sentence(goal)
        if not line or (lines and lines[-1].lower() == line.lower()):
            continue
        lines.append(line)
        if len(lines) >= 12:
            break
    if blocked and lines and any(word in blocked.lower() for word in ("financial", "payment", "purchase")):
        trimmed = re.split(
            r",?\s+then\s+(?:select|buy|purchase|order|checkout)\b",
            lines[-1],
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        lines[-1] = _sentence(trimmed)
    return lines, blocked


def _action_info(details: str) -> dict[str, object]:
    try:
        data = json.loads(details)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    target = data.get("target")
    params: object = {}
    if isinstance(target, str):
        try:
            parsed = json.loads(target)
        except json.JSONDecodeError:
            parsed = {}
        if isinstance(parsed, dict):
            params = parsed
    elif isinstance(target, dict):
        params = target
    return {"goal": data.get("goal") or "", "reason": data.get("reason") or "", "params": params}


def _sentence(text: str) -> str:
    cleaned = text.strip().rstrip(".")
    return f"{cleaned}." if cleaned else ""


def _stop_line(reason: str) -> str:
    lower = reason.lower()
    if any(word in lower for word in ("financial", "payment", "purchase", "buy")):
        return "Stop before payment or checkout. Do not buy or submit an order."
    if "password" in lower:
        return "Stop before typing or changing a password."
    if "delet" in lower:
        return "Stop before deleting anything."
    if "install" in lower:
        return "Stop before installing software."
    if any(word in lower for word in ("message", "email")):
        return "Stop before sending a message."
    return "Stop before the blocked action. Do not continue past it."


def _clean_task(text: str) -> tuple[str, bool]:
    fetched = _SAVED_NOTE.search(text) is not None
    body = _SAVED_NOTE.sub("", text)
    body = _PICTURE_NOTE.sub("", body).strip()
    if not body:
        raise SkillError("There is no recent task to save as a skill.")
    return body, fetched


def _parameterize(
    body: str, fetched: bool
) -> tuple[str, tuple[str, ...], dict[str, str], str]:
    """Turn one Ethereum address into {address} so the skill can run again."""

    unique = {item.lower() for item in _ANY_ADDRESS.findall(body)}
    if len(unique) == 1:
        body = _ANY_ADDRESS.sub("{address}", body)
        args = ("address",)
        checks = {"address": "ethereum"}
    elif "{address}" in body:
        args = ("address",)
        checks = {"address": "ethereum"}
    else:
        args = ()
        checks = {}
    fetch = "solidity" if fetched and args else ""
    return body.strip(), args, checks, fetch


def _name_from_task(body: str) -> str:
    plain = _ANY_ADDRESS.sub(" ", body)
    words = [
        word
        for word in re.findall(r"[a-z0-9]+", plain.lower())
        if word not in _NAME_SKIP and not word.isdigit()
    ]
    slug = "-".join(words[:4])[:48].strip("-")
    return slug or "task"


def _description(body: str) -> str:
    line = " ".join(body.split())
    if len(line) > 120:
        return line[:117].rstrip() + "..."
    return line or "Saved from the recent task."


def _skill_file(
    name: str,
    description: str,
    args: tuple[str, ...],
    checks: dict[str, str],
    fetch: str,
    body: str,
) -> str:
    lines = [f"name: {name}", f"description: {description}"]
    if args:
        lines.append("args: " + ", ".join(args))
    if checks:
        lines.append("validate: " + ", ".join(f"{key}={value}" for key, value in checks.items()))
    if fetch:
        lines.append(f"fetch: {fetch}")
    return "\n".join(lines) + "\n\n" + body.strip() + "\n"


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
