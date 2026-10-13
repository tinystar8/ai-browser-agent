"""Give the agent a prompt from the terminal.

Examples:
  ./prompt Open Google and search for Ethereum price
  ./prompt etherscan-contract 0x006FaD88c35D973A87E451CF8D000c7e83Dad409
  ./prompt --list
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.error
import urllib.request

from app.agent import Agent
from app.config import get_settings
from app.memory import MemoryStore
from app.safety import assess_task
from app.skills import Skill, SkillError, list_skills, render_skill

_API = "http://127.0.0.1:8000"


def main() -> None:
    """Parse the command line and run one prompt."""

    parser = argparse.ArgumentParser(
        prog="prompt",
        description="Send a plain-language prompt, or run a skill from skills/.",
    )
    parser.add_argument("words", nargs="*", help="Prompt text, or a skill name plus its arguments.")
    parser.add_argument("--list", action="store_true", help="Show skills you can run.")
    args = parser.parse_args()

    try:
        if args.list:
            _print_skills()
            return
        text = _prompt_from_words(args.words)
        _print_saved_files(text)
        if _api_is_up():
            _run_via_api(text)
        else:
            asyncio.run(_run_locally(text))
    except SkillError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(2) from exc


def _print_skills() -> None:
    skills = list_skills()
    if not skills:
        print("No skills yet. Add a markdown file in skills/.")
        return
    for skill in skills:
        arguments = " ".join(skill.args) if skill.args else "(no arguments)"
        print(f"{skill.name}: {skill.description} [{arguments}]")


def _prompt_from_words(words: list[str]) -> str:
    if not words:
        typed = input("Prompt: ").strip()
        if not typed:
            raise SkillError("Type a prompt, or run ./prompt --list")
        return _prompt_from_words(typed.split())

    skill = _match_skill(words[0])
    if skill is None:
        return " ".join(words)

    if not skill.args:
        body = render_skill(skill.name, {})
        extra = " ".join(words[1:]).strip()
        return f"{body}\n\n{extra}" if extra else body

    values: dict[str, str] = {}
    if len(skill.args) == 1 and len(words) >= 2:
        values[skill.args[0]] = " ".join(words[1:])
    else:
        if len(words) - 1 < len(skill.args):
            expected = " ".join(skill.args)
            raise SkillError(f"Usage: ./prompt {skill.name} {expected}")
        for name, value in zip(skill.args, words[1:], strict=False):
            values[name] = value
    return render_skill(skill.name, values)


def _print_saved_files(text: str) -> None:
    """Show source paths as soon as they are written, before the browser run."""

    paths = [
        line.removeprefix("Saved file: ").strip()
        for line in text.splitlines()
        if line.startswith("Saved file: ")
    ]
    if not paths:
        return
    print("Saved Solidity source:")
    for path in paths:
        print(path)


def _match_skill(name: str) -> Skill | None:
    key = name.strip().lower()
    for skill in list_skills():
        if skill.name.lower() == key or skill.path.stem.lower() == key:
            return skill
    return None


def _api_is_up() -> bool:
    try:
        with urllib.request.urlopen(f"{_API}/health", timeout=1) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _run_via_api(text: str) -> None:
    payload = json.dumps({"task": text}).encode()
    request = urllib.request.Request(
        f"{_API}/task",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            created = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SkillError(detail) from exc

    task_id = created["task_id"]
    print(f"task_id {task_id} ({created['status']})")
    if created["status"] != "running":
        if created.get("reason"):
            print(created["reason"])
        return

    while True:
        with urllib.request.urlopen(f"{_API}/task/{task_id}", timeout=30) as response:
            record = json.load(response)
        status = record["status"]
        if status in {"completed", "failed", "blocked"}:
            print(record.get("result") or "")
            if status != "completed":
                raise SystemExit(1)
            return
        time.sleep(3)


async def _run_locally(text: str) -> None:
    """Run one prompt when the API server is not already up."""

    settings = get_settings()
    memory = MemoryStore(settings.database_file, settings.action_log_file)
    await memory.init()
    decision = assess_task(text)
    if decision.blocked:
        task_id = await memory.create_task(text, "blocked", decision.reason)
        print(f"task_id {task_id} (blocked)")
        print(decision.reason)
        raise SystemExit(1)
    if await memory.has_active_task():
        raise SkillError("A browser task is already running.")

    task_id = await memory.create_task(text, "running")
    print(f"task_id {task_id} (running)")
    await Agent(memory, settings).execute(task_id)
    record = await memory.get_task(task_id)
    print(record.result if record and record.result else "")
    if record is None or record.status != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
