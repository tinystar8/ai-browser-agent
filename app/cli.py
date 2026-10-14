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
from app.skills import SkillError, expand_prompt, list_skills

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
        if _api_is_up():
            raw = " ".join(args.words).strip()
            if not raw:
                raw = input("Prompt: ").strip()
            if not raw:
                raise SkillError("Type a prompt, or run ./prompt --list")
            _run_via_api(raw)
        else:
            text = _prompt_from_words(args.words)
            _print_saved_files(text)
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

    return expand_prompt(" ".join(words))


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


def _api_is_up() -> bool:
    try:
        with urllib.request.urlopen(f"{_API}/health", timeout=1) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _run_via_api(text: str) -> None:
    payload = json.dumps({"message": text}).encode()
    request = urllib.request.Request(
        f"{_API}/chat",
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

    status = created["status"]
    saved = created.get("saved") or []
    if saved:
        print("Saved Solidity source:")
        for path in saved:
            print(path)
    if status == "steering":
        print(created.get("reply") or f"Sent to the running task {created.get('task_id')}.")
        return
    if status in {"idle", "waiting"}:
        reply = created.get("reply")
        if reply:
            print(reply)
        return
    task_id = created.get("task_id")
    print(f"task_id {task_id} ({status})")
    if status != "running" or task_id is None:
        reply = created.get("reply") or created.get("reason")
        if reply:
            print(reply)
        if status != "running":
            raise SystemExit(1)
        return

    while True:
        with urllib.request.urlopen(f"{_API}/task/{task_id}", timeout=30) as response:
            record = json.load(response)
        status = record["status"]
        if status in {"completed", "failed", "blocked"}:
            print(record.get("result") or "")
            print("The browser stays open. Send the next instruction in the chat, or run ./prompt close")
            if status != "completed":
                raise SystemExit(1)
            return
        time.sleep(3)


async def _run_locally(text: str) -> None:
    """Run one prompt when the API server is not already up."""

    settings = get_settings()
    memory = MemoryStore(settings.database_file, settings.action_log_file)
    await memory.init()
    if await memory.has_active_task():
        raise SkillError("A browser task is already running.")

    agent = Agent(memory, settings)
    failed = False
    current = text
    try:
        while True:
            decision = assess_task(current)
            if decision.blocked:
                task_id = await memory.create_task(current, "blocked", decision.reason)
                print(f"task_id {task_id} (blocked)")
                print(decision.reason)
                failed = True
                break
            if await memory.has_active_task():
                raise SkillError("A browser task is already running.")
            task_id = await memory.create_task(current, "running")
            print(f"task_id {task_id} (running)")
            await agent.execute(task_id)
            record = await memory.get_task(task_id)
            print(record.result if record and record.result else "")
            if record is None or record.status != "completed":
                failed = True
            if not agent.browser_is_open:
                break
            print("The browser stays open. Type the next instruction, or type close.")
            try:
                nxt = input("> ").strip()
            except EOFError:
                break
            if agent.wants_close(nxt):
                await agent.close_browser()
                print("Closed the browser.")
                break
            if not nxt:
                continue
            current = expand_prompt(nxt)
            failed = False
    finally:
        if agent.browser_is_open:
            await agent.close_browser()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
