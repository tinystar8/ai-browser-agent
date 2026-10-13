"""Browser Use agent driven by the local Qwen model.

Each step is logged, screenshotted, and checked by the safety gate before
Browser Use executes the model's actions. Calling stop() inside the step
callback aborts the step before the click, type, or navigation runs.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

from app.browser import build_browser
from app.config import PROJECT_ROOT, Settings, get_settings
from app.llm import build_agent_llm
from app.memory import MemoryStore, StoredTask
from app.safety import assess_action
from app.tools.filesystem import FilesystemError, inspect_path
from app.tools.shell import execute as run_shell
from app.tools.solidity import SolidityError, save_verified_source

logger = logging.getLogger("ai_browser_agent")

_SYSTEM_EXTRA = """
You are a local browser assistant for the owner of this computer.
Use the persistent Chromium profile. Do not ask for, type, or store passwords.
Do not delete files, send messages or email, make purchases or payments,
change passwords, or install software. If a task needs one of those, stop and
say that it is blocked.
Pass action arguments as objects.
Navigate: {"navigate": {"url": "https://example.com"}}.
Type into an element with the field name text, never value:
{"input": {"index": 2, "text": "Ethereum price"}}.
Click: {"click": {"index": 5}}.
When you finish, the done action must include a short factual result.
""".strip()


class Agent:
    """Runs one browser task at a time and writes the outcome to memory."""

    def __init__(self, memory: MemoryStore, settings: Settings | None = None) -> None:
        self.memory = memory
        self.settings = settings or get_settings()
        self._block_reason: str | None = None

    async def execute(self, task_id: int) -> None:
        """Run the stored task until it completes, fails, or is blocked."""

        record = await self.memory.get_task(task_id)
        if record is None:
            logger.error("Task %s disappeared before it could run.", task_id)
            return

        self._block_reason = None
        previous = await self.memory.recent_context(exclude_id=task_id)
        prompt = _with_memory(record.task, previous)
        await self.memory.add_action(task_id, "start", record.task, {"step": 0})

        browser: Any = None
        try:
            browser = build_browser(self.settings)
            llm = build_agent_llm(self.settings)
            tools = _build_tools(self.settings)
            holder: dict[str, Any] = {}

            async def on_step(browser_state: Any, model_output: Any, step: int) -> None:
                await self._on_step(task_id, browser_state, model_output, step, holder)

            browser_agent = _construct_browser_agent(
                task=prompt,
                llm=llm,
                browser=browser,
                tools=tools,
                settings=self.settings,
                on_step=on_step,
            )
            holder["agent"] = browser_agent
            history = await browser_agent.run(max_steps=self.settings.max_steps)
            if self._block_reason:
                await self.memory.finish_task(task_id, "blocked", self._block_reason)
                return
            result = _history_result(history)
            await self.memory.add_action(task_id, "done", "task finished", {"result": result[:500]})
            await self.memory.finish_task(task_id, "completed", result)
        except Exception as exc:
            if self._block_reason:
                await self.memory.finish_task(task_id, "blocked", self._block_reason)
                return
            logger.exception("Task %s failed", task_id)
            await self.memory.finish_task(task_id, "failed", f"{type(exc).__name__}: {exc}")
        finally:
            await _close_browser(browser)

    async def _on_step(
        self,
        task_id: int,
        browser_state: Any,
        model_output: Any,
        step: int,
        holder: dict[str, Any],
    ) -> None:
        """Log the step, save a screenshot, and stop before a dangerous action."""

        url = str(getattr(browser_state, "url", "") or "")
        await _save_screenshot(self.settings.screenshots_path, task_id, step, browser_state)

        goal = ""
        state = getattr(model_output, "current_state", None)
        if state is not None:
            goal = str(getattr(state, "next_goal", "") or "")

        for name, detail in _iter_actions(model_output):
            decision = assess_action(name, detail, goal)
            target = detail or url or name
            if decision.blocked:
                self._block_reason = decision.reason
                await self.memory.add_action(
                    task_id,
                    "blocked",
                    target,
                    {"step": step, "planned_action": name, "url": url, "reason": decision.reason},
                )
                active = holder.get("agent")
                if active is not None and hasattr(active, "stop"):
                    active.stop()
                return

            await self.memory.add_action(
                task_id,
                name,
                target,
                {"step": step, "url": url, "goal": goal},
            )


def _with_memory(task: str, previous: list[StoredTask]) -> str:
    """Append a short recall of earlier tasks. The stored task text stays unchanged."""

    if not previous:
        return task
    lines: list[str] = []
    for item in previous:
        snippet = " ".join((item.result or "").split())[:240]
        lines.append(f"- {item.task} [{item.status}] {snippet}".rstrip())
    recall = "\n".join(lines)
    return (
        f"{task}\n\n"
        "Earlier tasks for this user, oldest first. Use them only as context.\n"
        f"{recall}"
    )


def _iter_actions(model_output: Any) -> list[tuple[str, str]]:
    """Return (action name, parameter text) pairs from a model step."""

    actions = getattr(model_output, "action", None) or []
    found: list[tuple[str, str]] = []
    for action in actions:
        dumped: dict[str, Any]
        if hasattr(action, "model_dump"):
            dumped = action.model_dump(exclude_none=True)
        elif isinstance(action, dict):
            dumped = {key: value for key, value in action.items() if value is not None}
        else:
            found.append(("action", str(action)))
            continue
        if not dumped:
            continue
        name = next(iter(dumped))
        value = dumped[name]
        detail = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        found.append((str(name), detail))
    return found


def _history_result(history: Any) -> str:
    if history is None:
        return "Task finished without a result."
    final = history.final_result() if hasattr(history, "final_result") else None
    if final:
        return str(final)
    if hasattr(history, "extracted_content"):
        chunks = [str(item) for item in history.extracted_content() if item]
        if chunks:
            return "\n".join(chunks)
    return "Task finished without a text result."


async def _save_screenshot(directory: Path, task_id: int, step: int, browser_state: Any) -> None:
    raw = getattr(browser_state, "screenshot", None)
    if not raw:
        return
    try:
        encoded = str(raw)
        if encoded.startswith("data:"):
            encoded = encoded.split(",", 1)[1]
        payload = base64.b64decode(encoded)
        folder = directory / f"task-{task_id}"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"step-{int(step):03d}.png"
        target.write_bytes(payload)
    except Exception:
        logger.warning("Could not save screenshot for task %s step %s", task_id, step, exc_info=True)


async def _close_browser(browser: Any) -> None:
    if browser is None:
        return
    for method_name in ("kill", "close", "stop"):
        method = getattr(browser, method_name, None)
        if method is None:
            continue
        try:
            result = method()
            if hasattr(result, "__await__"):
                await result
            return
        except Exception:
            logger.debug("Browser %s failed", method_name, exc_info=True)


def _construct_browser_agent(
    task: str,
    llm: Any,
    browser: Any,
    tools: Any,
    settings: Settings,
    on_step: Any,
) -> Any:
    """Build a Browser Use agent, dropping kwargs unknown to older releases."""

    from browser_use import Agent as BrowserUseAgent

    if tools is None:
        logger.info("Starting without extra local tools.")

    kwargs: dict[str, Any] = {
        "task": task,
        "llm": llm,
        "browser": browser,
        "tools": tools,
        "use_vision": settings.vision_mode(),
        "use_thinking": settings.use_thinking,
        "extend_system_message": _SYSTEM_EXTRA,
        "register_new_step_callback": on_step,
        "max_failures": 3,
        "max_actions_per_step": 3,
        "llm_timeout": settings.llm_timeout,
        "step_timeout": settings.step_timeout,
        "use_judge": False,
        "enable_planning": False,
        "enable_signal_handler": False,
        "calculate_cost": False,
        "include_tool_call_examples": True,
        "max_history_items": 8,
        "file_system_path": str(settings.logs_path / "agent-files"),
    }
    if tools is None:
        kwargs.pop("tools")
    try:
        return BrowserUseAgent(**kwargs)
    except TypeError:
        logger.warning("This Browser Use version rejected some agent options; retrying with the core set.")
        core = {
            key: kwargs[key]
            for key in (
                "task",
                "llm",
                "browser",
                "tools",
                "use_vision",
                "extend_system_message",
                "register_new_step_callback",
                "max_failures",
            )
            if key in kwargs
        }
        return BrowserUseAgent(**core)


def _build_tools(settings: Settings) -> Any | None:
    """Register the read-only file tool and the allowlisted shell tool."""

    try:
        from browser_use import ActionResult, Tools
    except ImportError:
        try:
            from browser_use.agent.views import ActionResult
            from browser_use.tools.service import Tools
        except ImportError:
            logger.warning("Browser Use tools API is unavailable; running without local tools.")
            return None

    tools = Tools()
    root = PROJECT_ROOT

    @tools.action(
        "Read a UTF-8 text file, open a SQLite database such as database/agent.db, "
        "or list a directory inside the agent project. "
        "Cannot write or delete. path is relative to the project root."
    )
    async def read_workspace_file(path: str) -> Any:
        try:
            text = inspect_path(path, root)
        except FilesystemError as exc:
            text = f"Refused: {exc}"
        return ActionResult(extracted_content=text[:8000])

    @tools.action(
        "Download verified Ethereum mainnet Solidity for a 0x address and save each "
        "source file under contracts/<address>/ using its original file name. "
        "address is the contract address."
    )
    async def save_solidity_source(address: str) -> Any:
        try:
            saved = save_verified_source(address.strip(), root)
            text = saved.note()
        except SolidityError as exc:
            text = f"Refused: {exc}"
        return ActionResult(extracted_content=text[:8000])

    if settings.shell_enabled:

        @tools.action(
            "Run one local command. Allowed without confirmation: ls, pwd, git status, docker ps. "
            "rm, sudo, shutdown, and reboot are never run by the browser agent."
        )
        async def local_shell(command: str) -> Any:
            # confirm stays False so the model cannot approve rm, sudo, shutdown, or reboot.
            outcome = run_shell(command, confirm=False, cwd=root)
            message = outcome.output or outcome.reason or outcome.status
            return ActionResult(extracted_content=message[:8000])

    return tools
