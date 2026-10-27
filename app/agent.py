"""Browser Use agent driven by the local Qwen model.

Each step is logged, screenshotted, and checked by the safety gate before
Browser Use executes the model's actions. Calling stop() inside the step
callback aborts the step before the click, type, or navigation runs.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from pathlib import Path
from typing import Any

from app.browser import build_browser
from app.config import PROJECT_ROOT, Settings, get_settings
from app.llm import build_agent_llm
from app.memory import MemoryStore, StoredInstruction, StoredTask
from app.safety import assess_action
from app.tools.filesystem import FilesystemError, inspect_path
from app.tools.shell import execute as run_shell
from app.tools.solidity import SolidityError, save_verified_source

logger = logging.getLogger("ai_browser_agent")

_CLOSE_REQUEST = re.compile(
    r"^(?:please\s+)?(?:yes(?:,| )?\s*)?close(?:\s+it|\s+the\s+browser)?[.!]?$",
    re.IGNORECASE,
)
_KEEP_OPEN = (
    "Finished. The browser stays open. "
    "Send the next instruction, or say close to close it."
)

_SYSTEM_EXTRA = """
You are a local browser assistant for the owner of this computer.
Use the persistent Chromium profile. Do not ask for, type, or store passwords.
Do not delete files, send messages or email, make purchases or payments,
change passwords, or install software. If a task needs one of those, stop and
say that it is blocked.
For a Playwright-based browser agent, I recommend this workflow:
1. Detect whether another browser page or popup opened.
2. Determine which page is active.
3. Identify whether the interface uses an iframe or modal.
4. Capture a fresh screenshot of the correct browser context.
5. Use Playwright locators rather than screen coordinates whenever possible.
6. Resume automation when verification succeeds.
Pass action arguments as objects.
Navigate: {"navigate": {"url": "https://example.com"}}.
Type into an element with the field name text, never value:
{"input": {"index": 2, "text": "Ethereum price"}}.
Click an element by index: {"click": {"index": 5}}.
When the target sits inside a popup frame or image, click its point: {"click": {"coordinate_x": 420, "coordinate_y": 310}}. The center of the popup frame is not the target.
When you finish, the done action must include a short factual result.
Do not close the browser. The user decides when it closes.
A follow-up user request overrides the plan you were following.
If the Chromium window is already open, do the new instruction on the page
that is showing. Do not open another window, and do not leave that page
unless the instruction asks for a different one.
Standing instructions from a similar past situation also apply.
""".strip()

_MANUAL_EXTRA = """
Manual mode. The user's latest message is the only instruction. It may be code or other text.
Carry out that instruction exactly, then stop.
Do not add steps, do not continue into later work, and do not apply standing instructions.
If the instruction is a command or code to run on this computer, run it with local_shell and report the result.
If it is a browser instruction, do that in the open browser and then stop.
When the instruction is finished, the done action must include a short factual result.
""".strip()


class Agent:
    """Runs one browser task at a time and writes the outcome to memory."""

    def __init__(self, memory: MemoryStore, settings: Settings | None = None) -> None:
        self.memory = memory
        self.settings = settings or get_settings()
        self._block_reason: str | None = None
        self._running = False
        self._active_task_id: int | None = None
        self._active_task_text = ""
        self._last_url = ""
        self._inbox: list[tuple[str, list[Any]]] = []
        self._browser: Any = None
        self._browser_agent: Any = None
        self._close_requested = False
        self.mode = "auto"
        self._turn_manual = False
        self._turn_images: list[str] = []

    @property
    def is_running(self) -> bool:
        """True while a browser task is in progress."""

        return self._running

    @property
    def active_task_id(self) -> int | None:
        return self._active_task_id

    def situation(self) -> str:
        """The current task and page, used to remember what an instruction was about."""

        parts = [self._active_task_text.strip(), self._last_url.strip()]
        return " | ".join(part for part in parts if part)

    @property
    def browser_is_open(self) -> bool:
        """True after a task while Chromium is waiting for the next instruction."""

        return self._browser is not None

    def wants_close(self, text: str) -> bool:
        """True when the user is confirming that the browser should close."""

        return _CLOSE_REQUEST.fullmatch(text.strip()) is not None

    def request_close(self) -> None:
        """Stop the current step, then close Chromium when the task exits."""

        self._close_requested = True
        active = self._browser_agent
        if self._running and active is not None and hasattr(active, "stop"):
            active.stop()

    async def close_browser(self) -> None:
        """Close Chromium. Safe to call when it is already closed."""

        browser = self._browser
        self._browser = None
        self._browser_agent = None
        self._close_requested = False
        await _close_browser(browser)

    def set_mode(self, mode: str) -> None:
        """Switch between autonomous work and one-instruction manual work."""

        if mode in {"auto", "manual"}:
            self.mode = mode

    def arm_turn(self, *, manual: bool, image_urls: list[str]) -> None:
        """Remember how the next task should treat the user's instruction."""

        self._turn_manual = manual
        self._turn_images = list(image_urls)

    def steer(self, text: str, image_urls: list[str] | None = None) -> None:
        """Queue an instruction for the running browser task."""

        cleaned = text.strip()
        pictures = list(image_urls or [])
        if cleaned or pictures:
            self._inbox.append((cleaned, pictures))

    async def execute(self, task_id: int) -> None:
        """Run the stored task until it completes, fails, or is blocked."""

        record = await self.memory.get_task(task_id)
        if record is None:
            self._turn_manual = False
            self._turn_images = []
            logger.error("Task %s disappeared before it could run.", task_id)
            return

        manual = self._turn_manual
        image_urls = self._turn_images
        self._turn_manual = False
        self._turn_images = []
        self._block_reason = None
        self._running = True
        self._active_task_id = task_id
        self._active_task_text = record.task
        if manual:
            prompt = record.task
        else:
            previous = await self.memory.recent_context(exclude_id=task_id)
            guidance = await self.memory.relevant_instructions(record.task)
            prompt = _with_memory(record.task, previous, guidance)
        reused = self._browser is not None
        current_url = self._last_url
        if reused:
            showing = f" The page on screen is {current_url}." if current_url else ""
            prompt = (
                "The Chromium window is already open. Work in that window, "
                "on the page that is showing now."
                f"{showing} "
                "Do not open a new window. Do not navigate away unless this "
                "instruction asks for a different page. Do not close the browser.\n\n"
                + prompt
            )
        else:
            self._last_url = ""
        await self.memory.add_action(task_id, "start", record.task, {"step": 0})
        await self.memory.add_chat("agent", "Working on it.", task_id)

        try:
            if self._browser is None:
                self._browser = build_browser(self.settings)
            else:
                _prepare_kept_browser(self._browser)
            browser = self._browser
            llm = build_agent_llm(self.settings)
            tools = _build_tools(self.settings)
            holder: dict[str, Any] = {}

            async def on_step(browser_state: Any, model_output: Any, step: int) -> None:
                await self._on_step(task_id, browser_state, model_output, step, holder)

            async def on_start(browser_agent: Any) -> None:
                await self._apply_steering(task_id, browser_agent)

            browser_agent = _construct_browser_agent(
                task=prompt,
                llm=llm,
                browser=browser,
                tools=tools,
                settings=self.settings,
                on_step=on_step,
                directly_open_url=not reused,
                system_extra=_system_extra(manual),
                sample_images=_image_parts(image_urls),
            )
            holder["agent"] = browser_agent
            self._browser_agent = browser_agent
            history = await browser_agent.run(
                max_steps=self.settings.max_steps,
                on_step_start=on_start,
            )
            if self._block_reason:
                await self.memory.finish_task(task_id, "blocked", self._block_reason)
                await self.memory.add_chat("agent", self._block_reason, task_id)
                return
            result = _history_result(history)
            await self.memory.add_action(task_id, "done", "task finished", {"result": result[:500]})
            await self.memory.finish_task(task_id, "completed", result)
            await self.memory.add_chat("agent", result, task_id)
        except Exception as exc:
            if self._block_reason:
                await self.memory.finish_task(task_id, "blocked", self._block_reason)
                await self.memory.add_chat("agent", self._block_reason, task_id)
                return
            logger.exception("Task %s failed", task_id)
            message = f"{type(exc).__name__}: {exc}"
            await self.memory.finish_task(task_id, "failed", message)
            await self.memory.add_chat("agent", message, task_id)
        finally:
            self._inbox.clear()
            self._running = False
            self._active_task_id = None
            self._active_task_text = ""
            self._browser_agent = None
            if self._browser is not None:
                if self._close_requested:
                    await self.close_browser()
                    await self.memory.add_chat("agent", "Closed the browser.", task_id)
                else:
                    await self.memory.add_chat("agent", _KEEP_OPEN, task_id)

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
        if url:
            self._last_url = url
        steered = await self._apply_steering(task_id, holder.get("agent"), model_output)
        await _save_screenshot(self.settings.screenshots_path, task_id, step, browser_state)
        if steered:
            return

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
        if goal:
            await self.memory.add_chat("agent", goal, task_id)

    def _drain(self) -> list[tuple[str, list[str]]]:
        if not self._inbox:
            return []
        notes = self._inbox
        self._inbox = []
        return notes

    async def _apply_steering(
        self,
        task_id: int,
        browser_agent: Any,
        model_output: Any = None,
    ) -> bool:
        """Give queued user instructions to the model before the next action."""

        notes = self._drain()
        if not notes:
            return False
        texts = [text for text, _images in notes if text]
        pictures: list[str] = []
        for _text, images in notes:
            pictures.extend(images)
        manual = self.mode == "manual"
        _attach_images(browser_agent, _image_parts(pictures))
        _inject_instruction(browser_agent, texts or ["Use the attached picture."], manual)
        if model_output is not None and getattr(model_output, "action", None):
            # The model chose this action before it saw the new instruction.
            model_output.action = []
        shown = texts or ["the attached picture"]
        text = "Following your instruction: " + " | ".join(shown)
        await self.memory.add_chat("agent", text, task_id)
        return True


def _system_extra(manual: bool) -> str:
    """Safety and workflow text. Manual mode keeps the agent on one instruction."""

    if not manual:
        return _SYSTEM_EXTRA
    base = _SYSTEM_EXTRA.replace(
        "Standing instructions from a similar past situation also apply.",
        "Standing instructions do not apply. Do only the user's latest instruction.",
    )
    return base + "\n" + _MANUAL_EXTRA


def _image_parts(data_urls: list[str]) -> list[Any] | None:
    if not data_urls:
        return None
    from app.chat_images import content_parts

    parts = content_parts(data_urls)
    return parts or None


def _attach_images(browser_agent: Any, parts: list[Any] | None) -> None:
    """Show the user's picture on the next model step."""

    if browser_agent is None or not parts:
        return
    browser_agent.sample_images = parts
    manager = getattr(browser_agent, "_message_manager", None)
    if manager is not None:
        manager.sample_images = parts


def _inject_instruction(browser_agent: Any, notes: list[str], manual: bool = False) -> None:
    """Append a follow-up to the running Browser Use conversation."""

    if browser_agent is None:
        return
    manager = getattr(browser_agent, "_message_manager", None)
    if manager is None or not hasattr(manager, "add_new_task"):
        logger.warning("The running agent cannot accept a follow-up instruction.")
        return
    if manual:
        body = (
            "Manual mode. This is the only instruction. Carry it out and then stop:\n"
            + "\n".join(notes)
        )
    else:
        body = "Follow this instruction now. It overrides the previous plan:\n" + "\n".join(
            f"- {note}" for note in notes
        )
    manager.add_new_task(body)


def _with_memory(
    task: str,
    previous: list[StoredTask],
    guidance: list[StoredInstruction] | None = None,
) -> str:
    """Append a short recall of earlier tasks. The stored task text stays unchanged."""

    parts = [task]
    if guidance:
        lines = [
            f"- Situation: {item.situation}\n  Do this: {item.instruction}"
            for item in guidance
        ]
        parts.append(
            "Earlier instructions from the user for a similar situation. "
            "Make the same kind of decision when this case matches. "
            "Ignore an instruction when the situation is different.\n"
            + "\n".join(lines)
        )
    if previous:
        lines = []
        for item in previous:
            snippet = " ".join((item.result or "").split())[:240]
            lines.append(f"- {item.task} [{item.status}] {snippet}".rstrip())
        parts.append(
            "Earlier tasks for this user, oldest first. Use them only as context.\n"
            + "\n".join(lines)
        )
    return "\n\n".join(parts)


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


def _prepare_kept_browser(browser: Any) -> None:
    """Let the next command use the Chromium window that is already open.

    Browser Use stops the event bus when a task ends, but the window, the page,
    and the click handlers stay. Restart that same bus. A replacement bus has
    no handlers, so the next command cannot see the page.
    """

    bus = getattr(browser, "event_bus", None)
    if bus is None:
        return
    queue = getattr(bus, "event_queue", None)
    running = bool(getattr(bus, "_is_running", False))
    shutdown = bool(getattr(queue, "_is_shutdown", False)) if queue is not None else True
    if running and not shutdown:
        return
    bus.event_queue = None
    bus._on_idle = None
    bus._is_running = False
    bus._runloop_task = None
    if hasattr(browser, "_intentional_stop"):
        browser._intentional_stop = False


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
    directly_open_url: bool = True,
    system_extra: str | None = None,
    sample_images: list[Any] | None = None,
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
        "extend_system_message": system_extra or _SYSTEM_EXTRA,
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
        "directly_open_url": directly_open_url,
        "file_system_path": str(settings.logs_path / "agent-files"),
    }
    if sample_images:
        kwargs["sample_images"] = sample_images
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
                "directly_open_url",
            )
            if key in kwargs
        }
        return BrowserUseAgent(**core)


def _prefer_click_coordinates(tools: Any) -> None:
    """Use a supplied point when a click also names an element.

    The point is where the button is. The element index is often the popup frame.
    """

    original = getattr(tools, "_click_by_index", None)
    coordinate = getattr(tools, "_click_by_coordinate", None)
    if original is None or coordinate is None:
        return

    async def _click_by_index(params: Any, browser_session: Any) -> Any:
        if getattr(params, "coordinate_x", None) is not None and getattr(
            params, "coordinate_y", None
        ) is not None:
            return await coordinate(params, browser_session)
        return await original(params, browser_session)

    tools._click_by_index = _click_by_index


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
    if hasattr(tools, "set_coordinate_clicking"):
        tools.set_coordinate_clicking(True)
    _prefer_click_coordinates(tools)
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
