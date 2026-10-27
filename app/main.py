"""FastAPI server for the local browser agent."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from app.agent import Agent
from app.chat_images import ChatImageError, image_file, save_chat_images
from app.config import PROJECT_ROOT, Settings, get_settings
from app.memory import MemoryStore, utc_now
from app.models import (
    ActionView,
    HealthResponse,
    ShellRequest,
    ShellResponse,
    ChatMessage,
    ChatRequest,
    ChatState,
    SkillView,
    TaskCreateRequest,
    TaskCreateResponse,
    TaskStatusResponse,
    TaskSummary,
)
from app.safety import assess_task
from app.skills import SkillError, expand_prompt, list_skills, render_skill
from app.tools.shell import execute as run_shell

logger = logging.getLogger("ai_browser_agent")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Prepare storage, logging, and the single shared agent."""

    settings = get_settings()
    _configure_logging(settings)
    settings.screenshots_path.mkdir(parents=True, exist_ok=True)
    settings.profile_path.mkdir(parents=True, exist_ok=True)

    memory = MemoryStore(settings.database_file, settings.action_log_file)
    await memory.init()
    app.state.settings = settings
    app.state.memory = memory
    app.state.agent = Agent(memory, settings)
    app.state.jobs: set[asyncio.Task[None]] = set()
    app.state.submit_lock = asyncio.Lock()
    logger.info("Agent API ready on %s:%s", settings.host, settings.port)
    yield
    await app.state.agent.close_browser()


app = FastAPI(
    title="AI Browser Agent",
    summary="Local Qwen browser agent using Ollama, Browser Use, and Playwright.",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
async def chat_page() -> str:
    """Serve the chat page used to start and steer the agent."""

    page = PROJECT_ROOT / "app" / "static" / "index.html"
    return page.read_text(encoding="utf-8")


@app.get("/chat", response_model=ChatState)
async def read_chat(request: Request) -> ChatState:
    """Return the transcript and whether a browser task is running."""

    return await _chat_state(request)


@app.post("/chat", response_model=ChatState)
async def post_chat(body: ChatRequest, request: Request) -> ChatState:
    """Start a task, steer the one that is already running, or switch mode."""

    memory: MemoryStore = request.app.state.memory
    agent: Agent = request.app.state.agent
    settings: Settings = request.app.state.settings
    if body.mode in {"auto", "manual"}:
        agent.set_mode(body.mode)
    message = body.message.strip()
    try:
        stored_urls, data_urls = save_chat_images(
            body.images, settings.logs_path / "chat-images"
        )
    except ChatImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not message and not stored_urls:
        return await _chat_state(request)

    decision = assess_task(message)
    await memory.add_chat("user", message, agent.active_task_id, images=stored_urls)
    if decision.blocked:
        await memory.add_chat("agent", decision.reason, agent.active_task_id)
        state = await _chat_state(request, status="blocked")
        return state.model_copy(update={"reply": decision.reason})

    if message and agent.wants_close(message):
        if agent.is_running:
            agent.request_close()
            await memory.add_chat(
                "agent",
                "Stopping. I will close the browser when this step finishes.",
                agent.active_task_id,
            )
            state = await _chat_state(request, status="steering", task_id=agent.active_task_id)
            return state.model_copy(update={"reply": "Closing after this step."})
        if agent.browser_is_open:
            reply = "Closed the browser."
            await agent.close_browser()
        else:
            reply = "The browser is already closed."
        await memory.add_chat("agent", reply)
        state = await _chat_state(request, status="idle")
        return state.model_copy(update={"reply": reply})

    manual = agent.mode == "manual"
    async with request.app.state.submit_lock:
        if agent.is_running or await memory.has_active_task():
            if not manual:
                situation = agent.situation() or message
                await memory.remember_instruction(message or "Use the attached picture.", situation)
            agent.steer(message or "Use the attached picture.", data_urls)
            task_id = agent.active_task_id
            state = await _chat_state(request, status="steering", task_id=task_id)
            reply = "Sent. The agent will do only this." if manual else "Sent to the running agent."
            return state.model_copy(update={"reply": reply})

        try:
            text = _instruction_text(message, manual)
        except SkillError as exc:
            await memory.add_chat("agent", str(exc))
            state = await _chat_state(request, status="idle")
            return state.model_copy(update={"reply": str(exc)})
        if data_urls:
            text += "\n\nA picture is attached to this instruction. Use that picture."
        task_id = await memory.create_task(text, "running")
        agent.arm_turn(manual=manual, image_urls=data_urls)

    saved = _saved_paths(text)
    if saved:
        await memory.add_chat("agent", "Saved Solidity source:\n" + "\n".join(saved), task_id)
    job = asyncio.create_task(agent.execute(task_id), name=f"task-{task_id}")
    request.app.state.jobs.add(job)
    job.add_done_callback(request.app.state.jobs.discard)
    state = await _chat_state(request, status="running", task_id=task_id)
    return state.model_copy(update={"saved": saved})


@app.get("/chat/images/{name}")
async def chat_image(name: str, request: Request) -> FileResponse:
    """Serve one picture that was attached in the chat."""

    settings: Settings = request.app.state.settings
    path = image_file(settings.logs_path / "chat-images", name)
    if path is None:
        raise HTTPException(status_code=404, detail="Picture not found.")
    return FileResponse(path)


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Report that the API is up and whether Ollama answered."""

    settings: Settings = request.app.state.settings
    return HealthResponse(
        status="ok",
        ollama=await _ollama_status(settings),
        model=settings.openai_model,
    )


@app.post("/task", response_model=TaskCreateResponse, response_model_exclude_none=True)
async def create_task(body: TaskCreateRequest, request: Request) -> TaskCreateResponse:
    """Store a task and start it, or refuse it when the safety gate matches."""

    memory: MemoryStore = request.app.state.memory
    agent: Agent = request.app.state.agent
    try:
        text = _task_text(body)
    except SkillError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    decision = assess_task(text)

    if decision.blocked:
        task_id = await memory.create_task(text, "blocked", decision.reason)
        await memory.add_action(
            task_id,
            "blocked",
            decision.category or "safety",
            {"reason": decision.reason},
        )
        return TaskCreateResponse(task_id=task_id, status="blocked", reason=decision.reason)

    async with request.app.state.submit_lock:
        if await memory.has_active_task():
            raise HTTPException(status_code=409, detail="A browser task is already running.")
        task_id = await memory.create_task(text, "running")

    job = asyncio.create_task(agent.execute(task_id), name=f"task-{task_id}")
    request.app.state.jobs.add(job)
    job.add_done_callback(request.app.state.jobs.discard)
    return TaskCreateResponse(task_id=task_id, status="running")


@app.get("/skills", response_model=list[SkillView])
async def skills() -> list[SkillView]:
    """List prompt skills loaded from the skills directory."""

    return [
        SkillView(name=skill.name, description=skill.description, args=list(skill.args))
        for skill in list_skills()
    ]


@app.get("/task/{task_id}", response_model=TaskStatusResponse)
async def get_task(task_id: int, request: Request) -> TaskStatusResponse:
    """Return the status, result, and recorded actions for one task."""

    memory: MemoryStore = request.app.state.memory
    record = await memory.get_task(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Task not found.")
    return TaskStatusResponse(
        status=record.status,
        result=record.result,
        actions=[
            ActionView(
                id=item.id,
                action=item.action,
                details=item.details,
                timestamp=item.timestamp,
            )
            for item in record.actions
        ],
    )


@app.get("/tasks", response_model=list[TaskSummary])
async def list_tasks(request: Request, limit: int = 20) -> list[TaskSummary]:
    """Return recent tasks so callers can see what the agent remembers."""

    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100.")
    memory: MemoryStore = request.app.state.memory
    rows = await memory.list_tasks(limit)
    return [
        TaskSummary(
            id=row.id,
            task=row.task,
            status=row.status,
            created_at=row.created_at,
            finished_at=row.finished_at,
            result=row.result,
        )
        for row in rows
    ]


@app.post("/shell", response_model=ShellResponse, response_model_exclude_none=True)
async def shell(body: ShellRequest, request: Request) -> ShellResponse:
    """Run an allowlisted shell command. Dangerous commands need confirm=true."""

    settings: Settings = request.app.state.settings
    if not settings.shell_enabled:
        raise HTTPException(status_code=403, detail="The shell tool is disabled.")

    outcome = await asyncio.to_thread(
        run_shell,
        body.command,
        confirm=body.confirm,
        cwd=PROJECT_ROOT,
    )
    await asyncio.to_thread(
        _append_shell_log,
        settings.action_log_file,
        body.command,
        outcome.status,
        outcome.reason,
    )
    return ShellResponse(status=outcome.status, output=outcome.output, reason=outcome.reason)


async def _chat_state(
    request: Request,
    status: str | None = None,
    task_id: int | None = None,
) -> ChatState:
    """Build the transcript payload. An explicit status wins over the database."""

    memory: MemoryStore = request.app.state.memory
    agent: Agent = request.app.state.agent
    lines = await memory.list_chat()
    messages = [
        ChatMessage(
            id=line.id,
            role=line.role,
            text=line.text,
            task_id=line.task_id,
            created_at=line.created_at,
            images=list(line.images),
        )
        for line in lines
    ]
    if status is None:
        if agent.is_running:
            status = "running"
            task_id = agent.active_task_id
        elif agent.browser_is_open:
            status = "waiting"
            recent = await memory.list_tasks(1)
            task_id = recent[0].id if recent else None
        else:
            recent = await memory.list_tasks(1)
            if recent:
                status = recent[0].status
                task_id = recent[0].id
            else:
                status = "idle"
    return ChatState(status=status, mode=agent.mode, task_id=task_id, messages=messages)


def _instruction_text(message: str, manual: bool) -> str:
    """Manual mode keeps the chat text, including code. Auto mode can expand a skill."""

    if manual:
        return message or "Use the attached picture."
    if not message:
        return "Use the attached picture."
    return expand_prompt(message)


def _saved_paths(text: str) -> list[str]:
    return [
        line.removeprefix("Saved file: ").strip()
        for line in text.splitlines()
        if line.startswith("Saved file: ")
    ]


def _task_text(body: TaskCreateRequest) -> str:
    """Build the prompt from a free-text task, a skill, or both."""

    parts: list[str] = []
    if body.skill:
        parts.append(render_skill(body.skill, body.args))
    if body.task and body.task.strip():
        parts.append(body.task.strip())
    if not parts:
        raise SkillError("Give a task or a skill.")
    return "\n\n".join(parts)


def _append_shell_log(path: Path, command: str, status: str, reason: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {
        "time": utc_now(),
        "action": "shell",
        "target": command,
        "status": status,
    }
    if reason:
        record["reason"] = reason
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(json.dumps(record, ensure_ascii=False))


def _configure_logging(settings: Settings) -> None:
    settings.logs_path.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(message)s")
    file_handler = logging.FileHandler(settings.logs_path / "agent.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False


async def _ollama_status(settings: Settings) -> str:
    """Ask Ollama's OpenAI-compatible model list. Any HTTP response counts as up."""

    url = settings.openai_base_url.rstrip("/") + "/models"
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            await client.get(
                url,
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            )
        return "reachable"
    except httpx.HTTPError:
        return "unreachable"
