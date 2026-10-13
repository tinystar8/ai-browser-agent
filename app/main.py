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

from app.agent import Agent
from app.config import PROJECT_ROOT, Settings, get_settings
from app.memory import MemoryStore, utc_now
from app.models import (
    ActionView,
    HealthResponse,
    ShellRequest,
    ShellResponse,
    SkillView,
    TaskCreateRequest,
    TaskCreateResponse,
    TaskStatusResponse,
    TaskSummary,
)
from app.safety import assess_task
from app.skills import SkillError, list_skills, render_skill
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


app = FastAPI(
    title="AI Browser Agent",
    summary="Local Qwen browser agent using Ollama, Browser Use, and Playwright.",
    lifespan=lifespan,
)


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
