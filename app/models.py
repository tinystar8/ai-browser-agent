"""Pydantic models for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TaskCreateRequest(BaseModel):
    """Body of POST /task.

    Send a plain prompt in task, or a skill name plus the arguments that skill declares.
    """

    task: str | None = Field(default=None, max_length=8000)
    skill: str | None = None
    args: dict[str, str] = Field(default_factory=dict)


class SkillView(BaseModel):
    """One skill the caller can run."""

    name: str
    description: str
    args: list[str] = Field(default_factory=list)


class TaskCreateResponse(BaseModel):
    """Acknowledgement that a task was stored and either started or refused."""

    task_id: int
    status: str
    reason: str | None = None


class ActionView(BaseModel):
    """One recorded browser or safety action."""

    id: int
    action: str
    details: str | None
    timestamp: str


class TaskStatusResponse(BaseModel):
    """Body of GET /task/{id}."""

    status: str
    result: str | None = None
    actions: list[ActionView] = Field(default_factory=list)


class TaskSummary(BaseModel):
    """One row in the task history list."""

    id: int
    task: str
    status: str
    created_at: str
    finished_at: str | None = None
    result: str | None = None


class HealthResponse(BaseModel):
    """Body of GET /health."""

    status: str
    ollama: str
    model: str


class ShellRequest(BaseModel):
    """Body of POST /shell."""

    command: str = Field(min_length=1, max_length=500)
    confirm: bool = False


class ChatRequest(BaseModel):
    """Body of POST /chat."""

    message: str = Field(default="", max_length=16000)
    mode: Literal["auto", "manual"] | None = None
    images: list[str] = Field(default_factory=list, max_length=4)


class ChatMessage(BaseModel):
    """One line in the chat transcript."""

    id: int
    role: str
    text: str
    task_id: int | None = None
    created_at: str
    images: list[str] = Field(default_factory=list)


class ChatState(BaseModel):
    """What the chat page polls, and the reply to a sent message."""

    status: str
    mode: Literal["auto", "manual"] = "auto"
    task_id: int | None = None
    reply: str | None = None
    saved: list[str] = Field(default_factory=list)
    messages: list[ChatMessage] = Field(default_factory=list)


class ShellResponse(BaseModel):
    """Result of a shell tool call."""

    status: str
    output: str
    reason: str | None = None
