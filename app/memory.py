"""SQLite memory for tasks, actions, and the JSON action log.

The agent reads recent finished tasks and adds them to the next prompt so it
can use earlier results as context.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("ai_browser_agent")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    finished_at TEXT,
    result TEXT
);

CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL,
    action TEXT NOT NULL,
    details TEXT,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE INDEX IF NOT EXISTS idx_actions_task_id ON actions(task_id);

CREATE TABLE IF NOT EXISTS instructions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    instruction TEXT NOT NULL,
    situation TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    task_id INTEGER,
    created_at TEXT NOT NULL
);
"""

_RESULT_LIMIT = 50_000
_DETAILS_LIMIT = 20_000
_INSTRUCTION_LIMIT = 4_000
_CHAT_LIMIT = 4_000
_STOP_WORDS = frozenset(
    """
    a an the to of for and or in on at is it this that with from your my do be
    when then you not into about just like also only same case
    open search click read show look page find goto using
    """.split()
)


def utc_now() -> str:
    """Return a UTC timestamp without microseconds."""

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


@dataclass(frozen=True)
class StoredAction:
    """One row from the actions table."""

    id: int
    task_id: int
    action: str
    details: str | None
    timestamp: str


@dataclass(frozen=True)
class StoredInstruction:
    """One user instruction kept for a later similar situation."""

    id: int
    instruction: str
    situation: str
    created_at: str


@dataclass(frozen=True)
class ChatLine:
    """One line in the chat transcript."""

    id: int
    role: str
    text: str
    task_id: int | None
    created_at: str


@dataclass(frozen=True)
class StoredTask:
    """One task plus the actions recorded for it."""

    id: int
    task: str
    status: str
    created_at: str
    finished_at: str | None
    result: str | None
    actions: tuple[StoredAction, ...] = ()


class MemoryStore:
    """Async-safe wrapper around a single SQLite connection."""

    def __init__(self, database: Path, action_log: Path) -> None:
        self.database = database
        self.action_log = action_log
        self._lock = asyncio.Lock()
        self._conn: sqlite3.Connection | None = None

    async def init(self) -> None:
        """Create the database file and tables."""

        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.action_log.parent.mkdir(parents=True, exist_ok=True)

        def _open() -> sqlite3.Connection:
            conn = sqlite3.connect(self.database, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.executescript(_SCHEMA)
            conn.commit()
            return conn

        async with self._lock:
            self._conn = await asyncio.to_thread(_open)
        await self.fail_interrupted_tasks()

    async def fail_interrupted_tasks(self) -> None:
        """A restarted process cannot still be running an old browser task."""

        now = utc_now()

        def _update() -> None:
            self._connection().execute(
                """
                UPDATE tasks
                SET status = 'failed',
                    finished_at = ?,
                    result = 'Interrupted because the agent process stopped.'
                WHERE status = 'running'
                """,
                (now,),
            )
            self._connection().commit()

        async with self._lock:
            await asyncio.to_thread(_update)

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Memory store is not initialized.")
        return self._conn

    async def create_task(self, task: str, status: str, result: str | None = None) -> int:
        """Insert a task and return its id."""

        now = utc_now()
        finished = now if status in {"blocked", "completed", "failed"} else None
        stored_result = _clip(result, _RESULT_LIMIT)

        def _insert() -> int:
            cursor = self._connection().execute(
                """
                INSERT INTO tasks (task, status, created_at, finished_at, result)
                VALUES (?, ?, ?, ?, ?)
                """,
                (task, status, now, finished, stored_result),
            )
            self._connection().commit()
            return int(cursor.lastrowid)

        async with self._lock:
            return await asyncio.to_thread(_insert)

    async def finish_task(self, task_id: int, status: str, result: str) -> None:
        """Mark a task finished and store its result."""

        now = utc_now()
        stored_result = _clip(result, _RESULT_LIMIT)

        def _update() -> None:
            self._connection().execute(
                """
                UPDATE tasks
                SET status = ?, finished_at = ?, result = ?
                WHERE id = ?
                """,
                (status, now, stored_result, task_id),
            )
            self._connection().commit()

        async with self._lock:
            await asyncio.to_thread(_update)

    async def add_action(
        self,
        task_id: int,
        action: str,
        target: str,
        extra: dict[str, object] | None = None,
    ) -> None:
        """Store an action in SQLite and append the JSON log line."""

        now = utc_now()
        payload: dict[str, object] = {"target": target}
        if extra:
            payload.update(extra)
        details = _clip(json.dumps(payload, ensure_ascii=False), _DETAILS_LIMIT)
        record = {"time": now, "action": action, "target": target, "task_id": task_id}
        if extra:
            record.update(extra)
        line = json.dumps(record, ensure_ascii=False)

        def _write() -> None:
            self._connection().execute(
                """
                INSERT INTO actions (task_id, action, details, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (task_id, action, details, now),
            )
            self._connection().commit()
            with self.action_log.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

        async with self._lock:
            await asyncio.to_thread(_write)
        logger.info(line)

    async def has_active_task(self) -> bool:
        """Return True when a browser task is still running."""

        def _query() -> bool:
            row = self._connection().execute(
                "SELECT 1 FROM tasks WHERE status = 'running' LIMIT 1"
            ).fetchone()
            return row is not None

        async with self._lock:
            return await asyncio.to_thread(_query)

    async def get_task(self, task_id: int) -> StoredTask | None:
        """Return a task and its actions, or None."""

        def _query() -> StoredTask | None:
            row = self._connection().execute(
                "SELECT * FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
            if row is None:
                return None
            action_rows = self._connection().execute(
                """
                SELECT * FROM actions
                WHERE task_id = ?
                ORDER BY id ASC
                """,
                (task_id,),
            ).fetchall()
            actions = tuple(_action_from_row(item) for item in action_rows)
            return _task_from_row(row, actions)

        async with self._lock:
            return await asyncio.to_thread(_query)

    async def list_tasks(self, limit: int = 20) -> list[StoredTask]:
        """Return the most recent tasks without their action rows."""

        def _query() -> list[StoredTask]:
            rows = self._connection().execute(
                """
                SELECT * FROM tasks
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [_task_from_row(row, ()) for row in rows]

        async with self._lock:
            return await asyncio.to_thread(_query)

    async def recent_context(self, exclude_id: int, limit: int = 5) -> list[StoredTask]:
        """Return earlier finished tasks, oldest first, for the next prompt."""

        def _query() -> list[StoredTask]:
            rows = self._connection().execute(
                """
                SELECT * FROM tasks
                WHERE id != ?
                  AND status IN ('completed', 'failed', 'blocked')
                ORDER BY id DESC
                LIMIT ?
                """,
                (exclude_id, limit),
            ).fetchall()
            return [_task_from_row(row, ()) for row in reversed(rows)]

        async with self._lock:
            return await asyncio.to_thread(_query)

    async def remember_instruction(self, instruction: str, situation: str) -> None:
        """Store an instruction together with the situation it was given in."""

        now = utc_now()
        text = _clip(instruction.strip(), _INSTRUCTION_LIMIT) or ""
        case = _clip(situation.strip(), _INSTRUCTION_LIMIT) or text
        if not text:
            return

        def _insert() -> None:
            self._connection().execute(
                """
                INSERT INTO instructions (instruction, situation, created_at)
                VALUES (?, ?, ?)
                """,
                (text, case, now),
            )
            self._connection().commit()

        async with self._lock:
            await asyncio.to_thread(_insert)

    async def relevant_instructions(self, situation: str, limit: int = 6) -> list[StoredInstruction]:
        """Return earlier instructions whose situation shares words with this one."""

        def _query() -> list[StoredInstruction]:
            rows = self._connection().execute(
                """
                SELECT * FROM instructions
                ORDER BY id DESC
                LIMIT 200
                """
            ).fetchall()
            current = _tokens(situation)
            if not current:
                return []
            ranked: list[tuple[int, int, StoredInstruction]] = []
            for row in rows:
                item = _instruction_from_row(row)
                if item.instruction.strip().lower() == situation.strip().lower():
                    continue
                overlap = len(current & _tokens(f"{item.situation}\n{item.instruction}"))
                if overlap >= 2:
                    ranked.append((overlap, item.id, item))
            ranked.sort(key=lambda pair: (pair[0], pair[1]), reverse=True)
            chosen = [item for _, _, item in ranked[:limit]]
            return list(reversed(chosen))

        async with self._lock:
            return await asyncio.to_thread(_query)

    async def add_chat(self, role: str, text: str, task_id: int | None = None) -> None:
        """Append one chat line."""

        now = utc_now()
        stored = _clip(text.strip(), _CHAT_LIMIT) or ""
        if not stored:
            return

        def _insert() -> None:
            self._connection().execute(
                """
                INSERT INTO chat (role, text, task_id, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (role, stored, task_id, now),
            )
            self._connection().commit()

        async with self._lock:
            await asyncio.to_thread(_insert)

    async def list_chat(self, limit: int = 80) -> list[ChatLine]:
        """Return the latest chat lines, oldest first."""

        def _query() -> list[ChatLine]:
            rows = self._connection().execute(
                """
                SELECT * FROM (
                    SELECT * FROM chat
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
                """,
                (limit,),
            ).fetchall()
            return [_chat_from_row(row) for row in rows]

        async with self._lock:
            return await asyncio.to_thread(_query)


def _tokens(text: str) -> set[str]:
    """Words that can identify a situation. Short and common words are dropped."""

    return {
        word
        for word in re.findall(r"[a-z0-9]{4,}", text.lower())
        if word not in _STOP_WORDS
    }


def _clip(value: str | None, limit: int) -> str | None:
    if value is None:
        return None
    if len(value) <= limit:
        return value
    return value[:limit] + "...(truncated)"


def _task_from_row(row: sqlite3.Row, actions: tuple[StoredAction, ...]) -> StoredTask:
    return StoredTask(
        id=int(row["id"]),
        task=str(row["task"]),
        status=str(row["status"]),
        created_at=str(row["created_at"]),
        finished_at=row["finished_at"],
        result=row["result"],
        actions=actions,
    )


def _instruction_from_row(row: sqlite3.Row) -> StoredInstruction:
    return StoredInstruction(
        id=int(row["id"]),
        instruction=str(row["instruction"]),
        situation=str(row["situation"]),
        created_at=str(row["created_at"]),
    )


def _chat_from_row(row: sqlite3.Row) -> ChatLine:
    task_id = row["task_id"]
    return ChatLine(
        id=int(row["id"]),
        role=str(row["role"]),
        text=str(row["text"]),
        task_id=int(task_id) if task_id is not None else None,
        created_at=str(row["created_at"]),
    )


def _action_from_row(row: sqlite3.Row) -> StoredAction:
    return StoredAction(
        id=int(row["id"]),
        task_id=int(row["task_id"]),
        action=str(row["action"]),
        details=row["details"],
        timestamp=str(row["timestamp"]),
    )
