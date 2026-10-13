"""Read-only access to files inside the project workspace.

There is no delete or write function. Paths outside the workspace are refused.
SQLite database files are opened read-only and returned as text tables.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

_MAX_BYTES = 1_000_000
_MAX_DB_ROWS = 200
_MAX_CELL = 500
_SQLITE_MAGIC = b"SQLite format 3\x00"


class FilesystemError(Exception):
    """Raised when a filesystem request is unsafe or impossible."""


def _resolve(path: str, root: Path) -> Path:
    """Resolve path and require it to stay inside root."""

    root = root.resolve()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise FilesystemError(f"Path is outside the workspace: {path}")
    return resolved


def read_text(path: str, root: Path) -> str:
    """Read a UTF-8 text file, or the tables of a SQLite database, from the workspace."""

    resolved = _resolve(path, root)
    if not resolved.is_file():
        raise FilesystemError(f"File does not exist: {path}")
    size = resolved.stat().st_size
    if size > _MAX_BYTES:
        raise FilesystemError(f"File is too large to read ({size} bytes).")
    if _is_sqlite(resolved):
        return read_sqlite(resolved)
    return resolved.read_text(encoding="utf-8", errors="replace")


def list_directory(path: str, root: Path) -> list[str]:
    """List the names in a workspace directory."""

    resolved = _resolve(path, root)
    if not resolved.is_dir():
        raise FilesystemError(f"Directory does not exist: {path}")
    return sorted(entry.name for entry in resolved.iterdir())


def inspect_path(path: str, root: Path) -> str:
    """Read a text file, open a SQLite database, or list a directory."""

    resolved = _resolve(path, root)
    if resolved.is_dir():
        names = sorted(entry.name for entry in resolved.iterdir())
        return "\n".join(names)
    if resolved.is_file():
        return read_text(str(resolved), root)
    raise FilesystemError(f"Path does not exist: {path}")


def read_sqlite(path: Path) -> str:
    """Return every user table in a SQLite file. The connection is read-only."""

    uri = f"{path.resolve().as_uri()}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        raise FilesystemError(f"Could not open database: {exc}") from exc

    try:
        connection.row_factory = sqlite3.Row
        names = [
            str(row[0])
            for row in connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            )
        ]
        if not names:
            return f"{path.name}: SQLite database with no user tables."
        sections = [_render_table(connection, name) for name in names]
    except sqlite3.Error as exc:
        raise FilesystemError(f"Could not read database: {exc}") from exc
    finally:
        connection.close()
    return "\n\n".join(sections)


def _is_sqlite(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(len(_SQLITE_MAGIC)) == _SQLITE_MAGIC
    except OSError:
        return False


def _render_table(connection: sqlite3.Connection, name: str) -> str:
    quoted = '"' + name.replace('"', '""') + '"'
    columns = [str(row[1]) for row in connection.execute(f"PRAGMA table_info({quoted})")]
    total = int(connection.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0])
    rows = connection.execute(
        f"SELECT * FROM {quoted} LIMIT ?",
        (_MAX_DB_ROWS,),
    ).fetchall()
    header = " | ".join(columns)
    lines = [f"# {name} ({total} rows)", header]
    for row in rows:
        lines.append(" | ".join(_cell(row[column]) for column in columns))
    if total > len(rows):
        lines.append(f"... {total - len(rows)} more rows")
    return "\n".join(lines)


def _cell(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bytes):
        return f"<bytes {len(value)}>"
    text = str(value).replace("\n", " ").replace("|", "/")
    if len(text) > _MAX_CELL:
        return text[:_MAX_CELL] + "...(truncated)"
    return text
