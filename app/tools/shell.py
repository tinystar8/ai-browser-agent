"""Optional shell tool with an allowlist.

Commands that run without confirmation: ls, pwd, git status, docker ps.
Commands that run only when confirm=True: rm, sudo, shutdown, reboot.
Everything else is refused. The autonomous browser agent always calls this
with confirm=False, so it cannot run the confirmation-gated commands.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

_AUTO_SECOND = {
    "git": "status",
    "docker": "ps",
}
_CONFIRM = {"rm", "sudo", "shutdown", "reboot"}
_FORBIDDEN_CHARS = set(";&|$`><\n")
_ROOT_WIPE = {
    "/",
    "/*",
    "/bin",
    "/boot",
    "/dev",
    "/etc",
    "/lib",
    "/lib64",
    "/proc",
    "/root",
    "/sbin",
    "/sys",
    "/usr",
    "/var",
}


@dataclass(frozen=True)
class ShellOutcome:
    """Result of evaluating or running a command."""

    status: str
    output: str
    reason: str | None = None


def _reject_metacharacters(command: str) -> ShellOutcome | None:
    if any(char in command for char in _FORBIDDEN_CHARS):
        return ShellOutcome(
            status="blocked",
            output="",
            reason="Refused: shell metacharacters are not allowed. Pass a single command.",
        )
    return None


def _is_root_wipe(argv: list[str]) -> bool:
    """Refuse rm (including under sudo) that targets the system root."""

    if "rm" not in argv:
        return False
    if any(arg in {"--no-preserve-root"} for arg in argv):
        return True
    start = argv.index("rm") + 1
    for arg in argv[start:]:
        if arg.startswith("-"):
            continue
        if arg in _ROOT_WIPE or arg.rstrip("/") == "":
            return True
    return False


def _classify(argv: list[str]) -> ShellOutcome | None:
    """Return an outcome when the command must not run, otherwise None."""

    if not argv:
        return ShellOutcome(status="blocked", output="", reason="Refused: empty command.")

    program = argv[0]
    if program in {"ls", "pwd"}:
        if program == "pwd" and len(argv) > 1:
            return ShellOutcome(status="blocked", output="", reason="Refused: pwd takes no arguments.")
        return None

    expected = _AUTO_SECOND.get(program)
    if expected is not None:
        if len(argv) < 2 or argv[1] != expected:
            return ShellOutcome(
                status="blocked",
                output="",
                reason=f"Refused: only `{program} {expected}` is allowed for {program}.",
            )
        return None

    if program in _CONFIRM:
        return None

    return ShellOutcome(
        status="blocked",
        output="",
        reason=(
            "Refused: command is not on the allowlist. "
            "Automatic commands: ls, pwd, git status, docker ps. "
            "rm, sudo, shutdown, and reboot require confirm=true."
        ),
    )


def execute(command: str, *, confirm: bool, cwd: Path, timeout: int = 20) -> ShellOutcome:
    """Run a command if the safety policy allows it."""

    cleaned = command.strip()
    blocked = _reject_metacharacters(cleaned)
    if blocked is not None:
        return blocked

    try:
        argv = shlex.split(cleaned)
    except ValueError as exc:
        return ShellOutcome(status="blocked", output="", reason=f"Refused: {exc}")

    classified = _classify(argv)
    if classified is not None:
        return classified

    if _is_root_wipe(argv):
        return ShellOutcome(
            status="blocked",
            output="",
            reason="Refused: deleting the filesystem root is not allowed.",
        )

    if argv[0] in _CONFIRM and not confirm:
        return ShellOutcome(
            status="needs_confirmation",
            output="",
            reason=(
                f"`{argv[0]}` is not run automatically. "
                "Resend the same command with confirm=true if you intend to run it."
            ),
        )

    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ShellOutcome(
            status="failed",
            output="",
            reason=f"Command timed out after {timeout} seconds.",
        )
    except FileNotFoundError:
        return ShellOutcome(
            status="failed",
            output="",
            reason=f"Program not found: {argv[0]}",
        )

    output = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0:
        return ShellOutcome(
            status="failed",
            output=output.strip(),
            reason=f"Exit code {completed.returncode}",
        )
    return ShellOutcome(status="completed", output=output.strip(), reason=None)
