"""Save verified Solidity source under each file's original name.

The browser is a poor way to copy a multi-file contract. This module reads the
public Blockscout verification for Ethereum mainnet and writes every source
file under contracts/<address>/, keeping paths such as
lib/openzeppelin-contracts-v5/contracts/interfaces/IERC1967.sol.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import httpx

from app.tools.functions import write_function_report

_BLOCKSCOUT = "https://eth.blockscout.com/api/v2/smart-contracts/{address}"
_SOURCE_SUFFIXES = (".sol", ".vy", ".yul")
_MAX_FILE_BYTES = 1_500_000
_MAX_TOTAL_BYTES = 30_000_000
_USER_AGENT = "ai-browser-agent"


class SolidityError(Exception):
    """Raised when verified source cannot be saved."""


@dataclass(frozen=True)
class SavedSource:
    """Files written for one address, plus any implementation that was followed."""

    address: str
    name: str
    paths: tuple[str, ...]
    implementation: str | None = None
    implementation_name: str | None = None
    implementation_paths: tuple[str, ...] = ()

    def note(self) -> str:
        """Text appended to a skill prompt so the model can cite the files."""

        lines = [
            "Verified Solidity source is already saved on disk.",
            "Read a file with the workspace file tool if you need its contents.",
            "List these paths in the result. Do not paste full source into the result.",
            "",
        ]
        lines.extend(f"Saved file: {path}" for path in self.paths)
        if any(path.endswith("/functions.txt") for path in self.paths):
            lines.append("")
            lines.append(
                "functions.txt lists every function as read or write. "
                "Address inputs are marked. No-argument read functions include the on-chain value, "
                "and any returned address is listed at the bottom of that file."
            )
        if self.implementation and self.implementation_paths:
            label = self.implementation_name or "implementation"
            lines.append("")
            lines.append(f"This contract is a proxy. Implementation {label} at {self.implementation}:")
            lines.extend(f"Saved file: {path}" for path in self.implementation_paths)
        return "\n".join(lines)


def save_verified_source(address: str, root: Path) -> SavedSource:
    """Download verified source for address and write each file by its path."""

    record = _fetch(address)
    name = str(record.get("name") or "Contract")
    paths = _write_files(address, _collect_files(record, name), root)

    implementation = _implementation_address(record)
    implementation_name: str | None = None
    implementation_paths: tuple[str, ...] = ()
    impl_record: dict[str, object] | None = None
    if implementation and implementation.lower() != address.lower():
        try:
            fetched = _fetch(implementation)
            impl_name = str(fetched.get("name") or "Contract")
            implementation_paths = _write_files(
                implementation,
                _collect_files(fetched, impl_name),
                root,
            )
            implementation_name = impl_name
            impl_record = fetched
        except SolidityError:
            implementation = None

    report = write_function_report(
        call_address=address,
        abi_record=impl_record or record,
        directory=root / "contracts" / address,
        root=root,
        contract_name=implementation_name or name,
        implementation=implementation if impl_record is not None else None,
    )
    if report:
        paths = paths + (report,)

    return SavedSource(
        address=address,
        name=name,
        paths=paths,
        implementation=implementation,
        implementation_name=implementation_name,
        implementation_paths=implementation_paths,
    )


def _fetch(address: str) -> dict[str, object]:
    url = _BLOCKSCOUT.format(address=address)
    try:
        with httpx.Client(
            timeout=40,
            headers={"Accept": "application/json", "User-Agent": _USER_AGENT},
            follow_redirects=True,
        ) as client:
            response = client.get(url)
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as exc:
        raise SolidityError(f"Could not download verified source for {address}: {exc}") from exc

    if not isinstance(payload, dict):
        raise SolidityError(f"Unexpected source response for {address}.")
    if not payload.get("is_verified"):
        raise SolidityError(f"{address} is not verified, so there is no Solidity source to save.")
    return payload


def _implementation_address(record: dict[str, object]) -> str | None:
    implementations = record.get("implementations")
    if not isinstance(implementations, list) or not implementations:
        return None
    first = implementations[0]
    if not isinstance(first, dict):
        return None
    address = first.get("address_hash") or first.get("address")
    if not isinstance(address, str) or not address.startswith("0x"):
        return None
    return address


def _collect_files(record: dict[str, object], contract_name: str) -> list[tuple[str, str]]:
    """Return (relative path, source text) pairs from one verification record."""

    files: list[tuple[str, str]] = []
    source = record.get("source_code")
    source_text = source if isinstance(source, str) else ""
    bundled = _standard_json_sources(source_text)
    if bundled:
        files.extend(bundled)
    else:
        file_path = record.get("file_path")
        relative = file_path if isinstance(file_path, str) and file_path.strip() else f"{contract_name}.sol"
        if source_text.strip():
            files.append((relative, source_text))

    extras = record.get("additional_sources")
    if isinstance(extras, list):
        for item in extras:
            if not isinstance(item, dict):
                continue
            path = item.get("file_path")
            content = item.get("source_code")
            if isinstance(path, str) and isinstance(content, str) and content.strip():
                files.append((path, content))
    if not files:
        raise SolidityError("The contract is verified but the response had no source files.")
    return files


def _standard_json_sources(source: str) -> list[tuple[str, str]] | None:
    """Parse a Standard JSON source bundle. Plain Solidity returns None."""

    stripped = source.strip()
    if stripped.startswith("{{") and stripped.endswith("}}"):
        stripped = stripped[1:-1]
    if not stripped.startswith("{"):
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    sources = payload.get("sources") if isinstance(payload, dict) else None
    if not isinstance(sources, dict):
        return None
    files: list[tuple[str, str]] = []
    for path, entry in sources.items():
        if not isinstance(path, str):
            continue
        if isinstance(entry, dict) and isinstance(entry.get("content"), str):
            files.append((path, entry["content"]))
        elif isinstance(entry, str):
            files.append((path, entry))
    return files or None


def _write_files(address: str, files: list[tuple[str, str]], root: Path) -> tuple[str, ...]:
    folder = (root / "contracts" / address).resolve()
    root_resolved = root.resolve()
    if not folder.is_relative_to(root_resolved):
        raise SolidityError("Refusing to write source outside the project.")

    written: list[str] = []
    seen: set[str] = set()
    total = 0
    for raw_path, content in files:
        relative = _safe_source_path(raw_path)
        key = relative.as_posix()
        if key in seen:
            continue
        seen.add(key)
        data = content.encode("utf-8")
        if len(data) > _MAX_FILE_BYTES:
            raise SolidityError(f"{key} is too large to save ({len(data)} bytes).")
        total += len(data)
        if total > _MAX_TOTAL_BYTES:
            raise SolidityError("Verified source is larger than the save limit.")
        target = (folder / relative).resolve()
        if not target.is_relative_to(folder):
            raise SolidityError(f"Refusing source path: {raw_path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        written.append(target.relative_to(root_resolved).as_posix())
    if not written:
        raise SolidityError("No Solidity files were saved.")
    return tuple(written)


def _safe_source_path(raw: str) -> Path:
    """Keep a verifier file path and reject anything that could escape the folder."""

    cleaned = raw.replace("\\", "/").strip()
    if not cleaned or cleaned.startswith("/") or "\x00" in cleaned:
        raise SolidityError(f"Refusing source path: {raw}")
    parts = [part for part in cleaned.split("/") if part not in {"", "."}]
    if not parts or any(part == ".." for part in parts):
        raise SolidityError(f"Refusing source path: {raw}")
    if ":" in parts[0]:
        raise SolidityError(f"Refusing source path: {raw}")
    if not parts[-1].lower().endswith(_SOURCE_SUFFIXES):
        raise SolidityError(f"Refusing non-source file: {raw}")
    return Path(*parts)
