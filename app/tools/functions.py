"""List a contract's read and write functions, including address values.

Etherscan's Read and Write tabs are the verified ABI. Read functions are
view or pure. Write functions are the ones that send a transaction.
No-argument read functions are called on-chain so an address return value
is the real address, not a guess from the source.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

_USER_AGENT = "ai-browser-agent"
_RPCS = (
    "https://ethereum.publicnode.com",
    "https://eth.drpc.org",
    "https://rpc.ankr.com/eth",
    "https://cloudflare-eth.com",
)
_MAX_CALLS = 80

# Keccak-f[1600] rotation offsets, indexed [x][y].
_ROTATION = (
    (0, 36, 3, 41, 18),
    (1, 44, 10, 45, 2),
    (62, 6, 43, 15, 61),
    (28, 55, 25, 21, 56),
    (27, 20, 39, 8, 14),
)
_ROUND_CONSTANTS = (
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
)
_MASK64 = (1 << 64) - 1


@dataclass(frozen=True)
class _Function:
    name: str
    signature: str
    selector: str
    kind: str
    inputs: tuple[tuple[str, str], ...]
    outputs: tuple[str, ...]
    value: str | None = None
    addresses: tuple[str, ...] = ()


def write_function_report(
    call_address: str,
    abi_record: dict[str, object],
    directory: Path,
    root: Path,
    contract_name: str,
    implementation: str | None,
) -> str | None:
    """Write functions.txt and return its path relative to the project root."""

    functions = _functions_from_abi(abi_record)
    if not functions:
        return None
    values = _read_values(call_address, [item for item in functions if item.kind == "read" and not item.inputs])
    filled = tuple(_with_value(item, values.get(item.selector)) for item in functions)
    text = _render(call_address, contract_name, implementation, filled)
    target = (directory / "functions.txt").resolve()
    root_resolved = root.resolve()
    if not target.is_relative_to(root_resolved):
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target.relative_to(root_resolved).as_posix()


def keccak256(data: bytes) -> bytes:
    """Ethereum Keccak-256. This is not hashlib.sha3_256."""

    rate = 136
    state = [0] * 25
    padded = bytearray(data)
    padded.append(0x01)
    while len(padded) % rate != rate - 1:
        padded.append(0)
    padded.append(0x80)
    for offset in range(0, len(padded), rate):
        block = padded[offset : offset + rate]
        for index in range(rate // 8):
            state[index] ^= int.from_bytes(block[index * 8 : (index + 1) * 8], "little")
        _permute(state)
    output = bytearray()
    while len(output) < 32:
        for index in range(rate // 8):
            output += state[index].to_bytes(8, "little")
            if len(output) >= 32:
                break
        if len(output) < 32:
            _permute(state)
    return bytes(output[:32])


def _functions_from_abi(record: dict[str, object]) -> tuple[_Function, ...]:
    raw = record.get("abi")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return ()
    if not isinstance(raw, list):
        return ()
    found: list[_Function] = []
    for item in raw:
        if not isinstance(item, dict) or item.get("type") != "function":
            continue
        name = str(item.get("name") or "")
        if not name:
            continue
        inputs = _components(item.get("inputs"))
        outputs = tuple(_type_name(entry) for entry in _entries(item.get("outputs")))
        signature = f"{name}({','.join(kind for _, kind in inputs)})"
        selector = "0x" + keccak256(signature.encode()).hex()[:8]
        state = str(item.get("stateMutability") or "")
        if state in {"view", "pure"} or item.get("constant") is True:
            kind = "read"
        else:
            kind = "write"
        found.append(_Function(name, signature, selector, kind, inputs, outputs))
    return tuple(found)


def _with_value(item: _Function, value: str | None) -> _Function:
    if item.kind != "read" or item.inputs or value is None:
        addresses = _input_addresses(item)
        return _Function(
            item.name,
            item.signature,
            item.selector,
            item.kind,
            item.inputs,
            item.outputs,
            None,
            addresses,
        )
    decoded, addresses = _decode_outputs(item.outputs, value)
    return _Function(
        item.name,
        item.signature,
        item.selector,
        item.kind,
        item.inputs,
        item.outputs,
        decoded,
        addresses,
    )


def _input_addresses(item: _Function) -> tuple[str, ...]:
    """Names of inputs whose ABI type is address. These are not on-chain values."""

    return tuple(f"{name or 'arg'}: {kind}" for name, kind in item.inputs if _contains_address(kind))


def _read_values(address: str, functions: list[_Function]) -> dict[str, str]:
    """Call no-argument read functions. A failed endpoint leaves values blank."""

    chosen = functions[:_MAX_CALLS]
    if not chosen:
        return {}
    calls = []
    for index, item in enumerate(chosen):
        calls.append(
            {
                "jsonrpc": "2.0",
                "id": index,
                "method": "eth_call",
                "params": [{"to": address, "data": item.selector}, "latest"],
            }
        )
    payload = _rpc_batch(calls)
    if payload is None:
        return {}
    values: dict[str, str] = {}
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        index = entry.get("id")
        if not isinstance(index, int) or index < 0 or index >= len(chosen):
            continue
        result = entry.get("result")
        if isinstance(result, str) and result.startswith("0x"):
            values[chosen[index].selector] = result
        elif "error" in entry:
            values[chosen[index].selector] = "reverted"
    return values


def _rpc_batch(calls: list[dict[str, Any]]) -> list[Any] | None:
    headers = {"Accept": "application/json", "User-Agent": _USER_AGENT}
    for url in _RPCS:
        try:
            with httpx.Client(timeout=40, headers=headers, follow_redirects=True) as client:
                response = client.post(url, json=calls)
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, json.JSONDecodeError):
            continue
        if isinstance(payload, list):
            return payload
    return None


def _render(
    call_address: str,
    contract_name: str,
    implementation: str | None,
    functions: tuple[_Function, ...],
) -> str:
    reads = [item for item in functions if item.kind == "read"]
    writes = [item for item in functions if item.kind == "write"]
    lines = [f"Contract: {contract_name}", f"Called at: {call_address}"]
    if implementation:
        lines.append(f"Read as proxy. Implementation: {implementation}")
    lines.append("read = view or pure. write = transaction. Address values come from read calls.")
    lines.append("")
    lines.extend(_section("Read", reads))
    lines.append("")
    lines.extend(_section("Write", writes))
    address_lines = [
        f"{item.name}() = {address}"
        for item in reads
        if not item.inputs
        for address in item.addresses
        if address.startswith("0x")
    ]
    if address_lines:
        lines.append("")
        lines.append("Addresses returned by read functions")
        lines.extend(address_lines)
    return "\n".join(lines) + "\n"


def _section(title: str, functions: list[_Function]) -> list[str]:
    lines = [title]
    if not functions:
        lines.append("(none)")
        return lines
    for item in functions:
        output = ""
        if item.outputs:
            output = " -> " + ", ".join(item.outputs)
        value = ""
        if item.value:
            value = f" = {item.value}"
        elif item.kind == "read" and item.inputs:
            value = " (needs arguments)"
        lines.append(f"{item.kind}  {item.signature}  {item.selector}{output}{value}")
        for address in item.addresses:
            if address.startswith("0x"):
                lines.append(f"  address: {address}")
            else:
                lines.append(f"  address input: {address}")
    return lines


def _decode_outputs(types: tuple[str, ...], raw: str) -> tuple[str, tuple[str, ...]]:
    if raw == "reverted":
        return "reverted", ()
    data = bytes.fromhex(raw[2:] if raw.startswith("0x") else raw)
    if not types:
        return "()", ()
    if not data:
        return "empty", ()
    try:
        values, addresses = _decode_tuple(types, data, 0)
    except (ValueError, IndexError):
        return raw, ()
    shown = ", ".join(_show(value) for value in values)
    return shown, tuple(addresses)


def _decode_tuple(types: tuple[str, ...] | list[str], data: bytes, offset: int) -> tuple[list[Any], list[str]]:
    values: list[Any] = []
    addresses: list[str] = []
    cursor = offset
    for kind in types:
        if _is_dynamic(kind):
            jump = int.from_bytes(data[cursor : cursor + 32], "big")
            value, found = _decode_dynamic(kind, data, offset + jump)
            cursor += 32
        else:
            value, found, cursor = _decode_static(kind, data, cursor)
        values.append(value)
        addresses.extend(found)
    return values, addresses


def _decode_static(kind: str, data: bytes, offset: int) -> tuple[Any, list[str], int]:
    if kind == "address":
        word = data[offset : offset + 32]
        address = _checksum("0x" + word[-20:].hex())
        return address, [address], offset + 32
    if kind == "bool":
        return bool(data[offset + 31]), [], offset + 32
    if kind.startswith("uint"):
        return int.from_bytes(data[offset : offset + 32], "big"), [], offset + 32
    if kind.startswith("int"):
        raw = int.from_bytes(data[offset : offset + 32], "big")
        if raw >= 1 << 255:
            raw -= 1 << 256
        return raw, [], offset + 32
    if kind.startswith("bytes") and kind != "bytes":
        width = int(kind[5:])
        return "0x" + data[offset : offset + width].hex(), [], offset + 32
    if kind.startswith("(") and kind.endswith(")") and not _is_dynamic(kind):
        inner = _split_tuple(kind[1:-1])
        values, found = _decode_tuple(inner, data, offset)
        return values, found, offset + 32 * len(inner)
    word = data[offset : offset + 32]
    return "0x" + word.hex(), [], offset + 32


def _decode_dynamic(kind: str, data: bytes, offset: int) -> tuple[Any, list[str]]:
    if kind in {"bytes", "string"}:
        length = int.from_bytes(data[offset : offset + 32], "big")
        blob = data[offset + 32 : offset + 32 + length]
        if kind == "string":
            return blob.decode("utf-8", errors="replace"), []
        return "0x" + blob.hex(), []
    if kind.endswith("[]"):
        length = int.from_bytes(data[offset : offset + 32], "big")
        inner = kind[:-2]
        values: list[Any] = []
        found: list[str] = []
        cursor = offset + 32
        for _ in range(length):
            if _is_dynamic(inner):
                jump = int.from_bytes(data[cursor : cursor + 32], "big")
                value, nested = _decode_dynamic(inner, data, offset + 32 + jump)
                cursor += 32
            else:
                value, nested, cursor = _decode_static(inner, data, cursor)
            values.append(value)
            found.extend(nested)
        return values, found
    if kind.startswith("(") and kind.endswith(")"):
        values, found = _decode_tuple(_split_tuple(kind[1:-1]), data, offset)
        return values, found
    value, found, _ = _decode_static(kind, data, offset)
    return value, found


def _is_dynamic(kind: str) -> bool:
    if kind in {"string", "bytes"} or kind.endswith("[]"):
        return True
    if kind.endswith("]") and "[" in kind and not kind.endswith("[]"):
        inner = kind[: kind.rfind("[")]
        return _is_dynamic(inner)
    if kind.startswith("(") and kind.endswith(")"):
        return any(_is_dynamic(part) for part in _split_tuple(kind[1:-1]))
    return False


def _is_address_type(kind: str) -> bool:
    if kind == "address":
        return True
    if kind.endswith("[]"):
        return _is_address_type(kind[:-2])
    return False


def _contains_address(kind: str) -> bool:
    """True when a value of this ABI type holds an address, including inside a tuple."""

    if _is_address_type(kind):
        return True
    if "(" not in kind:
        return False
    inner = kind[kind.find("(") + 1 : kind.rfind(")")]
    return any(_contains_address(part) for part in _split_tuple(inner))


def _split_tuple(body: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(body):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(body[start:index])
            start = index + 1
    if body[start:]:
        parts.append(body[start:])
    return parts


def _components(raw: object) -> tuple[tuple[str, str], ...]:
    return tuple((str(entry.get("name") or "arg"), _type_name(entry)) for entry in _entries(raw))


def _entries(raw: object) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    return [entry for entry in raw if isinstance(entry, dict)]


def _type_name(entry: dict[str, Any]) -> str:
    kind = str(entry.get("type") or "")
    if kind.startswith("tuple"):
        inner = ",".join(_type_name(item) for item in _entries(entry.get("components")))
        return f"({inner}){kind[5:]}"
    return kind


def _checksum(address: str) -> str:
    hexed = address.lower().removeprefix("0x")
    digest = keccak256(hexed.encode()).hex()
    chars = [char.upper() if char.isalpha() and int(nibble, 16) >= 8 else char for char, nibble in zip(hexed, digest)]
    return "0x" + "".join(chars)


def _show(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(_show(item) for item in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _permute(state: list[int]) -> None:
    for constant in _ROUND_CONSTANTS:
        columns = [
            state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20]
            for x in range(5)
        ]
        mixed = [columns[(x - 1) % 5] ^ _rotate(columns[(x + 1) % 5], 1) for x in range(5)]
        for index in range(25):
            state[index] ^= mixed[index % 5]
        rotated = [0] * 25
        for x in range(5):
            for y in range(5):
                rotated[y + 5 * ((2 * x + 3 * y) % 5)] = _rotate(state[x + 5 * y], _ROTATION[x][y])
        for y in range(5):
            for x in range(5):
                current = rotated[x + 5 * y]
                state[x + 5 * y] = (
                    current ^ ((~rotated[((x + 1) % 5) + 5 * y]) & rotated[((x + 2) % 5) + 5 * y])
                ) & _MASK64
        state[0] ^= constant


def _rotate(value: int, bits: int) -> int:
    if bits == 0:
        return value
    return ((value << bits) | (value >> (64 - bits))) & _MASK64
