"""Save pictures attached to a chat message and turn them into model inputs."""

from __future__ import annotations

import base64
import binascii
import re
import secrets
from pathlib import Path

_DATA_URL = re.compile(
    r"^data:image/(png|jpeg|jpg|gif|webp);base64,([A-Za-z0-9+/=\s]+)$",
    re.IGNORECASE,
)
_NAME = re.compile(r"^[0-9a-f]{16}\.(png|jpg|gif|webp)$")
_MAX_BYTES = 4_000_000
_MEDIA = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
}


class ChatImageError(ValueError):
    """The attachment is not a picture this chat can store."""


def save_chat_images(payloads: list[str], directory: Path) -> tuple[list[str], list[str]]:
    """Write each data URL and return public paths plus the original data URLs."""

    if len(payloads) > 4:
        raise ChatImageError("Attach up to 4 pictures.")
    directory.mkdir(parents=True, exist_ok=True)
    urls: list[str] = []
    kept: list[str] = []
    for payload in payloads:
        ext, raw, data_url = _decode(payload)
        name = f"{secrets.token_hex(8)}.{ext}"
        (directory / name).write_bytes(raw)
        urls.append(f"/chat/images/{name}")
        kept.append(data_url)
    return urls, kept


def image_file(directory: Path, name: str) -> Path | None:
    """Return a stored picture path, or None when the name is not a stored picture."""

    if _NAME.fullmatch(name) is None:
        return None
    path = (directory / name).resolve()
    root = directory.resolve()
    if path.parent != root or not path.is_file():
        return None
    return path


def content_parts(data_urls: list[str]) -> list[object]:
    """Build Browser Use image parts. An empty list means there is nothing to attach."""

    if not data_urls:
        return []
    from browser_use.llm.messages import ContentPartImageParam, ContentPartTextParam, ImageURL

    parts: list[object] = [ContentPartTextParam(text="Picture attached by the user:")]
    for url in data_urls:
        header = url.split(",", 1)[0].lower()
        media = "image/png"
        for kind, mime in _MEDIA.items():
            if kind in header:
                media = mime
                break
        parts.append(
            ContentPartImageParam(
                image_url=ImageURL(url=url, media_type=media, detail="auto")
            )
        )
    return parts


def _decode(payload: str) -> tuple[str, bytes, str]:
    match = _DATA_URL.fullmatch(payload.strip())
    if match is None:
        raise ChatImageError("Attach a PNG, JPEG, GIF, or WebP picture.")
    kind = match.group(1).lower()
    ext = "jpg" if kind == "jpeg" else kind
    try:
        raw = base64.b64decode(match.group(2), validate=True)
    except binascii.Error as exc:
        raise ChatImageError("That picture could not be read.") from exc
    if not raw or len(raw) > _MAX_BYTES:
        raise ChatImageError("Each picture must be under 4 MB.")
    sniffed = _sniff(raw)
    if sniffed is None or sniffed != ext:
        raise ChatImageError("That file is not a PNG, JPEG, GIF, or WebP picture.")
    media = _MEDIA[kind]
    encoded = base64.b64encode(raw).decode("ascii")
    return ext, raw, f"data:{media};base64,{encoded}"


def _sniff(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "webp"
    return None
