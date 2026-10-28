"""Local Qwen connection through Ollama's OpenAI-compatible API.

LangChain's ChatOpenAI is the client. Browser Use 0.3+ does not accept
LangChain models directly, so LocalOllamaChat adapts that same client to the
Browser Use chat protocol. No request is sent to api.openai.com.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_openai import ChatOpenAI

from app.config import Settings, get_settings

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def build_langchain_llm(settings: Settings | None = None) -> ChatOpenAI:
    """Build the LangChain client pointed at local Ollama."""

    cfg = settings or get_settings()
    return ChatOpenAI(
        model=cfg.openai_model,
        api_key=cfg.openai_api_key,
        base_url=cfg.openai_base_url,
        temperature=0.1,
        max_tokens=8192,
        timeout=cfg.llm_timeout,
        max_retries=2,
    )


_ANSWER_SYSTEM = """
Answer the user's question about the attached screenshot.
When they ask for a coordinate, measure it on a 1000 by 1000 grid.
coordinate_x runs from the left edge (0) to the right edge (1000).
coordinate_y runs from the top edge (0) to the bottom edge (1000).
Write the point as coordinate_x=<number>, coordinate_y=<number>.
If several objects match, use the first one in reading order, left to right then top to bottom.
Name the object and the point. Do not invent a point you cannot see.
Do not return a browser action.
""".strip()

_GRID_POINT = re.compile(
    r"coordinate_x\s*[:=]\s*(\d+(?:\.\d+)?)\s*,?\s*coordinate_y\s*[:=]\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def grid_to_page(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    """Map a 0–1000 model point onto page pixels. 1000 is the far edge."""

    return round(x / 1000 * width), round(y / 1000 * height)


def rewrite_grid_coordinates(text: str, width: int, height: int) -> str:
    """Replace the model's 0–1000 point with the page pixels a click uses."""

    if width <= 0 or height <= 0:
        return text

    def replace(match: re.Match[str]) -> str:
        page_x, page_y = grid_to_page(float(match.group(1)), float(match.group(2)), width, height)
        return f"coordinate_x={page_x}, coordinate_y={page_y}"

    rewritten, count = _GRID_POINT.subn(replace, text)
    if count == 0:
        return text
    first = _GRID_POINT.search(rewritten)
    if first is None:
        return rewritten
    return f"{rewritten.rstrip()}\nclick {first.group(1)} {first.group(2)}"


def _image_size(data_url: str) -> tuple[int, int] | None:
    """Width and height of a data URL, in pixels."""

    import base64
    import io

    raw = data_url.split(",", 1)[-1]
    try:
        blob = base64.b64decode(raw, validate=False)
    except Exception:
        return None
    try:
        from PIL import Image

        with Image.open(io.BytesIO(blob)) as image:
            width, height = image.size
    except Exception:
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


async def answer_about_image(
    question: str, data_urls: list[str], settings: Settings | None = None
) -> str:
    """Ask the local model about a screenshot and return page-pixel coordinates."""

    from langchain_core.messages import HumanMessage, SystemMessage

    content: list[dict[str, object]] = [{"type": "text", "text": question}]
    for url in data_urls[:2]:
        content.append({"type": "image_url", "image_url": {"url": url}})
    response = await build_langchain_llm(settings).ainvoke(
        [SystemMessage(content=_ANSWER_SYSTEM), HumanMessage(content=content)]
    )
    text = _THINK_BLOCK.sub("", _message_text(getattr(response, "content", response))).strip()
    if not text:
        return "I could not see an answer in the picture."
    size = _image_size(data_urls[0]) if data_urls else None
    if size is None:
        return text
    return rewrite_grid_coordinates(text, size[0], size[1])


def build_agent_llm(settings: Settings | None = None) -> LocalOllamaChat:
    """Return the Browser Use model wrapper around the LangChain client."""

    return LocalOllamaChat(build_langchain_llm(settings), (settings or get_settings()).openai_model)


class LocalOllamaChat:
    """Browser Use chat model backed by LangChain's OpenAI-compatible client."""

    # Browser Use skips its own cloud key check when this is true.
    _verified_api_keys = True

    def __init__(self, chat: ChatOpenAI, model: str) -> None:
        self.chat = chat
        self.model = model

    @property
    def provider(self) -> str:
        return "ollama"

    @property
    def name(self) -> str:
        return self.model

    @property
    def model_name(self) -> str:
        return self.model

    async def ainvoke(
        self,
        messages: list[Any],
        output_format: type[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Call Ollama and, when requested, parse a JSON object into output_format."""

        from browser_use.llm.exceptions import ModelProviderError
        from browser_use.llm.views import ChatInvokeCompletion

        langchain_messages = _to_langchain_messages(messages)
        try:
            response = await self.chat.ainvoke(langchain_messages)
        except Exception as exc:
            raise ModelProviderError(message=f"Ollama request failed: {exc}", model=self.name) from exc

        text = _message_text(getattr(response, "content", response))
        if output_format is None:
            return _completion(ChatInvokeCompletion, text)

        try:
            parsed = output_format.model_validate_json(_extract_json(text))
        except Exception as exc:
            snippet = text[:500].replace("\n", " ")
            raise ModelProviderError(
                message=f"Could not parse model JSON ({exc}). Output started: {snippet}",
                model=self.name,
            ) from exc
        return _completion(ChatInvokeCompletion, parsed)


def _completion(kind: type[Any], payload: Any) -> Any:
    """Build a chat completion, tolerating a required stop_reason field."""

    try:
        return kind(completion=payload, usage=None)
    except TypeError:
        return kind(completion=payload, usage=None, stop_reason=None)


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                chunks.append(str(item.get("text", "")))
            else:
                chunks.append(str(getattr(item, "text", item)))
        return "\n".join(chunk for chunk in chunks if chunk)
    return str(content)


def _extract_json(text: str) -> str:
    """Pull the first JSON object out of a local-model reply.

    Qwen often wraps the object in thinking tags or Markdown fences. Browser Use
    needs the bare object so it can validate the action schema.
    """

    cleaned = _THINK_BLOCK.sub("", text).strip()
    cleaned = _FENCE.sub("", cleaned).strip()
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("model did not return a JSON object")
    obj, _end = json.JSONDecoder().raw_decode(cleaned[start:])
    return json.dumps(normalize_agent_output(obj))


# Local models often invent parameter names. Browser Use rejects the whole
# step when the input action uses "value" instead of the required "text".
_INPUT_ALIASES = ("input_text", "type_text", "type")
_TEXT_KEYS = ("value", "query", "content")


def normalize_agent_output(payload: object) -> object:
    """Rewrite known Qwen action mistakes into the Browser Use schema."""

    if not isinstance(payload, dict):
        return payload
    actions = payload.get("action")
    if not isinstance(actions, list):
        return payload
    normalized = dict(payload)
    normalized["action"] = [_normalize_action(item) for item in actions]
    return normalized


def _normalize_action(item: object) -> object:
    if not isinstance(item, dict):
        return item
    rewritten: dict[str, object] = {}
    for name, params in item.items():
        action_name = "input" if name in _INPUT_ALIASES else name
        if action_name == "input" and isinstance(params, dict):
            params = _input_params(params)
        rewritten[action_name] = params
    return rewritten


def _input_params(params: dict[str, object]) -> dict[str, object]:
    """Require the field name Browser Use validates: input.text."""

    updated = dict(params)
    if "text" not in updated:
        for alias in _TEXT_KEYS:
            value = updated.get(alias)
            if isinstance(value, str):
                updated["text"] = updated.pop(alias)
                break
    else:
        updated.pop("value", None)
    return updated


def _content_to_langchain(message: object) -> str | list[dict[str, object]]:
    content = getattr(message, "content", "")
    if isinstance(content, str) or content is None:
        return "" if content is None else content
    if not isinstance(content, list):
        return str(content)

    parts: list[dict[str, object]] = []
    for part in content:
        part_type = getattr(part, "type", None)
        if part_type == "text":
            parts.append({"type": "text", "text": str(getattr(part, "text", ""))})
        elif part_type == "image_url":
            image = getattr(part, "image_url", None)
            url = getattr(image, "url", str(image))
            detail = getattr(image, "detail", "auto")
            parts.append({"type": "image_url", "image_url": {"url": url, "detail": detail}})
        else:
            text = getattr(part, "text", None)
            if text:
                parts.append({"type": "text", "text": str(text)})
    return parts


def _to_langchain_messages(messages: list[Any]) -> list[Any]:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    converted: list[Any] = []
    for message in messages:
        role = message.__class__.__name__.lower()
        content = _content_to_langchain(message)
        if "system" in role:
            text = content if isinstance(content, str) else "\n".join(
                str(part.get("text", "")) for part in content
            )
            converted.append(SystemMessage(content=text))
        elif "assistant" in role:
            text = content if isinstance(content, str) else "\n".join(
                str(part.get("text", "")) for part in content
            )
            converted.append(AIMessage(content=text))
        else:
            converted.append(HumanMessage(content=content))
    return converted
