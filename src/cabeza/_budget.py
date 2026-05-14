"""Token-budget utilities shared by context managers and the window guard.

These helpers are deliberately backend-agnostic: they accept any tiktoken-like
encoder (with ``.encode``/``.decode``) or ``None`` (falls back to a word-count
estimate).
"""

from __future__ import annotations

import json
from typing import Any, Optional

try:  # pragma: no cover - optional dependency
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None


def default_encoder():
    """Return a default cl100k_base encoder, or ``None`` if tiktoken is missing."""
    if tiktoken is None:
        return None
    return tiktoken.get_encoding("cl100k_base")


def stringify_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
            else:
                text = str(item)
            if text:
                parts.append(str(text))
        return "\n".join(parts).strip()
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def stringify_tool_calls(tool_calls: Any) -> str:
    if not tool_calls:
        return ""
    if isinstance(tool_calls, str):
        return tool_calls
    try:
        return json.dumps(tool_calls, ensure_ascii=False)
    except TypeError:
        return str(tool_calls)


def stringify_message(message: dict, *, include_native_fields: bool = True) -> str:
    """Render a message into a single string suitable for token counting.

    When ``include_native_fields`` is True, reasoning fields and structured
    ``tool_calls`` are included so the budget tracks what actually goes on the
    wire to the model.
    """
    content = stringify_content(message.get("content")).strip()
    if not include_native_fields:
        return content

    parts: list[str] = []
    reasoning = stringify_content(
        message.get("reasoning_content")
        or message.get("reasoning")
        or message.get("reasoning_details")
        or message.get("thinking")
    ).strip()
    tool_calls_text = stringify_tool_calls(message.get("tool_calls")).strip()

    if reasoning:
        parts.append(reasoning)
    if content:
        parts.append(content)
    if tool_calls_text:
        parts.append(tool_calls_text)
    return "\n\n".join(parts).strip()


def count_text_tokens(text: str, encoder=None) -> int:
    if not text:
        return 0
    if encoder is None:
        return max(1, (len(text) + 3) // 4)
    return len(encoder.encode(text))


def count_messages(
    messages: list[dict],
    encoder=None,
    *,
    include_native_fields: bool = True,
) -> int:
    total = 0
    for message in messages:
        text = stringify_message(message, include_native_fields=include_native_fields)
        if text:
            total += count_text_tokens(text, encoder)
    return total


def find_first_user_idx(messages: list[dict]) -> Optional[int]:
    for idx, message in enumerate(messages):
        if message.get("role") == "user":
            return idx
    return None
