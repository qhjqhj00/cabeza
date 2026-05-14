"""Discard-tool-response context strategy.

Replaces the content of old tool-response messages with a short placeholder
once the prompt exceeds the soft budget. The latest tool-response round is
always preserved so the model can still continue from a coherent state.

Handles both native ``role="tool"`` messages and Qwen-style ``<tool_response>``
blocks wrapped inside a ``role="user"`` message.
"""

from __future__ import annotations

import re

from cabeza.base import ContextManager

from cabeza._budget import count_messages, default_encoder, stringify_content


NATIVE_TRUNCATED = "[tool_response_truncated]"
QWEN_TRUNCATED = f"<tool_response>\n{NATIVE_TRUNCATED}\n</tool_response>\n"

_QWEN_TOOL_RESPONSE_RE = re.compile(
    r"<tool_response>\s*(.*?)\s*</tool_response>",
    flags=re.IGNORECASE | re.DOTALL,
)


class DiscardToolStrategy(ContextManager):
    """Replace older tool-response payloads with a placeholder string."""

    def __init__(
        self,
        *,
        max_input_tokens: int,
        family: str = "qwen",
        encoder=None,
        include_native_fields: bool = True,
        label: str = "DiscardTool",
    ) -> None:
        self._max_input_tokens = max(0, int(max_input_tokens))
        self._family = (family or "").strip().lower() or "qwen"
        self._encoder = encoder if encoder is not None else default_encoder()
        self._include_native_fields = include_native_fields
        self._label = label

    def reset(self) -> None:
        return None

    # ---- detection helpers -------------------------------------------------

    @staticmethod
    def _is_native_tool_response(msg: dict) -> bool:
        return msg.get("role") == "tool"

    def _is_qwen_tool_response(self, msg: dict) -> bool:
        if self._family != "qwen" or msg.get("role") != "user":
            return False
        content = stringify_content(msg.get("content")).strip()
        return content.lower().startswith("<tool_response>") and bool(
            _QWEN_TOOL_RESPONSE_RE.search(content)
        )

    def _is_tool_response(self, msg: dict) -> bool:
        return self._is_native_tool_response(msg) or self._is_qwen_tool_response(msg)

    def _is_already_truncated(self, msg: dict) -> bool:
        content = stringify_content(msg.get("content")).strip()
        if self._is_qwen_tool_response(msg):
            blocks = _QWEN_TOOL_RESPONSE_RE.findall(content)
            return bool(blocks) and all(b.strip() == NATIVE_TRUNCATED for b in blocks)
        return content == NATIVE_TRUNCATED

    def _truncated_content(self, msg: dict) -> tuple[str, int]:
        if not self._is_qwen_tool_response(msg):
            return NATIVE_TRUNCATED, 1
        content = stringify_content(msg.get("content"))
        block_count = len(_QWEN_TOOL_RESPONSE_RE.findall(content)) or 1
        return QWEN_TRUNCATED * block_count, block_count

    def _tool_response_count(self, msg: dict) -> int:
        if not self._is_tool_response(msg):
            return 0
        if not self._is_qwen_tool_response(msg):
            return 1
        content = stringify_content(msg.get("content"))
        return len(_QWEN_TOOL_RESPONSE_RE.findall(content)) or 1

    def _latest_round_indices(self, messages: list[dict]) -> set[int]:
        last_idx: int | None = None
        for idx, msg in enumerate(messages):
            if self._is_tool_response(msg):
                last_idx = idx
        if last_idx is None:
            return set()
        keep = {last_idx}
        idx = last_idx - 1
        while idx >= 0 and self._is_tool_response(messages[idx]):
            keep.add(idx)
            idx -= 1
        return keep

    # ---- ContextManager API -----------------------------------------------

    def process(self, messages: list[dict], state) -> list[dict]:
        if not messages or self._max_input_tokens <= 0:
            return messages

        total = count_messages(
            messages,
            self._encoder,
            include_native_fields=self._include_native_fields,
        )
        if total <= self._max_input_tokens:
            return messages

        managed = [dict(m) for m in messages]
        truncated_count = 0
        preserved = self._latest_round_indices(managed)
        preserved_count = sum(self._tool_response_count(managed[i]) for i in preserved)

        for idx, msg in enumerate(managed):
            if not self._is_tool_response(msg):
                continue
            if idx in preserved:
                continue
            if self._is_already_truncated(msg):
                continue
            msg["content"], n = self._truncated_content(msg)
            truncated_count += n

        new_total = count_messages(
            managed,
            self._encoder,
            include_native_fields=self._include_native_fields,
        )
        if truncated_count > 0:
            print(
                f"[{self._label}] truncated {truncated_count} older tool responses "
                f"(preserved={preserved_count}; {total} -> {new_total} tokens, "
                f"step={state.step}, tool_rounds={state.tool_rounds})."
            )
        else:
            print(
                f"[{self._label}] threshold reached but no older tool responses "
                f"available to discard (preserved={preserved_count}; "
                f"{total} -> {new_total} tokens)."
            )

        if new_total > self._max_input_tokens:
            print(
                f"[{self._label}] Warning: still over budget after discarding."
            )
        return managed
