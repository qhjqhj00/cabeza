"""Recent-K context strategy.

Keep only the K most recent assistant/tool-response "turns" once the prompt
exceeds the soft budget. Trailing messages that do not belong to a closed
tool turn (e.g. a finalization user message) are preserved.

Tool-turn detection works for both native ``role="tool"`` messages and
Qwen-style ``<tool_response>`` blocks wrapped in ``role="user"``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from cabeza.base import ContextManager

from cabeza._budget import (
    count_messages,
    default_encoder,
    find_first_user_idx,
    stringify_content,
)


_QWEN_TOOL_MARKERS = (
    "<tool_call",
    "</tool_call>",
    "<function=",
    "</function>",
    "<parameter=",
    "</parameter>",
)
_QWEN_TOOL_RESPONSE_RE = re.compile(
    r"<tool_response>\s*.*?\s*</tool_response>",
    flags=re.IGNORECASE | re.DOTALL,
)

_DROPPED_NOTICE_TITLE = "## Context Management Notice"
_DROPPED_NOTICE = (
    f"\n\n{_DROPPED_NOTICE_TITLE}\n"
    "Older tool-call/tool-response turns were removed to fit the context budget. "
    "The retained tool results are only the most recent evidence, not a complete "
    "record of all searches or visits. Do not assume earlier leads were fully "
    "resolved just because they are absent; continue verifying uncertain facts "
    "when needed."
)


@dataclass
class _TurnGroup:
    messages: list[dict]


class RecentKStrategy(ContextManager):
    def __init__(
        self,
        *,
        max_input_tokens: int,
        keep_recent_turns: int = 10,
        family: str = "qwen",
        encoder=None,
        include_native_fields: bool = True,
        label: str = "RecentK",
    ) -> None:
        self._max_input_tokens = max(0, int(max_input_tokens))
        self._keep_recent_turns = max(0, int(keep_recent_turns))
        self._family = (family or "").strip().lower() or "qwen"
        self._encoder = encoder if encoder is not None else default_encoder()
        self._include_native_fields = include_native_fields
        self._label = label

    def reset(self) -> None:
        return None

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

        protected, groups, trailing = self._split_messages(messages)
        if not groups:
            print(
                f"[{self._label}] threshold reached but no closed tool turns available."
            )
            return messages

        kept = groups[-self._keep_recent_turns:] if self._keep_recent_turns > 0 else []
        dropped = len(groups) - len(kept)
        if dropped > 0:
            protected = self._with_dropped_notice(protected)

        managed = list(protected)
        for group in kept:
            managed.extend(group.messages)
        managed.extend(trailing)

        new_total = count_messages(
            managed,
            self._encoder,
            include_native_fields=self._include_native_fields,
        )

        if dropped > 0:
            print(
                f"[{self._label}] kept {len(kept)} most recent turns and dropped "
                f"{dropped} older turns ({total} -> {new_total} tokens, "
                f"step={state.step}, tool_rounds={state.tool_rounds})."
            )
        else:
            print(
                f"[{self._label}] threshold reached but history already fits "
                f"within k={self._keep_recent_turns}."
            )

        if new_total > self._max_input_tokens:
            print(
                f"[{self._label}] Warning: still over budget after recent-k trim."
            )
        return managed

    # ---- internals --------------------------------------------------------

    def _split_messages(
        self, messages: list[dict]
    ) -> tuple[list[dict], list[_TurnGroup], list[dict]]:
        user_idx = find_first_user_idx(messages)
        if user_idx is None:
            return list(messages), [], []

        protected = list(messages[: user_idx + 1])
        rest = list(messages[user_idx + 1:])

        groups: list[_TurnGroup] = []
        trailing: list[dict] = []
        current: list[dict] = []
        current_is_tool_turn = False
        has_response = False

        def flush() -> None:
            nonlocal current, current_is_tool_turn, has_response
            if not current:
                return
            if current_is_tool_turn and has_response:
                groups.append(_TurnGroup(messages=list(current)))
            else:
                trailing.extend(current)
            current = []
            current_is_tool_turn = False
            has_response = False

        for msg in rest:
            role = msg.get("role")
            if role == "assistant":
                flush()
                current = [msg]
                current_is_tool_turn = self._assistant_starts_tool_turn(msg)
                continue

            if current and current_is_tool_turn and self._is_tool_response(msg):
                current.append(msg)
                has_response = True
                continue

            flush()
            trailing.append(msg)

        flush()
        return protected, groups, trailing

    def _assistant_starts_tool_turn(self, msg: dict) -> bool:
        if msg.get("role") != "assistant":
            return False
        tool_calls = msg.get("tool_calls") or []
        if isinstance(tool_calls, list) and len(tool_calls) > 0:
            return True
        if self._family != "qwen":
            return False
        content = stringify_content(msg.get("content")).strip().lower()
        return any(marker in content for marker in _QWEN_TOOL_MARKERS)

    def _is_tool_response(self, msg: dict) -> bool:
        if msg.get("role") == "tool":
            return True
        if self._family != "qwen" or msg.get("role") != "user":
            return False
        content = stringify_content(msg.get("content")).strip()
        return content.lower().startswith("<tool_response>") and bool(
            _QWEN_TOOL_RESPONSE_RE.search(content)
        )

    def _with_dropped_notice(self, protected: list[dict]) -> list[dict]:
        if self._family != "qwen" or not protected:
            return protected
        user_idx = find_first_user_idx(protected)
        if user_idx is None:
            return protected
        managed = [dict(m) for m in protected]
        content = stringify_content(managed[user_idx].get("content"))
        if _DROPPED_NOTICE_TITLE in content:
            return managed
        managed[user_idx]["content"] = f"{content}{_DROPPED_NOTICE}"
        return managed
