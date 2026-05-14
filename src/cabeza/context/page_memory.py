"""Page-memory context strategy.

When the prompt exceeds the soft budget, the entire live window after the
first user message is flushed into one or more pages of an external
``PageStore``. The pages are summarized into bullets that get injected back
into the first user message under an "Exploration Memory" section so the
agent still knows what was explored. Full raw content can be retrieved on
demand via the ``memory`` tool.

This is the "v2" page-based memory strategy from the original SAM codebase,
generalized to work with either a single-agent ``PageStore`` or a thread-safe
``SharedPageStore`` (for swarm/fugue setups).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from cabeza.base import ContextManager

from cabeza._budget import (
    count_messages,
    count_text_tokens,
    default_encoder,
    find_first_user_idx,
    stringify_content,
    stringify_tool_calls,
)

if TYPE_CHECKING:
    from cabeza.memory.page_store import PageStore
    from cabeza.memory.summarizer import Summarizer


MEMORY_SECTION_MARKER = (
    "\n\n"
    "## Exploration Memory\n"
    "Below is a high-level summary of your earlier exploration windows, organized by pages. "
    "Earlier live context may have been removed to fit the context window, but these page "
    "summaries tell you what was explored before.\n"
    "If you need to rely on past exploration details or verify them, use the `memory` tool "
    "to inspect only the relevant pages.\n"
)
TRUNCATION_NOTICE_TAG = "[Context Window Notice]"
DEFAULT_PAGE_MAX_TOKENS = 30_000

_QWEN_TOOL_MARKERS = (
    "<tool_call",
    "</tool_call>",
    "<function=",
    "</function>",
    "<parameter=",
    "</parameter>",
)


# ---------------------------------------------------------------------------
# Message-grouping helpers (used by both single and shared page memory)
# ---------------------------------------------------------------------------


@dataclass
class _PageChunk:
    raw_content: str
    entry_count: int
    token_count: int


@dataclass
class _TurnGroup:
    messages: list[dict]


def _assistant_has_tool_call(message: dict) -> bool:
    tool_calls = message.get("tool_calls") or []
    if isinstance(tool_calls, list) and len(tool_calls) > 0:
        return True
    if message.get("role") != "assistant":
        return False
    content = stringify_content(message.get("content")).strip().lower()
    return any(marker in content for marker in _QWEN_TOOL_MARKERS)


def _stringify_message_full(message: dict) -> str:
    parts: list[str] = []
    reasoning = stringify_content(
        message.get("reasoning_content")
        or message.get("reasoning")
        or message.get("reasoning_details")
        or message.get("thinking")
    ).strip()
    content = stringify_content(message.get("content")).strip()
    tool_calls = stringify_tool_calls(message.get("tool_calls")).strip()

    if reasoning:
        parts.append(f"[Reasoning]\n{reasoning}")
    if content:
        parts.append(content)
    if tool_calls:
        parts.append(f"[Tool Calls]\n{tool_calls}")

    if message.get("role") == "tool":
        meta = []
        name = str(message.get("name") or "").strip()
        call_id = str(message.get("tool_call_id") or "").strip()
        if name:
            meta.append(f"name: {name}")
        if call_id:
            meta.append(f"tool_call_id: {call_id}")
        if meta:
            parts.append("[Tool Metadata]\n" + "\n".join(meta))

    return "\n\n".join(p for p in parts if p).strip()


def _message_label(message: dict, idx: int) -> str:
    content = stringify_content(message.get("content")).strip()
    role = message.get("role", "unknown")
    if role == "tool":
        return f"Tool Result {idx}"
    if role == "assistant" and _assistant_has_tool_call(message):
        return f"Assistant Tool Call {idx}"
    if role == "user" and content.startswith("<tool_response>"):
        return f"Tool Response {idx}"
    return f"{str(role).title()} {idx}"


def _serialize_window_entries(messages: list[dict]) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for idx, msg in enumerate(messages, start=1):
        rendered = _stringify_message_full(msg)
        if not rendered:
            continue
        entries.append((f"### {_message_label(msg, idx)}", rendered))
    return entries


def _split_entry_segments(
    header: str,
    content: str,
    max_tokens: int,
    encoder,
) -> list[str]:
    combined = f"{header}\n{content}"
    if count_text_tokens(combined, encoder) <= max_tokens:
        return [combined]

    if encoder is not None:
        tokens = encoder.encode(content)
        if not tokens:
            return [combined]
        segments: list[str] = []
        cursor = 0
        part_idx = 1
        while cursor < len(tokens):
            part_header = header if part_idx == 1 else f"{header} (cont. {part_idx})"
            allowed = max(1, max_tokens - count_text_tokens(f"{part_header}\n", encoder))
            chunk = tokens[cursor: cursor + allowed]
            segments.append(f"{part_header}\n{encoder.decode(chunk)}")
            cursor += len(chunk)
            part_idx += 1
        return segments

    words = str(content or "").split()
    if not words:
        return [combined]
    segments = []
    cursor = 0
    part_idx = 1
    while cursor < len(words):
        part_header = header if part_idx == 1 else f"{header} (cont. {part_idx})"
        allowed = max(1, max_tokens - count_text_tokens(f"{part_header}\n", encoder))
        chunk_words = words[cursor: cursor + allowed]
        segments.append(f"{part_header}\n{' '.join(chunk_words)}")
        cursor += len(chunk_words)
        part_idx += 1
    return segments


def _build_message_page_chunks(
    messages: list[dict],
    page_max_tokens: int,
    encoder,
) -> list[_PageChunk]:
    entries = _serialize_window_entries(messages)
    if not entries:
        return []

    chunks: list[_PageChunk] = []
    sep_tokens = count_text_tokens("\n\n", encoder)
    current: list[str] = []
    current_tokens = 0
    current_entries = 0

    for header, content in entries:
        for segment in _split_entry_segments(header, content, page_max_tokens, encoder):
            seg_tokens = count_text_tokens(segment, encoder)
            join_tokens = sep_tokens if current else 0
            if current and current_tokens + join_tokens + seg_tokens > page_max_tokens:
                raw = "\n\n".join(current).strip()
                chunks.append(
                    _PageChunk(
                        raw_content=raw,
                        entry_count=current_entries,
                        token_count=count_text_tokens(raw, encoder),
                    )
                )
                current = []
                current_tokens = 0
                current_entries = 0
            if current:
                current_tokens += sep_tokens
            current.append(segment)
            current_tokens += seg_tokens
            current_entries += 1

    if current:
        raw = "\n\n".join(current).strip()
        chunks.append(
            _PageChunk(
                raw_content=raw,
                entry_count=current_entries,
                token_count=count_text_tokens(raw, encoder),
            )
        )
    return chunks


def _merge_page_units(units: list[_PageChunk], encoder) -> _PageChunk:
    parts = [u.raw_content.strip() for u in units if u.raw_content.strip()]
    raw = "\n\n".join(parts).strip()
    return _PageChunk(
        raw_content=raw,
        entry_count=sum(u.entry_count for u in units),
        token_count=count_text_tokens(raw, encoder),
    )


def _build_turn_group_page_chunks(
    groups: list[_TurnGroup],
    page_max_tokens: int,
    encoder,
) -> list[_PageChunk]:
    chunks: list[_PageChunk] = []
    current: list[_PageChunk] = []
    current_tokens = 0
    sep_tokens = count_text_tokens("\n\n", encoder)

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        chunks.append(_merge_page_units(current, encoder))
        current = []
        current_tokens = 0

    for group in groups:
        group_chunks = _build_message_page_chunks(group.messages, page_max_tokens, encoder)
        if not group_chunks:
            continue
        if len(group_chunks) > 1:
            flush()
            chunks.extend(group_chunks)
            continue
        gc = group_chunks[0]
        join_tokens = sep_tokens if current else 0
        if current and current_tokens + join_tokens + gc.token_count > page_max_tokens:
            flush()
        if current:
            current_tokens += sep_tokens
        current.append(gc)
        current_tokens += gc.token_count

    flush()
    return chunks


def _group_live_window_messages(messages: list[dict]) -> list[_TurnGroup]:
    groups: list[_TurnGroup] = []
    current: list[dict] = []
    for msg in messages:
        if msg.get("role") == "assistant":
            if current:
                groups.append(_TurnGroup(messages=current))
            current = [msg]
            continue
        if current:
            current.append(msg)
        else:
            current = [msg]
    if current:
        groups.append(_TurnGroup(messages=current))
    return groups


def _build_truncation_notice(dropped_windows: int) -> str:
    if dropped_windows <= 0:
        return ""
    return (
        "To fit the context window, earlier exploration windows were truncated and saved "
        "into Exploration Memory pages. Use the `memory` tool to retrieve them when needed."
    )


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class PageMemoryStrategy(ContextManager):
    """Flush overflowed live windows into a ``PageStore`` as memory pages.

    The strategy keeps the first user message + system prompt, summarizes the
    live window into one or more pages via the supplied ``Summarizer``, and
    rewrites the first user message to include the rendered shared-memory
    section. Works with either a single-owner ``PageStore`` or a thread-safe
    ``SharedPageStore``.
    """

    def __init__(
        self,
        *,
        page_store: "PageStore",
        summarizer: "Summarizer",
        max_input_tokens: int,
        page_max_tokens: int = DEFAULT_PAGE_MAX_TOKENS,
        encoder=None,
        owner: str = "",
        label: str = "PageMemory",
    ) -> None:
        self._store = page_store
        self._summarizer = summarizer
        self._max_input_tokens = max(0, int(max_input_tokens))
        self._page_max_tokens = max(256, int(page_max_tokens))
        self._encoder = encoder if encoder is not None else default_encoder()
        self._owner = owner or ""
        self._label = label
        self._original_user_content: Optional[str] = None
        self._dropped_windows = 0
        self._flush_round = 0

    # ---- ContextManager API -----------------------------------------------

    def reset(self) -> None:
        self._original_user_content = None
        self._dropped_windows = 0
        self._flush_round = 0

    def process(self, messages: list[dict], state) -> list[dict]:
        if not messages:
            return messages

        self._sync_first_user_message(messages)

        if self._max_input_tokens <= 0:
            return messages

        if count_messages(messages, self._encoder) <= self._max_input_tokens:
            return messages

        protected = self._keep_only_system_and_first_user(messages)
        window = self._extract_live_window_messages(messages)

        if window:
            pages = self._flush_window_to_memory(window)
            self._dropped_windows += 1
            owner_tag = f"[{self._owner}] " if self._owner else ""
            page_desc = ", ".join(str(p) for p in pages) if pages else "none"
            print(
                f"{owner_tag}[{self._label}] flushed live window "
                f"({len(window)} msgs) -> pages [{page_desc}]; cleared live context."
            )
        else:
            print(
                f"[{self._label}] threshold reached with empty live window; "
                "clearing anyway."
            )

        self._sync_first_user_message(protected)
        if count_messages(protected, self._encoder) > self._max_input_tokens:
            print(f"[{self._label}] Warning: still over budget after flush.")
        return protected

    # ---- finalization API (used by swarm/fugue) ---------------------------

    def flush_remaining(
        self,
        messages: list[dict],
        *,
        drop_final_assistant: bool = True,
    ) -> list[int]:
        """Flush any pending live-window messages to memory at finish time."""
        window = self._extract_live_window_messages(messages)
        if not window:
            return []
        if (
            drop_final_assistant
            and window
            and window[-1].get("role") == "assistant"
            and not _assistant_has_tool_call(window[-1])
        ):
            window = window[:-1]
        if not window:
            return []
        return self._flush_window_to_memory(window, label_prefix="Final Window")

    # ---- internals --------------------------------------------------------

    def _sync_first_user_message(self, messages: list[dict]) -> None:
        user_idx = find_first_user_idx(messages)
        if user_idx is None:
            return
        if self._original_user_content is None:
            self._original_user_content = stringify_content(messages[user_idx].get("content"))

        content = self._original_user_content or ""
        notice = _build_truncation_notice(self._dropped_windows)
        if notice:
            content += f"\n\n{TRUNCATION_NOTICE_TAG}\n{notice}"
        shared = self._store.render(owner=self._owner)
        if shared:
            content += MEMORY_SECTION_MARKER + shared
        messages[user_idx]["content"] = content

    @staticmethod
    def _keep_only_system_and_first_user(messages: list[dict]) -> list[dict]:
        user_idx = find_first_user_idx(messages)
        if user_idx is None:
            return list(messages)
        return list(messages[: user_idx + 1])

    @staticmethod
    def _extract_live_window_messages(messages: list[dict]) -> list[dict]:
        user_idx = find_first_user_idx(messages)
        if user_idx is None:
            return []
        return list(messages[user_idx + 1:])

    def _flush_window_to_memory(
        self,
        window_messages: list[dict],
        *,
        label_prefix: str = "Window",
    ) -> list[int]:
        self._flush_round += 1
        flush_id = self._flush_round
        turn_groups = _group_live_window_messages(window_messages)
        chunks = _build_turn_group_page_chunks(
            turn_groups,
            page_max_tokens=self._page_max_tokens,
            encoder=self._encoder,
        )
        if not chunks:
            return []

        page_payloads: list[dict] = []
        for chunk_idx, chunk in enumerate(chunks, start=1):
            raw = (
                f"### {label_prefix} {flush_id} / Chunk {chunk_idx}\n"
                f"{chunk.raw_content.strip() or '[Empty chunk]'}"
            )
            summary = self._summarizer.summarize_window(
                raw,
                question=self._original_user_content or "",
            )
            page_payloads.append(
                {
                    "summary": summary,
                    "raw_content": raw,
                    "token_count": count_text_tokens(raw, self._encoder),
                    "entry_count": chunk.entry_count,
                    "owner": self._owner,
                    "flush_round": flush_id,
                    "chunk_index": chunk_idx,
                }
            )
        return self._store.append(page_payloads)
