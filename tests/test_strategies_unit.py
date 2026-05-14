"""Unit tests for the five context-management strategies.

Each strategy is exercised directly on a synthetic message list — no live
LLM, deterministic, and fast. Covers:

  - the strategy fires when the prompt exceeds the soft budget
  - the strategy is a no-op when under budget
  - the resulting message list is structurally well-formed

The page_memory test additionally uses a stub PageStore + Summarizer so the
test doesn't touch the network.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _helpers import run_probes  # noqa: E402

from cabeza._budget import count_messages, default_encoder  # noqa: E402
from cabeza.base import RunState  # noqa: E402
from cabeza.context.discard_all import DiscardAllStrategy  # noqa: E402
from cabeza.context.discard_tool import DiscardToolStrategy, NATIVE_TRUNCATED  # noqa: E402
from cabeza.context.page_memory import PageMemoryStrategy  # noqa: E402
from cabeza.context.recent_k import RecentKStrategy  # noqa: E402


def _state() -> RunState:
    return RunState()


def _system_msg() -> dict:
    return {"role": "system", "content": "You are a helpful assistant."}


def _user_msg() -> dict:
    return {"role": "user", "content": "Original question: when was ICML first held?"}


def _tool_turn(idx: int, body_chars: int = 400) -> list[dict]:
    """A native-style tool turn: assistant tool_call + role=tool response."""
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"call_{idx}",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"query": ["x"]}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": f"call_{idx}",
            "name": "search",
            "content": "X" * body_chars,
        },
    ]


def _build_messages(n_turns: int = 4, body_chars: int = 400) -> list[dict]:
    msgs = [_system_msg(), _user_msg()]
    for i in range(n_turns):
        msgs.extend(_tool_turn(i, body_chars=body_chars))
    return msgs


def _tokens(messages: list[dict]) -> int:
    return count_messages(messages, default_encoder())


# ---------------------------------------------------------------------------
# DiscardAll
# ---------------------------------------------------------------------------


def probe_discard_all_under_budget() -> tuple[bool, str]:
    cm = DiscardAllStrategy(max_input_tokens=10_000)
    msgs = _build_messages(2, body_chars=100)
    out = cm.process(list(msgs), _state())
    ok = out == msgs  # no-op
    return ok, f"in={_tokens(msgs)} out={_tokens(out)} unchanged={ok}"


def probe_discard_all_over_budget() -> tuple[bool, str]:
    cm = DiscardAllStrategy(max_input_tokens=50)
    msgs = _build_messages(3, body_chars=400)
    out = cm.process(list(msgs), _state())
    roles = [m.get("role") for m in out]
    ok = roles == ["system", "user"] and "Original question" in out[1]["content"]
    return ok, f"in={_tokens(msgs)} out_roles={roles}"


# ---------------------------------------------------------------------------
# DiscardTool
# ---------------------------------------------------------------------------


def probe_discard_tool_under_budget() -> tuple[bool, str]:
    cm = DiscardToolStrategy(max_input_tokens=10_000, family="qwen")
    msgs = _build_messages(2, body_chars=50)
    out = cm.process(list(msgs), _state())
    ok = all(a == b for a, b in zip(out, msgs))
    return ok, f"in={_tokens(msgs)} out={_tokens(out)} unchanged={ok}"


def probe_discard_tool_over_budget() -> tuple[bool, str]:
    cm = DiscardToolStrategy(max_input_tokens=200, family="qwen")
    msgs = _build_messages(3, body_chars=400)
    out = cm.process(list(msgs), _state())
    # Tool messages from earlier turns should be replaced; latest preserved.
    tool_msgs = [m for m in out if m.get("role") == "tool"]
    truncated = [m for m in tool_msgs if m["content"] == NATIVE_TRUNCATED]
    kept = [m for m in tool_msgs if m["content"] != NATIVE_TRUNCATED]
    ok = len(truncated) >= 1 and len(kept) == 1
    return ok, f"truncated={len(truncated)} preserved={len(kept)} total_tool={len(tool_msgs)}"


# ---------------------------------------------------------------------------
# RecentK
# ---------------------------------------------------------------------------


def probe_recent_k_under_budget() -> tuple[bool, str]:
    cm = RecentKStrategy(max_input_tokens=10_000, keep_recent_turns=1, family="qwen")
    msgs = _build_messages(2, body_chars=50)
    out = cm.process(list(msgs), _state())
    ok = out == msgs
    return ok, f"in_msgs={len(msgs)} out_msgs={len(out)} unchanged={ok}"


def probe_recent_k_over_budget() -> tuple[bool, str]:
    cm = RecentKStrategy(max_input_tokens=200, keep_recent_turns=1, family="qwen")
    msgs = _build_messages(4, body_chars=400)
    out = cm.process(list(msgs), _state())
    assistants = sum(1 for m in out if m.get("role") == "assistant")
    tools = sum(1 for m in out if m.get("role") == "tool")
    # Each turn is one assistant+tool pair. keep_recent_turns=1 means exactly 1 pair.
    ok = assistants == 1 and tools == 1
    return ok, f"kept_assistants={assistants} kept_tools={tools} (expected 1/1)"


# ---------------------------------------------------------------------------
# PageMemory (synthetic store + summarizer; no network)
# ---------------------------------------------------------------------------


class _FakeStore:
    def __init__(self) -> None:
        self.pages: list[dict] = []
        self.calls = 0

    def append(self, payloads: list[dict]) -> list[int]:
        self.calls += 1
        nums: list[int] = []
        for p in payloads:
            n = len(self.pages) + 1
            self.pages.append({"page_num": n, "bullets": [p["summary"]]})
            nums.append(n)
        return nums

    def render(self, *, owner: str = "") -> str:
        if not self.pages:
            return ""
        return "\n\n".join(
            f"Page {p['page_num']}:\n- {p['bullets'][0]}" for p in self.pages
        )

    def snapshot(self) -> list[dict]:
        return list(self.pages)

    def __len__(self) -> int:
        return len(self.pages)


class _FakeSummarizer:
    def __init__(self) -> None:
        self.calls = 0

    def summarize_window(self, content: str, question: str = "") -> str:
        self.calls += 1
        return f"summary-{self.calls}"


def probe_page_memory_no_flush_under_budget() -> tuple[bool, str]:
    store = _FakeStore()
    cm = PageMemoryStrategy(
        page_store=store,
        summarizer=_FakeSummarizer(),
        max_input_tokens=10_000,
        page_max_tokens=2_000,
        owner="A",
    )
    msgs = _build_messages(2, body_chars=50)
    out = cm.process(list(msgs), _state())
    user_idx = next(i for i, m in enumerate(out) if m.get("role") == "user")
    # No flush should happen → no Exploration Memory marker in the user msg.
    ok = "Exploration Memory" not in out[user_idx]["content"] and len(store.pages) == 0
    return ok, f"pages={len(store.pages)} user_msg_has_memory={'Exploration Memory' in out[user_idx]['content']}"


def probe_page_memory_flush_on_overflow() -> tuple[bool, str]:
    store = _FakeStore()
    summarizer = _FakeSummarizer()
    cm = PageMemoryStrategy(
        page_store=store,
        summarizer=summarizer,
        max_input_tokens=200,
        page_max_tokens=2_000,
        owner="A",
    )
    msgs = _build_messages(3, body_chars=400)
    out = cm.process(list(msgs), _state())
    roles = [m.get("role") for m in out]
    user_idx = next(i for i, m in enumerate(out) if m.get("role") == "user")
    has_memory_section = "Exploration Memory" in out[user_idx]["content"]
    ok = (
        roles == ["system", "user"]
        and len(store.pages) >= 1
        and summarizer.calls >= 1
        and has_memory_section
    )
    return ok, (
        f"out_roles={roles} pages={len(store.pages)} "
        f"summarizer_calls={summarizer.calls} memory_in_user={has_memory_section}"
    )


def probe_page_memory_flush_remaining() -> tuple[bool, str]:
    """``flush_remaining`` writes leftover live messages as a 'Final Window'."""
    store = _FakeStore()
    cm = PageMemoryStrategy(
        page_store=store,
        summarizer=_FakeSummarizer(),
        max_input_tokens=100_000,
        page_max_tokens=2_000,
        owner="A",
    )
    msgs = _build_messages(2, body_chars=100)
    # Initialize internal state (records original_user_content):
    cm.process(list(msgs), _state())
    # Now call flush_remaining with pending live-window messages:
    flushed = cm.flush_remaining(msgs, drop_final_assistant=False)
    ok = len(flushed) >= 1 and len(store.pages) >= 1
    return ok, f"flushed_page_nums={flushed} store_pages={len(store.pages)}"


# ---------------------------------------------------------------------------
# Strategy registry sanity
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# RollingSummary (with a fake OpenAI-compatible client)
# ---------------------------------------------------------------------------


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs) -> _FakeResponse:
        self.calls += 1
        return _FakeResponse(f"SUMMARY-{self.calls}")


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.chat = _FakeChat()


def _summary_strategy(max_input_tokens: int):
    from cabeza.context.summary import RollingSummaryStrategy

    cm = RollingSummaryStrategy(
        model="m",
        api_key="EMPTY",
        base_url="http://localhost:9",
        max_input_tokens=max_input_tokens,
        request_timeout=5.0,
    )
    # Swap out the live OpenAI client for a fake so process() never touches the network.
    cm._client = _FakeOpenAIClient()
    return cm


def probe_summary_noop() -> tuple[bool, str]:
    cm = _summary_strategy(10_000)
    msgs = _build_messages(2, body_chars=50)
    out = cm.process(list(msgs), _state())
    # Under budget — should be a no-op (no summary section injected).
    user_idx = next(i for i, m in enumerate(out) if m.get("role") == "user")
    ok = (
        "Previous Context Summary" not in out[user_idx]["content"]
        and cm._client.chat.completions.calls == 0
    )
    return ok, f"summary_calls={cm._client.chat.completions.calls}"


def probe_summary_fires() -> tuple[bool, str]:
    cm = _summary_strategy(150)
    msgs = _build_messages(3, body_chars=400)
    out = cm.process(list(msgs), _state())
    user_idx = next(i for i, m in enumerate(out) if m.get("role") == "user")
    has_summary = "Previous Context Summary" in out[user_idx]["content"]
    has_token = "SUMMARY-1" in out[user_idx]["content"]
    ok = has_summary and has_token and cm._client.chat.completions.calls == 1
    return ok, (
        f"summary_calls={cm._client.chat.completions.calls} "
        f"has_section={has_summary} has_token={has_token}"
    )


def probe_registry_lookup() -> tuple[bool, str]:
    from cabeza.context import get_strategy, list_strategies

    expected = {"summary", "recent_k", "discard_tool", "discard_all", "page_memory"}
    actual = set(list_strategies())
    missing = expected - actual
    factories = {name: get_strategy(name) for name in expected}
    ok = not missing and all(callable(f) for f in factories.values())
    return ok, f"missing={missing} factories_callable={all(callable(f) for f in factories.values())}"


PROBES = {
    "discard_all_noop": probe_discard_all_under_budget,
    "discard_all_fires": probe_discard_all_over_budget,
    "discard_tool_noop": probe_discard_tool_under_budget,
    "discard_tool_fires": probe_discard_tool_over_budget,
    "recent_k_noop": probe_recent_k_under_budget,
    "recent_k_fires": probe_recent_k_over_budget,
    "summary_noop": probe_summary_noop,
    "summary_fires": probe_summary_fires,
    "page_memory_noop": probe_page_memory_no_flush_under_budget,
    "page_memory_fires": probe_page_memory_flush_on_overflow,
    "page_memory_final_flush": probe_page_memory_flush_remaining,
    "registry": probe_registry_lookup,
}


def main(argv: list[str]) -> int:
    return run_probes(PROBES, argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
