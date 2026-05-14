"""Exercise each of cabeza's context-management strategies end-to-end.

For each strategy, build an Agent driving the Serper SearchTool with a very
low soft budget so the strategy actually fires within a few rounds, then
verify both that the agent produces an answer and that the strategy did
something observable (compressed history, dropped messages, flushed pages,
or rewrote the first user message).
"""

from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _helpers import (  # noqa: E402
    DEEPSEEK_KEY,
    SERPER,
    memory_config,
    qwen_agent_kwargs,
    run_probes,
    search_tool,
    short,
)


QUESTION = (
    "Use the search tool to investigate the history of the ICML conference. "
    "Search at least three different queries (different angles, different "
    "phrasings). Then answer this single question: in what year was the FIRST "
    "ICML conference held? Reply with just the 4-digit year."
)


def _make_agent(strategy: str, **extra):
    from cabeza import Agent

    kw = qwen_agent_kwargs(
        context_management=strategy,
        context_management_tokens=3_000,
        max_steps=5,
    )
    kw.update(extra)
    return Agent(tools=[search_tool()], **kw)


def _run_and_capture(agent, prompt: str) -> tuple[str, str, int]:
    """Run the agent, capture stdout, return (final, log, message_count)."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = agent.run_verbose(prompt)
    log = buf.getvalue()
    sys.stdout.write(log)
    return result.prediction, log, len(result.messages)


# ---- probes ---------------------------------------------------------------


def probe_no_strategy() -> tuple[bool, str]:
    """Sanity baseline: no context management at all."""
    from cabeza import Agent

    kw = qwen_agent_kwargs(context_management=None, max_steps=4)
    agent = Agent(tools=[search_tool()], **kw)
    final, log, n = _run_and_capture(agent, QUESTION)
    ok = bool(final.strip()) and "[ContextMgmt]" not in log
    return ok, f"final={short(final, 100)} | messages={n}"


def probe_summary() -> tuple[bool, str]:
    """Rolling-summary strategy — uses DeepSeek as the summarizer."""
    if not DEEPSEEK_KEY:
        return False, "DEEPSEEK_API_KEY missing"
    agent = _make_agent("summary")
    final, log, _ = _run_and_capture(agent, QUESTION)
    fired = "[ContextSummary]" in log and "Collapsed history" in log
    ok = bool(final.strip()) and fired
    return ok, f"final={short(final, 80)} fired={fired}"


def probe_recent_k() -> tuple[bool, str]:
    """Recent-k tool-turn trimming — purely structural, no aux LLM."""
    agent = _make_agent("recent_k", keep_recent_turns=1)
    final, log, _ = _run_and_capture(agent, QUESTION)
    fired = "[RecentK]" in log and "kept" in log
    ok = bool(final.strip()) and fired
    return ok, f"final={short(final, 80)} fired={fired}"


def probe_discard_tool() -> tuple[bool, str]:
    """Discard old tool responses — purely structural."""
    agent = _make_agent("discard_tool")
    final, log, _ = _run_and_capture(agent, QUESTION)
    fired = "[DiscardTool]" in log
    ok = bool(final.strip()) and fired
    return ok, f"final={short(final, 80)} fired={fired}"


def probe_discard_all() -> tuple[bool, str]:
    """Discard everything except system + first user."""
    agent = _make_agent("discard_all")
    final, log, _ = _run_and_capture(agent, QUESTION)
    fired = "[DiscardAll]" in log and "threshold reached" in log
    ok = bool(final.strip()) and fired
    return ok, f"final={short(final, 80)} fired={fired}"


def probe_page_memory() -> tuple[bool, str]:
    """Page-memory strategy — DeepSeek-backed summarizer + memory tool."""
    if not DEEPSEEK_KEY:
        return False, "DEEPSEEK_API_KEY missing"
    agent = _make_agent("page_memory", memory_config=memory_config())
    final, log, _ = _run_and_capture(agent, QUESTION)
    fired = "[PageMemory]" in log and "flushed live window" in log
    ok = bool(final.strip()) and fired
    return ok, f"final={short(final, 80)} fired={fired}"


PROBES = {
    "no_strategy": probe_no_strategy,
    "summary": probe_summary,
    "recent_k": probe_recent_k,
    "discard_tool": probe_discard_tool,
    "discard_all": probe_discard_all,
    "page_memory": probe_page_memory,
}


def main(argv: list[str]) -> int:
    if not SERPER:
        print("[!] SERPER_API_KEY missing in cabeza/.env — strategy tests cannot run.")
        return 2
    return run_probes(PROBES, argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
