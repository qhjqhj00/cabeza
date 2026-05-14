"""End-to-end live smoke tests for each context-management strategy.

These are deliberately small: they run a 1–3 round agent loop where the soft
budget is set so small that the strategy *must* fire after the first tool
turn. The main agent uses DeepSeek (much faster + more reliable on simple
prompts than the local Qwen vLLM) and the auxiliary memory model is also
DeepSeek so the whole loop completes in seconds.

The Agent factory's ``family="deepseek"`` adapter requires explicit model and
base_url, both of which we pass from .env.

These are not correctness tests — they're wiring tests. We verify only:

  - the agent completes without raising,
  - a non-empty final answer is produced (or the harness fallback string),
  - the strategy log marker shows up in stdout.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _helpers import (  # noqa: E402
    DEEPSEEK_KEY,
    DEEPSEEK_MODEL,
    DEEPSEEK_URL,
    SERPER,
    memory_config,
    run_probes,
    search_tool,
    short,
)


QUESTION = (
    "Use the search tool to look up 'cabeza' (the Spanish word for head) and "
    "the percussion instrument 'cabasa'. Then in one sentence, say which one "
    "is the musical instrument."
)


def _agent(strategy: str, **extra):
    """DeepSeek-driven Agent with a tiny soft budget that forces ``strategy`` to fire."""
    from cabeza import Agent

    if not DEEPSEEK_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY missing")
    kw = dict(
        model=DEEPSEEK_MODEL,
        base_url=DEEPSEEK_URL,
        api_key=DEEPSEEK_KEY,
        family="deepseek",
        tools=[search_tool()],
        max_steps=4,
        max_time_seconds=120,
        context_window_tokens=32_000,
        context_management=strategy,
        # Very small budget so the strategy *must* fire after the first tool turn.
        context_management_tokens=800,
        temperature=0.0,
        timeout=90.0,
        enable_thinking=False,
        # If the strategy uses an LLM (summary/page_memory), point it at DeepSeek too.
        summary_model=DEEPSEEK_MODEL,
        summary_base_url=DEEPSEEK_URL,
        summary_api_key=DEEPSEEK_KEY,
    )
    kw.update(extra)
    return Agent(**kw)


def _run(agent, prompt: str) -> tuple[str, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = agent.run_verbose(prompt)
    log = buf.getvalue()
    sys.stdout.write(log)
    return result.prediction, log


def probe_recent_k() -> tuple[bool, str]:
    agent = _agent("recent_k", keep_recent_turns=1)
    final, log = _run(agent, QUESTION)
    fired = "[RecentK]" in log
    ok = bool(final.strip()) and fired
    return ok, f"fired={fired} final={short(final, 100)}"


def probe_discard_tool() -> tuple[bool, str]:
    agent = _agent("discard_tool")
    final, log = _run(agent, QUESTION)
    fired = "[DiscardTool]" in log
    ok = bool(final.strip()) and fired
    return ok, f"fired={fired} final={short(final, 100)}"


def probe_discard_all() -> tuple[bool, str]:
    agent = _agent("discard_all")
    final, log = _run(agent, QUESTION)
    fired = "[DiscardAll]" in log
    ok = bool(final.strip()) and fired
    return ok, f"fired={fired} final={short(final, 100)}"


def probe_summary() -> tuple[bool, str]:
    agent = _agent("summary")
    final, log = _run(agent, QUESTION)
    fired = "[ContextSummary]" in log
    ok = bool(final.strip()) and fired
    return ok, f"fired={fired} final={short(final, 100)}"


def probe_page_memory() -> tuple[bool, str]:
    agent = _agent("page_memory", memory_config=memory_config())
    final, log = _run(agent, QUESTION)
    fired = "[PageMemory]" in log
    ok = bool(final.strip()) and fired
    return ok, f"fired={fired} final={short(final, 100)}"


PROBES = {
    "recent_k": probe_recent_k,
    "discard_tool": probe_discard_tool,
    "discard_all": probe_discard_all,
    "summary": probe_summary,
    "page_memory": probe_page_memory,
}


def main(argv: list[str]) -> int:
    if not SERPER or not DEEPSEEK_KEY:
        print("[!] SERPER_API_KEY and DEEPSEEK_API_KEY required in cabeza/.env.")
        return 2
    return run_probes(PROBES, argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
