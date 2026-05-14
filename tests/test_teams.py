"""Live tests for the three team / swarm orchestrators.

All three modes run on DeepSeek (fast + tool-call reliable). ``fugue`` uses
DeepSeek as both the peer-agent model and the auxiliary memory summarizer.
Tests verify only that the team orchestrator runs to completion, produces a
non-empty prediction, and surfaces the expected ``extras`` payload.
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


def _deepseek_base_kwargs() -> dict:
    return dict(
        model=DEEPSEEK_MODEL,
        base_url=DEEPSEEK_URL,
        api_key=DEEPSEEK_KEY,
        family="deepseek",
        max_steps=6,
        max_time_seconds=180,
        context_window_tokens=32_000,
        temperature=0.0,
        timeout=90.0,
        enable_thinking=False,
    )


def _capture_run(agent, prompt: str):
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = agent.run_verbose(prompt)
    log = buf.getvalue()
    sys.stdout.write(log)
    return result, log


# ---------------------------------------------------------------------------
# naive: decompose → parallel subagents → synthesize
# ---------------------------------------------------------------------------


def probe_naive() -> tuple[bool, str]:
    """Two subagents, no tools — pure LLM reasoning, fast."""
    if not DEEPSEEK_KEY:
        return False, "DEEPSEEK_API_KEY missing"
    from cabeza import Agent
    from cabeza.agent import TeamConfig

    agent = Agent(
        team="naive",
        team_config=TeamConfig(mode="naive", size=2, meta_max_steps=4, sub_max_steps=3),
        tools=[],
        **_deepseek_base_kwargs(),
    )
    result, _ = _capture_run(
        agent,
        "What are the two most distinct meanings of the Spanish word 'cabeza'? "
        "Answer in one short sentence.",
    )
    extras = result.extras or {}
    sub_results = extras.get("subagent_results") or []
    ok = (
        bool(result.prediction.strip())
        and extras.get("team_mode") == "naive"
        and isinstance(extras.get("decomposition"), list)
        and len(extras["decomposition"]) == 2
        and len(sub_results) == 2
    )
    return ok, (
        f"team_mode={extras.get('team_mode')} "
        f"decomp={len(extras.get('decomposition', []))} "
        f"subagents={len(sub_results)} "
        f"final={short(result.prediction, 80)}"
    )


# ---------------------------------------------------------------------------
# swarm: dynamic create_subagent + assign_task
# ---------------------------------------------------------------------------


def probe_swarm() -> tuple[bool, str]:
    """Meta with create_subagent + assign_task, single dispatch."""
    if not DEEPSEEK_KEY:
        return False, "DEEPSEEK_API_KEY missing"
    from cabeza import Agent
    from cabeza.agent import TeamConfig

    agent = Agent(
        team="swarm",
        team_config=TeamConfig(
            mode="swarm",
            size=2,
            meta_max_steps=6,
            sub_max_steps=3,
            subagent_workers=2,
        ),
        tools=[],
        **_deepseek_base_kwargs(),
    )
    result, _ = _capture_run(
        agent,
        "Create one specialized subagent named 'definer' whose role is to "
        "define Spanish words. Then assign it the task: define the word "
        "'cabeza' in one short sentence. Report the subagent's answer.",
    )
    extras = result.extras or {}
    created = extras.get("created_subagents") or []
    sub_results = extras.get("subagent_results") or []
    ok = (
        bool(result.prediction.strip())
        and extras.get("team_mode") == "swarm"
        and len(created) >= 1
        and len(sub_results) >= 1
    )
    return ok, (
        f"team_mode={extras.get('team_mode')} "
        f"created={len(created)} subagent_results={len(sub_results)} "
        f"final={short(result.prediction, 80)}"
    )


# ---------------------------------------------------------------------------
# fugue: peer team + shared page memory
# ---------------------------------------------------------------------------


def probe_fugue() -> tuple[bool, str]:
    """Two DeepSeek peers sharing a page-memory store."""
    if not DEEPSEEK_KEY or not SERPER:
        return False, "DEEPSEEK_API_KEY and SERPER_API_KEY required"
    from cabeza import Agent
    from cabeza.agent import TeamConfig

    agent = Agent(
        team="fugue",
        team_config=TeamConfig(mode="fugue", size=2),
        tools=[search_tool()],
        context_management="page_memory",
        context_management_tokens=4_000,
        memory_config=memory_config(),
        **_deepseek_base_kwargs(),
    )
    result, _ = _capture_run(
        agent,
        "Use the search tool to confirm: is 'cabasa' a percussion instrument? "
        "Answer in one short sentence.",
    )
    extras = result.extras or {}
    agent_results = extras.get("agent_results") or []
    ok = (
        bool(result.prediction.strip())
        and extras.get("team_mode") == "fugue"
        and len(agent_results) == 2
    )
    return ok, (
        f"team_mode={extras.get('team_mode')} "
        f"agents={len(agent_results)} "
        f"shared_pages={extras.get('shared_memory_page_count', 0)} "
        f"final={short(result.prediction, 80)}"
    )


PROBES = {
    "naive": probe_naive,
    "swarm": probe_swarm,
    "fugue": probe_fugue,
}


def main(argv: list[str]) -> int:
    if not DEEPSEEK_KEY:
        print("[!] DEEPSEEK_API_KEY required.")
        return 2
    return run_probes(PROBES, argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
