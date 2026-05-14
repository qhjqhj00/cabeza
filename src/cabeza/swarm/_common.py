"""Shared helpers used by all swarm orchestrators."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from cabeza._budget import default_encoder
from cabeza.types import RunResult, Step, ToolCall

if TYPE_CHECKING:
    from cabeza.agent import Agent, TeamConfig


@dataclass(frozen=True)
class AgentSpec:
    """One concrete agent slot in a swarm."""

    index: int
    name: str
    family: str
    model: str
    base_url: str
    api_key: str


def _expand(values: list[str], default: str, n: int, field: str) -> list[str]:
    cleaned = [v.strip() for v in values if v and v.strip()]
    if not cleaned:
        cleaned = [str(default or "").strip()]
    if len(cleaned) == 1:
        return cleaned * n
    if len(cleaned) != n:
        raise ValueError(
            f"{field} must provide either 1 value or exactly {n}; got {len(cleaned)}."
        )
    return cleaned


def parse_agent_specs(agent: "Agent") -> list[AgentSpec]:
    """Expand TeamConfig per-agent CSVs into concrete ``AgentSpec`` slots."""
    cfg = agent.team_config
    if cfg is None:
        # Single homogeneous team made from the main agent's config.
        size = 3
        families = [agent.family] * size
        models = [agent.model] * size
        urls = [agent.base_url] * size
        keys = [agent.api_key] * size
    else:
        size = max(1, int(cfg.size or 1))
        families = _expand(cfg.families, agent.family, size, "families")
        models = _expand(cfg.models, agent.model, size, "models")
        urls = _expand(cfg.base_urls, agent.base_url, size, "base_urls")
        keys = _expand(cfg.api_keys, agent.api_key, size, "api_keys")

    out: list[AgentSpec] = []
    for i in range(size):
        out.append(
            AgentSpec(
                index=i,
                name=f"Agent-{i}",
                family=families[i].lower(),
                model=models[i],
                base_url=urls[i],
                api_key=keys[i] or "EMPTY",
            )
        )
    return out


def clone_agent_for_spec(agent: "Agent", spec: AgentSpec, *, team: Optional["TeamConfig"] = None) -> "Agent":
    """Return a shallow-copied ``Agent`` rebound to ``spec``.

    Memory/context/tool configuration is preserved so each member of the swarm
    behaves the same as a standalone ``Agent`` of the parent shape.
    """
    new = copy.copy(agent)
    new.family = spec.family
    new.model = spec.model
    new.base_url = spec.base_url
    new.api_key = spec.api_key
    new.team_mode = None
    new.team_config = team
    return new


def build_summarizer_from_memory_config(agent: "Agent"):
    """Convenience accessor for the agent's auxiliary summarizer."""
    return agent._build_summarizer()  # noqa: SLF001 (internal helper used in same package)


def extract_run_result(
    *,
    question: str,
    gizmo_agent,
    started_elapsed: float,
    extras: Optional[dict] = None,
) -> RunResult:
    """Convert a finished family agent into a cabeza ``RunResult``."""
    prediction = "No answer found."
    termination = "no answer found"
    if gizmo_agent.trajectory:
        last = gizmo_agent.trajectory[-1]
        if last.final_content:
            prediction = last.final_content
            termination = "answer (final assistant response)"

    trajectory = [
        Step(
            step=s.step,
            reasoning=s.reasoning,
            final_content=s.final_content,
            tool_calls=[
                ToolCall(name=tc.name, args=tc.args, result=tc.result)
                for tc in s.tool_calls
            ],
        )
        for s in gizmo_agent.trajectory
    ]
    messages = [
        {"role": "system", "content": gizmo_agent.system_prompt},
        *gizmo_agent.messages,
    ]
    return RunResult(
        question=question,
        prediction=prediction,
        final_content=prediction,
        termination=termination,
        total_rounds=gizmo_agent.state.step,
        tool_call_rounds=gizmo_agent.state.tool_rounds,
        total_process_time=started_elapsed,
        messages=messages,
        trajectory=trajectory,
        extras=dict(extras or {}),
    )


def encoder():
    return default_encoder()
