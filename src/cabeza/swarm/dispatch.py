"""Single entry point that dispatches to the right swarm orchestrator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from cabeza.types import RunResult

if TYPE_CHECKING:
    from cabeza.agent import Agent


def run_team(
    agent: "Agent",
    *,
    question: str,
    owner: str = "agent",
    trajectory_log_path: Optional[str] = None,
    memory_log_path: Optional[str] = None,
) -> RunResult:
    mode = (agent.team_mode or "").strip().lower()
    if mode == "naive":
        from cabeza.swarm.naive import run_naive

        return run_naive(agent, question=question)
    if mode == "swarm":
        from cabeza.swarm.swarm import run_swarm

        return run_swarm(agent, question=question)
    if mode == "fugue":
        from cabeza.swarm.fugue import run_fugue

        return run_fugue(agent, question=question)
    raise ValueError(
        f"Unknown team mode {mode!r}. Expected 'naive', 'swarm', or 'fugue'."
    )
