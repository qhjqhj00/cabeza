"""Swarm / multi-agent orchestrators.

Three modes:

- ``naive``  — meta decomposes the question into N subproblems, dispatches them
               to parallel subagents, then synthesizes the final answer.
- ``swarm``  — meta is given ``create_subagent`` and ``assign_task`` tools and
               dynamically builds + dispatches subagents (OpenAI Swarm style).
- ``fugue``  — N peer agents work on the same question concurrently, sharing
               a page-memory store; no central meta.
"""

from cabeza.swarm.naive import run_naive
from cabeza.swarm.swarm import run_swarm
from cabeza.swarm.fugue import run_fugue
from cabeza.swarm.dispatch import run_team

__all__ = ["run_naive", "run_swarm", "run_fugue", "run_team"]
