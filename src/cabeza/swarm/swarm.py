"""Dynamic swarm orchestrator (OpenAI Swarm style).

The meta-agent is given two tools:

- ``create_subagent(name, system_prompt)`` — register a new specialized subagent.
- ``assign_task(agent, prompt)``           — dispatch a prompt to a registered subagent.

Multiple ``assign_task`` calls in one assistant turn run in parallel.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Optional

from cabeza.tools.base import BaseTool

from cabeza.swarm._common import (
    AgentSpec,
    clone_agent_for_spec,
    extract_run_result,
    parse_agent_specs,
)
from cabeza.swarm._parallel import install_parallel_tool_executor
from cabeza.types import RunResult

if TYPE_CHECKING:
    from cabeza.agent import Agent


MAX_SUBAGENTS = 8


CREATE_SUBAGENT_PARAMETERS = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "Unique name for this agent configuration"},
        "system_prompt": {
            "type": "string",
            "description": "System prompt defining the agent's role, capabilities, and boundaries",
        },
    },
    "required": ["name", "system_prompt"],
}

ASSIGN_TASK_PARAMETERS = {
    "type": "object",
    "properties": {
        "agent": {"type": "string", "description": "Specify which created agent to use."},
        "prompt": {"type": "string", "description": "The task for the agent to perform"},
    },
    "required": ["agent", "prompt"],
}


@dataclass
class _CreatedSubagent:
    name: str
    system_prompt: str
    subagent_id: int


class _SwarmRuntime:
    def __init__(
        self,
        *,
        parent: "Agent",
        sub_specs: list[AgentSpec],
        question: str,
        max_subagents: int,
    ) -> None:
        self._parent = parent
        self._sub_specs = sub_specs
        self._question = question
        self._max_subagents = max(1, min(int(max_subagents), MAX_SUBAGENTS))
        self._lock = threading.Lock()
        self._created: dict[str, _CreatedSubagent] = {}
        self._assignment_counter = 0
        self.assignment_results: list[dict] = []

    def create_subagent(self, *, name: str, system_prompt: str) -> str:
        name = (name or "").strip()
        if not name:
            return "[create_subagent] Error: name is required."
        prompt = (system_prompt or "").strip()
        if not prompt:
            return "[create_subagent] Error: system_prompt is required."
        with self._lock:
            if name in self._created:
                self._created[name].system_prompt = prompt
                return f"[create_subagent] Updated existing subagent '{name}'."
            if len(self._created) >= self._max_subagents:
                return (
                    f"[create_subagent] Error: maximum of {self._max_subagents} subagents reached."
                )
            sa = _CreatedSubagent(name=name, system_prompt=prompt, subagent_id=len(self._created))
            self._created[name] = sa
        return f"[create_subagent] Created '{name}' ({sa.subagent_id + 1}/{self._max_subagents})."

    def assign_task(self, *, agent: str, prompt: str) -> str:
        name = (agent or "").strip()
        task = (prompt or "").strip()
        if not name:
            return "[assign_task] Error: agent is required."
        if not task:
            return "[assign_task] Error: prompt is required."
        with self._lock:
            sa = self._created.get(name)
            if sa is None:
                available = ", ".join(sorted(self._created)) or "(none)"
                return f"[assign_task] Error: unknown subagent '{name}'. Available: {available}."
            assignment_id = self._assignment_counter
            self._assignment_counter += 1

        result = self._run_subagent(sa, prompt=task, assignment_id=assignment_id)
        with self._lock:
            self.assignment_results.append(result)
        report = result.get("prediction", "") or result.get("error", "") or "No report."
        return f"[assign_task] {sa.name} completed assignment {assignment_id}.\n\n{report}"

    def created_subagents_payload(self) -> list[dict]:
        with self._lock:
            return [
                {"subagent_id": sa.subagent_id, "name": sa.name, "system_prompt": sa.system_prompt}
                for sa in sorted(self._created.values(), key=lambda s: s.subagent_id)
            ]

    def _run_subagent(self, sa: _CreatedSubagent, *, prompt: str, assignment_id: int) -> dict:
        started = time.time()
        spec = self._sub_specs[sa.subagent_id % len(self._sub_specs)]
        member = clone_agent_for_spec(self._parent, spec)
        cfg = self._parent.team_config
        if cfg is not None:
            if cfg.sub_max_steps:
                member.max_steps = cfg.sub_max_steps
            if cfg.sub_context_window_tokens:
                member.context_window_tokens = cfg.sub_context_window_tokens

        bundle = member.build_single_agent(owner=sa.name)
        base_system = bundle.agent.system_prompt
        bundle.agent.system_prompt = (
            f"{base_system}\n\n# Specialized Subagent Role\n{sa.system_prompt}"
        ).strip()

        user_prompt = f"""
You are subagent '{sa.name}'.

Original question:
{self._question}

Orchestrator assignment:
{prompt}

Use the tools available to you. Return one concise message to the orchestrator
with key findings, evidence, useful URLs or calculations, a candidate answer
if justified, and any uncertainty.
""".strip()

        error = ""
        try:
            bundle.agent.run(user_prompt)
            prediction, termination = _extract_final_content(bundle.agent)
        except Exception as exc:  # noqa: BLE001
            prediction = ""
            termination = f"error: {exc}"
            error = str(exc)

        elapsed = time.time() - started
        result = {
            "subagent_id": sa.subagent_id,
            "assignment_id": assignment_id,
            "name": sa.name,
            "spec": asdict(spec),
            "system_prompt": sa.system_prompt,
            "assigned_task": {"agent": sa.name, "prompt": prompt},
            "prediction": prediction,
            "termination": termination,
            "rounds": bundle.agent.state.step,
            "tool_call_rounds": bundle.agent.state.tool_rounds,
            "total_process_time": elapsed,
        }
        if error:
            result["error"] = error
        return result


def _extract_final_content(gizmo_agent) -> tuple[str, str]:
    if gizmo_agent.trajectory:
        last = gizmo_agent.trajectory[-1]
        if getattr(last, "final_content", ""):
            return last.final_content, "answer (final assistant response)"
    return "No answer found.", "no answer found"


class _CreateSubagentTool(BaseTool):
    def __init__(self, runtime: _SwarmRuntime) -> None:
        super().__init__(
            name="create_subagent",
            description="Create a custom subagent with a specific system prompt and reusable name.",
            parameters=CREATE_SUBAGENT_PARAMETERS,
        )
        self._runtime = runtime

    def execute(self, name: str, system_prompt: str) -> str:
        return self._runtime.create_subagent(name=name, system_prompt=system_prompt)


class _AssignTaskTool(BaseTool):
    def __init__(self, runtime: _SwarmRuntime) -> None:
        super().__init__(
            name="assign_task",
            description=(
                "Launch a new agent.\nUsage notes:\n"
                "1. You can launch multiple agents concurrently in the same response, to maximize performance.\n"
                "2. When the agent is done, it will return a single message back to you."
            ),
            parameters=ASSIGN_TASK_PARAMETERS,
        )
        self._runtime = runtime

    def execute(self, agent: str, prompt: str) -> str:
        return self._runtime.assign_task(agent=agent, prompt=prompt)


def _resolve_meta_spec(agent: "Agent") -> AgentSpec:
    cfg = agent.team_config
    return AgentSpec(
        index=-1,
        name="Meta",
        family=(cfg.meta_family if cfg and cfg.meta_family else agent.family).lower(),
        model=cfg.meta_model if cfg and cfg.meta_model else agent.model,
        base_url=cfg.meta_base_url if cfg and cfg.meta_base_url else agent.base_url,
        api_key=cfg.meta_api_key if cfg and cfg.meta_api_key else agent.api_key,
    )


def _meta_system_prompt(question: str, max_subagents: int) -> str:
    return f"""
You are the orchestrator for a dynamic swarm. You may create and schedule up to
{max_subagents} reusable specialized subagents.

Tools:
- create_subagent(name, system_prompt): create a custom subagent with a unique reusable name.
- assign_task(agent, prompt): dispatch work to a created subagent. You may call assign_task
  multiple times in the same response; independent assignments will run concurrently.

Policy:
- Create no more than {max_subagents} subagents.
- Give each subagent a focused role and clear boundaries.
- Parallelize independent assignments whenever useful.
- After subagent reports return, synthesize the final answer yourself.

Original question:
{question}
""".strip()


def _meta_initial_prompt(question: str, max_subagents: int) -> str:
    return f"""
Solve the question using the swarm.

First create up to {max_subagents} useful subagents with create_subagent. Then use
assign_task to dispatch independent investigations — launch multiple concurrently when possible.
Finally, integrate all returned reports into the final answer.

Original question:
{question}
""".strip()


def run_swarm(agent: "Agent", *, question: str) -> RunResult:
    cfg = agent.team_config
    max_sub = max(1, min(int((cfg.size if cfg else 3) or 3), MAX_SUBAGENTS))
    sub_specs = parse_agent_specs(agent)

    meta_spec = _resolve_meta_spec(agent)
    meta_agent = clone_agent_for_spec(agent, meta_spec)
    if cfg is not None:
        meta_agent.max_steps = cfg.meta_max_steps or meta_agent.max_steps
        if cfg.meta_context_window_tokens:
            meta_agent.context_window_tokens = cfg.meta_context_window_tokens

    runtime = _SwarmRuntime(
        parent=agent,
        sub_specs=sub_specs,
        question=question,
        max_subagents=max_sub,
    )
    bundle = meta_agent.build_single_agent(
        owner="Meta",
        extra_tools=[_CreateSubagentTool(runtime), _AssignTaskTool(runtime)],
    )
    bundle.agent.system_prompt = _meta_system_prompt(question, max_sub)
    install_parallel_tool_executor(
        bundle.agent,
        tool_name="assign_task",
        max_workers=max(1, min(int((cfg.subagent_workers if cfg else max_sub) or max_sub), max_sub)),
    )

    started = time.time()
    bundle.agent.run(_meta_initial_prompt(question, max_sub))
    elapsed = time.time() - started

    extras = {
        "team_mode": "swarm",
        "created_subagents": runtime.created_subagents_payload(),
        "subagent_results": sorted(
            runtime.assignment_results,
            key=lambda x: int(x.get("assignment_id", 0)),
        ),
        "meta_spec": asdict(meta_spec),
    }
    return extract_run_result(
        question=question,
        gizmo_agent=bundle.agent,
        started_elapsed=elapsed,
        extras=extras,
    )
