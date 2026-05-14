"""Fugue swarm orchestrator.

N peer agents work concurrently on the same question, sharing a single
``SharedPageStore``. Each agent uses page-memory context management against
the shared store so teammates can see each other's exploration summaries and
recall raw pages via the ``memory`` tool.
"""

from __future__ import annotations

import concurrent.futures
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Optional

from cabeza.memory.page_store import SharedPageStore
from cabeza.memory.summarizer import Summarizer
from cabeza.swarm._common import (
    AgentSpec,
    clone_agent_for_spec,
    parse_agent_specs,
)
from cabeza.types import RunResult, Step, ToolCall

if TYPE_CHECKING:
    from cabeza.agent import Agent


def _build_summarizer_for(member: "Agent") -> Summarizer:
    # Use the member's configured memory model (falls back to its base model).
    cfg = member.memory_config
    return Summarizer(
        model=cfg.summarizer_model or member.model,
        base_url=cfg.summarizer_base_url or member.base_url,
        api_key=cfg.summarizer_api_key or member.api_key,
        temperature=cfg.summarizer_temperature,
        max_response_tokens=cfg.summarizer_max_response_tokens,
        request_timeout_s=member.llm_config.timeout,
    )


def _run_member(
    parent: "Agent",
    spec: AgentSpec,
    *,
    question: str,
    shared_store: SharedPageStore,
) -> dict:
    member = clone_agent_for_spec(parent, spec)
    # Force page-memory mode on the member; the shared store is supplied so
    # all members write into the same place.
    member.memory_mode = "page"
    if member.context_management not in {"page_memory", None}:
        # If a user explicitly chose a different strategy, respect it; otherwise
        # default to page_memory for the team.
        pass
    else:
        member.context_management = "page_memory"

    member_summarizer = _build_summarizer_for(member)

    started = time.time()
    result: RunResult
    try:
        result = member.run_verbose(
            question=question,
            owner=spec.name,
            shared_page_store=shared_store,
            shared_summarizer=member_summarizer,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = time.time() - started
        return {
            "agent_id": spec.index,
            "name": spec.name,
            "spec": asdict(spec),
            "prediction": "",
            "termination": f"error: {exc}",
            "rounds": 0,
            "tool_call_rounds": 0,
            "total_process_time": elapsed,
            "error": str(exc),
        }

    return {
        "agent_id": spec.index,
        "name": spec.name,
        "spec": asdict(spec),
        "prediction": result.prediction,
        "termination": result.termination,
        "rounds": result.total_rounds,
        "tool_call_rounds": result.tool_call_rounds,
        "total_process_time": result.total_process_time,
        "messages": result.messages,
        "trajectory": [_step_to_dict(s) for s in result.trajectory],
    }


def _step_to_dict(step: Step) -> dict:
    return {
        "step": step.step,
        "reasoning": step.reasoning,
        "final_content": step.final_content,
        "tool_calls": [
            {"name": tc.name, "args": tc.args, "result": tc.result}
            for tc in step.tool_calls
        ],
    }


def _select_final_answer(agent_results: list[dict]) -> tuple[str, str, int]:
    """Pick a final prediction from the team.

    Strategy: prefer the first member that produced a non-empty final answer
    (sorted by agent_id for determinism). Returns ``(prediction, termination, picked_id)``.
    """
    sorted_results = sorted(agent_results, key=lambda r: int(r.get("agent_id", 0)))
    for r in sorted_results:
        prediction = (r.get("prediction") or "").strip()
        if prediction and prediction != "No answer found.":
            return prediction, r.get("termination", ""), int(r.get("agent_id", 0))
    if sorted_results:
        r = sorted_results[0]
        return r.get("prediction", "No answer found."), r.get("termination", "no answer found"), int(r.get("agent_id", 0))
    return "No answer found.", "no answer found", -1


def run_fugue(agent: "Agent", *, question: str) -> RunResult:
    cfg = agent.team_config
    specs = parse_agent_specs(agent)

    shared_store = SharedPageStore(
        max_pages_per_owner=agent.memory_config.max_pages_per_owner,
    )

    started = time.time()
    agent_results: list[Optional[dict]] = [None] * len(specs)

    with ThreadPoolExecutor(max_workers=len(specs), thread_name_prefix="cabeza-fugue") as executor:
        futures: dict[concurrent.futures.Future, int] = {}
        for spec in specs:
            f = executor.submit(_run_member, agent, spec, question=question, shared_store=shared_store)
            futures[f] = spec.index
        for f in concurrent.futures.as_completed(futures):
            idx = futures[f]
            try:
                agent_results[idx] = f.result()
            except Exception as exc:  # noqa: BLE001
                agent_results[idx] = {
                    "agent_id": idx,
                    "name": f"Agent-{idx}",
                    "prediction": "",
                    "termination": f"error: {exc}",
                    "error": str(exc),
                }

    elapsed = time.time() - started
    prediction, termination, picked_id = _select_final_answer([r for r in agent_results if r])

    extras: dict[str, Any] = {
        "team_mode": "fugue",
        "agent_results": agent_results,
        "shared_memory_pages": shared_store.snapshot(),
        "shared_memory_page_count": len(shared_store),
        "picked_agent_id": picked_id,
    }
    # Build a minimal cabeza RunResult; messages/trajectory are folded under extras.
    return RunResult(
        question=question,
        prediction=prediction,
        final_content=prediction,
        termination=termination,
        total_rounds=sum(int((r or {}).get("rounds", 0)) for r in agent_results),
        tool_call_rounds=sum(int((r or {}).get("tool_call_rounds", 0)) for r in agent_results),
        total_process_time=elapsed,
        messages=[],
        trajectory=[],
        extras=extras,
    )
