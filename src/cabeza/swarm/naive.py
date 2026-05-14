"""Naive multi-agent orchestrator.

Meta-agent decomposes the question into N subproblems, dispatches them to
parallel subagents, then synthesizes the final answer using the subagent
reports as evidence.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Optional

from cabeza.swarm._common import (
    AgentSpec,
    clone_agent_for_spec,
    extract_run_result,
    parse_agent_specs,
)
from cabeza.types import RunResult

if TYPE_CHECKING:
    from cabeza.agent import Agent


META_SYSTEM_PROMPT = (
    "You coordinate parallel search agents: first decompose the task, then "
    "synthesize their evidence into a final answer. Be concise, faithful to "
    "evidence, and follow any requested output format exactly."
)


def _decompose_prompt(question: str, *, n: int) -> str:
    return f"""
You are the meta-agent for a parallel research team. This is only the planning
phase, not the final-answer phase.
Split the question into exactly {n} complementary subproblems for independent subagents.

Output contract:
- Return a raw JSON array only. Do not use Markdown fences.
- The response must start with "[" and end with "]".
- Each array item must be an object with exactly these keys: "title" and "task".
- Do not include any final-answer format.

Requirements:
- Each task must be self-contained and include enough original constraints for a search agent to work without seeing your hidden reasoning.
- The tasks must divide the original problem into different useful pieces, not restate the entire question with different wording.
- Choose the decomposition that best fits the question: separate constraints, entities, dates, sources, relationship chains, timelines, clue clusters, or answer attributes as appropriate.
- Each task must have a clear deliverable the final meta-agent can combine.

Original question:
{question}
""".strip()


def _subagent_prompt(question: str, task: dict, idx: int, *, n: int) -> str:
    title = task.get("title") or f"Subtask {idx + 1}"
    text = task.get("task") or ""
    return f"""
You are SearchAgent-{idx + 1} in a {n}-agent parallel research baseline.
Your job is to complete only your assigned investigation, not to solve the
whole question by yourself.

Original question:
{question}

Your assigned investigation:
Title: {title}
Task: {text}

Operating rules:
- Solve your assigned subproblem thoroughly. Use the original question as context for constraints.
- Prioritize the deliverable requested by your assigned investigation.
- Do not let adjacent subproblems dominate your work. If you find useful facts outside your assigned lane, include them briefly as handoff notes.

Return a concise evidence-backed report that includes:
- the conclusion or partial answer for your assigned subproblem;
- useful sources, URLs, or tool evidence when available;
- candidate facts, extracted attributes, relationships, or contradictions relevant to your subproblem;
- explicit uncertainty and what the meta-agent should combine with other reports.
""".strip()


def _synthesis_prompt(
    question: str,
    decomposition: list[dict],
    subagent_results: list[dict],
    *,
    report_max_tokens: int,
) -> str:
    rendered: list[str] = []
    for r in sorted(subagent_results, key=lambda x: int(x.get("subagent_id", 0))):
        sid = int(r.get("subagent_id", 0))
        task = decomposition[sid] if sid < len(decomposition) else {}
        report = r.get("prediction", "") or r.get("error", "") or "No report."
        report = _truncate_tokens(report, report_max_tokens)
        rendered.append(
            "\n".join(
                [
                    f"## SearchAgent-{sid + 1}",
                    f"Assigned task: {task.get('title', '')}",
                    str(task.get("task", "")).strip(),
                    "",
                    "Report:",
                    report,
                    "",
                    f"Termination: {r.get('termination', '')}",
                ]
            ).strip()
        )
    reports_text = "\n\n".join(rendered)
    return f"""
You are the meta-agent in the final synthesis phase.
Combine the complementary subagent reports into the final answer. You may use
the available tools to fill gaps, resolve conflicts, or verify important
evidence before producing the final answer.

Original question:
{question}

Subagent reports:
{reports_text}

Instructions:
- Treat each subagent report as evidence for its assigned subproblem.
- First align the subproblem conclusions, constraints, and sources across reports, then infer the final answer.
- Resolve disagreements using the strongest evidence in the reports.
- If the reports leave a critical gap or conflict, use tools to research that specific missing piece instead of redoing every subagent's work.
- Do not invent sources or facts not supported by the reports or the original question.
""".strip()


def _truncate_tokens(text: str, max_tokens: int) -> str:
    text = str(text or "")
    if max_tokens <= 0:
        return text
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        toks = enc.encode(text)
        if len(toks) <= max_tokens:
            return text
        return enc.decode(toks[:max_tokens]) + "\n\n[truncated]"
    except ImportError:
        words = text.split()
        if len(words) <= max_tokens:
            return text
        return " ".join(words[:max_tokens]) + "\n\n[truncated]"


def _extract_json_payload(text: str) -> Any:
    text = str(text or "").strip()
    if not text:
        raise ValueError("empty decomposition")
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = [fenced.group(1).strip()] if fenced else []
    candidates.append(text)
    array_match = re.search(r"(\[\s*\{.*\}\s*\])", text, flags=re.DOTALL)
    if array_match:
        candidates.append(array_match.group(1))
    last_error: Optional[Exception] = None
    for c in candidates:
        try:
            return json.loads(c)
        except Exception as exc:
            last_error = exc
    raise ValueError(f"could not parse JSON decomposition: {last_error}")


def _normalize_tasks(payload: Any, *, n: int, question: str) -> list[dict]:
    if isinstance(payload, dict):
        for key in ("tasks", "subtasks", "agents"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    tasks: list[dict] = []
    if isinstance(payload, list):
        for idx, item in enumerate(payload):
            if isinstance(item, dict):
                task = item.get("task") or item.get("description") or item.get("question") or ""
                title = item.get("title") or item.get("name") or f"Subtask {idx + 1}"
            else:
                task, title = str(item), f"Subtask {idx + 1}"
            if str(task).strip():
                tasks.append({"id": idx, "title": str(title).strip(), "task": str(task).strip()})
    if len(tasks) >= n:
        return tasks[:n]
    fallback = [
        "Find primary evidence and authoritative sources relevant to the whole question.",
        "Search for complementary sources, edge cases, and conflicting evidence.",
        "Verify the likely answer and gather details needed for the required output format.",
    ]
    while len(tasks) < n:
        idx = len(tasks)
        focus = fallback[idx % len(fallback)]
        tasks.append(
            {"id": idx, "title": f"Fallback Subtask {idx + 1}", "task": f"{focus}\n\nOriginal question:\n{question}"}
        )
    return tasks


def _resolve_meta_spec(agent: "Agent") -> AgentSpec:
    cfg = agent.team_config
    if cfg is None or not (cfg.meta_model or cfg.meta_family):
        return AgentSpec(
            index=-1,
            name="Meta",
            family=agent.family,
            model=agent.model,
            base_url=agent.base_url,
            api_key=agent.api_key,
        )
    return AgentSpec(
        index=-1,
        name="Meta",
        family=(cfg.meta_family or agent.family).lower(),
        model=cfg.meta_model or agent.model,
        base_url=cfg.meta_base_url or agent.base_url,
        api_key=cfg.meta_api_key or agent.api_key,
    )


def _run_one_agent(
    member_agent: "Agent",
    *,
    prompt: str,
    system_prompt: Optional[str] = None,
    extra_tools: Optional[list] = None,
) -> tuple[Any, float, str, str]:
    bundle = member_agent.build_single_agent(extra_tools=extra_tools or None)
    if system_prompt is not None:
        bundle.agent.system_prompt = system_prompt
    started = time.time()
    bundle.agent.run(prompt)
    elapsed = time.time() - started
    prediction = "No answer found."
    termination = "no answer found"
    if bundle.agent.trajectory:
        last = bundle.agent.trajectory[-1]
        if last.final_content:
            prediction = last.final_content
            termination = "answer (final assistant response)"
    return bundle.agent, elapsed, prediction, termination


def run_naive(agent: "Agent", *, question: str) -> RunResult:
    cfg = agent.team_config
    n = max(1, int((cfg.size if cfg else 3) or 3))

    meta_spec = _resolve_meta_spec(agent)
    meta_agent = clone_agent_for_spec(agent, meta_spec)
    if cfg is not None:
        meta_agent.max_steps = cfg.meta_max_steps or meta_agent.max_steps
        if cfg.meta_context_window_tokens:
            meta_agent.context_window_tokens = cfg.meta_context_window_tokens

    started = time.time()
    decomp_text, _ = _run_meta_phase(
        meta_agent,
        system_prompt=META_SYSTEM_PROMPT,
        prompt=_decompose_prompt(question, n=n),
    )
    try:
        decomposition = _normalize_tasks(_extract_json_payload(decomp_text), n=n, question=question)
    except Exception as exc:  # noqa: BLE001
        print(f"[naive] decomposition parse failed: {exc}; using fallback tasks.")
        decomposition = _normalize_tasks([], n=n, question=question)

    sub_specs = parse_agent_specs(agent)
    workers = max(1, min((cfg.subagent_workers if cfg else n) or n, n))
    subagent_results: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _run_subagent,
                agent=agent,
                spec=sub_specs[idx % len(sub_specs)],
                question=question,
                task=task,
                idx=idx,
                n=n,
            ): idx
            for idx, task in enumerate(decomposition)
        }
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                subagent_results.append(future.result())
            except Exception as exc:  # noqa: BLE001
                print(f"[naive] SearchAgent-{idx + 1} failed: {exc}")
                subagent_results.append(
                    {
                        "subagent_id": idx,
                        "name": f"SearchAgent-{idx + 1}",
                        "assigned_task": decomposition[idx],
                        "prediction": "",
                        "termination": f"error: {exc}",
                        "error": str(exc),
                    }
                )

    subagent_results.sort(key=lambda x: int(x.get("subagent_id", 0)))
    synth_text, synth_agent = _run_meta_phase(
        meta_agent,
        system_prompt=META_SYSTEM_PROMPT,
        prompt=_synthesis_prompt(
            question,
            decomposition,
            subagent_results,
            report_max_tokens=int(cfg.subagent_report_max_tokens) if cfg else 8192,
        ),
        extra_tools=list(agent.tools),
    )
    elapsed = time.time() - started

    extras = {
        "team_mode": "naive",
        "decomposition": decomposition,
        "subagent_results": subagent_results,
        "meta_spec": asdict(meta_spec),
    }
    result = extract_run_result(
        question=question,
        gizmo_agent=synth_agent,
        started_elapsed=elapsed,
        extras=extras,
    )
    result.prediction = synth_text or result.prediction
    result.final_content = result.prediction
    return result


def _run_meta_phase(
    meta_agent: "Agent",
    *,
    system_prompt: str,
    prompt: str,
    extra_tools: Optional[list] = None,
) -> tuple[str, Any]:
    bundle = meta_agent.build_single_agent(extra_tools=extra_tools or [])
    bundle.agent.system_prompt = system_prompt
    bundle.agent.run(prompt)
    prediction = ""
    if bundle.agent.trajectory:
        last = bundle.agent.trajectory[-1]
        if last.final_content:
            prediction = last.final_content
    return prediction, bundle.agent


def _run_subagent(
    *,
    agent: "Agent",
    spec: AgentSpec,
    question: str,
    task: dict,
    idx: int,
    n: int,
) -> dict:
    member = clone_agent_for_spec(agent, spec)
    cfg = agent.team_config
    if cfg is not None:
        if cfg.sub_max_steps:
            member.max_steps = cfg.sub_max_steps
        if cfg.sub_context_window_tokens:
            member.context_window_tokens = cfg.sub_context_window_tokens

    gizmo_agent, elapsed, prediction, termination = _run_one_agent(
        member,
        prompt=_subagent_prompt(question, task, idx, n=n),
    )
    return {
        "subagent_id": idx,
        "name": f"SearchAgent-{idx + 1}",
        "spec": asdict(spec),
        "assigned_task": task,
        "prediction": prediction,
        "termination": termination,
        "rounds": gizmo_agent.state.step,
        "tool_call_rounds": gizmo_agent.state.tool_rounds,
        "total_process_time": elapsed,
    }
