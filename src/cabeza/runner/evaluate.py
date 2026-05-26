"""Parallel, resumable evaluation runner."""

from __future__ import annotations

import concurrent.futures
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Iterable, Optional

from cabeza.datasets.base import Dataset
from cabeza.prompts import format_question
from cabeza.types import Example, RunResult


def _load_completed_ids(path: str) -> tuple[set[str], int, int]:
    completed: set[str] = set()
    existing = 0
    malformed = 0
    if not os.path.exists(path):
        return completed, existing, malformed
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            existing += 1
            item_id = record.get("id")
            if item_id is not None:
                completed.add(str(item_id))
    return completed, existing, malformed


def _filter_pending(items: Iterable[Example], completed: set[str]) -> tuple[list[Example], int]:
    pending: list[Example] = []
    skipped = 0
    for item in items:
        item_id = item.get("id")
        if item_id is not None and str(item_id) in completed:
            skipped += 1
            continue
        pending.append(item)
    return pending, skipped


def _result_to_record(item: Example, result: RunResult) -> dict:
    return {
        "id": item.get("id", ""),
        "question": item.get("question", ""),
        "answer": item.get("golden_answers", ""),
        "prediction": result.prediction,
        "termination": result.termination,
        "total_rounds": result.total_rounds,
        "tool_call_rounds": result.tool_call_rounds,
        "total_process_time": result.total_process_time,
        "messages": result.messages,
        "trajectory": [_step_dict(s) for s in result.trajectory],
        "extras": _jsonify(result.extras),
    }


def _step_dict(step) -> dict:
    return {
        "step": step.step,
        "reasoning": step.reasoning,
        "final_content": step.final_content,
        "tool_calls": [
            {"name": tc.name, "args": tc.args, "result": tc.result}
            for tc in step.tool_calls
        ],
    }


def _jsonify(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonify(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonify(v) for v in value]
    return value


def evaluate(
    agent,
    dataset: Dataset,
    *,
    out: str,
    workers: int = 4,
    resume: bool = True,
    answer_format: Optional[str] = None,
    log_dir: Optional[str] = None,
    progress: Optional[Callable[[int, int, dict], None]] = None,
) -> dict:
    """Run ``agent`` over every example in ``dataset`` and stream JSONL records to ``out``.

    Parameters
    ----------
    agent:
        A ``cabeza.Agent`` instance.
    dataset:
        Any object satisfying ``cabeza.datasets.Dataset``.
    out:
        Path to the JSONL results file.
    workers:
        Number of concurrent questions to process.
    resume:
        When True and ``out`` already exists, skip examples whose ``id`` is
        already present in the file.
    answer_format:
        Wraps the question with a benchmark answer-format template before
        passing it to the agent. Accepts ``"browsecomp"``, ``"hle"``,
        ``"gisa"``, ``"free"``, or ``None`` (no wrapping). When ``None``, the function consults
        ``dataset.spec.answer_format`` if available.
    log_dir:
        Optional directory for per-example trajectory logs.
    progress:
        Optional callback invoked as ``(completed_in_run, total, record)``
        after each item is written.

    Returns
    -------
    dict
        Summary with ``completed``, ``skipped``, ``total``, ``out``.
    """
    out_dir = os.path.dirname(os.path.abspath(out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    items = list(dataset)
    total = len(items)

    completed_ids: set[str] = set()
    existing = malformed = 0
    if resume:
        completed_ids, existing, malformed = _load_completed_ids(out)
    pending, skipped = _filter_pending(items, completed_ids)

    if malformed:
        print(f"[evaluate] ignoring {malformed} malformed lines in {out}")
    if skipped:
        print(f"[evaluate] resuming; skipping {skipped}/{total} already-completed items")

    if answer_format is None:
        spec = getattr(dataset, "spec", None)
        answer_format = getattr(spec, "answer_format", None)

    mode = "a" if (resume and os.path.exists(out)) else "w"

    def _process(item: Example) -> dict:
        try:
            question_text = format_question(item["question"], answer_format=answer_format)
            traj_path = None
            if log_dir:
                safe_id = str(item.get("id", "")).replace("/", "_") or "item"
                traj_path = os.path.join(log_dir, f"{safe_id}.json")
            result = agent.run_verbose(
                question_text,
                owner="agent",
                trajectory_log_path=traj_path,
            )
            return _result_to_record(item, result)
        except Exception as exc:  # noqa: BLE001
            print(f"[evaluate] error on item {item.get('id', '?')}: {exc}")
            return {
                "id": item.get("id", ""),
                "question": item.get("question", ""),
                "answer": item.get("golden_answers", ""),
                "error": str(exc),
                "termination": f"error: {exc}",
            }

    started = time.time()
    completed_in_run = 0
    with open(out, mode, encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
            futures = [executor.submit(_process, item) for item in pending]
            for future in concurrent.futures.as_completed(futures):
                try:
                    record = future.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"[evaluate] future failed: {exc}")
                    continue
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                completed_in_run += 1
                if progress is not None:
                    progress(completed_in_run, total, record)
                else:
                    print(f"[evaluate] completed {skipped + completed_in_run}/{total}")
    elapsed = time.time() - started

    return {
        "completed": completed_in_run,
        "skipped": skipped,
        "existing": existing,
        "total": total,
        "elapsed": elapsed,
        "out": out,
    }
