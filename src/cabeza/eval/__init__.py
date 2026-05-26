"""Accuracy scoring for cabeza-style prediction JSONL files.

Bundled scorers:

- ``_qa``         : QA-style benchmarks (``bc`` / ``bc_zh`` / ``bcp`` / ``hle``).
                    Uses an OpenAI-compatible LLM-as-a-judge, plus a
                    conservative exact-match fallback when the judge cannot
                    return a parseable verdict.
- ``_widesearch`` : WideSearch (``ws``). Aligns the prediction's CSV-style
                    answer against the gold tables in ``cabeza/data/ws/gold``
                    and emits per-row / per-column / aggregate metrics.
- ``_deepsearchqa`` : DeepSearchQA (``dsqa``). Reproduces the official
                    LLM-as-judge prompt and precision / recall / F1 aggregation.
- ``_gisa``       : GISA (``gisa``). Reproduces the official deterministic
                    TSV/CSV scorer for item, set, list, and table answers.

Programmatic entry points:

    from cabeza.eval import score
    summary = score(
        benchmark="bc",
        input_files=["results/bc_qwen.jsonl"],
        judge_model="deepseek-v4-flash",
        judge_base_url="https://api.deepseek.com/v1",
        judge_api_key="sk-...",
    )

CLI:

    cabeza score --benchmark bc \\
        --input_file results/bc_qwen.jsonl \\
        --judge_model deepseek-v4-flash \\
        --judge_base_url https://api.deepseek.com/v1 \\
        --judge_api_key $DEEPSEEK_API_KEY
"""

from __future__ import annotations

from typing import Any, Sequence

from cabeza.eval._dispatch import (
    ALL_BENCHMARKS,
    DSQA_BENCHMARKS,
    GISA_BENCHMARKS,
    QA_BENCHMARKS,
    WS_BENCHMARKS,
    main as run_cli,
)


def score(
    *,
    benchmark: str,
    input_files: Sequence[str],
    judge_model: str = "",
    judge_base_url: str = "",
    judge_api_key: str = "",
    eval_data_path: str = "",
    gold_dir: str = "",
    extra_argv: Sequence[str] | None = None,
) -> int:
    """Convenience wrapper around the CLI dispatcher.

    Builds the ``--benchmark/--input_file`` argv that the underlying evaluator
    modules expect, then dispatches. ``judge_model`` is required by QA,
    WideSearch, and DeepSearchQA; GISA ignores it because its official scorer is
    deterministic.

    Returns the process exit code (``0`` on success). For richer programmatic
    access (cache + scored + summary files), pass output paths via
    ``extra_argv`` as additional CLI flags, e.g.
    ``["--summary_output", "summary.json"]``.
    """
    benchmark_key = benchmark.strip().lower()
    argv: list[str] = ["--benchmark", benchmark_key]
    for path in input_files:
        argv += ["--input_file", path]
    if benchmark_key not in GISA_BENCHMARKS:
        argv += ["--judge_model", judge_model]
    if judge_base_url:
        argv += ["--judge_base_url", judge_base_url]
    if judge_api_key:
        argv += ["--judge_api_key", judge_api_key]
    if eval_data_path:
        argv += ["--eval_data_path", eval_data_path]
    if gold_dir and benchmark_key in (WS_BENCHMARKS | GISA_BENCHMARKS):
        argv += ["--gold_dir", gold_dir]
    if extra_argv:
        argv += list(extra_argv)
    run_cli(argv)
    return 0


__all__ = [
    "score",
    "run_cli",
    "ALL_BENCHMARKS",
    "DSQA_BENCHMARKS",
    "GISA_BENCHMARKS",
    "QA_BENCHMARKS",
    "WS_BENCHMARKS",
]
