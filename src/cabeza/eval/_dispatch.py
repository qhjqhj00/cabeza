#!/usr/bin/env python3
"""Unified evaluator entry point for cabeza.

Dispatch:
- ``bc`` / ``bc_zh`` / ``bcp`` / ``hle``  → :mod:`cabeza.eval._qa`
- ``ws`` / ``widesearch`` / ``wide_search`` → :mod:`cabeza.eval._widesearch`
- ``dsqa`` / ``deepsearchqa`` → :mod:`cabeza.eval._deepsearchqa`
- ``gisa`` → :mod:`cabeza.eval._gisa`
"""

from __future__ import annotations

import sys
from typing import Sequence

WS_BENCHMARKS = {"ws", "widesearch", "wide_search"}
DSQA_BENCHMARKS = {"dsqa", "deepsearchqa", "deepsearch_qa"}
GISA_BENCHMARKS = {"gisa"}
QA_BENCHMARKS = {
    "bc",
    "bc_zh",
    "bc-zh",
    "browsecomp",
    "browsecomp_en",
    "browsecomp_en_full",
    "browsecomp_zh",
    "browsecomp-zh",
    "browsecompzh",
    "bcp",
    "browsecompplus",
    "browsecomp_plus",
    "hle",
}
ALL_BENCHMARKS = sorted(WS_BENCHMARKS | DSQA_BENCHMARKS | GISA_BENCHMARKS | QA_BENCHMARKS)


def extract_benchmark(argv: Sequence[str]) -> str | None:
    for idx, arg in enumerate(argv):
        if arg == "--benchmark" and idx + 1 < len(argv):
            return argv[idx + 1].strip().lower()
        if arg.startswith("--benchmark="):
            return arg.split("=", 1)[1].strip().lower()
    return None


def strip_benchmark_arg(argv: Sequence[str]) -> list[str]:
    stripped: list[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "--benchmark":
            skip_next = True
            continue
        if arg.startswith("--benchmark="):
            continue
        stripped.append(arg)
    return stripped


def print_help() -> None:
    print(
        "Usage:\n"
        "  cabeza score --benchmark {bc,bc_zh,bcp,hle,ws,dsqa,gisa} "
        "[benchmark-specific args...]\n\n"
        "Examples:\n"
        "  cabeza score --benchmark hle --input_file /path/to/run.jsonl "
        "--judge_model Qwen3-32B --judge_base_url http://localhost:8004/v1 --judge_api_key EMPTY\n"
        "  cabeza score --benchmark ws --input_file /path/to/run.jsonl "
        "--judge_model Qwen3-32B --judge_base_url http://localhost:8004/v1 --judge_api_key EMPTY\n\n"
        "  cabeza score --benchmark dsqa --input_file /path/to/run.jsonl "
        "--judge_model gemini-2.5-flash --judge_base_url https://... --judge_api_key ...\n"
        "  cabeza score --benchmark gisa --input_file /path/to/run.jsonl\n\n"
        "Notes:\n"
        "  - QA benchmarks (`bc`, `bc_zh`, `bcp`, `hle`) are handled by `cabeza.eval._qa`.\n"
        "  - WideSearch (`ws`) is handled by `cabeza.eval._widesearch`.\n"
        "  - DeepSearchQA (`dsqa`) is handled by `cabeza.eval._deepsearchqa`.\n"
        "  - GISA (`gisa`) is handled by `cabeza.eval._gisa`.\n"
        "  - Multiple `--input_file` are supported for QA pass@k and WideSearch best-of-k.\n"
    )


def run_module(main_fn, argv: Sequence[str]) -> None:
    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0], *argv]
        main_fn()
    finally:
        sys.argv = original_argv


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in {"-h", "--help"}:
        print_help()
        return

    benchmark = extract_benchmark(argv)
    if benchmark is None:
        raise SystemExit(
            "Missing --benchmark. Valid values: " + ", ".join(ALL_BENCHMARKS)
        )

    if benchmark in WS_BENCHMARKS:
        from cabeza.eval import _widesearch

        run_module(_widesearch.main, strip_benchmark_arg(argv))
        return

    if benchmark in DSQA_BENCHMARKS:
        from cabeza.eval import _deepsearchqa

        run_module(_deepsearchqa.main, strip_benchmark_arg(argv))
        return

    if benchmark in GISA_BENCHMARKS:
        from cabeza.eval import _gisa

        run_module(_gisa.main, strip_benchmark_arg(argv))
        return

    if benchmark in QA_BENCHMARKS:
        from cabeza.eval import _qa

        run_module(_qa.main, argv)
        return

    raise SystemExit(
        f"Unsupported --benchmark={benchmark}. Valid values: " + ", ".join(ALL_BENCHMARKS)
    )


if __name__ == "__main__":
    main()
