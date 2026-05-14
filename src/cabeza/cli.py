"""``cabeza`` command-line interface.

Subcommands:

    cabeza run    "<question>"   — single-question one-off
    cabeza eval                  — full-dataset benchmark
    cabeza list                  — list strategies / team modes / datasets

All knobs map 1:1 to ``Agent`` constructor parameters.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

from cabeza.agent import Agent, MemoryConfig, TeamConfig, _resolve_api_key
from cabeza.context import list_strategies
from cabeza.datasets import builtin_specs, load
from cabeza.runner import evaluate
from cabeza.tools import build_toolset


_TOOL_PRESETS = {"none", "web", "corpus", "hle"}


def _split_csv(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _autodetect_provider() -> Optional[dict]:
    """Infer model/base_url/family from whichever provider env vars are set.

    Preference order: DeepSeek → OpenAI → local Qwen vLLM. Returns ``None``
    when nothing is configured (the caller then requires explicit --model).
    """
    if os.environ.get("DEEPSEEK_API_KEY"):
        return {
            "family": "deepseek",
            "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            "label": "DeepSeek",
        }
    if os.environ.get("OPENAI_API_KEY"):
        return {
            "family": "gpt",
            "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "label": "OpenAI",
        }
    if os.environ.get("QWEN_BASE_URL"):
        return {
            "family": "qwen",
            "model": os.environ.get("QWEN_MODEL", "Qwen3-30B-A3B-Instruct-2507"),
            "base_url": os.environ["QWEN_BASE_URL"],
            "label": "Qwen (local vLLM)",
        }
    return None


def _autodetect_tool_preset() -> str:
    """Pick a tool preset from env: 'web' when both Serper and Jina keys exist."""
    if os.environ.get("SERPER_API_KEY") and os.environ.get("JINA_API_KEY"):
        return "web"
    return "none"


def _resolve_provider_args(args: argparse.Namespace) -> None:
    """Fill model/base_url/family from env when --model was not supplied.

    Mutates ``args`` in place and prints a one-line summary of what was picked.
    """
    if args.model:
        # Explicit --model: keep everything as given (family defaults to qwen
        # via argparse if the user did not pass --family). Still resolve the
        # key from env so tool wiring has a concrete value.
        if not args.family:
            args.family = "qwen"
        args.api_key = _resolve_api_key(args.api_key, args.family)
        return

    auto = _autodetect_provider()
    if auto is None:
        raise SystemExit(
            "No --model supplied and no provider env vars detected. Either pass "
            "--model/--base-url/--family explicitly, or set one of "
            "DEEPSEEK_API_KEY / OPENAI_API_KEY / QWEN_BASE_URL."
        )
    args.model = args.model or auto["model"]
    args.base_url = args.base_url or auto["base_url"]
    args.family = args.family or auto["family"]
    # Resolve the key from the family env var now so downstream tool wiring
    # (e.g. the visit-tool extractor LLM) has a concrete key to use.
    args.api_key = _resolve_api_key(args.api_key, args.family)

    # Auto-enable web tools when the user did not ask for a preset and the
    # Serper + Jina keys are both present.
    auto_tool_note = ""
    if (args.tools or "none") == "none":
        detected = _autodetect_tool_preset()
        if detected != "none":
            args.tools = detected
            auto_tool_note = " + web tools (Serper + Jina)"

    print(
        f"[cabeza] auto-config: {auto['label']} / {args.model} "
        f"@ {args.base_url}{auto_tool_note}"
    )


def _build_agent_from_args(args: argparse.Namespace) -> Agent:
    # tools
    tools = []
    preset = (args.tools or "").lower()
    if preset == "web":
        tools = build_toolset(
            "web",
            search_api_key=args.search_api_key or os.environ.get("SEARCH_API_KEY", ""),
            visit_api_key=args.jina_api_key or os.environ.get("JINA_API_KEY", ""),
            visit_llm_api_key=args.visit_llm_api_key or args.api_key,
            visit_llm_base_url=args.visit_llm_base_url or args.base_url,
            visit_llm_model=args.visit_llm_model or args.model,
            visit_max_content_tokens=args.visit_max_content_tokens,
            include_scholar=args.include_scholar,
            include_python=args.include_python,
            scholar_api_key=args.scholar_api_key or os.environ.get("SCHOLAR_API_KEY", "") or None,
        )
    elif preset == "corpus":
        tools = build_toolset(
            "corpus",
            index_path=args.corpus_index_path,
            corpus_path=args.corpus_data_path,
            embed_api_url=args.embed_api_url,
            embed_model=args.embed_model,
            embed_api_key=args.embed_api_key,
            top_k=args.corpus_top_k,
            snippet_max_tokens=args.snippet_max_tokens,
            visit_llm_api_key=args.visit_llm_api_key or args.api_key,
            visit_llm_base_url=args.visit_llm_base_url or args.base_url,
            visit_llm_model=args.visit_llm_model or args.model,
            include_scholar=args.include_scholar,
            include_python=args.include_python,
            scholar_api_key=args.scholar_api_key or os.environ.get("SCHOLAR_API_KEY", "") or None,
        )
    elif preset == "hle":
        tools = build_toolset(
            "hle",
            search_api_key=args.search_api_key or os.environ.get("SEARCH_API_KEY", ""),
            visit_api_key=args.jina_api_key or os.environ.get("JINA_API_KEY", ""),
            visit_llm_api_key=args.visit_llm_api_key or args.api_key,
            visit_llm_base_url=args.visit_llm_base_url or args.base_url,
            visit_llm_model=args.visit_llm_model or args.model,
            visit_max_content_tokens=args.visit_max_content_tokens,
            scholar_api_key=args.scholar_api_key or os.environ.get("SCHOLAR_API_KEY", "") or None,
        )

    # team config
    team_config: Optional[TeamConfig] = None
    if args.team:
        team_config = TeamConfig(
            mode=args.team,
            size=int(args.team_size or 3),
            families=_split_csv(args.team_families),
            models=_split_csv(args.team_models),
            base_urls=_split_csv(args.team_base_urls),
            api_keys=_split_csv(args.team_api_keys),
            subagent_workers=args.subagent_workers,
            subagent_report_max_tokens=args.subagent_report_max_tokens,
            meta_family=args.meta_family or None,
            meta_model=args.meta_model or None,
            meta_base_url=args.meta_base_url or None,
            meta_api_key=args.meta_api_key or None,
            meta_max_steps=args.meta_max_steps,
            meta_context_window_tokens=args.meta_context_window_tokens,
            sub_max_steps=args.sub_max_steps,
            sub_context_window_tokens=args.sub_context_window_tokens,
        )

    memory_config = MemoryConfig(
        kind="page",
        summarizer_model=args.memory_model or None,
        summarizer_base_url=args.memory_base_url or None,
        summarizer_api_key=args.memory_api_key or None,
        page_max_tokens=args.page_max_tokens,
        memory_tool_max_pages=args.memory_tool_max_pages,
        max_pages_per_owner=args.memory_max_pages_per_owner,
    )

    return Agent(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        family=args.family,
        tools=tools,
        system_prompt=args.system_prompt or None,
        max_steps=args.max_steps,
        max_time_seconds=args.max_time_seconds,
        context_management=args.context_management or None,
        context_management_tokens=args.context_management_tokens,
        context_window_tokens=args.context_window_tokens,
        summary_model=args.summary_model or None,
        summary_base_url=args.summary_base_url or None,
        summary_api_key=args.summary_api_key or None,
        keep_recent_turns=args.keep_recent_turns,
        memory=args.memory or None,
        memory_config=memory_config,
        team=args.team or None,
        team_config=team_config,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        max_output_tokens=args.max_output_tokens,
        timeout=args.timeout,
        stream=args.stream,
        enable_thinking=args.enable_thinking,
        reasoning_effort=args.reasoning_effort or None,
        verbose=bool(getattr(args, "verbose", False)),
    )


def _add_agent_args(p: argparse.ArgumentParser) -> None:
    # model / base_url / family are optional: when omitted they are inferred
    # from provider env vars (DEEPSEEK_API_KEY / OPENAI_API_KEY / QWEN_BASE_URL).
    p.add_argument("--model", default="", help="LLM name. Omit to auto-detect from env.")
    p.add_argument("--base-url", default="", help="OpenAI-compatible base URL. Omit to auto-detect.")
    p.add_argument("--api-key", default=None, help="Provider key. Omit to read the family env var.")
    p.add_argument(
        "--family",
        default="",
        choices=["", "qwen", "glm", "kimi", "deepseek", "gpt", "gpt_oss", "gpt-oss"],
        help="Agent family. Omit to auto-detect alongside --model.",
    )
    p.add_argument("--system-prompt", default="")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--max-time-seconds", type=float, default=None)

    verbose_group = p.add_mutually_exclusive_group()
    verbose_group.add_argument(
        "--verbose", dest="verbose", action="store_true", default=None,
        help="Stream step/tool/final to stdout (default on for `run`, off for `eval`).",
    )
    verbose_group.add_argument(
        "--no-verbose", dest="verbose", action="store_false",
        help="Silence the step-by-step trace.",
    )

    p.add_argument(
        "--context-management",
        default="",
        choices=["", "summary", "recent_k", "discard_tool", "discard_all", "page_memory"],
    )
    p.add_argument("--context-management-tokens", type=int, default=64_000)
    p.add_argument("--context-window-tokens", type=int, default=128_000)
    p.add_argument("--keep-recent-turns", type=int, default=10)

    p.add_argument("--summary-model", default="")
    p.add_argument("--summary-base-url", default="")
    p.add_argument("--summary-api-key", default="")

    p.add_argument("--memory", default="", choices=["", "page"])
    p.add_argument("--memory-model", default="")
    p.add_argument("--memory-base-url", default="")
    p.add_argument("--memory-api-key", default="")
    p.add_argument("--page-max-tokens", type=int, default=30_000)
    p.add_argument("--memory-tool-max-pages", type=int, default=5)
    p.add_argument("--memory-max-pages-per-owner", type=int, default=None)

    p.add_argument("--team", default="", choices=["", "naive", "swarm", "fugue"])
    p.add_argument("--team-size", type=int, default=3)
    p.add_argument("--team-families", default="")
    p.add_argument("--team-models", default="")
    p.add_argument("--team-base-urls", default="")
    p.add_argument("--team-api-keys", default="")
    p.add_argument("--subagent-workers", type=int, default=None)
    p.add_argument("--subagent-report-max-tokens", type=int, default=8192)
    p.add_argument("--meta-family", default="")
    p.add_argument("--meta-model", default="")
    p.add_argument("--meta-base-url", default="")
    p.add_argument("--meta-api-key", default="")
    p.add_argument("--meta-max-steps", type=int, default=8)
    p.add_argument("--meta-context-window-tokens", type=int, default=None)
    p.add_argument("--sub-max-steps", type=int, default=None)
    p.add_argument("--sub-context-window-tokens", type=int, default=None)

    p.add_argument("--tools", default="none", choices=sorted(_TOOL_PRESETS))
    p.add_argument("--search-api-key", default="")
    p.add_argument("--jina-api-key", default="")
    p.add_argument("--scholar-api-key", default="")
    p.add_argument("--visit-llm-model", default="")
    p.add_argument("--visit-llm-base-url", default="")
    p.add_argument("--visit-llm-api-key", default="")
    p.add_argument("--visit-max-content-tokens", type=int, default=95_000)
    p.add_argument("--include-scholar", action="store_true")
    p.add_argument("--include-python", action="store_true")
    p.add_argument("--corpus-index-path", default="")
    p.add_argument("--corpus-data-path", default="")
    p.add_argument("--embed-api-url", default="")
    p.add_argument("--embed-model", default="")
    p.add_argument("--embed-api-key", default="EMPTY")
    p.add_argument("--corpus-top-k", type=int, default=5)
    p.add_argument("--snippet-max-tokens", type=int, default=2048)

    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-output-tokens", type=int, default=None)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--stream", action="store_true", default=None)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--reasoning-effort", default="")


def _cmd_run(args: argparse.Namespace) -> int:
    _resolve_provider_args(args)
    if args.verbose is None:          # one-off command: stream by default
        args.verbose = True
    agent = _build_agent_from_args(args)
    result = agent.run_verbose(args.question, owner="cli")
    print("===== prediction =====")
    print(result.prediction)
    print()
    print(
        f"[stats] rounds={result.total_rounds} tool_rounds={result.tool_call_rounds} "
        f"time={result.total_process_time:.1f}s termination={result.termination}"
    )
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    _resolve_provider_args(args)
    if args.verbose is None:          # batch command: silent by default
        args.verbose = False
    agent = _build_agent_from_args(args)
    limit = int(args.eval_num) if args.eval_num and int(args.eval_num) > 0 else None
    dataset = load(args.dataset, path=args.dataset_path or None, limit=limit)
    answer_format = args.answer_format or None
    summary = evaluate(
        agent,
        dataset,
        out=args.out,
        workers=args.workers,
        resume=not args.no_resume,
        answer_format=answer_format,
        log_dir=args.log_dir or None,
    )
    print("[evaluate] " + json.dumps(summary, ensure_ascii=False))
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    print("context strategies:")
    for s in list_strategies():
        print(f"  - {s}")
    print()
    print("team modes:")
    for s in ["naive", "swarm", "fugue"]:
        print(f"  - {s}")
    print()
    print("datasets:")
    for s in builtin_specs():
        print(f"  - {s.name}: {s.description}  ({s.default_path})")
    _ = args
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="cabeza", description="Cabeza CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_parser = sub.add_parser("run", help="Run a single question.")
    run_parser.add_argument("question")
    _add_agent_args(run_parser)
    run_parser.set_defaults(func=_cmd_run)

    eval_parser = sub.add_parser("eval", help="Run benchmark evaluation.")
    eval_parser.add_argument("--dataset", required=True, help="Builtin dataset name (e.g. bc, bcp, ws, hle).")
    eval_parser.add_argument("--dataset-path", default="", help="Override the dataset JSONL path.")
    eval_parser.add_argument(
        "--eval-num",
        type=int,
        default=0,
        help="Evaluate only the first N items from the dataset (0 = all).",
    )
    eval_parser.add_argument("--out", required=True, help="Output results.jsonl path.")
    eval_parser.add_argument("--log-dir", default="", help="Optional per-question log directory.")
    eval_parser.add_argument("--workers", type=int, default=4)
    eval_parser.add_argument("--no-resume", action="store_true")
    eval_parser.add_argument("--answer-format", default="", choices=["", "browsecomp", "hle", "free"])
    _add_agent_args(eval_parser)
    eval_parser.set_defaults(func=_cmd_eval)

    list_parser = sub.add_parser("list", help="List available strategies, team modes, and datasets.")
    list_parser.set_defaults(func=_cmd_list)

    # `cabeza score` — LLM-judge accuracy scoring for cabeza-style prediction JSONL files.
    # All flags after the subcommand are forwarded to cabeza.eval (QA or WideSearch
    # dispatch). We use parse_known_args() rather than register every flag because
    # the underlying evaluator already has a rich argparse surface.
    score_parser = sub.add_parser(
        "score",
        help="Score a prediction JSONL via the LLM judge (bc/bc_zh/bcp/hle/ws).",
        add_help=False,
    )
    score_parser.set_defaults(func=_cmd_score)

    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "score":
        args = argparse.Namespace(cmd="score", func=_cmd_score, score_argv=argv[1:])
        return int(_cmd_score(args))

    args = parser.parse_args(argv)
    return int(args.func(args))


def _cmd_score(args: argparse.Namespace) -> int:
    from cabeza.eval import run_cli

    run_cli(args.score_argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
