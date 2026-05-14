"""``Agent`` — the public entry point.

``Agent`` is a thin factory that assembles a family agent with the
requested context-management strategy, memory subsystem, hard window guard,
and (optionally) a swarm orchestrator. The same ``Agent`` instance can be
called once per question via ``run``/``run_verbose`` or driven across a
whole dataset via ``cabeza.runner.evaluate``.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from cabeza.base import BaseAgent, LLMConfig

from cabeza._budget import default_encoder
from cabeza.context import get_strategy
from cabeza.context.hard_window import install_hard_window_guard
from cabeza.memory.page_store import PageStore, SharedPageStore
from cabeza.memory.summarizer import Summarizer
from cabeza.memory.tool import MemoryTool
from cabeza.types import RunResult, Step, ToolCall


_AGENT_FAMILY_ALIASES = {
    "qwen": "qwen",
    "glm": "glm",
    "zhipu": "glm",
    "kimi": "kimi",
    "moonshot": "kimi",
    "deepseek": "deepseek",
    "ds": "deepseek",
    "gpt": "gpt",
    "gpt_oss": "gpt_oss",
    "gpt-oss": "gpt_oss",
}


def _resolve_family(family: str) -> str:
    raw = (family or "qwen").strip().lower()
    if raw not in _AGENT_FAMILY_ALIASES:
        raise ValueError(
            f"Unsupported agent family {family!r}. "
            f"Expected one of: {sorted(set(_AGENT_FAMILY_ALIASES.values()))}"
        )
    return _AGENT_FAMILY_ALIASES[raw]


# Per-family environment-variable conventions used when an explicit
# ``api_key`` / ``base_url`` is not supplied.
_FAMILY_ENV_KEYS: dict[str, tuple[str, ...]] = {
    "qwen":     ("QWEN_API_KEY",     "DASHSCOPE_API_KEY"),
    "glm":      ("GLM_API_KEY",      "ZHIPU_API_KEY"),
    "kimi":     ("KIMI_API_KEY",     "MOONSHOT_API_KEY"),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "gpt":      ("OPENAI_API_KEY",),
    "gpt_oss":  ("OPENAI_API_KEY",),
}
_FAMILY_ENV_BASE_URLS: dict[str, tuple[str, ...]] = {
    "qwen":     ("QWEN_BASE_URL",),
    "glm":      ("GLM_BASE_URL",),
    "kimi":     ("KIMI_BASE_URL",),
    "deepseek": ("DEEPSEEK_BASE_URL",),
    "gpt":      ("OPENAI_BASE_URL",),
    "gpt_oss":  ("OPENAI_BASE_URL",),
}


def _resolve_api_key(api_key: Optional[str], family: str) -> str:
    """Resolve an API key from explicit kwarg → family env var → "EMPTY".

    The "EMPTY" default keeps local vLLM-style endpoints happy when no key
    is needed at all.
    """
    if api_key:
        return api_key
    for name in _FAMILY_ENV_KEYS.get(family, ()):
        value = os.environ.get(name)
        if value:
            return value
    return "EMPTY"


def _resolve_base_url(base_url: Optional[str], family: str) -> str:
    """Resolve a base URL from explicit kwarg → family env var → empty string.

    An empty string falls through to the family adapter's built-in default
    (which often checks the same env vars itself; this layer just centralizes
    the lookup so the Agent surface is consistent).
    """
    if base_url:
        return base_url
    for name in _FAMILY_ENV_BASE_URLS.get(family, ()):
        value = os.environ.get(name)
        if value:
            return value
    return ""


def _agent_class(family: str):
    from cabeza.agents.deepseek import DeepSeekAgent
    from cabeza.agents.glm import GLMAgent
    from cabeza.agents.gpt import GPTAgent
    from cabeza.agents.gpt_oss import GPTOssAgent
    from cabeza.agents.kimi import KimiAgent
    from cabeza.agents.qwen import QwenAgent

    return {
        "qwen": QwenAgent,
        "glm": GLMAgent,
        "kimi": KimiAgent,
        "deepseek": DeepSeekAgent,
        "gpt": GPTAgent,
        "gpt_oss": GPTOssAgent,
    }[family]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MemoryConfig:
    """Configuration for the page-memory subsystem."""

    kind: str = "page"  # currently only "page" is supported
    summarizer_model: Optional[str] = None
    summarizer_base_url: Optional[str] = None
    summarizer_api_key: Optional[str] = None
    page_max_tokens: int = 30_000
    memory_tool_max_pages: int = 5
    max_pages_per_owner: Optional[int] = None  # only used by shared stores
    summarizer_temperature: float = 0.3
    summarizer_max_response_tokens: int = 8192


@dataclass
class TeamConfig:
    """Configuration for swarm/multi-agent mode."""

    mode: str  # "naive" / "swarm" / "fugue"
    size: int = 3
    families: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    base_urls: list[str] = field(default_factory=list)
    api_keys: list[str] = field(default_factory=list)
    subagent_workers: Optional[int] = None
    subagent_report_max_tokens: int = 8192
    # meta-vs-sub overrides (naive / swarm)
    meta_family: Optional[str] = None
    meta_model: Optional[str] = None
    meta_base_url: Optional[str] = None
    meta_api_key: Optional[str] = None
    meta_max_steps: int = 8
    meta_context_window_tokens: Optional[int] = None
    sub_max_steps: Optional[int] = None
    sub_context_window_tokens: Optional[int] = None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class Agent:
    """Configurable single-agent (or swarm) inference harness.

    Parameters in brief — see the README for examples.

    LLM:
        ``model``, ``base_url``, ``api_key``, ``family``.

    Tools:
        ``tools`` — a list of ``BaseTool`` instances. Use
        ``cabeza.tools.build_toolset`` for presets.

    Context management:
        ``context_management`` ∈ ``{None, "summary", "recent_k",
        "discard_tool", "discard_all", "page_memory"}``.
        ``context_management_tokens`` is the soft budget that triggers it;
        ``context_window_tokens`` is the hard budget (forces final answer).

    Memory:
        ``memory=None`` or ``"page"``. ``memory_config`` for full control.
        Page memory is implicitly enabled when ``context_management=
        "page_memory"`` or when ``team="fugue"``.

    Swarm:
        ``team=None | "naive" | "swarm" | "fugue"`` plus ``team_config``.

    Verbose:
        ``verbose=True`` streams a step-by-step trace to stdout — every LLM
        emission (tool call or final answer) and every tool result. Use
        ``agent.on(event, fn)`` to register custom lifecycle hooks alongside
        (or instead of) the built-in verbose printer.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        family: str = "qwen",
        system_prompt: Optional[str] = None,
        tools: Optional[list] = None,
        max_steps: int = 200,
        max_time_seconds: Optional[float] = None,
        max_tool_rounds: Optional[int] = None,
        # context management
        context_management: Optional[str] = None,
        context_management_tokens: int = 64_000,
        context_window_tokens: int = 128_000,
        context_strategy_kwargs: Optional[dict] = None,
        # rolling-summary specifics
        summary_model: Optional[str] = None,
        summary_base_url: Optional[str] = None,
        summary_api_key: Optional[str] = None,
        # recent-k specifics
        keep_recent_turns: int = 10,
        # memory
        memory: Optional[str] = None,
        memory_config: Optional[MemoryConfig] = None,
        # swarm
        team: Optional[str] = None,
        team_config: Optional[TeamConfig] = None,
        # llm generation
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        timeout: float = 120.0,
        stream: Optional[bool] = None,
        enable_thinking: bool = False,
        reasoning_effort: Optional[str] = None,
        extra_body: Optional[dict] = None,
        # introspection
        verbose: bool = False,
    ) -> None:
        self.family = _resolve_family(family)
        self.model = model
        # api_key / base_url: explicit kwargs win; otherwise fall back to the
        # family-conventional env vars (e.g. DEEPSEEK_API_KEY for "deepseek",
        # OPENAI_API_KEY for "gpt"). See `.env.example` for the full list.
        self.base_url = _resolve_base_url(base_url, self.family)
        self.api_key = _resolve_api_key(api_key, self.family)
        self.system_prompt = system_prompt
        self.tools = list(tools or [])
        self.max_steps = int(max_steps)
        self.max_time_seconds = max_time_seconds
        self.max_tool_rounds = max_tool_rounds

        self.context_management = context_management
        self.context_management_tokens = int(context_management_tokens or 0)
        self.context_window_tokens = int(context_window_tokens or 0)
        self.context_strategy_kwargs = dict(context_strategy_kwargs or {})

        self.summary_model = summary_model
        self.summary_base_url = summary_base_url
        self.summary_api_key = summary_api_key
        self.keep_recent_turns = int(keep_recent_turns)

        self.memory_mode = self._resolve_memory_mode(memory, context_management, team)
        self.memory_config = memory_config or MemoryConfig()

        self.team_mode = team
        self.team_config = team_config

        self.llm_config = LLMConfig(
            max_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            timeout=float(timeout),
            stream=stream,
            enable_thinking=bool(enable_thinking),
            reasoning_effort=reasoning_effort or None,
            extra_body=extra_body,
        )

        # Lifecycle hooks — applied to the Gizmo agent inside build_single_agent.
        # Each entry is (event_name, callback). Use ``agent.on(event, fn)`` to
        # add more after construction.
        self._hooks: list[tuple[str, Callable]] = []
        self.verbose = bool(verbose)
        if self.verbose:
            self._install_verbose_hooks()

    # ---- hooks ------------------------------------------------------------

    def on(self, event: str, fn: Callable) -> "Agent":
        """Register a lifecycle hook on every underlying Gizmo agent built by this factory.

        Supported events:
            ``before_llm(state, messages)``           — fired before every LLM call.
            ``after_llm(state, parsed)``              — fired after every LLM call.
            ``after_tool(state, name, args, result)`` — fired after each tool execution.
            ``should_stop(state) -> Optional[str]``   — fired before every loop turn;
                a non-empty return triggers a stop with that reason.

        Returns ``self`` for chaining.
        """
        if event not in {"before_llm", "after_llm", "after_tool", "should_stop"}:
            raise ValueError(f"Unknown hook event: {event!r}")
        self._hooks.append((event, fn))
        return self

    def _install_verbose_hooks(self) -> None:
        """Built-in step-by-step printer used when ``verbose=True``."""

        def _short(s: Any, n: int = 160) -> str:
            text = " ".join(str(s or "").split())
            return text if len(text) <= n else text[: n - 1] + "…"

        def on_after_llm(state, parsed):
            tool_calls = parsed.get("tool_calls") or []
            if tool_calls:
                for tc in tool_calls:
                    name = tc["function"]["name"]
                    args = tc["function"].get("arguments", {})
                    print(f"[cabeza] step {state.step} → {name}({_short(args, 200)})")
                return
            final = (parsed.get("final_content") or "").strip()
            if final:
                print(f"[cabeza] step {state.step} ✓ final: {_short(final, 220)}")

        def on_after_tool(state, name, args, result):
            print(f"[cabeza] step {state.step} ← {name}: {_short(result, 200)}")

        self.on("after_llm", on_after_llm)
        self.on("after_tool", on_after_tool)

    # ---- properties -------------------------------------------------------

    @staticmethod
    def _resolve_memory_mode(
        memory: Optional[str],
        context_management: Optional[str],
        team: Optional[str],
    ) -> Optional[str]:
        if memory is not None:
            return memory
        if context_management == "page_memory":
            return "page"
        if team == "fugue":
            return "page"
        return None

    # ---- single-shot run --------------------------------------------------

    def run(self, question: str, **kwargs: Any) -> str:
        """Run the agent on ``question`` and return only the final answer string."""
        return self.run_verbose(question, **kwargs).prediction

    def run_verbose(
        self,
        question: str,
        *,
        owner: str = "agent",
        shared_page_store: Optional[SharedPageStore] = None,
        shared_summarizer: Optional[Summarizer] = None,
        trajectory_log_path: Optional[str] = None,
        memory_log_path: Optional[str] = None,
    ) -> RunResult:
        """Run the agent on ``question`` and return a structured ``RunResult``.

        For swarm/fugue modes, the caller can pre-build the shared page store
        and summarizer and inject them so multiple Agent instances participate
        in the same team. Otherwise both are constructed per-run.
        """
        if self.team_mode is not None:
            from cabeza.swarm import run_team

            return run_team(
                self,
                question=question,
                owner=owner,
                trajectory_log_path=trajectory_log_path,
                memory_log_path=memory_log_path,
            )

        return self._run_single_agent(
            question=question,
            owner=owner,
            shared_page_store=shared_page_store,
            shared_summarizer=shared_summarizer,
            trajectory_log_path=trajectory_log_path,
            memory_log_path=memory_log_path,
        )

    # ---- internal: single-agent builder ----------------------------------

    def build_single_agent(
        self,
        *,
        owner: str = "agent",
        shared_page_store: Optional[SharedPageStore] = None,
        shared_summarizer: Optional[Summarizer] = None,
        memory_log_path: Optional[str] = None,
        extra_tools: Optional[list] = None,
    ) -> "_AgentBundle":
        """Build (but do not run) the underlying agent + auxiliary objects.

        Useful for swarm orchestrators that need to drive the same plumbing
        with custom prompts.
        """
        family = self.family
        cls = _agent_class(family)

        system_prompt = self.system_prompt
        if system_prompt is None:
            from cabeza.prompts import default_system_prompt

            system_prompt = default_system_prompt(family)

        agent_tools = list(self.tools)
        if extra_tools:
            agent_tools.extend(extra_tools)

        # Page memory plumbing (built first so we can append the memory tool).
        page_store: Optional[PageStore] = None
        summarizer: Optional[Summarizer] = None
        memory_tool: Optional[MemoryTool] = None
        if self.memory_mode == "page":
            page_store = shared_page_store if shared_page_store is not None else PageStore()
            summarizer = shared_summarizer if shared_summarizer is not None else self._build_summarizer()
            if memory_log_path is not None:
                summarizer.set_trajectory_autosave(memory_log_path)
            memory_tool = MemoryTool(
                summarizer=summarizer,
                page_store=page_store,
                max_pages=self.memory_config.memory_tool_max_pages,
                team_aware=isinstance(page_store, SharedPageStore),
            )
            agent_tools.append(memory_tool)

        agent = cls(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            system_prompt=system_prompt,
            tools=agent_tools,
            max_steps=self.max_steps,
            max_time_seconds=self.max_time_seconds,
            max_tool_rounds=self.max_tool_rounds,
            llm_config=self.llm_config,
        )

        encoder = default_encoder()
        context_manager = self._build_context_manager(
            agent=agent,
            owner=owner,
            page_store=page_store,
            summarizer=summarizer,
            encoder=encoder,
        )
        if context_manager is not None:
            agent.use(context_manager)

        install_hard_window_guard(
            agent,
            max_input_tokens=self.context_window_tokens,
            encoder=encoder,
        )

        # Replay any lifecycle hooks (verbose printer + user callbacks) onto
        # the freshly built Gizmo agent.
        for event, fn in self._hooks:
            agent.on(event, fn)

        return _AgentBundle(
            agent=agent,
            page_store=page_store,
            summarizer=summarizer,
            memory_tool=memory_tool,
            context_manager=context_manager,
            owner=owner,
        )

    def _build_context_manager(
        self,
        *,
        agent: BaseAgent,
        owner: str,
        page_store: Optional[PageStore],
        summarizer: Optional[Summarizer],
        encoder,
    ):
        strategy = self.context_management
        if strategy is None or strategy == "none":
            return None

        factory = get_strategy(strategy)
        kwargs = dict(self.context_strategy_kwargs)
        soft_budget = self.context_management_tokens or self.context_window_tokens

        if strategy == "summary":
            return factory(
                model=self.model,
                api_key=self.summary_api_key or self.api_key,
                base_url=self.summary_base_url or self.base_url,
                max_input_tokens=soft_budget,
                summary_model=self.summary_model or self.model,
                request_timeout=self.llm_config.timeout,
                **kwargs,
            )

        if strategy == "recent_k":
            return factory(
                max_input_tokens=soft_budget,
                keep_recent_turns=self.keep_recent_turns,
                family=self.family,
                encoder=encoder,
                **kwargs,
            )

        if strategy == "discard_tool":
            return factory(
                max_input_tokens=soft_budget,
                family=self.family,
                encoder=encoder,
                **kwargs,
            )

        if strategy == "discard_all":
            return factory(
                max_input_tokens=soft_budget,
                encoder=encoder,
                **kwargs,
            )

        if strategy == "page_memory":
            assert page_store is not None and summarizer is not None, (
                "page_memory strategy requires the memory subsystem to be enabled"
            )
            return factory(
                page_store=page_store,
                summarizer=summarizer,
                max_input_tokens=soft_budget,
                page_max_tokens=self.memory_config.page_max_tokens,
                encoder=encoder,
                owner=owner,
                **kwargs,
            )

        # Custom strategy — caller-supplied factory. Forward best-effort kwargs.
        return factory(max_input_tokens=soft_budget, encoder=encoder, **kwargs)

    def _build_summarizer(self) -> Summarizer:
        cfg = self.memory_config
        model = cfg.summarizer_model or self.model
        base_url = cfg.summarizer_base_url or self.base_url
        api_key = cfg.summarizer_api_key or self.api_key
        return Summarizer(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=cfg.summarizer_temperature,
            max_response_tokens=cfg.summarizer_max_response_tokens,
            request_timeout_s=self.llm_config.timeout,
        )

    # ---- internal: single-agent runner -----------------------------------

    def _run_single_agent(
        self,
        *,
        question: str,
        owner: str,
        shared_page_store: Optional[SharedPageStore],
        shared_summarizer: Optional[Summarizer],
        trajectory_log_path: Optional[str],
        memory_log_path: Optional[str],
    ) -> RunResult:
        bundle = self.build_single_agent(
            owner=owner,
            shared_page_store=shared_page_store,
            shared_summarizer=shared_summarizer,
            memory_log_path=memory_log_path,
        )

        started = time.time()
        try:
            bundle.agent.run(question)
        finally:
            # Flush any pending live window into pages when page memory is on.
            if (
                self.memory_mode == "page"
                and bundle.context_manager is not None
                and hasattr(bundle.context_manager, "flush_remaining")
            ):
                try:
                    has_final = bool(
                        bundle.agent.trajectory
                        and getattr(bundle.agent.trajectory[-1], "final_content", "")
                    )
                    bundle.context_manager.flush_remaining(
                        bundle.agent.messages,
                        drop_final_assistant=has_final,
                    )
                except Exception as exc:  # pragma: no cover - logging only
                    print(f"[{owner}] page-memory flush failed: {exc}")
            if trajectory_log_path:
                try:
                    bundle.agent.save_trajectory(trajectory_log_path)
                except Exception as exc:  # pragma: no cover - logging only
                    print(f"[{owner}] trajectory save failed: {exc}")

        elapsed = time.time() - started

        prediction = "No answer found."
        termination = "no answer found"
        if bundle.agent.trajectory:
            last = bundle.agent.trajectory[-1]
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
            for s in bundle.agent.trajectory
        ]
        messages = [
            {"role": "system", "content": bundle.agent.system_prompt},
            *bundle.agent.messages,
        ]
        extras: dict[str, Any] = {}
        if bundle.page_store is not None:
            extras["memory_page_count"] = len(bundle.page_store)
            extras["memory_pages"] = bundle.page_store.snapshot()

        return RunResult(
            question=question,
            prediction=prediction,
            final_content=prediction,
            termination=termination,
            total_rounds=bundle.agent.state.step,
            tool_call_rounds=bundle.agent.state.tool_rounds,
            total_process_time=elapsed,
            messages=messages,
            trajectory=trajectory,
            extras=extras,
        )


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


@dataclass
class _AgentBundle:
    agent: BaseAgent
    page_store: Optional[PageStore]
    summarizer: Optional[Summarizer]
    memory_tool: Optional[MemoryTool]
    context_manager: Any
    owner: str
