import copy
import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from openai import OpenAI

"""
Base classes shared by every agent family.

``BaseAgent`` provides the LLM-call / tool-call / response-parsing / tool-
execution scaffolding. Family adapters (Qwen, GLM, Kimi, DeepSeek, GPT,
GPT-OSS) override the hooks they need to match their provider's chat
template and tool-call conventions.
"""


@dataclass
class LLMConfig:
    """Generation parameters for one LLM call.

    Attributes:
        max_tokens:       Max output tokens (chat.completions field; ``GPTAgent``
                          maps this to the Responses API's ``max_output_tokens``).
        max_output_tokens: Responses API ``max_output_tokens`` field.
        temperature:      Sampling temperature.
        top_p:            Nucleus sampling parameter.
        seed:             Random seed for reproducibility.
        timeout:          HTTP timeout in seconds; also used as the OpenAI
                          client timeout.
        stream:           Whether to stream chat.completions; ``None`` lets the
                          family adapter pick a provider-safe default.
        enable_thinking:  Enable vLLM "thinking" mode by injecting
                          ``chat_template_kwargs.enable_thinking=True`` into
                          ``extra_body``.
        extra_body:       Extra request-body fields passed through to the
                          OpenAI API (e.g. vLLM extensions); merged with any
                          fields produced by ``enable_thinking``.
        store:            Responses API ``store`` field.
        truncation:       Responses API ``truncation`` field.
        parallel_tool_calls: Responses API ``parallel_tool_calls`` field.
        tool_choice:      Responses API ``tool_choice`` field.
        reasoning:        Responses API ``reasoning`` object.
        reasoning_effort: Responses API ``reasoning.effort`` field.
        reasoning_summary: Responses API ``reasoning.summary`` field.
        text:             Responses API ``text`` object.
        text_verbosity:   Responses API ``text.verbosity`` field.
        text_format:      Responses API ``text.format`` field.
        include:          Responses API ``include`` field.
        metadata:         Responses API ``metadata`` field.
        service_tier:     Responses API ``service_tier`` field.
        prompt_cache_key: Responses API ``prompt_cache_key`` field.
        safety_identifier: Responses API ``safety_identifier`` field.
    """
    max_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    seed: Optional[int] = None
    timeout: float = 120.0
    stream: Optional[bool] = None
    enable_thinking: bool = False
    extra_body: Optional[dict] = None
    store: Optional[bool] = None
    truncation: Optional[str] = None
    parallel_tool_calls: Optional[bool] = None
    tool_choice: Optional[object] = None
    reasoning: Optional[dict] = None
    reasoning_effort: Optional[str] = None
    reasoning_summary: Optional[str] = None
    text: Optional[dict] = None
    text_verbosity: Optional[str] = None
    text_format: Optional[dict] = None
    include: Optional[list[str]] = None
    metadata: Optional[dict] = None
    service_tier: Optional[str] = None
    prompt_cache_key: Optional[str] = None
    safety_identifier: Optional[str] = None


@dataclass
class RunState:
    """Per-run mutable state, reset at the start of every ``run`` / ``run_verbose`` call.

    Attributes:
        step:         Number of LLM calls completed so far (final-answer round included).
        tool_rounds:  Number of rounds in which at least one tool was invoked.
        elapsed:      Seconds since the current run started (computed live).
        stop_reason:  Stop reason — empty string for normal termination,
                      otherwise one of ``"max_steps"`` / ``"timeout"`` /
                      ``"max_tool_rounds"`` / ``"token_budget"``.
    """
    step: int = 0
    tool_rounds: int = 0
    start_time: float = field(default_factory=time.time)
    stop_reason: str = ""

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time


class ContextManager(ABC):
    """Pluggable message-list transformer applied before every LLM call.

    Each registered ``ContextManager`` receives the full message list (system
    message first, then the running conversation) and returns a transformed
    list that the next manager — and finally the LLM — sees. Common uses:

        - token budget:       count tokens, truncate or compress when over budget.
        - history trim:       drop old messages by turn or time window.
        - message rebuild:    reformat structure (e.g. merge multi-turn tool messages).
        - truncation notice:  inject a placeholder message announcing trimmed history.

    Hook vs ContextManager:
        Hooks (``before_llm`` / ``after_llm`` / ``after_tool`` / ``should_stop``)
        are lightweight observer callbacks with no return value.
        ``ContextManager`` is a message-rewriting pipeline that returns the
        transformed list.

    Order of operations on every LLM call:
        1. ``_apply_context_managers(messages)``  → every ContextManager in order.
        2. ``_fire("before_llm", state, ...)``    → hooks see the rewritten list.
        3. ``client.chat.completions.create()``   → the actual LLM request.
    """

    @abstractmethod
    def process(self, messages: list[dict], state: "RunState") -> list[dict]:
        """Transform the full message list and return the new version.

        Args:
            messages: Current full message list — first entry is the system
                      message, the rest are in chronological order.
            state:    Current ``RunState``; ``step`` / ``tool_rounds`` /
                      ``elapsed`` are readable.

        Returns:
            The transformed message list. It is passed to the next
            ``ContextManager`` or to the LLM. Return the original list
            unchanged for a no-op. The returned list is also synced back to
            the agent's internal ``messages`` attribute before the request is
            sent.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Called at the start of every ``run()``; reset internal state here.

        The default is a no-op. Subclasses with per-run state (token counters,
        round counts, cached summaries, …) should override.
        """


@dataclass
class ToolCallRecord:
    name: str
    args: dict
    result: str


@dataclass
class TrajectoryStep:
    step: int
    reasoning: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    final_content: str = ""


class BaseAgent:
    _PARSE_RETRY_LIMIT = 1
    _FINALIZE_FOLLOWUP_LIMIT = 4
    _FINAL_ANSWER_ONLY_SYSTEM_SUFFIX = (
        "The conversation has reached the configured context window. "
        "You must now provide one final answer based only on the information "
        "already present in the conversation. Do not call tools, do not request "
        "tool use, and do not continue the investigation."
    )
    _FINAL_ANSWER_ONLY_USER_PROMPT = (
        "The conversation has reached the configured context window. Provide "
        "your best final answer now based only on what is already available. "
        "Do not call any tools."
    )

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        system_prompt: str = "You are a helpful assistant.",
        tools: Optional[list] = None,
        max_steps: int = 200,
        max_time_seconds: Optional[float] = None,
        max_tool_rounds: Optional[int] = None,
        llm_config: Optional[LLMConfig] = None,
    ):
        self.model = model
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.tools = {tool.name: tool for tool in (tools or [])}
        self.max_steps = max_steps
        self.max_time_seconds = max_time_seconds
        self.max_tool_rounds = max_tool_rounds
        self.messages: list[dict] = []
        self.trajectory: list[TrajectoryStep] = []
        self.state: RunState = RunState()
        self.llm_config = llm_config or LLMConfig()
        self._force_final_answer_once = False
        self._force_final_answer_reason = ""
        self._force_final_answer_prompt = ""

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.llm_config.timeout,
        )

        # Hook registry — one ordered callback list per event.
        self._hooks: dict[str, list[Callable]] = {
            "before_llm":   [],   # (state, messages) -> None
            "after_llm":    [],   # (state, parsed)   -> None
            "after_tool":   [],   # (state, tool_name, args, result) -> None
            "should_stop":  [],   # (state)            -> Optional[str]
        }

        # ContextManager pipeline; managers run in registration order before each LLM call.
        self._context_managers: list[ContextManager] = []

    # ------------------------------------------------------------------
    # Hook registration API
    # ------------------------------------------------------------------

    def on(self, event: str, fn: Callable) -> "BaseAgent":
        """Register a lifecycle hook. Returns ``self`` for chaining.

        Events:
            before_llm(state, messages)           Fired before every LLM call; ``messages``
                                                  may be mutated in place.
            after_llm(state, parsed)              Fired after every LLM call; ``parsed``
                                                  carries ``final_content`` /
                                                  ``reasoning_content`` / ``tool_calls``.
            after_tool(state, name, args, result) Fired after each tool execution.
            should_stop(state) -> Optional[str]   Fired at the top of every loop iteration.
                                                  Returning a non-empty string triggers a
                                                  stop; the first non-empty return wins.

        Example::

            agent.on("after_llm", lambda s, p: print(p["reasoning_content"]))
            agent.on("should_stop", lambda s: "custom" if s.step > 5 else None)
        """
        if event not in self._hooks:
            raise ValueError(f"Unknown hook event: {event!r}. "
                             f"Valid events: {list(self._hooks)}")
        self._hooks[event].append(fn)
        return self

    def _fire(self, event: str, *args):
        """Fire every hook for ``event``; for ``should_stop``, return the first non-empty result."""
        for fn in self._hooks[event]:
            result = fn(*args)
            if event == "should_stop" and result:
                return result
        return None

    # ------------------------------------------------------------------
    # ContextManager registration API
    # ------------------------------------------------------------------

    def use(self, cm: ContextManager) -> "BaseAgent":
        """Register a ``ContextManager``. Returns ``self`` for chaining.

        Managers form a pipeline in registration order; before every LLM call
        each manager receives the full message list (system message first,
        followed by the running conversation history) and returns a possibly-
        transformed list that the next manager — and finally the LLM — sees.

        Example::

            agent.use(TokenBudgetManager(max_tokens=4096))
                 .use(TruncationNoticeManager(notice="[history truncated]"))
        """
        self._context_managers.append(cm)
        return self

    def _apply_context_managers(self, messages: list[dict]) -> list[dict]:
        """Run every registered ContextManager in order; return the final list."""
        for cm in self._context_managers:
            messages = cm.process(messages, self.state)
        return messages

    def _persist_processed_messages(self, messages: list[dict]) -> None:
        """Mirror the post-ContextManager message list back to ``self.messages``."""
        if not messages:
            self.messages = []
            return

        if messages[0].get("role") == "system":
            self.messages = list(messages[1:])
            return

        self.messages = list(messages)

    def _use_native_tools(self) -> bool:
        return False

    def request_final_answer_once(
        self,
        reason: str = "token_budget",
        prompt: Optional[str] = None,
    ) -> None:
        """Force the next LLM call to be a single final-answer-only call."""
        self._force_final_answer_once = True
        self._force_final_answer_reason = reason or "token_budget"
        self._force_final_answer_prompt = (
            prompt or self._FINAL_ANSWER_ONLY_USER_PROMPT
        )

    def _is_final_answer_only_call(self) -> bool:
        return bool(self._force_final_answer_once)

    def _final_answer_system_prompt(self) -> str:
        return f"{self.system_prompt}\n\n{self._FINAL_ANSWER_ONLY_SYSTEM_SUFFIX}"

    def _current_tools(self) -> dict:
        if self._is_final_answer_only_call():
            return {}
        return self.tools

    def _prepare_final_answer_only_messages(self, messages: list[dict]) -> list[dict]:
        prepared = [dict(message) for message in messages]
        if prepared and prepared[0].get("role") == "system":
            prepared[0]["content"] = self._final_answer_system_prompt()
        else:
            prepared.insert(
                0,
                {
                    "role": "system",
                    "content": self._final_answer_system_prompt(),
                },
            )

        prompt = self._force_final_answer_prompt or self._FINAL_ANSWER_ONLY_USER_PROMPT
        if not prepared or prepared[-1].get("content") != prompt:
            prepared.append({"role": "user", "content": prompt})
        return prepared

    @staticmethod
    def _strip_tool_request_markup(text: str) -> str:
        text = str(text or "")
        text = re.sub(
            r"<tool_call>\s*.*?\s*</tool_call>",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return text.strip()

    @classmethod
    def _content_from_assistant_messages(parsed: dict) -> str:
        parts: list[str] = []
        for message in parsed.get("assistant_messages") or []:
            content = message.get("content")
            if isinstance(content, str):
                content = cls._strip_tool_request_markup(content)
                if content:
                    parts.append(content)
        return "\n\n".join(parts).strip()

    def _coerce_final_answer_only_parsed(self, parsed: dict) -> dict:
        final_content = (
            self._strip_tool_request_markup(parsed.get("final_content") or "")
            or self._content_from_assistant_messages(parsed)
            or parsed.get("reasoning_content")
            or f"[stopped: {self._force_final_answer_reason or 'token_budget'}]"
        )
        coerced = dict(parsed)
        coerced["assistant_message"] = {"role": "assistant", "content": final_content}
        coerced["assistant_messages"] = [coerced["assistant_message"]]
        coerced["tool_calls"] = []
        coerced["final_content"] = final_content
        coerced["retryable"] = False
        coerced["retry_reason"] = ""
        return coerced

    def _consume_final_answer_only_call(self, parsed: dict) -> dict:
        coerced = self._coerce_final_answer_only_parsed(parsed)
        self.state.stop_reason = self._force_final_answer_reason or "token_budget"
        self._force_final_answer_once = False
        self._force_final_answer_reason = ""
        self._force_final_answer_prompt = ""
        return coerced

    def _build_extra_body(self) -> Optional[dict]:
        """Merge the ``enable_thinking`` flag into the user-supplied ``extra_body``."""
        cfg = self.llm_config
        base = copy.deepcopy(cfg.extra_body) if cfg.extra_body else {}

        if cfg.enable_thinking:
            tmpl = base.setdefault("chat_template_kwargs", {})
            tmpl["enable_thinking"] = True

        return base or None

    def _default_stream(self) -> bool:
        return False

    def _should_stream_response(self) -> bool:
        if self.llm_config.stream is not None:
            return bool(self.llm_config.stream)
        return self._default_stream()

    def _prepare_request_kwargs(self, kwargs: dict) -> dict:
        """Subclass hook: adjust OpenAI-compatible request kwargs before sending."""
        return kwargs

    def _prepare_messages_for_llm(self, messages: list[dict]) -> list[dict]:
        """Subclass hook: validate / normalize the message list right before the request."""
        return messages

    def _call_llm(self):
        raw_messages = [{"role": "system", "content": self.system_prompt}] + self.messages
        messages = self._apply_context_managers(raw_messages)
        if self._is_final_answer_only_call():
            messages = self._prepare_final_answer_only_messages(messages)
        messages = self._prepare_messages_for_llm(messages)
        self._persist_processed_messages(messages)
        cfg = self.llm_config

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
        }

        if cfg.max_tokens is not None:
            kwargs["max_tokens"] = cfg.max_tokens
        if cfg.temperature is not None:
            kwargs["temperature"] = cfg.temperature
        if cfg.seed is not None:
            kwargs["seed"] = cfg.seed

        extra_body = self._build_extra_body()
        if extra_body:
            kwargs["extra_body"] = extra_body

        if self._use_native_tools():
            tools = [tool.to_schema() for tool in self._current_tools().values()] or None
            if tools:
                kwargs["tools"] = tools

        return self.client.chat.completions.create(**kwargs)

    def _execute_tool(self, tool_name: str, tool_args: dict) -> str:
        tool = self.tools.get(tool_name)
        if not tool:
            return f"Error: tool '{tool_name}' not found"

        try:
            result = tool.execute(**tool_args)
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return f"Error while executing tool '{tool_name}': {str(e)}"

    def _parse_response(self, raw_content: str) -> dict:
        raise NotImplementedError

    def _build_tool_result_messages(self, tool_name: str, tool_result: str) -> list[dict]:
        raise NotImplementedError

    def _parse_llm_response(self, response) -> dict:
        message = response.choices[0].message
        raw_content = getattr(message, "content", "") or ""
        return self._parse_response(raw_content)

    @staticmethod
    def _assistant_messages_from_parsed(parsed: dict) -> list[dict]:
        assistant_messages = parsed.get("assistant_messages")
        if assistant_messages:
            return list(assistant_messages)
        return [parsed["assistant_message"]]

    def _should_retry_parsed_response(self, parsed: dict) -> tuple[bool, str]:
        if parsed.get("retryable"):
            return True, str(parsed.get("retry_reason") or "retryable_parse")
        return False, ""

    @staticmethod
    def _log_retry(reason: str, *, phase: str = "run") -> None:
        print(f"[Recovery] {phase}: {reason}. Retrying with same context.")

    def _execute_parsed_tool_calls(self, step: TrajectoryStep, tool_calls: list[dict]) -> None:
        for tc in tool_calls:
            tool_name = tc["function"]["name"]
            tool_args = tc["function"].get("arguments", {})
            if not isinstance(tool_args, dict):
                tool_args = {}

            tool_result = self._execute_tool(tool_name, tool_args)
            step.tool_calls.append(
                ToolCallRecord(name=tool_name, args=tool_args, result=tool_result)
            )
            self._fire("after_tool", self.state, tool_name, tool_args, tool_result)

            for message in self._build_tool_result_messages(tool_name, tool_result):
                self.messages.append(message)

        self.state.tool_rounds += 1

    def _on_stop(self, stop_reason: str) -> Optional[str]:
        """Return the user message injected on the final round when a stop condition fires.

        ``None`` skips injection. Subclasses may override to customize the
        finalization prompt per stop reason.
        """
        prompts = {
            "max_steps":      "You have reached the maximum number of steps. Please provide your best answer now based on what you have found so far.",
            "timeout":        "Time is running out. Please provide your best answer now based on what you have found so far.",
            "max_tool_rounds": "You have used the maximum number of tool calls. Please provide your final answer now based on what you have gathered.",
            "token_budget":   "The conversation has reached the context length limit. Please provide your best answer now based on what you have found so far.",
        }
        return prompts.get(stop_reason)

    def _check_stop(self) -> str:
        """Check stop conditions; return the stop-reason string, or ``""`` when none fired.

        Both built-in conditions and ``should_stop`` hooks are evaluated;
        built-in conditions take precedence over hook returns.
        """
        if self.max_time_seconds and self.state.elapsed >= self.max_time_seconds:
            return "timeout"
        if self.max_tool_rounds is not None and self.state.tool_rounds >= self.max_tool_rounds:
            return "max_tool_rounds"
        hook_reason = self._fire("should_stop", self.state)
        return hook_reason or ""

    def _finalize(self, stop_reason: str) -> dict:
        """After a stop condition fires, optionally inject a final prompt and call the LLM once more."""
        self.state.stop_reason = stop_reason
        stop_msg = self._on_stop(stop_reason)
        if not stop_msg:
            return {"final_content": f"[stopped: {stop_reason}]", "reasoning_content": "", "tool_calls": []}

        self.messages.append({"role": "user", "content": stop_msg})
        retry_budget = self._PARSE_RETRY_LIMIT

        for _ in range(self._FINALIZE_FOLLOWUP_LIMIT):
            self.state.step += 1
            self._fire("before_llm", self.state, self.messages)
            resp = self._call_llm()
            parsed = self._parse_llm_response(resp)
            self._fire("after_llm", self.state, parsed)

            if self._is_final_answer_only_call():
                parsed = self._consume_final_answer_only_call(parsed)
                self.messages.extend(self._assistant_messages_from_parsed(parsed))
                step = TrajectoryStep(
                    step=self.state.step,
                    reasoning=parsed.get("reasoning_content", ""),
                    final_content=parsed.get("final_content", ""),
                )
                self.trajectory.append(step)
                return parsed

            should_retry, retry_reason = self._should_retry_parsed_response(parsed)
            if should_retry and retry_budget > 0:
                retry_budget -= 1
                self._log_retry(retry_reason, phase="finalize")
                continue

            retry_budget = self._PARSE_RETRY_LIMIT
            self.messages.extend(self._assistant_messages_from_parsed(parsed))
            step = TrajectoryStep(
                step=self.state.step,
                reasoning=parsed.get("reasoning_content", ""),
            )

            tool_calls = parsed["tool_calls"]
            if not tool_calls:
                step.final_content = parsed.get("final_content", "")
                self.trajectory.append(step)
                return parsed

            self._execute_parsed_tool_calls(step, tool_calls)
            self.trajectory.append(step)

        return {"final_content": f"[stopped: {stop_reason}]", "reasoning_content": "", "tool_calls": []}

    def _run_loop(self, user_input: str) -> dict:
        self.messages = []
        self.trajectory = []
        self.state = RunState()
        self._force_final_answer_once = False
        self._force_final_answer_reason = ""
        self._force_final_answer_prompt = ""
        for cm in self._context_managers:
            cm.reset()
        self.messages.append({"role": "user", "content": user_input})

        retry_budget = self._PARSE_RETRY_LIMIT
        step_idx = 0
        while step_idx < self.max_steps:
            # Check stop conditions before every LLM call.
            stop_reason = self._check_stop()
            if stop_reason:
                return self._finalize(stop_reason)

            step_idx += 1
            self.state.step = step_idx

            self._fire("before_llm", self.state, self.messages)
            resp = self._call_llm()
            parsed = self._parse_llm_response(resp)
            self._fire("after_llm", self.state, parsed)

            if self._is_final_answer_only_call():
                parsed = self._consume_final_answer_only_call(parsed)
                self.messages.extend(self._assistant_messages_from_parsed(parsed))
                step = TrajectoryStep(
                    step=self.state.step,
                    reasoning=parsed.get("reasoning_content", ""),
                    final_content=parsed.get("final_content", ""),
                )
                self.trajectory.append(step)
                return parsed

            tool_calls = parsed["tool_calls"]
            final_content = parsed["final_content"]
            reasoning = parsed.get("reasoning_content", "")

            should_retry, retry_reason = self._should_retry_parsed_response(parsed)
            if should_retry and retry_budget > 0:
                retry_budget -= 1
                self._log_retry(retry_reason)
                continue

            retry_budget = self._PARSE_RETRY_LIMIT
            self.messages.extend(self._assistant_messages_from_parsed(parsed))
            step = TrajectoryStep(step=self.state.step, reasoning=reasoning)

            if not tool_calls:
                step.final_content = final_content
                self.trajectory.append(step)
                self.state.stop_reason = ""
                return parsed

            self._execute_parsed_tool_calls(step, tool_calls)
            self.trajectory.append(step)

        return self._finalize("max_steps")

    def run(self, user_input: str) -> str:
        parsed = self._run_loop(user_input)
        return parsed.get("final_content") or ""

    def run_verbose(self, user_input: str) -> dict:
        """Run the agent and return the full parsed result: ``final_content``, ``reasoning_content``, ``tool_calls``."""
        return self._run_loop(user_input)

    def print_trajectory(self) -> None:
        """Print the full trajectory for debugging."""
        s = self.state
        print(f"\n[RunState] steps={s.step}  tool_rounds={s.tool_rounds}  "
              f"elapsed={s.elapsed:.1f}s  stop_reason={s.stop_reason or 'normal'}")
        for step in self.trajectory:
            print(f"\n{'='*60}")
            print(f"Step {step.step}")
            print(f"{'='*60}")
            if step.reasoning:
                print(f"[Reasoning]\n{step.reasoning}")
            if step.tool_calls:
                for tc in step.tool_calls:
                    print(f"\n[Tool Call] {tc.name}")
                    print(f"  args:   {json.dumps(tc.args, ensure_ascii=False)}")
                    print(f"  result: {tc.result}")
            if step.final_content:
                print(f"\n[Final Answer]\n{step.final_content}")

    def save_trajectory(self, path: str) -> None:
        """Save the full conversation trajectory (system → final answer) to a JSON file."""
        full_messages = [
            {"role": "system", "content": self.system_prompt},
            *self.messages,
        ]
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(full_messages, f, ensure_ascii=False, indent=2)


class NativeToolChatAgent(BaseAgent):
    """OpenAI-compatible chat.completions agent using structured tool calls."""

    def _use_native_tools(self) -> bool:
        return True

    def _default_stream(self) -> bool:
        # GLM/Kimi thinking responses can be very large. Streaming avoids provider
        # and proxy timeouts while preserving the same parsed message shape.
        return True

    @staticmethod
    def _get_field(obj: Any, name: str, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    @classmethod
    def _normalize_content(cls, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    text = item
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content") or ""
                else:
                    text = str(item)
                if text:
                    parts.append(str(text))
            return "".join(parts).strip()
        return str(content).strip()

    @staticmethod
    def _safe_load_arguments(raw_args: Any) -> Any:
        if isinstance(raw_args, (dict, list, int, float, bool)):
            return raw_args
        if raw_args is None:
            return {}
        if not isinstance(raw_args, str):
            return raw_args

        text = raw_args.strip()
        if not text:
            return {}

        try:
            return json.loads(text)
        except Exception:
            return raw_args

    @staticmethod
    def _serialize_arguments(arguments: Any) -> str:
        if isinstance(arguments, str):
            return arguments
        return json.dumps(arguments or {}, ensure_ascii=False)

    @classmethod
    def _to_plain_data(cls, value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            return cls._to_plain_data(value.model_dump())
        if isinstance(value, dict):
            return {key: cls._to_plain_data(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._to_plain_data(item) for item in value]
        if isinstance(value, tuple):
            return [cls._to_plain_data(item) for item in value]
        return copy.deepcopy(value)

    def _collect_stream_response(self, stream: Any) -> dict:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        reasoning_details_parts: list[Any] = []
        tool_calls_by_index: dict[int, dict[str, Any]] = {}
        role = "assistant"

        for chunk in stream:
            choices = self._get_field(chunk, "choices", []) or []
            for choice in choices:
                delta = self._get_field(choice, "delta", {}) or {}
                delta_role = self._get_field(delta, "role")
                if delta_role:
                    role = delta_role

                reasoning_piece = (
                    self._get_field(delta, "reasoning_content")
                    or self._get_field(delta, "reasoning")
                    or self._get_field(delta, "thinking")
                )
                if reasoning_piece:
                    reasoning_parts.append(str(reasoning_piece))

                reasoning_details_piece = self._get_field(delta, "reasoning_details")
                if reasoning_details_piece:
                    plain_reasoning_details = self._to_plain_data(reasoning_details_piece)
                    if isinstance(plain_reasoning_details, list):
                        reasoning_details_parts.extend(plain_reasoning_details)
                    else:
                        reasoning_details_parts.append(plain_reasoning_details)

                content_piece = self._get_field(delta, "content")
                if content_piece:
                    content_parts.append(str(content_piece))

                raw_tool_calls = self._get_field(delta, "tool_calls") or []
                for fallback_index, tool_call in enumerate(raw_tool_calls):
                    index = self._get_field(tool_call, "index", fallback_index)
                    try:
                        index = int(index)
                    except Exception:
                        index = fallback_index

                    state = tool_calls_by_index.setdefault(
                        index,
                        {
                            "id": "",
                            "type": "function",
                            "function": {
                                "name": "",
                                "arguments": "",
                            },
                        },
                    )

                    tool_call_id = self._get_field(tool_call, "id")
                    if tool_call_id:
                        state["id"] = tool_call_id
                    tool_call_type = self._get_field(tool_call, "type")
                    if tool_call_type:
                        state["type"] = tool_call_type

                    function = self._get_field(tool_call, "function", {}) or {}
                    tool_name = (
                        self._get_field(function, "name")
                        or self._get_field(tool_call, "name")
                    )
                    if tool_name:
                        state["function"]["name"] += str(tool_name)
                    arguments = self._get_field(function, "arguments")
                    if arguments:
                        state["function"]["arguments"] += str(arguments)

        message: dict[str, Any] = {
            "role": role,
            "content": "".join(content_parts),
        }
        if reasoning_parts:
            message["reasoning_content"] = "".join(reasoning_parts)
        if reasoning_details_parts:
            message["reasoning_details"] = reasoning_details_parts

        tool_calls = []
        for _, tool_call in sorted(tool_calls_by_index.items()):
            if not tool_call["function"].get("name"):
                continue
            cleaned = {
                "type": tool_call.get("type") or "function",
                "function": {
                    "name": tool_call["function"].get("name", ""),
                    "arguments": tool_call["function"].get("arguments", ""),
                },
            }
            if tool_call.get("id"):
                cleaned["id"] = tool_call["id"]
            tool_calls.append(cleaned)
        if tool_calls:
            message["tool_calls"] = tool_calls

        return {"choices": [{"message": message}]}

    def _call_llm(self):
        raw_messages = [{"role": "system", "content": self.system_prompt}] + self.messages
        messages = self._apply_context_managers(raw_messages)
        messages = self._prepare_messages_for_llm(messages)
        if self._is_final_answer_only_call():
            messages = self._prepare_final_answer_only_messages(messages)
            messages = self._prepare_messages_for_llm(messages)
        self._persist_processed_messages(messages)
        cfg = self.llm_config

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }

        if cfg.max_tokens is not None:
            kwargs["max_tokens"] = cfg.max_tokens
        if cfg.temperature is not None:
            kwargs["temperature"] = cfg.temperature
        if cfg.top_p is not None:
            kwargs["top_p"] = cfg.top_p
        if cfg.seed is not None:
            kwargs["seed"] = cfg.seed
        if cfg.tool_choice is not None and not self._is_final_answer_only_call():
            kwargs["tool_choice"] = copy.deepcopy(cfg.tool_choice)

        extra_body = self._build_extra_body()
        if extra_body:
            kwargs["extra_body"] = extra_body

        tools = [tool.to_schema() for tool in self._current_tools().values()] or None
        if tools:
            kwargs["tools"] = tools

        if self._should_stream_response():
            kwargs["stream"] = True

        kwargs = self._prepare_request_kwargs(kwargs)
        response = self.client.chat.completions.create(**kwargs)
        if kwargs.get("stream"):
            return self._collect_stream_response(response)
        return response

    def _parse_native_tool_calls(self, raw_tool_calls: Any) -> tuple[list[dict], list[dict]]:
        if not raw_tool_calls:
            return [], []

        parsed_tool_calls: list[dict] = []
        history_tool_calls: list[dict] = []

        for tool_call in raw_tool_calls:
            function = self._get_field(tool_call, "function", {}) or {}
            tool_name = (
                self._get_field(function, "name")
                or self._get_field(tool_call, "name")
                or ""
            ).strip()
            if not tool_name:
                continue

            raw_arguments = self._get_field(function, "arguments")
            if raw_arguments is None:
                raw_arguments = self._get_field(tool_call, "arguments")

            tool_call_id = self._get_field(tool_call, "id")
            tool_call_type = self._get_field(tool_call, "type") or "function"

            parsed_tool_call = {
                "type": tool_call_type,
                "function": {
                    "name": tool_name,
                    "arguments": self._safe_load_arguments(raw_arguments),
                },
            }
            history_tool_call = {
                "type": tool_call_type,
                "function": {
                    "name": tool_name,
                    "arguments": self._serialize_arguments(raw_arguments),
                },
            }

            if tool_call_id:
                parsed_tool_call["id"] = tool_call_id
                history_tool_call["id"] = tool_call_id

            parsed_tool_calls.append(parsed_tool_call)
            history_tool_calls.append(history_tool_call)

        return parsed_tool_calls, history_tool_calls

    def _parse_response_message(self, msg: Any) -> dict:
        raw_content = self._normalize_content(self._get_field(msg, "content"))
        reasoning_details = self._to_plain_data(
            self._get_field(msg, "reasoning_details")
        )
        reasoning_content = self._normalize_content(
            self._get_field(msg, "reasoning_content")
            or self._get_field(msg, "reasoning")
            or self._get_field(msg, "thinking")
        )
        tool_calls, history_tool_calls = self._parse_native_tool_calls(
            self._get_field(msg, "tool_calls")
        )

        if tool_calls and raw_content and not reasoning_content:
            reasoning_content = raw_content

        assistant_message: dict[str, Any] = {"role": "assistant"}
        if raw_content or tool_calls or not history_tool_calls:
            assistant_message["content"] = raw_content
        if reasoning_details is not None:
            assistant_message["reasoning_details"] = reasoning_details
        elif reasoning_content:
            assistant_message["reasoning_content"] = reasoning_content
        if history_tool_calls:
            assistant_message["tool_calls"] = history_tool_calls

        final_content = "" if tool_calls else raw_content

        return {
            "assistant_message": assistant_message,
            "assistant_messages": [assistant_message],
            "tool_calls": tool_calls,
            "reasoning_content": reasoning_content,
            "reasoning_details": reasoning_details,
            "final_content": final_content,
        }

    def _parse_llm_response(self, response) -> dict:
        choices = self._get_field(response, "choices", []) or []
        if not choices:
            return self._parse_response_message({})
        first_choice = choices[0]
        message = self._get_field(first_choice, "message", {})
        return self._parse_response_message(message)

    def _parse_response(self, raw_content: str) -> dict:
        raw_content = self._normalize_content(raw_content)
        assistant_message = {"role": "assistant", "content": raw_content}
        return {
            "assistant_message": assistant_message,
            "assistant_messages": [assistant_message],
            "tool_calls": [],
            "reasoning_content": "",
            "final_content": raw_content,
        }

    def _build_tool_result_messages(
        self,
        tool_name: str,
        tool_result: str,
        *,
        tool_call_id: Optional[str] = None,
    ) -> list[dict]:
        raise NotImplementedError

    def _execute_parsed_tool_calls(self, step: TrajectoryStep, tool_calls: list[dict]) -> None:
        for tc in tool_calls:
            tool_name = tc["function"]["name"]
            tool_args = tc["function"].get("arguments", {})
            if not isinstance(tool_args, dict):
                tool_args = {}

            tool_result = self._execute_tool(tool_name, tool_args)
            step.tool_calls.append(
                ToolCallRecord(name=tool_name, args=tool_args, result=tool_result)
            )
            self._fire("after_tool", self.state, tool_name, tool_args, tool_result)

            for message in self._build_tool_result_messages(
                tool_name,
                tool_result,
                tool_call_id=tc.get("id"),
            ):
                self.messages.append(message)

        self.state.tool_rounds += 1
