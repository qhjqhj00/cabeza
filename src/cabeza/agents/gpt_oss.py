import ast
import json
import re
from typing import Any, Optional

from cabeza.base import (
    BaseAgent,
    RunState,
    ToolCallRecord,
    TrajectoryStep,
)
from cabeza.prompts._defaults import GPT_OSS_SYSTEM_PROMPT


class GPTOssAgent(BaseAgent):
    """Agent adapter for the gpt-oss chat template.

    Design goals:
    - Preserve the same ``run`` / ``run_verbose`` / trajectory behavior as
      ``QwenAgent``.
    - Prefer native tool calling and rely on the server-side chat template
      to render tool declarations.
    - Handle two response shapes interchangeably:
        1. OpenAI/vLLM has already parsed structured ``tool_calls`` /
           ``thinking`` fields.
        2. The model emits raw gpt-oss template token fragments directly.
    """

    _ANALYSIS_PATTERN = re.compile(
        r"(?:<\|start\|>assistant)?<\|channel\|>analysis<\|message\|>"
        r"(.*?)(?:<\|end\|>|<\|return\|>)",
        re.DOTALL,
    )
    _FINAL_PATTERN = re.compile(
        r"(?:<\|start\|>assistant)?<\|channel\|>final<\|message\|>"
        r"(.*?)(?:<\|end\|>|<\|return\|>)",
        re.DOTALL,
    )
    _TOOL_PATTERN = re.compile(
        r"(?:<\|start\|>assistant)?\s*to=functions\.([^\s<]+)"
        r"<\|channel\|>commentary(?: [^<]*)?<\|message\|>"
        r"(.*?)(?:<\|call\|>|<\|end\|>|<\|return\|>)",
        re.DOTALL,
    )
    _TEMPLATE_MARKERS = (
        "<|start|>",
        "<|channel|>",
        "<|message|>",
        "<|end|>",
        "<|return|>",
    )

    def __init__(self, *args, system_prompt: str = GPT_OSS_SYSTEM_PROMPT, **kwargs):
        super().__init__(*args, system_prompt=system_prompt, **kwargs)

    def _use_native_tools(self) -> bool:
        return True

    @classmethod
    def _contains_template_markup(cls, text: str) -> bool:
        return bool(text and any(marker in text for marker in cls._TEMPLATE_MARKERS))

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
                    if not text and isinstance(item.get("text"), dict):
                        text = (
                            item["text"].get("value")
                            or item["text"].get("content")
                            or ""
                        )
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

        for loader in (json.loads, ast.literal_eval):
            try:
                return loader(text)
            except Exception:
                continue

        return raw_args

    @classmethod
    def _extract_tool_name(cls, tool_call: Any) -> str:
        function = cls._get_field(tool_call, "function", {}) or {}
        return (
            cls._get_field(function, "name")
            or cls._get_field(tool_call, "name")
            or ""
        ).strip()

    @classmethod
    def _build_assistant_history_messages(
        cls,
        tool_calls: list[dict],
        reasoning_content: str,
        final_content: str,
    ) -> list[dict]:
        if tool_calls:
            message: dict[str, Any] = {
                "role": "assistant",
                "tool_calls": [tool_calls[0]],
            }
            if reasoning_content:
                message["thinking"] = reasoning_content
            return [message]

        return [{"role": "assistant", "content": final_content}]

    @classmethod
    def _validate_parsed_response_shape(
        cls,
        *,
        tool_calls: list[dict],
        reasoning_content: str,
        final_content: str,
    ) -> None:
        if tool_calls and final_content:
            raise ValueError(
                "gpt-oss assistant output cannot contain both tool calls and final content "
                "in the same turn."
            )
        if len(tool_calls) > 1:
            raise ValueError(
                "gpt-oss assistant output must contain at most one tool call per response."
            )
        if reasoning_content and cls._contains_template_markup(reasoning_content):
            raise ValueError(
                "assistant reasoning stored in history must not already contain gpt-oss "
                "template channel markers."
            )

    def _validate_messages_for_template(self, messages: list[dict]) -> None:
        awaiting_tool_result = False

        for idx, message in enumerate(messages):
            role = message.get("role")

            if idx == 0 and role in {"system", "developer"}:
                continue
            if role in {"system", "developer"}:
                raise ValueError(
                    "gpt-oss template only supports a system/developer message in the first "
                    "position."
                )
            if role == "user":
                if awaiting_tool_result:
                    raise ValueError(
                        "tool result must immediately follow an assistant tool-call message."
                    )
                continue
            if role == "tool":
                if not awaiting_tool_result:
                    raise ValueError(
                        "tool messages must immediately follow an assistant message with a "
                        "single tool call."
                    )
                awaiting_tool_result = False
                continue
            if role != "assistant":
                raise ValueError(f"Unsupported gpt-oss message role: {role!r}")

            if awaiting_tool_result:
                raise ValueError(
                    "assistant tool-call messages must be followed by a tool message before "
                    "the next assistant turn."
                )

            content = self._normalize_content(message.get("content"))
            thinking = self._normalize_content(message.get("thinking"))
            if self._contains_template_markup(content):
                raise ValueError(
                    "assistant.content already contains gpt-oss channel markers; pass plain "
                    "text and let the official template render it."
                )
            if self._contains_template_markup(thinking):
                raise ValueError(
                    "assistant.thinking already contains gpt-oss channel markers; pass plain "
                    "text and let the official template render it."
                )

            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                continue
            if len(tool_calls) != 1:
                raise ValueError(
                    "gpt-oss official template assumes at most one tool call per assistant "
                    "message."
                )
            if content and thinking:
                raise ValueError(
                    "assistant messages with tool_calls must use either content or thinking, "
                    "not both."
                )
            if not self._extract_tool_name(tool_calls[0]):
                raise ValueError("assistant tool call is missing a function name.")
            awaiting_tool_result = True

        if awaiting_tool_result:
            raise ValueError(
                "assistant tool-call message is missing its following tool result."
            )

    def _prepare_messages_for_llm(self, messages: list[dict]) -> list[dict]:
        self._validate_messages_for_template(messages)
        return messages

    def _parse_native_tool_calls(self, raw_tool_calls: Any) -> list[dict]:
        if not raw_tool_calls:
            return []

        results: list[dict] = []
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

            parsed_tool_call = {
                "function": {
                    "name": tool_name,
                    "arguments": self._safe_load_arguments(raw_arguments),
                }
            }

            tool_call_id = self._get_field(tool_call, "id")
            if tool_call_id:
                parsed_tool_call["id"] = tool_call_id

            results.append(parsed_tool_call)

        return results

    def _parse_raw_response(self, raw_content: str) -> dict:
        raw_content = (raw_content or "").strip()
        if not raw_content:
            return {
                "assistant_message": {"role": "assistant", "content": ""},
                "assistant_messages": [{"role": "assistant", "content": ""}],
                "tool_calls": [],
                "reasoning_content": "",
                "final_content": "",
            }

        tool_calls: list[dict] = []
        for tool_name, raw_arguments in self._TOOL_PATTERN.findall(raw_content):
            tool_calls.append(
                {
                    "function": {
                        "name": tool_name.strip(),
                        "arguments": self._safe_load_arguments(raw_arguments),
                    }
                }
            )

        reasoning_blocks = [
            block.strip() for block in self._ANALYSIS_PATTERN.findall(raw_content) if block.strip()
        ]
        reasoning_content = "\n\n".join(reasoning_blocks)

        final_blocks = [
            block.strip() for block in self._FINAL_PATTERN.findall(raw_content) if block.strip()
        ]
        final_content = "\n\n".join(final_blocks)

        if not final_content and not tool_calls:
            cleaned = self._ANALYSIS_PATTERN.sub("", raw_content)
            cleaned = self._TOOL_PATTERN.sub("", cleaned)
            cleaned = self._FINAL_PATTERN.sub(r"\1", cleaned)
            cleaned = cleaned.replace("<|end|>", "").replace("<|return|>", "")
            final_content = cleaned.strip()

        self._validate_parsed_response_shape(
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
            final_content=final_content,
        )
        assistant_messages = self._build_assistant_history_messages(
            tool_calls,
            reasoning_content,
            final_content,
        )
        assistant_message = assistant_messages[0]

        return {
            "assistant_message": assistant_message,
            "assistant_messages": assistant_messages,
            "tool_calls": tool_calls,
            "reasoning_content": reasoning_content,
            "final_content": final_content,
        }

    def _extract_reasoning_content(
        self,
        msg: Any,
        raw_content: str,
        *,
        tool_calls: list[dict],
        raw_parsed: Optional[dict] = None,
    ) -> str:
        explicit_reasoning = ""
        for field in ("reasoning_content", "reasoning", "thinking"):
            value = self._normalize_content(self._get_field(msg, field))
            if value:
                explicit_reasoning = value
                break

        if explicit_reasoning:
            if tool_calls and raw_content and not self._contains_template_markup(raw_content):
                raise ValueError(
                    "assistant message with tool_calls cannot contain both plain-text content "
                    "and thinking/reasoning fields."
                )
            return explicit_reasoning

        parsed = raw_parsed or self._parse_raw_response(raw_content)
        parsed_reasoning = parsed.get("reasoning_content", "")
        if parsed_reasoning:
            return parsed_reasoning

        if tool_calls and raw_content and not self._contains_template_markup(raw_content):
            return raw_content.strip()

        return ""

    def _parse_response_message(self, msg: Any) -> dict:
        raw_content = self._normalize_content(self._get_field(msg, "content"))
        native_tool_calls = self._parse_native_tool_calls(self._get_field(msg, "tool_calls"))
        raw_parsed = self._parse_raw_response(raw_content)

        tool_calls = native_tool_calls or raw_parsed["tool_calls"]
        reasoning_content = self._extract_reasoning_content(
            msg,
            raw_content,
            tool_calls=tool_calls,
            raw_parsed=raw_parsed,
        )
        final_content = raw_parsed["final_content"]
        if tool_calls and not raw_parsed["tool_calls"] and not self._contains_template_markup(raw_content):
            final_content = ""

        if not final_content and not tool_calls:
            final_content = raw_content.strip()

        self._validate_parsed_response_shape(
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
            final_content=final_content,
        )
        assistant_messages = self._build_assistant_history_messages(
            tool_calls,
            reasoning_content,
            final_content,
        )
        assistant_message = assistant_messages[0]

        return {
            "assistant_message": assistant_message,
            "assistant_messages": assistant_messages,
            "tool_calls": tool_calls,
            "reasoning_content": reasoning_content,
            "final_content": final_content,
        }

    def _parse_response(self, raw_content: str) -> dict:
        return self._parse_raw_response(raw_content)

    def _build_tool_result_messages(
        self,
        tool_name: str,
        tool_result: str,
        tool_call_id: Optional[str] = None,
    ) -> list[dict]:
        message = {
            "role": "tool",
            "content": tool_result,
        }
        if tool_call_id:
            message["tool_call_id"] = tool_call_id
        return [message]

    def _finalize(self, stop_reason: str) -> dict:
        """On stop, run one more round formatted in gpt-oss's message structure to elicit the final answer."""
        self.state.stop_reason = stop_reason
        stop_msg = self._on_stop(stop_reason)
        if stop_msg:
            self.messages.append({"role": "user", "content": stop_msg})
            resp = self._call_llm()
            parsed = self._parse_response_message(resp.choices[0].message)
            if parsed["tool_calls"]:
                raise ValueError(
                    "stop-finalization response must be a final assistant answer, not a tool "
                    "call."
                )
            self.messages.extend(parsed["assistant_messages"])
            step = TrajectoryStep(
                step=self.state.step + 1,
                reasoning=parsed.get("reasoning_content", ""),
                final_content=parsed.get("final_content", ""),
            )
            self.trajectory.append(step)
            return parsed

        return {
            "final_content": f"[stopped: {stop_reason}]",
            "reasoning_content": "",
            "tool_calls": [],
        }

    def _run_loop(self, user_input: str) -> dict:
        self.messages = []
        self.trajectory = []
        self.state = RunState()
        for cm in self._context_managers:
            cm.reset()
        self.messages.append({"role": "user", "content": user_input})

        for step_idx in range(self.max_steps):
            stop_reason = self._check_stop()
            if stop_reason:
                return self._finalize(stop_reason)

            self.state.step = step_idx + 1

            self._fire("before_llm", self.state, self.messages)
            resp = self._call_llm()
            msg = resp.choices[0].message
            parsed = self._parse_response_message(msg)

            assistant_messages = parsed.get("assistant_messages") or [parsed["assistant_message"]]
            tool_calls = parsed["tool_calls"]
            final_content = parsed["final_content"]
            reasoning = parsed.get("reasoning_content", "")

            self._fire("after_llm", self.state, parsed)

            step = TrajectoryStep(step=self.state.step, reasoning=reasoning)

            if not tool_calls:
                self.messages.extend(assistant_messages)
                step.final_content = final_content
                self.trajectory.append(step)
                self.state.stop_reason = ""
                return parsed

            assistant_message = assistant_messages[0]
            tc = tool_calls[0]
            self.messages.append(assistant_message)
            tool_name = tc["function"]["name"]
            tool_args = tc["function"].get("arguments", {})
            if not isinstance(tool_args, dict):
                tool_args = {}

            tool_result = self._execute_tool(tool_name, tool_args)
            step.tool_calls.append(
                ToolCallRecord(name=tool_name, args=tool_args, result=tool_result)
            )
            self._fire("after_tool", self.state, tool_name, tool_args, tool_result)

            tool_call_id = tc.get("id")
            for message in self._build_tool_result_messages(
                tool_name,
                tool_result,
                tool_call_id=tool_call_id,
            ):
                self.messages.append(message)

            self.state.tool_rounds += 1
            self.trajectory.append(step)

        return self._finalize("max_steps")
