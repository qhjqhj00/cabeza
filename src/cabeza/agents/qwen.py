import ast
import json
import re
from typing import Any

from cabeza.base import BaseAgent, ToolCallRecord
from cabeza.prompts._defaults import QWEN_SYSTEM_PROMPT


class QwenAgent(BaseAgent):
    """Qwen adapter that uses prompt-injected XML tools and local parsing.

    This intentionally does not send OpenAI-compatible ``tools`` to vLLM. The
    Qwen chat template can render tool schemas into the system prompt, but when
    vLLM reasoning/tool parsers are disabled we need to reproduce that contract
    locally and parse raw assistant content ourselves.
    """

    _TOOL_MARKERS = (
        "<tool_call",
        "</tool_call>",
        "<function=",
        "</function>",
        "<parameter=",
        "</parameter>",
    )

    _TOOL_INSTRUCTION = """# Tools

You have access to the following functions:

<tools>
{tools_text}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>"""

    def __init__(self, *args, system_prompt: str = QWEN_SYSTEM_PROMPT, **kwargs):
        super().__init__(*args, system_prompt=system_prompt, **kwargs)
        self.system_prompt = self._build_system_prompt()

    def _default_stream(self) -> bool:
        return False

    def _tool_schema_for_prompt(self, tool) -> dict:
        return tool.to_schema()

    def _build_tools_text(self) -> str:
        tool_blocks = []
        for tool in self._current_tools().values():
            tool_blocks.append(
                json.dumps(
                    self._tool_schema_for_prompt(tool),
                    ensure_ascii=False,
                )
            )
        return "\n".join(tool_blocks)

    def _build_system_prompt(self) -> str:
        tools_text = self._build_tools_text()
        if "{tool_des}" in self.system_prompt:
            replacement = f"<tools>\n{tools_text}\n</tools>" if tools_text else ""
            return self.system_prompt.format(tool_des=replacement)

        if not tools_text:
            return self.system_prompt

        tool_instruction = self._TOOL_INSTRUCTION.format(tools_text=tools_text).strip()
        return f"{tool_instruction}\n\n{self.system_prompt}"

    def _final_answer_system_prompt(self) -> str:
        prompt = self.system_prompt
        tool_header = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
        if prompt.startswith(tool_header):
            important_end = "</IMPORTANT>"
            important_idx = prompt.find(important_end)
            if important_idx >= 0:
                prompt = prompt[important_idx + len(important_end) :].lstrip()
        return f"{prompt}\n\n{self._FINAL_ANSWER_ONLY_SYSTEM_SUFFIX}"

    @staticmethod
    def _normalize_content(content: Any) -> str:
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
    def _get_field(obj: Any, name: str, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    def _call_llm(self):
        raw_messages = [{"role": "system", "content": self.system_prompt}] + self.messages
        messages = self._apply_context_managers(raw_messages)
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

        extra_body = self._build_extra_body()
        if extra_body:
            kwargs["extra_body"] = extra_body

        if self._should_stream_response():
            kwargs["stream"] = True

        kwargs = self._prepare_request_kwargs(kwargs)
        response = self.client.chat.completions.create(**kwargs)
        if kwargs.get("stream"):
            return self._collect_stream_response(response)
        return response

    def _collect_stream_response(self, stream: Any) -> dict:
        content_parts: list[str] = []
        role = "assistant"

        for chunk in stream:
            choices = self._get_field(chunk, "choices", []) or []
            for choice in choices:
                delta = self._get_field(choice, "delta", {}) or {}
                delta_role = self._get_field(delta, "role")
                if delta_role:
                    role = delta_role

                content_piece = self._get_field(delta, "content")
                if content_piece:
                    content_parts.append(str(content_piece))

                reasoning_piece = (
                    self._get_field(delta, "reasoning_content")
                    or self._get_field(delta, "reasoning")
                    or self._get_field(delta, "thinking")
                )
                if reasoning_piece:
                    content_parts.append(str(reasoning_piece))

        return {
            "choices": [
                {
                    "message": {
                        "role": role,
                        "content": "".join(content_parts),
                    }
                }
            ]
        }

    @classmethod
    def _parse_reasoning_content(cls, content: str) -> str:
        content = (content or "").strip()
        if not content:
            return ""

        think_pattern = re.compile(
            r"<think>\s*(.*?)\s*</think>",
            re.IGNORECASE | re.DOTALL,
        )
        matches = think_pattern.findall(content)
        if matches:
            return "\n\n".join(match.strip() for match in matches if match.strip())

        close_tag = re.search(r"</think>", content, re.IGNORECASE)
        if close_tag:
            return content[: close_tag.start()].strip()

        return ""

    @classmethod
    def _remove_reasoning_content(cls, content: str) -> str:
        content = (content or "").strip()
        if not content:
            return ""

        cleaned = re.sub(
            r"<think>\s*.*?\s*</think>",
            "",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if cleaned != content:
            return cleaned.strip()

        close_tag = re.search(r"</think>", content, re.IGNORECASE)
        if close_tag:
            return content[close_tag.end() :].strip()

        return cleaned.strip()

    @staticmethod
    def _parse_parameter_value(raw_value: str):
        value = (raw_value or "").strip()
        if not value:
            return ""

        for loader in (json.loads, ast.literal_eval):
            try:
                return loader(value)
            except Exception:
                continue

        return value

    @staticmethod
    def _serialize_parameter_value(value: Any) -> str:
        if isinstance(value, (dict, list, tuple, int, float, bool)) or value is None:
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    @classmethod
    def _contains_tool_markup(cls, content: str) -> bool:
        lowered = (content or "").lower()
        return any(marker in lowered for marker in cls._TOOL_MARKERS)

    def _parse_tool_calls(self, content: str) -> list[dict]:
        content = (content or "").strip()
        if not content:
            return []

        tool_pattern = re.compile(
            r"<tool_call>\s*(.*?)\s*</tool_call>",
            re.IGNORECASE | re.DOTALL,
        )
        function_pattern = re.compile(
            r"<function=([^\n>]+)>\s*(.*?)\s*</function>",
            re.IGNORECASE | re.DOTALL,
        )
        parameter_pattern = re.compile(
            r"<parameter=([^\n>]+)>\s*(.*?)\s*</parameter>",
            re.IGNORECASE | re.DOTALL,
        )

        results: list[dict] = []
        for tool_block in tool_pattern.findall(content):
            for function_match in function_pattern.finditer(tool_block):
                tool_name = function_match.group(1).strip()
                if tool_name not in self._current_tools():
                    continue

                arguments = {}
                function_body = function_match.group(2)
                for param_name, param_value in parameter_pattern.findall(function_body):
                    arguments[param_name.strip()] = self._parse_parameter_value(
                        param_value
                    )

                results.append(
                    {
                        "function": {
                            "name": tool_name,
                            "arguments": arguments,
                        }
                    }
                )

        return results

    def _remove_tool_calls(self, content: str) -> str:
        content = (content or "").strip()
        if not content:
            return ""

        return re.sub(
            r"<tool_call>\s*.*?\s*</tool_call>",
            "",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()

    def _extract_final_content(self, raw_content: str) -> str:
        content = self._remove_reasoning_content(raw_content)
        content = self._remove_tool_calls(content)
        return content.strip()

    @staticmethod
    def _looks_like_formatted_final_answer(content: str) -> bool:
        lowered = (content or "").lower()
        return "exact answer:" in lowered and "confidence:" in lowered

    def _looks_like_unclosed_thinking(self, content: str) -> bool:
        content = (content or "").strip()
        if not content:
            return False

        lowered = content.lower()
        if "</think>" in lowered or self._looks_like_formatted_final_answer(content):
            return False
        if self._contains_tool_markup(content):
            return False
        if "<think" in lowered:
            return True

        # With Qwen thinking enabled, vLLM returns generated text after the
        # chat-template "<think>\n" prefix. Without a generated "</think>", the
        # text is still hidden reasoning and should not be accepted as an answer.
        return bool(self.llm_config.enable_thinking)

    def _build_history_tool_xml(self, tool_calls: list[dict]) -> str:
        blocks: list[str] = []
        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            tool_name = function.get("name", "")
            arguments = function.get("arguments", {}) or {}
            if not tool_name:
                continue

            lines = ["<tool_call>", f"<function={tool_name}>"]
            for name, value in arguments.items():
                lines.extend(
                    [
                        f"<parameter={name}>",
                        self._serialize_parameter_value(value),
                        "</parameter>",
                    ]
                )
            lines.extend(["</function>", "</tool_call>"])
            blocks.append("\n".join(lines))
        return "\n".join(blocks)

    def _parse_response(self, raw_content: str) -> dict:
        raw_content = self._normalize_content(raw_content)
        reasoning_content = self._parse_reasoning_content(raw_content)
        visible_content = self._remove_reasoning_content(raw_content)
        tool_calls = self._parse_tool_calls(visible_content)
        if not tool_calls:
            tool_calls = self._parse_tool_calls(raw_content)
        final_content = self._extract_final_content(raw_content)

        malformed_tool_call = not tool_calls and self._contains_tool_markup(raw_content)
        unclosed_thinking = self._looks_like_unclosed_thinking(raw_content)
        if tool_calls or malformed_tool_call or unclosed_thinking:
            final_content = ""

        if not final_content and self._looks_like_formatted_final_answer(reasoning_content):
            final_content = reasoning_content

        assistant_message = {
            "role": "assistant",
            "content": raw_content,
        }

        return {
            "assistant_message": assistant_message,
            "assistant_messages": [assistant_message],
            "tool_calls": tool_calls,
            "reasoning_content": reasoning_content,
            "final_content": final_content,
            "retryable": malformed_tool_call or unclosed_thinking,
            "retry_reason": (
                "malformed_tool_call_xml"
                if malformed_tool_call
                else "unclosed_qwen_thinking"
                if unclosed_thinking
                else ""
            ),
        }

    def _parse_llm_response(self, response) -> dict:
        choices = self._get_field(response, "choices", []) or []
        if not choices:
            return self._parse_response("")
        first_choice = choices[0]
        message = self._get_field(first_choice, "message", {})
        raw_content = self._get_field(message, "content") or ""
        return self._parse_response(raw_content)

    def _build_tool_result_messages(self, tool_name: str, tool_result: str) -> list[dict]:
        return [
            {
                "role": "user",
                "content": f"<tool_response>\n{tool_result}\n</tool_response>\n",
            }
        ]

    def _execute_parsed_tool_calls(self, step, tool_calls: list[dict]) -> None:
        tool_response_blocks: list[str] = []
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
            tool_response_blocks.append(
                f"<tool_response>\n{tool_result}\n</tool_response>"
            )

        if tool_response_blocks:
            self.messages.append(
                {
                    "role": "user",
                    "content": "\n".join(tool_response_blocks) + "\n",
                }
            )

        self.state.tool_rounds += 1
