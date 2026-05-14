import copy
import json
from typing import Any, Optional

from cabeza.base import (
    BaseAgent,
    RunState,
    ToolCallRecord,
    TrajectoryStep,
)
from cabeza.prompts._defaults import GPT_SYSTEM_PROMPT


class GPTAgent(BaseAgent):
    """Official GPT adapter built on top of the OpenAI Responses API."""

    def __init__(
        self,
        *args,
        system_prompt: str = GPT_SYSTEM_PROMPT,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            system_prompt=system_prompt,
            base_url=base_url or "https://api.openai.com/v1",
            **kwargs,
        )

    @staticmethod
    def _get_field(obj: Any, name: str, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    @classmethod
    def _to_python(cls, value: Any):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, list):
            return [cls._to_python(item) for item in value]
        if isinstance(value, tuple):
            return [cls._to_python(item) for item in value]
        if isinstance(value, dict):
            return {k: cls._to_python(v) for k, v in value.items()}

        for method_name in ("model_dump", "to_dict", "dict"):
            method = getattr(value, method_name, None)
            if callable(method):
                try:
                    dumped = method(exclude_none=True)
                except TypeError:
                    dumped = method()
                return cls._to_python(dumped)

        if hasattr(value, "__dict__"):
            return {
                key: cls._to_python(val)
                for key, val in vars(value).items()
                if not key.startswith("_")
            }

        return value

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

        for loader in (json.loads,):
            try:
                return loader(text)
            except Exception:
                continue

        return raw_args

    @classmethod
    def _serialize_arguments(cls, arguments: Any) -> str:
        if isinstance(arguments, str):
            return arguments
        return json.dumps(arguments or {}, ensure_ascii=False)

    @classmethod
    def _text_from_content_part(cls, part: Any) -> str:
        if part is None:
            return ""
        if isinstance(part, str):
            return part

        part_type = str(cls._get_field(part, "type") or "")
        if part_type in {"input_text", "output_text", "text", "summary_text"}:
            text = cls._get_field(part, "text")
            if isinstance(text, str):
                return text
            if text is not None:
                return cls._text_from_content_part(text)

        if part_type == "refusal":
            refusal = cls._get_field(part, "refusal") or cls._get_field(part, "text")
            if isinstance(refusal, str):
                return refusal

        text = cls._get_field(part, "text")
        if isinstance(text, str):
            return text

        return ""

    @classmethod
    def _normalize_message_text(cls, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if not isinstance(content, list):
            return str(content).strip()

        parts: list[str] = []
        for item in content:
            text = cls._text_from_content_part(item).strip()
            if text:
                parts.append(text)
        return "".join(parts).strip()

    @classmethod
    def _extract_reasoning_text(cls, item: Any) -> str:
        summary = cls._get_field(item, "summary")
        if summary:
            text = cls._normalize_message_text(summary)
            if text:
                return text

        content = cls._get_field(item, "content")
        if content:
            return cls._normalize_message_text(content)

        text = cls._get_field(item, "text")
        if text:
            return cls._normalize_message_text(text)

        return ""

    @staticmethod
    def _is_instruction_message(item: dict) -> bool:
        return (
            item.get("type") == "message"
            and item.get("role") in {"system", "developer"}
        )

    @staticmethod
    def _ensure_message_item(role: str, content: Any) -> dict:
        return {
            "type": "message",
            "role": role,
            "content": content,
        }

    def _tool_to_response_schema(self, tool) -> dict:
        if hasattr(tool, "to_response_schema"):
            return tool.to_response_schema()
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }

    def _normalize_history_item(self, item: dict) -> list[dict]:
        normalized = copy.deepcopy(item)
        item_type = normalized.get("type")
        role = normalized.get("role")

        if item_type == "message":
            if role not in {"system", "developer", "user", "assistant"}:
                raise ValueError(f"Unsupported Responses message role: {role!r}")
            return [normalized]

        if item_type == "function_call_output":
            call_id = normalized.get("call_id") or normalized.get("tool_call_id")
            if not call_id:
                raise ValueError("function_call_output item is missing call_id.")
            normalized["call_id"] = call_id
            normalized.pop("tool_call_id", None)
            return [normalized]

        if item_type in {"function_call", "reasoning"}:
            return [normalized]

        if item_type:
            return [normalized]

        if role == "tool":
            call_id = normalized.get("call_id") or normalized.get("tool_call_id")
            if not call_id:
                raise ValueError("tool role item must include call_id/tool_call_id.")
            return [
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": normalized.get("output", normalized.get("content", "")),
                }
            ]

        if role in {"system", "developer", "user", "assistant"}:
            items: list[dict] = []
            content = normalized.get("content")
            thinking = normalized.get("thinking")

            if content is not None or role != "assistant":
                items.append(self._ensure_message_item(role, content or ""))

            tool_calls = normalized.get("tool_calls") or []
            for idx, tool_call in enumerate(tool_calls):
                function = self._get_field(tool_call, "function", {}) or {}
                tool_name = (
                    self._get_field(function, "name")
                    or self._get_field(tool_call, "name")
                    or ""
                ).strip()
                if not tool_name:
                    raise ValueError("assistant tool_call is missing a function name.")

                call_id = (
                    self._get_field(tool_call, "call_id")
                    or self._get_field(tool_call, "id")
                    or f"call_{len(self.messages)}_{idx}"
                )

                raw_arguments = self._get_field(function, "arguments")
                if raw_arguments is None:
                    raw_arguments = self._get_field(tool_call, "arguments")

                items.append(
                    {
                        "type": "function_call",
                        "id": self._get_field(tool_call, "id"),
                        "call_id": call_id,
                        "name": tool_name,
                        "arguments": self._serialize_arguments(raw_arguments),
                    }
                )

            if thinking:
                items.insert(
                    0,
                    {
                        "type": "reasoning",
                        "summary": [
                            {
                                "type": "summary_text",
                                "text": str(thinking),
                            }
                        ],
                    },
                )

            return items

        raise ValueError(f"Unsupported history item for GPTAgent: {normalized!r}")

    def _prepare_messages_for_llm(self, messages: list[dict]) -> list[dict]:
        normalized: list[dict] = []
        for item in messages:
            normalized.extend(self._normalize_history_item(item))
        return normalized

    def _persist_processed_messages(self, messages: list[dict]) -> None:
        stripped = list(messages)
        while stripped and self._is_instruction_message(stripped[0]):
            stripped = stripped[1:]
        self.messages = stripped

    def _split_instructions_and_input(self, items: list[dict]) -> tuple[Optional[Any], list[dict]]:
        instruction_items: list[dict] = []
        body_items = list(items)

        while body_items and self._is_instruction_message(body_items[0]):
            instruction_items.append(copy.deepcopy(body_items.pop(0)))

        if not instruction_items:
            return None, body_items

        if len(instruction_items) == 1:
            content = instruction_items[0].get("content")
            if isinstance(content, str):
                return content, body_items

        return instruction_items, body_items

    def _build_reasoning_config(self) -> Optional[dict]:
        cfg = self.llm_config
        reasoning = copy.deepcopy(cfg.reasoning) if cfg.reasoning else {}
        if cfg.reasoning_effort is not None:
            reasoning["effort"] = cfg.reasoning_effort
        if cfg.reasoning_summary is not None:
            reasoning["summary"] = cfg.reasoning_summary
        return reasoning or None

    def _build_text_config(self) -> Optional[dict]:
        cfg = self.llm_config
        text = copy.deepcopy(cfg.text) if cfg.text else {}
        if cfg.text_verbosity is not None:
            text["verbosity"] = cfg.text_verbosity
        if cfg.text_format is not None:
            text["format"] = copy.deepcopy(cfg.text_format)
        return text or None

    def _call_llm(self):
        raw_messages = [self._ensure_message_item("developer", self.system_prompt), *self.messages]
        messages = self._apply_context_managers(raw_messages)
        messages = self._prepare_messages_for_llm(messages)
        self._persist_processed_messages(messages)

        instructions, input_items = self._split_instructions_and_input(messages)
        cfg = self.llm_config

        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
        }

        if instructions is not None:
            kwargs["instructions"] = instructions

        max_output_tokens = cfg.max_output_tokens
        if max_output_tokens is None:
            max_output_tokens = cfg.max_tokens
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens
        if cfg.temperature is not None:
            kwargs["temperature"] = cfg.temperature
        if cfg.top_p is not None:
            kwargs["top_p"] = cfg.top_p
        if cfg.store is not None:
            kwargs["store"] = cfg.store
        if cfg.truncation is not None:
            kwargs["truncation"] = cfg.truncation
        if cfg.parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = cfg.parallel_tool_calls
        if cfg.tool_choice is not None:
            kwargs["tool_choice"] = copy.deepcopy(cfg.tool_choice)
        if cfg.include is not None:
            kwargs["include"] = list(cfg.include)
        if cfg.metadata is not None:
            kwargs["metadata"] = copy.deepcopy(cfg.metadata)
        if cfg.service_tier is not None:
            kwargs["service_tier"] = cfg.service_tier
        if cfg.prompt_cache_key is not None:
            kwargs["prompt_cache_key"] = cfg.prompt_cache_key
        if cfg.safety_identifier is not None:
            kwargs["safety_identifier"] = cfg.safety_identifier

        reasoning = self._build_reasoning_config()
        if reasoning:
            kwargs["reasoning"] = reasoning

        text = self._build_text_config()
        if text:
            kwargs["text"] = text

        tools = [self._tool_to_response_schema(tool) for tool in self.tools.values()] or None
        if tools:
            kwargs["tools"] = tools

        if cfg.extra_body:
            kwargs["extra_body"] = copy.deepcopy(cfg.extra_body)

        return self.client.responses.create(**kwargs)

    def _extract_tool_calls(self, output_items: list[dict]) -> tuple[list[dict], bool]:
        tool_calls: list[dict] = []
        malformed = False

        for item in output_items:
            if item.get("type") != "function_call":
                continue

            tool_name = str(item.get("name") or "").strip()
            call_id = str(item.get("call_id") or "").strip()
            if not tool_name or not call_id:
                malformed = True
                continue

            tool_calls.append(
                {
                    "id": item.get("id"),
                    "call_id": call_id,
                    "status": item.get("status"),
                    "function": {
                        "name": tool_name,
                        "arguments": self._safe_load_arguments(item.get("arguments")),
                    },
                }
            )

        return tool_calls, malformed

    def _parse_response(self, response) -> dict:
        output_items = [
            item for item in self._to_python(self._get_field(response, "output", [])) or []
            if isinstance(item, dict)
        ]
        tool_calls, malformed_tool_call = self._extract_tool_calls(output_items)

        reasoning_blocks: list[str] = []
        message_blocks: list[str] = []
        assistant_message = None

        for item in output_items:
            item_type = item.get("type")
            if assistant_message is None and item_type == "message":
                assistant_message = copy.deepcopy(item)
            if item_type == "reasoning":
                reasoning_text = self._extract_reasoning_text(item)
                if reasoning_text:
                    reasoning_blocks.append(reasoning_text)
            elif item_type == "message" and item.get("role") == "assistant":
                message_text = self._normalize_message_text(item.get("content"))
                if message_text:
                    message_blocks.append(message_text)

        message_text = "\n\n".join(block for block in message_blocks if block).strip()
        reasoning_content = "\n\n".join(block for block in reasoning_blocks if block).strip()

        if tool_calls:
            if message_text:
                reasoning_content = (
                    f"{reasoning_content}\n\n{message_text}".strip()
                    if reasoning_content else message_text
                )
            final_content = ""
        else:
            final_content = message_text
            if not final_content:
                final_content = self._normalize_message_text(
                    self._get_field(response, "output_text")
                )

        if assistant_message is None:
            assistant_message = (
                copy.deepcopy(output_items[0]) if output_items else self._ensure_message_item("assistant", final_content)
            )

        return {
            "assistant_message": assistant_message,
            "output_items": copy.deepcopy(output_items),
            "history_items": copy.deepcopy(output_items),
            "tool_calls": tool_calls,
            "reasoning_content": reasoning_content,
            "final_content": final_content,
            "response_id": self._get_field(response, "id"),
            "response_status": self._get_field(response, "status"),
            "retryable": malformed_tool_call,
            "retry_reason": "malformed_function_call_item" if malformed_tool_call else "",
        }

    def _build_tool_result_messages(
        self,
        tool_name: str,
        tool_result: str,
        *,
        call_id: Optional[str] = None,
    ) -> list[dict]:
        if not call_id:
            raise ValueError(f"Tool call '{tool_name}' is missing call_id.")
        return [
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": tool_result,
            }
        ]

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
                call_id=tc.get("call_id") or tc.get("id"),
            ):
                self.messages.append(message)

        self.state.tool_rounds += 1

    def _finalize(self, stop_reason: str) -> dict:
        self.state.stop_reason = stop_reason
        stop_msg = self._on_stop(stop_reason)
        if not stop_msg:
            return {
                "final_content": f"[stopped: {stop_reason}]",
                "reasoning_content": "",
                "tool_calls": [],
                "output_items": [],
                "history_items": [],
            }

        self.messages.append({"role": "user", "content": stop_msg})
        retry_budget = self._PARSE_RETRY_LIMIT

        for _ in range(self._FINALIZE_FOLLOWUP_LIMIT):
            self.state.step += 1
            self._fire("before_llm", self.state, self.messages)
            response = self._call_llm()
            parsed = self._parse_response(response)
            self._fire("after_llm", self.state, parsed)

            should_retry, retry_reason = self._should_retry_parsed_response(parsed)
            if should_retry and retry_budget > 0:
                retry_budget -= 1
                self._log_retry(retry_reason, phase="finalize")
                continue

            retry_budget = self._PARSE_RETRY_LIMIT
            self.messages.extend(parsed["history_items"])
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

        return {
            "final_content": f"[stopped: {stop_reason}]",
            "reasoning_content": "",
            "tool_calls": [],
            "output_items": [],
            "history_items": [],
        }

    def _run_loop(self, user_input: str) -> dict:
        self.messages = []
        self.trajectory = []
        self.state = RunState()
        for cm in self._context_managers:
            cm.reset()
        self.messages.append({"role": "user", "content": user_input})

        retry_budget = self._PARSE_RETRY_LIMIT
        step_idx = 0
        while step_idx < self.max_steps:
            stop_reason = self._check_stop()
            if stop_reason:
                return self._finalize(stop_reason)

            step_idx += 1
            self.state.step = step_idx

            self._fire("before_llm", self.state, self.messages)
            response = self._call_llm()
            parsed = self._parse_response(response)
            tool_calls = parsed["tool_calls"]
            final_content = parsed["final_content"]
            reasoning = parsed.get("reasoning_content", "")

            self._fire("after_llm", self.state, parsed)

            should_retry, retry_reason = self._should_retry_parsed_response(parsed)
            if should_retry and retry_budget > 0:
                retry_budget -= 1
                self._log_retry(retry_reason)
                continue

            retry_budget = self._PARSE_RETRY_LIMIT
            self.messages.extend(parsed["history_items"])
            step = TrajectoryStep(step=self.state.step, reasoning=reasoning)

            if not tool_calls:
                step.final_content = final_content
                self.trajectory.append(step)
                self.state.stop_reason = ""
                return parsed

            self._execute_parsed_tool_calls(step, tool_calls)
            self.trajectory.append(step)

        return self._finalize("max_steps")

    def save_trajectory(self, path: str) -> None:
        payload = {
            "instructions": self.system_prompt,
            "input": self.messages,
        }
        path_dir = path.rsplit("/", 1)[0] if "/" in path else ""
        if path_dir:
            import os

            os.makedirs(path_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
