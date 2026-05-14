import copy
import os
from typing import Optional

from cabeza.base import NativeToolChatAgent
from cabeza.prompts._defaults import DEEPSEEK_SYSTEM_PROMPT


class DeepSeekAgent(NativeToolChatAgent):
    """DeepSeek official OpenAI-compatible chat/tool-calling adapter."""

    def __init__(
        self,
        *args,
        system_prompt: str = DEEPSEEK_SYSTEM_PROMPT,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            system_prompt=system_prompt,
            base_url=(
                base_url
                or os.environ.get("DEEPSEEK_BASE_URL")
                or os.environ.get("TRANSIT_BASE_URL")
                or ""
            ),
            **kwargs,
        )

    @staticmethod
    def _normalize_reasoning_effort(reasoning_effort: Optional[str]) -> Optional[str]:
        effort = str(reasoning_effort or "").strip().lower()
        if not effort:
            return "high"
        aliases = {
            "max": "high",
            "xhigh": "high",
            "extra_high": "high",
            "extra-high": "high",
        }
        effort = aliases.get(effort, effort)
        if effort in {"low", "medium", "high"}:
            return effort
        return None

    def _build_extra_body(self) -> Optional[dict]:
        cfg = self.llm_config
        base = copy.deepcopy(cfg.extra_body) if cfg.extra_body else {}

        thinking = base.setdefault("thinking", {})
        thinking.setdefault("type", "enabled" if cfg.enable_thinking else "disabled")

        return base or None

    def _prepare_request_kwargs(self, kwargs: dict) -> dict:
        cfg = self.llm_config
        if cfg.enable_thinking:
            effort = self._normalize_reasoning_effort(cfg.reasoning_effort)
            if effort:
                kwargs.setdefault("reasoning_effort", effort)
            # DeepSeek V4 thinking mode documents sampling knobs as unsupported.
            kwargs.pop("temperature", None)
            kwargs.pop("top_p", None)
        return kwargs

    def _build_tool_result_messages(
        self,
        tool_name: str,
        tool_result: str,
        *,
        tool_call_id: Optional[str] = None,
    ) -> list[dict]:
        message = {
            "role": "tool",
            "content": tool_result,
        }
        if tool_call_id:
            message["tool_call_id"] = tool_call_id
        return [message]
