import copy
import os
from typing import Optional

from cabeza.base import NativeToolChatAgent
from cabeza.prompts._defaults import GLM_SYSTEM_PROMPT


class GLMAgent(NativeToolChatAgent):
    """GLM official OpenAI-compatible chat / function-calling adapter."""

    def __init__(
        self,
        *args,
        system_prompt: str = GLM_SYSTEM_PROMPT,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            system_prompt=system_prompt,
            base_url=(
                base_url
                or os.environ.get("GLM_BASE_URL")
                or os.environ.get("TRANSIT_BASE_URL")
                or "http://localhost:8001/v1"
            ),
            **kwargs,
        )

    def _default_stream(self) -> bool:
        # The transit GLM endpoint currently exposes OpenAI-compatible chat
        # completions but may not support SSE streaming reliably.
        return False

    def _uses_openrouter_reasoning(self) -> bool:
        base_url = str(getattr(self, "base_url", "") or "").lower()
        model_name = str(self.model or "").strip().lower()
        return "openrouter.ai" in base_url or model_name.startswith("z-ai/")

    def _build_extra_body(self) -> Optional[dict]:
        cfg = self.llm_config
        base = copy.deepcopy(cfg.extra_body) if cfg.extra_body else {}

        if cfg.enable_thinking:
            if self._uses_openrouter_reasoning():
                reasoning = base.setdefault("reasoning", {})
                reasoning.setdefault("enabled", True)
            else:
                thinking = base.setdefault("thinking", {})
                thinking.setdefault("type", "enabled")
                thinking.setdefault("clear_thinking", False)

        return base or None

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
