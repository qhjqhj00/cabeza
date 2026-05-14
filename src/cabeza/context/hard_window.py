"""Hard context-window guard.

Forces the agent to emit one final tool-free answer when the prompt is about
to exceed the configured context window. Sits at the end of the
ContextManager pipeline so it sees the prompt after every other strategy has
already trimmed it.
"""

from __future__ import annotations

from typing import Optional

from cabeza.base import ContextManager

from cabeza._budget import count_messages, default_encoder


class HardContextWindowGuard(ContextManager):
    """Trigger ``agent.request_final_answer_once`` when over the hard limit."""

    def __init__(
        self,
        agent,
        *,
        max_input_tokens: int,
        encoder=None,
        include_native_fields: bool = True,
        label: str = "HardContextWindow",
    ) -> None:
        self._agent = agent
        self._max_input_tokens = max(0, int(max_input_tokens or 0))
        self._encoder = encoder if encoder is not None else default_encoder()
        self._include_native_fields = include_native_fields
        self._label = label
        self._triggered = False

    def reset(self) -> None:
        self._triggered = False

    def process(self, messages: list[dict], state) -> list[dict]:
        if self._max_input_tokens <= 0 or self._triggered:
            return messages
        if getattr(self._agent, "_force_final_answer_once", False):
            return messages

        total = count_messages(
            messages,
            self._encoder,
            include_native_fields=self._include_native_fields,
        )
        if total < self._max_input_tokens:
            return messages

        self._triggered = True
        print(
            f"[{self._label}] {total} tokens >= hard limit {self._max_input_tokens}; "
            "forcing one final answer without tool calls."
        )
        self._agent.request_final_answer_once(
            "token_budget",
            prompt=(
                "The conversation has reached the configured context window. "
                "Provide your best final answer now based only on what is already "
                "available. Do not call any tools."
            ),
        )
        return messages


def install_hard_window_guard(
    agent,
    *,
    max_input_tokens: int,
    encoder=None,
    include_native_fields: bool = True,
    label: str = "HardContextWindow",
) -> Optional[HardContextWindowGuard]:
    """Attach a ``HardContextWindowGuard`` to ``agent`` and return it.

    Returns ``None`` when ``max_input_tokens`` is non-positive.
    """
    if int(max_input_tokens or 0) <= 0:
        return None
    guard = HardContextWindowGuard(
        agent,
        max_input_tokens=max_input_tokens,
        encoder=encoder,
        include_native_fields=include_native_fields,
        label=label,
    )
    agent.use(guard)
    return guard
