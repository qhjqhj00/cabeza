"""Discard-all context strategy.

Once the prompt exceeds the configured budget, keep only the system message
and the first user message; drop everything else.
"""

from __future__ import annotations

from cabeza.base import ContextManager

from cabeza._budget import count_messages, default_encoder


class DiscardAllStrategy(ContextManager):
    """Keep only system + first user message when over budget."""

    def __init__(
        self,
        *,
        max_input_tokens: int,
        encoder=None,
        include_native_fields: bool = True,
        label: str = "DiscardAll",
    ) -> None:
        self._max_input_tokens = max(0, int(max_input_tokens))
        self._encoder = encoder if encoder is not None else default_encoder()
        self._include_native_fields = include_native_fields
        self._label = label
        self._discard_count = 0

    def reset(self) -> None:
        self._discard_count = 0

    def process(self, messages: list[dict], state) -> list[dict]:
        if not messages or self._max_input_tokens <= 0:
            return messages

        total_tokens = count_messages(
            messages,
            self._encoder,
            include_native_fields=self._include_native_fields,
        )
        if total_tokens <= self._max_input_tokens:
            return messages

        protected = self._protected_messages(messages)
        new_total = count_messages(
            protected,
            self._encoder,
            include_native_fields=self._include_native_fields,
        )
        dropped = max(0, len(messages) - len(protected))
        self._discard_count += 1

        print(
            f"[{self._label}] threshold reached; kept only system + first user "
            f"({total_tokens} -> {new_total} tokens, "
            f"dropped={dropped}, discard_count={self._discard_count}, "
            f"step={state.step}, tool_rounds={state.tool_rounds})."
        )

        if new_total > self._max_input_tokens:
            print(
                f"[{self._label}] Warning: system + first user still exceeds the budget."
            )
        return protected

    @staticmethod
    def _protected_messages(messages: list[dict]) -> list[dict]:
        protected: list[dict] = []
        system = next((m for m in messages if m.get("role") == "system"), None)
        if system is not None:
            protected.append(dict(system))
        first_user = next((m for m in messages if m.get("role") == "user"), None)
        if first_user is not None:
            protected.append(dict(first_user))
        if protected:
            return protected
        return [dict(messages[0])]
