"""Rolling-summary context strategy.

Collapses the running conversation into a rolling summary tucked into the
first user message when the prompt exceeds the soft token budget. Delegates
the actual summarization to a dedicated LLM client.
"""

from __future__ import annotations

import re
from typing import Optional

from cabeza.context._rolling_summary import RollingSummaryContextManager


def _strip_thinking(content: str) -> str:
    """Remove `<think>...</think>` blocks from summarizer output.

    Some thinking-enabled models leak their scratchpad into the summary; this
    keeps the rolling summary itself clean without changing the model call.
    """
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
        return content[close_tag.end():].strip()
    return cleaned.strip()


class _SanitizingRollingSummary(RollingSummaryContextManager):
    def _summarize_messages(self, messages: list[dict]) -> str:
        summary = super()._summarize_messages(messages)
        return _strip_thinking(summary)


def RollingSummaryStrategy(
    *,
    model: str,
    api_key: str,
    base_url: str,
    max_input_tokens: int,
    summary_model: Optional[str] = None,
    summary_prompt: Optional[str] = None,
    summary_max_tokens: int = 2048,
    summary_temperature: float = 0.0,
    request_timeout: float = 120.0,
    include_native_fields: bool = True,
    summary_extra_body: Optional[dict] = None,
    sanitize_thinking: bool = True,
):
    """Build a rolling-summary context manager.

    ``sanitize_thinking`` (default ``True``) strips leaked ``<think>`` blocks
    from the rewritten summary.
    """
    cls = _SanitizingRollingSummary if sanitize_thinking else RollingSummaryContextManager
    return cls(
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_input_tokens=int(max_input_tokens),
        summary_model=summary_model,
        summary_prompt=summary_prompt,
        summary_max_tokens=int(summary_max_tokens),
        summary_temperature=float(summary_temperature),
        request_timeout=float(request_timeout),
        include_structured_fields=bool(include_native_fields),
        summary_extra_body=summary_extra_body,
    )
