"""Prompt helpers.

Cabeza is benchmark-agnostic at the agent level. These helpers exist purely
for ergonomic reuse of common answer-format templates and the family-default
system prompts vendored under ``cabeza.prompts._defaults``.
"""

from __future__ import annotations

from typing import Optional


BROWSECOMP_TEMPLATE = """\
{question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}"""


HLE_TEMPLATE = """\
{question}

Your response should be in the following format:
Explanation: {{your explanation for your answer choice}}
Answer: {{your chosen answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}"""


def format_question(question: str, *, answer_format: Optional[str] = None) -> str:
    """Wrap a bare question in the requested answer-format template.

    ``answer_format`` accepts ``"browsecomp"``, ``"hle"``, or ``None``/``"free"``
    (returns the question unchanged).
    """
    text = (question or "").strip()
    if not answer_format or answer_format == "free":
        return text
    if answer_format == "browsecomp":
        return BROWSECOMP_TEMPLATE.format(question=text)
    if answer_format == "hle":
        return HLE_TEMPLATE.format(question=text)
    raise ValueError(f"Unknown answer_format {answer_format!r}")


def default_system_prompt(family: str, *, benchmark: Optional[str] = None) -> str:
    """Return the default system prompt for ``family``/benchmark.

    The cabeza Agent normally lets the family adapter inject its built-in
    default when ``system_prompt`` is ``None``; this helper exposes the same
    defaults for callers that want to customize on top of them.
    """
    from cabeza.prompts import _defaults as P

    family = (family or "qwen").lower()
    if family == "qwen":
        if benchmark == "bc":
            return P.BC_QWEN_SYSTEM_PROMPT
        if benchmark == "bc_zh":
            return P.BC_QWEN_SYSTEM_PROMPT
        if benchmark == "bcp":
            return P.BCP_QWEN_SYSTEM_PROMPT
        if benchmark == "ws":
            return P.WS_QWEN_SYSTEM_PROMPT
        if benchmark == "hle":
            return P.HLE_QWEN_SYSTEM_PROMPT
        return P.QWEN_SYSTEM_PROMPT
    if family == "glm":
        return P.GLM_SYSTEM_PROMPT
    if family == "kimi":
        if benchmark == "hle":
            return P.build_hle_kimi_system_prompt()
        return P.KIMI_SYSTEM_PROMPT
    if family == "deepseek":
        return P.DEEPSEEK_SYSTEM_PROMPT
    return P.QWEN_SYSTEM_PROMPT
