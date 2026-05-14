"""Shared types used across cabeza."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NotRequired, TypedDict


class Example(TypedDict):
    """Minimal dataset record consumed by ``Agent.run`` / ``evaluate``."""

    id: NotRequired[str]
    question: str
    golden_answers: NotRequired[Any]
    language: NotRequired[str]
    metadata: NotRequired[dict]


@dataclass
class ToolCall:
    name: str
    args: dict
    result: str


@dataclass
class Step:
    step: int
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_content: str = ""


@dataclass
class RunResult:
    """Structured result returned by ``Agent.run_verbose``."""

    question: str
    prediction: str
    final_content: str
    termination: str
    total_rounds: int
    tool_call_rounds: int
    total_process_time: float
    messages: list[dict] = field(default_factory=list)
    trajectory: list[Step] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)
