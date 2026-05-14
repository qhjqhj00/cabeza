"""Pluggable context-management strategies.

Each strategy is a ``cabeza.base.ContextManager`` subclass that
operates on the running message list before every LLM call. Strategies are
registered by string key and can be looked up via ``get_strategy``.

Built-in strategies:
    "summary"       — RollingSummaryStrategy (LLM-rewritten summary in user msg)
    "recent_k"      — RecentKStrategy        (keep K most recent tool turns)
    "discard_tool"  — DiscardToolStrategy    (replace old tool responses with placeholder)
    "discard_all"   — DiscardAllStrategy     (drop everything except system + first user)
    "page_memory"   — PageMemoryStrategy     (flush overflow into page memory)

Custom strategies can be registered with ``register_strategy``.
"""

from __future__ import annotations

from typing import Any, Callable

from cabeza.context.discard_all import DiscardAllStrategy
from cabeza.context.discard_tool import DiscardToolStrategy
from cabeza.context.hard_window import HardContextWindowGuard
from cabeza.context.page_memory import PageMemoryStrategy
from cabeza.context.recent_k import RecentKStrategy
from cabeza.context.summary import RollingSummaryStrategy

# A strategy factory takes a kwargs dict and returns a ContextManager.
StrategyFactory = Callable[..., Any]

_REGISTRY: dict[str, StrategyFactory] = {
    "summary": RollingSummaryStrategy,
    "recent_k": RecentKStrategy,
    "discard_tool": DiscardToolStrategy,
    "discard_all": DiscardAllStrategy,
    "page_memory": PageMemoryStrategy,
}


def register_strategy(name: str, factory: StrategyFactory) -> None:
    """Register a custom context-management strategy.

    The factory is invoked as ``factory(**strategy_kwargs)`` when the Agent
    wires its context pipeline.
    """
    _REGISTRY[name] = factory


def get_strategy(name: str) -> StrategyFactory:
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown context strategy {name!r}. Available: {available}")
    return _REGISTRY[name]


def list_strategies() -> list[str]:
    return sorted(_REGISTRY)


__all__ = [
    "DiscardAllStrategy",
    "DiscardToolStrategy",
    "HardContextWindowGuard",
    "PageMemoryStrategy",
    "RecentKStrategy",
    "RollingSummaryStrategy",
    "register_strategy",
    "get_strategy",
    "list_strategies",
]
