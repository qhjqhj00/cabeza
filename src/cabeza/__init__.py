"""Cabeza — inference harness for long-horizon agentic search.

Public surface:

    from cabeza import Agent
    agent = Agent(model=..., base_url=..., api_key=..., family="qwen", ...)
    answer = agent.run("question text")

    from cabeza.datasets import load
    ds = load("bc")

    from cabeza.runner import evaluate
    evaluate(agent, ds, out="results.jsonl")
"""

from cabeza.agent import Agent
from cabeza.types import Example, RunResult

__all__ = ["Agent", "Example", "RunResult"]
__version__ = "0.1.0"
