"""Patch an agent's tool-call executor to run a chosen tool in parallel.

Used by the dynamic swarm: when the meta-agent emits multiple ``assign_task``
calls in one assistant turn, run them concurrently instead of sequentially.
"""

from __future__ import annotations

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from types import MethodType

from cabeza.base import ToolCallRecord


def install_parallel_tool_executor(agent, *, tool_name: str, max_workers: int) -> None:
    """Run ``tool_name`` calls in parallel when an assistant turn emits >1.

    Falls back to the agent's default sequential executor if the turn mixes
    other tools.
    """
    original_execute_tool = agent._execute_tool
    workers = max(1, int(max_workers))

    def execute_one(tool_call: dict):
        name = tool_call["function"]["name"]
        args = tool_call["function"].get("arguments", {})
        if not isinstance(args, dict):
            args = {}
        result = original_execute_tool(name, args)
        return tool_call, name, args, result

    def execute_parsed_tool_calls(self, step, tool_calls):
        targets = [tc for tc in tool_calls if tc.get("function", {}).get("name") == tool_name]
        if len(targets) <= 1 or len(targets) != len(tool_calls):
            return self.__class__._execute_parsed_tool_calls(self, step, tool_calls)

        results: dict[int, tuple] = {}
        with ThreadPoolExecutor(max_workers=min(workers, len(targets))) as executor:
            future_to_idx = {executor.submit(execute_one, tc): i for i, tc in enumerate(tool_calls)}
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    tc = tool_calls[idx]
                    name = tc.get("function", {}).get("name", "")
                    args = tc.get("function", {}).get("arguments", {})
                    if not isinstance(args, dict):
                        args = {}
                    results[idx] = (tc, name, args, f"Error while executing tool '{name}': {exc}")

        for idx in range(len(tool_calls)):
            tc, name, args, result = results[idx]
            step.tool_calls.append(ToolCallRecord(name=name, args=args, result=result))
            self._fire("after_tool", self.state, name, args, result)
            try:
                messages = self._build_tool_result_messages(name, result, tool_call_id=tc.get("id"))
            except TypeError:
                messages = self._build_tool_result_messages(name, result)
            for message in messages:
                self.messages.append(message)
        self.state.tool_rounds += 1

    agent._execute_parsed_tool_calls = MethodType(execute_parsed_tool_calls, agent)
