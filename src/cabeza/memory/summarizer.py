"""Auxiliary LLM that produces page summaries and answers ``memory`` lookups.

The summarizer keeps an in-memory call trace (``self.trajectory``) so the
caller can dump it alongside the agent trajectory for analysis.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Optional

from openai import OpenAI


_SUMMARIZE_SYSTEM = (
    "You are a memory manager for a research agent. Your job is to compress the prior "
    "conversation and tool-use history into a concise working memory that helps the next agent "
    "continue the task without rereading the full transcript.\n\n"

    "Write a factual summary of what has already been explored, tried, confirmed, and left unresolved. "
    "Preserve only information that is useful for continuing the work. Omit chit-chat, stylistic details, "
    "and repeated content unless it affects the task.\n\n"

    "Prioritize:\n"
    "1. The user's goal, constraints, and preferences.\n"
    "2. Key facts established during the conversation.\n"
    "3. Tools used and the most important results from them.\n"
    "4. Partial conclusions, promising leads, and failed approaches.\n"
    "5. Open questions, uncertainties, and what still needs to be done next.\n\n"

    "Requirements:\n"
    "- Be concise but information-dense.\n"
    "- Be factual; do not invent details.\n"
    "- Distinguish confirmed findings from tentative inferences.\n"
    "- Focus on continuation value.\n"
    "- Prefer bullets over full sentences when efficient.\n"
    "- Output only the summary."
)

_SUMMARIZE_USER_TEMPLATE = (
    "Previous conversation and tool-use history:\n"
    "{window_content}\n\n"
    "Summarize it for continuation. Output only the summary."
)

_CONSULT_SYSTEM = (
    "You are a research assistant. Given a research goal and retrieved pages from past "
    "explorations, extract the information that is relevant to the goal and produce a "
    "concise, focused summary.\n\n"
    "Rules:\n"
    "1. Keep only information directly relevant to the goal. Preserve important facts, findings, dates, names, and evidence when present.\n"
    "2. Incorporate prior extracted results when provided. Do not drop previously established key information unless contradicted or irrelevant.\n"
    "3. Add important new information from the current page, avoiding repetition.\n"
    "4. Distinguish clearly between confirmed information and uncertain or incomplete information.\n"
    "5. Be concise, factual, and information-dense.\n"
    "6. Output only the extracted information and summary."
)

_CONSULT_USER_TEMPLATE = (
    "Research goal:\n{goal}\n\n"
    "Previous extracted results:\n{previous_summary}\n\n"
    "Current page:\n{page_content}\n\n"
    "Integrate the previous results with the current page, keeping only information relevant "
    "to the goal. Output only the updated extracted information and summary."
)


def _strip_thinking(raw: str) -> str:
    text = (raw or "").strip()
    lower = text.lower()
    end_tag = "</think>"
    if end_tag in lower:
        idx = lower.rfind(end_tag)
        after = text[idx + len(end_tag):].strip()
        if after:
            return after
        return re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    if "<think>" in lower:
        return text[: lower.find("<think>")].strip()
    return text


class Summarizer:
    """Auxiliary LLM used by ``PageMemoryStrategy`` and ``MemoryTool``."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str = "EMPTY",
        max_response_tokens: int = 8192,
        temperature: float = 0.3,
        request_timeout_s: float = 120.0,
        max_retries: int = 6,
        retry_sleep_s: float = 3.0,
        trajectory_autosave_path: Optional[str] = None,
    ) -> None:
        self.model = model
        self.max_response_tokens = max_response_tokens
        self.temperature = float(temperature)
        self.max_retries = max(1, int(max_retries))
        self.retry_sleep_s = max(0.0, float(retry_sleep_s))
        self.trajectory: list[dict] = []
        self.trajectory_autosave_path = trajectory_autosave_path

        client_kwargs: dict = {"api_key": api_key, "base_url": base_url}
        if request_timeout_s is not None:
            client_kwargs["timeout"] = request_timeout_s
        self._client = OpenAI(**client_kwargs)

    # ---- public API --------------------------------------------------------

    def summarize_window(self, window_content: str, question: str = "") -> str:
        _ = question  # accepted for symmetry with downstream prompt templates
        messages = [
            {"role": "system", "content": _SUMMARIZE_SYSTEM},
            {
                "role": "user",
                "content": _SUMMARIZE_USER_TEMPLATE.format(
                    window_content=window_content or "[Empty window]"
                ),
            },
        ]
        raw = self._call_llm(
            messages,
            call_type="summarize_window",
            metadata={"window_chars": len(window_content or "")},
        )
        return self._parse_summary(raw)

    def consult_pages(
        self,
        *,
        pages: list[int],
        goal: str,
        memory_pages: list[dict],
    ) -> str:
        if not memory_pages:
            return "[Memory] No exploration history available yet."
        available = [p["page_num"] for p in memory_pages]
        selected = [p for p in memory_pages if p["page_num"] in pages]
        if not selected:
            return f"[Memory] None of the requested pages found. Available pages: {available}"

        by_num = {p["page_num"]: p for p in selected}
        ordered = [by_num[n] for n in pages if n in by_num]

        running = "None yet. No earlier page has been summarized."
        for idx, page in enumerate(ordered, start=1):
            messages = [
                {"role": "system", "content": _CONSULT_SYSTEM},
                {
                    "role": "user",
                    "content": _CONSULT_USER_TEMPLATE.format(
                        goal=goal,
                        previous_summary=running,
                        page_content=page.get("raw_content") or "[Empty page]",
                    ),
                },
            ]
            result = self._call_llm(
                messages,
                call_type="consult_pages_step",
                metadata={
                    "pages": pages,
                    "goal": goal,
                    "current_page": page["page_num"],
                    "step_index": idx,
                    "step_count": len(ordered),
                },
            )
            if not result or not result.strip():
                continue
            parsed = self._parse_summary(result)
            if parsed:
                running = parsed

        if running == "No relevant information extracted yet.":
            return "[Memory] Failed to summarize content."
        return running

    # ---- trajectory I/O ----------------------------------------------------

    def reset_trajectory(self) -> None:
        self.trajectory = []

    def set_trajectory_autosave(self, path: Optional[str]) -> None:
        self.trajectory_autosave_path = path

    def save_trajectory(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {"model_name": self.model, "trajectory": self.trajectory}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    # ---- internals ---------------------------------------------------------

    @staticmethod
    def _parse_summary(raw: str) -> str:
        cleaned = _strip_thinking(raw).strip()
        if not cleaned:
            return "Explored a new direction"
        lines = cleaned.splitlines()
        if lines and lines[0].strip().upper().startswith("BULLET:"):
            lines[0] = lines[0].split(":", 1)[1].strip()
            cleaned = "\n".join(lines).strip()
        return cleaned or "Explored a new direction"

    def _call_llm(
        self,
        messages: list[dict],
        *,
        call_type: str,
        metadata: Optional[dict] = None,
    ) -> str:
        last_error: Optional[str] = None
        errors: list[str] = []
        attempts = 0
        final = ""
        for attempt in range(self.max_retries):
            attempts = attempt + 1
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_response_tokens,
                    temperature=self.temperature,
                )
                content = resp.choices[0].message.content
                if content and content.strip():
                    final = content.strip()
                    break
                last_error = "Empty response"
                errors.append(last_error)
            except Exception as exc:
                last_error = str(exc)
                errors.append(last_error)
                print(f"[Summarizer] error (attempt {attempt + 1}/{self.max_retries}): {last_error}")
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_sleep_s)
        if not final:
            print(f"[Summarizer] All retries failed: {last_error}")

        self._record_trajectory(
            messages=messages,
            response=final,
            call_type=call_type,
            attempts=attempts,
            errors=errors,
            metadata=metadata,
        )
        return final

    def _record_trajectory(
        self,
        *,
        messages: list[dict],
        response: str,
        call_type: str,
        attempts: int,
        errors: list[str],
        metadata: Optional[dict] = None,
    ) -> None:
        entry = {
            "call_index": len(self.trajectory) + 1,
            "call_type": call_type,
            "attempts": attempts,
            "errors": errors,
            "messages": [*messages, {"role": "assistant", "content": response}],
            "response": response,
            "metadata": metadata or {},
            "timestamp": time.time(),
        }
        self.trajectory.append(entry)
        if self.trajectory_autosave_path:
            try:
                self.save_trajectory(self.trajectory_autosave_path)
            except Exception as exc:
                print(f"[Summarizer] failed to autosave trajectory: {exc}")
