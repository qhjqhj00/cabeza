"""``memory`` tool exposed to the agent for on-demand page recall."""

from __future__ import annotations

import copy
import json
import re
from typing import TYPE_CHECKING

from cabeza.tools.base import BaseTool

if TYPE_CHECKING:
    from cabeza.memory.page_store import PageStore
    from cabeza.memory.summarizer import Summarizer


_DESCRIPTION_SINGLE = (
    "Recall and summarize the raw content of relevant past explorations from your memory pages. "
    "Use this tool when: "
    "(1) Earlier conversation turns are no longer available in the current context — call memory "
    "to recover detailed content from those earlier rounds; "
    "(2) You need to look up specific details and verify or cross-check the content stored in the "
    "Exploration Memory section. "
    "First read the bullet-point summaries under each page, then choose ONLY the pages whose "
    "exploration directions are directly relevant to your current goal."
)

_DESCRIPTION_TEAM = (
    "Recall and summarize the raw content of relevant past explorations from your memory pages. "
    "Use this tool when: "
    "(1) Earlier conversation turns are no longer available in the current context — call memory "
    "to recover detailed content from those earlier rounds; "
    "(2) You need information that was already obtained through a previous search, visit, or "
    "exploration — retrieve it from memory instead of repeating the same query; "
    "(3) You want to inspect relevant exploration content produced by your teammates — the memory "
    "tool can retrieve teammate pages as well. "
    "First read the bullet-point summaries under each page, then choose ONLY the pages whose "
    "exploration directions are directly relevant to your current goal."
)


def _parameters_for(max_pages: int, *, team: bool) -> dict:
    pages_desc = (
        f"Page numbers to retrieve (max {max_pages}). Read the per-page summaries in "
        "Exploration Memory and select only the relevant ones"
        + (", including teammate pages when needed." if team else ".")
    )
    return {
        "type": "object",
        "properties": {
            "pages": {
                "type": "array",
                "items": {"type": "integer"},
                "maxItems": max_pages,
                "description": pages_desc,
            },
            "goal": {
                "type": "string",
                "description": "The specific information you want to extract from these pages.",
            },
        },
        "required": ["pages", "goal"],
    }


class MemoryTool(BaseTool):
    """Page recall tool. Works with any ``PageStore`` (single or shared)."""

    def __init__(
        self,
        *,
        summarizer: "Summarizer",
        page_store: "PageStore",
        max_pages: int = 5,
        team_aware: bool = False,
    ) -> None:
        description = _DESCRIPTION_TEAM if team_aware else _DESCRIPTION_SINGLE
        super().__init__(
            name="memory",
            description=description,
            parameters=copy.deepcopy(_parameters_for(max_pages, team=team_aware)),
        )
        self._summarizer = summarizer
        self._store = page_store
        self._max_pages = max(1, int(max_pages))

    def execute(self, pages, goal: str) -> str:
        if not pages:
            return "[Memory] Error: 'pages' parameter is required."
        if not goal:
            return "[Memory] Error: 'goal' parameter is required."

        try:
            parsed = self._parse_pages(pages)
        except Exception as exc:
            return f"[Memory] Error: invalid 'pages' parameter: {exc}"

        if not parsed:
            return "[Memory] Error: no valid page numbers found in 'pages'."

        if len(parsed) > self._max_pages:
            print(f"[Memory] {len(parsed)} pages requested; capping to first {self._max_pages}.")
            parsed = parsed[: self._max_pages]

        try:
            return self._summarizer.consult_pages(
                pages=parsed,
                goal=goal,
                memory_pages=self._store.snapshot(),
            )
        except Exception as exc:
            return f"[Memory] Error: {exc}"

    @staticmethod
    def _parse_pages(pages) -> list[int]:
        if isinstance(pages, list):
            raw = pages
        elif isinstance(pages, int):
            raw = [pages]
        elif isinstance(pages, str):
            text = pages.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                raw = parsed
            elif isinstance(parsed, int):
                raw = [parsed]
            else:
                tokens = re.split(r"[\s,]+", text.strip("[]"))
                raw = [t for t in tokens if t]
        else:
            raw = [pages]

        result: list[int] = []
        for page in raw:
            if isinstance(page, int):
                result.append(page)
                continue
            text = str(page).strip()
            if not text:
                continue
            result.append(int(text))
        return result
