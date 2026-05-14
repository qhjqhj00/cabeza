"""LocalVisitTool — offline document access against a local corpus cache + vLLM LLM API.

Behavior:
    - Documents are addressed by URL — specifically, URLs returned by an
      earlier ``LocalSearchTool`` query.
    - Document content is read straight from the ``LocalSearchTool``
      in-memory cache; no online endpoint is touched.
    - Unknown URLs (never searched, or missing from the corpus) return an
      error immediately; there is no online fallback.
    - Accepts a single URL or a list of URLs; every URL is extracted against
      the same ``goal``.
    - ``goal`` is required — the local vLLM LLM API is always invoked to
      extract evidence/summary that is relevant to the goal.

Constructor parameters:
    llm_api_key:  vLLM LLM API key (usually ``"EMPTY"`` for local deploys).
    llm_base_url: vLLM LLM API base URL (e.g. ``http://localhost:8001/v1``).
    llm_model:    LLM model name.
    search_tool:  An initialized ``LocalSearchTool`` instance — supplies the
                  shared docid → text cache.
"""

import json

from cabeza.prompts._tool_prompts import EXTRACTOR_PROMPT
from cabeza.tools.base import BaseTool
from openai import OpenAI

_LOCAL_VISIT_PARAMETERS = {
    "type": "object",
    "properties": {
        "url": {
            "type": ["string", "array"],
            "items": {
                "type": "string"
                },
            "minItems": 1,
            "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
    },
    "goal": {
            "type": "string",
            "description": "The goal of the visit for webpage(s)."
    }
    },
    "required": ["url", "goal"]
}

_LOCAL_VISIT_DESCRIPTION = (
    "Visit webpage(s) and return the summary of the content."
)


class LocalVisitTool(BaseTool):
    """Visit tool backed purely by the local corpus cache (no Jina fallback).

    Content is retrieved by URL from the linked LocalSearchTool's in-memory corpus
    store. Only URLs that appeared in a prior search result are accessible; if the
    URL is not found, an error is returned immediately.

    Args:
        llm_api_key:       API key for the auxiliary LLM (usually "EMPTY" for local vllm).
        llm_base_url:      Base URL of the vllm LLM API.
        llm_model:         Model name to use for evidence extraction.
        search_tool:       The LocalSearchTool instance whose corpus cache is used.
        llm_max_retries:   Number of retries on LLM call failure.
        max_content_chars: Maximum characters of corpus text sent to the LLM.
    """

    MAX_CONTENT_CHARS = 100000

    def __init__(
        self,
        llm_api_key: str,
        llm_base_url: str,
        llm_model: str,
        search_tool,
        llm_max_retries: int = 2,
        max_content_chars: int = MAX_CONTENT_CHARS,
    ):
        super().__init__(
            name="visit",
            description=_LOCAL_VISIT_DESCRIPTION,
            parameters=_LOCAL_VISIT_PARAMETERS,
        )
        self.search_tool = search_tool
        self.llm_max_retries = llm_max_retries
        self.max_content_chars = max_content_chars
        self._llm = OpenAI(api_key=llm_api_key, base_url=llm_base_url)
        self._llm_model = llm_model

    # ------------------------------------------------------------------
    # evidence extraction
    # ------------------------------------------------------------------

    def _extract_evidence(self, content: str, goal: str) -> str:
        messages = [
            {
                "role": "user",
                "content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal),
            }
        ]
        for attempt in range(self.llm_max_retries):
            try:
                resp = self._llm.chat.completions.create(
                    model=self._llm_model,
                    messages=messages,
                    temperature=0.7,
                )
                raw = resp.choices[0].message.content or ""
                try:
                    json.loads(raw)
                    return raw
                except Exception:
                    left, right = raw.find("{"), raw.rfind("}")
                    if left != -1 and right != -1 and left <= right:
                        return raw[left : right + 1]
                    return raw
            except Exception as e:
                if attempt == self.llm_max_retries - 1:
                    return f"[LocalVisitTool] LLM extraction error: {e}"
        return ""

    # ------------------------------------------------------------------
    # execute
    # ------------------------------------------------------------------

    def execute(self, url, goal: str) -> str:
        try:
            urls = self.coerce_str_list(url, field_name="url", extract_urls=True)
        except ValueError as e:
            return f"[LocalVisitTool] {e}"

        results = []
        for u in urls:
            text = self.search_tool.get_text_by_url(u)
            if text is None:
                results.append(
                    f"[LocalVisitTool] URL not found in corpus cache: {u!r}. "
                    "Make sure to run a search first and use a URL from the search results."
                )
                continue

            content = text[: self.max_content_chars]
            evidence = self._extract_evidence(content, goal)
            results.append(f"URL: {u}\nEvidence:\n{evidence}")

        return "\n=======\n".join(results)
