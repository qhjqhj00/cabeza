"""
GoogleScholarTool — Google Scholar via the official Serper API.

- POST https://google.serper.dev/scholar with ``X-API-KEY`` header and a JSON
  body ``{"q": "..."}``.
- Tool schema: name is ``google_scholar``, argument is ``query``.
- Accepts one or more queries and returns a readable text summary for each.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from cabeza.tools.base import BaseTool

_GOOGLE_SCHOLAR_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "array",
            "items": {
                "type": "string",
                "description": "The search query.",
            },
            "minItems": 1,
            "description": "The list of search queries for Google Scholar.",
        }
    },
    "required": ["query"],
    "additionalProperties": False,
}

_GOOGLE_SCHOLAR_DESCRIPTION = (
    "Leverage Google Scholar to retrieve relevant information from academic "
    "publications. Accepts multiple queries."
)

_SCHOLAR_URL = "https://google.serper.dev/scholar"
_DEFAULT_MAX_RETRIES = 5


class _RetryableScholarError(Exception):
    pass


class _ScholarRequestError(Exception):
    pass


class GoogleScholarTool(BaseTool):
    """Google Scholar tool backed by the rag.ac.cn SERP proxy."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        endpoint: str = _SCHOLAR_URL,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        max_workers: int = 3,
    ):
        super().__init__(
            name="google_scholar",
            description=_GOOGLE_SCHOLAR_DESCRIPTION,
            parameters=_GOOGLE_SCHOLAR_PARAMETERS,
        )
        self.api_key = api_key or os.environ.get("SERPER_API_KEY") or os.environ.get("SERPER_KEY_ID")
        self.endpoint = endpoint
        self.max_retries = max(1, int(max_retries))
        self.max_workers = max(1, int(max_workers))

    @staticmethod
    def _retry_delay(attempt: int) -> float:
        return min(max(0.5 * (2 ** (attempt - 1)), 1.0), 5.0)

    @staticmethod
    def _is_chinese(text: str) -> bool:
        return any("\u4e00" <= c <= "\u9fff" for c in text)

    def _build_payload(self, query: str) -> dict:
        payload: dict = {"q": query, "page": 1}
        if self._is_chinese(query):
            payload.update({"gl": "cn", "hl": "zh-cn"})
        else:
            payload.update({"gl": "us", "hl": "en"})
        return payload

    def _request_results(self, query: str) -> dict:
        headers = {
            "X-API-KEY": self.api_key or "",
            "Content-Type": "application/json",
        }
        payload = self._build_payload(query)
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            print(f"[GoogleScholar] Searching: {query}")
            try:
                resp = requests.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=(5, 30),
                )

                if resp.status_code == 429 or resp.status_code >= 500:
                    raise _RetryableScholarError(f"Server Error: {resp.status_code}")
                try:
                    data = resp.json()
                except ValueError as exc:
                    raise _ScholarRequestError("Scholar response was not valid JSON") from exc

                error_message = str(data.get("error", "") or "").strip()
                if resp.status_code in {401, 403}:
                    raise PermissionError(error_message or f"Scholar request failed: {resp.status_code}")
                if resp.status_code >= 400:
                    raise _ScholarRequestError(
                        error_message or f"Scholar request failed: {resp.status_code}"
                    )
                if error_message:
                    raise _ScholarRequestError(error_message)

                return data
            except (requests.RequestException, _RetryableScholarError) as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                print(
                    f"[GoogleScholarTool] retry {attempt}/{self.max_retries} for "
                    f"query {query!r}: {exc}"
                )
                time.sleep(self._retry_delay(attempt))
            except Exception as exc:
                last_error = exc
                break

        if last_error is None:
            last_error = RuntimeError("Unknown scholar search failure")
        raise last_error

    @staticmethod
    def _stringify_publication_info(value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            summary = value.get("summary")
            if summary:
                return str(summary)
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)

    def _format_results(self, query: str, results: dict) -> str:
        organic = results.get("organic") or []
        web_snippets: list[str] = []

        for idx, page in enumerate(organic, 1):
            title = str(page.get("title", "No Title"))
            pdf_url = str(page.get("pdfUrl", "") or "")
            link = str(page.get("link", "") or "")
            best_link = pdf_url or link

            header = (
                f"{idx}. [{title}]({best_link})"
                if best_link
                else f"{idx}. {title}\nLink: no available link"
            )

            publication_info = self._stringify_publication_info(
                page.get("publicationInfo")
            )
            year = page.get("year")
            cited_by = page.get("citedBy")
            snippet = str(page.get("snippet", "") or "")

            lines = [header]
            if publication_info:
                lines.append(f"publicationInfo: {publication_info}")
            if year is not None:
                lines.append(f"Date published: {year}")
            if cited_by is not None:
                lines.append(f"citedBy: {cited_by}")
            if snippet:
                lines.append(snippet.replace("Your browser can't play this video.", "").strip())

            web_snippets.append("\n".join(line for line in lines if line))

        return (
            f"A Google scholar for '{query}' found {len(web_snippets)} results:\n\n"
            "## Scholar Results\n"
            + "\n\n".join(web_snippets)
        )

    def _single_search(self, query: str) -> str:
        try:
            results = self._request_results(query)
            if not results.get("organic"):
                return (
                    f"No results found for '{query}'. "
                    "Try with a more general query."
                )
            return self._format_results(query, results)
        except PermissionError as exc:
            return f"Scholar Error: PermissionError: {exc}"
        except _ScholarRequestError as exc:
            return f"Scholar Error: {_ScholarRequestError.__name__}: {exc}"
        except Exception as exc:
            print(f"[GoogleScholarTool] {type(exc).__name__}: {exc}")
            return f"Scholar Error: {type(exc).__name__}: {exc}"

    def execute(self, query) -> str:
        if not self.api_key:
            print("[GoogleScholarTool] Missing Serper API key.")
            return (
                "[GoogleScholarTool] Missing Serper API key. "
                "Set SERPER_API_KEY or pass api_key explicitly."
            )

        try:
            queries = self.coerce_str_list(query, field_name="query")
        except ValueError as exc:
            return f"[GoogleScholarTool] {exc}"

        results_map: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(queries))) as executor:
            future_to_query = {
                executor.submit(self._single_search, q): q for q in queries
            }
            for future in as_completed(future_to_query):
                q = future_to_query[future]
                try:
                    results_map[q] = future.result()
                except Exception as exc:
                    results_map[q] = f"Scholar Error: {type(exc).__name__}: {exc}"

        return "\n=======\n".join(results_map.get(q, "Error") for q in queries)
