"""
SearchTool — online search via the official Serper API.

- POST https://google.serper.dev/search with ``X-API-KEY`` header and a
  JSON body ``{"q": "...", "gl": ..., "hl": ...}``.
- Tool schema: name is ``search``, argument is ``query`` (a JSON array of
  query strings).
- Accepts one or more queries and returns a Markdown-style snippet block for
  each result list.
- Repairs common malformed batched query strings produced by XML tool callers,
  such as missing brackets around ``"query 1", "query 2"`` fragments.
"""

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from cabeza.tools.base import BaseTool

_SEARCH_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "A JSON array of search query strings, e.g. "
                "[\"query one\", \"query two\"]. Do not pass comma-separated "
                "plain text; keep the square brackets and quotes."
            ),
        }
    },
    "required": ["query"],
}

_SEARCH_DESCRIPTION = (
    "Performs batched web searches. The 'query' argument must be a JSON array "
    "of strings, for example [\"query one\", \"query two\"]. Do not omit the "
    "square brackets or quotes. The tool retrieves the top results for each "
    "query in one call."
)

_SERPER_URL = "https://google.serper.dev/search"
_MAX_SEARCH_RETRIES = 8


class _RetryableSearchError(Exception):
    pass


class SearchTool(BaseTool):
    """Batch Google search tool backed by the official Serper API."""

    _QUOTED_QUERY_DELIMITER_RE = re.compile(r"""(?<=["'])\s*,\s*(?=["'])""")
    _TRAILING_QUOTED_QUERY_DELIMITER_RE = re.compile(r"""(?<!^)(?<![\[,])["']\s*,\s*["']""")

    def __init__(self, api_key: str | None = None, *, endpoint: str = _SERPER_URL):
        super().__init__(
            name="search",
            description=_SEARCH_DESCRIPTION,
            parameters=_SEARCH_PARAMETERS,
        )
        # Explicit kwarg → SERPER_API_KEY env var. Raise if neither is set.
        resolved = api_key or os.environ.get("SERPER_API_KEY", "")
        if not resolved:
            raise ValueError(
                "SearchTool requires a Serper API key. "
                "Pass api_key=... or set SERPER_API_KEY in the environment."
            )
        self.api_key = resolved
        self.endpoint = endpoint

    @staticmethod
    def _is_chinese(text: str) -> bool:
        return any("\u4e00" <= c <= "\u9fff" for c in text)

    @staticmethod
    def _retry_delay(attempt: int) -> float:
        return min(max(0.5 * (2 ** (attempt - 1)), 1.0), 5.0)

    def _build_payload(self, query: str) -> dict:
        payload: dict = {"q": query, "page": 1}
        if self._is_chinese(query):
            payload.update({"gl": "cn", "hl": "zh-cn"})
        else:
            payload.update({"gl": "us", "hl": "en"})
        return payload

    def _request_results(self, query: str) -> dict:
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }
        payload = self._build_payload(query)
        last_error: Exception | None = None

        for attempt in range(1, _MAX_SEARCH_RETRIES + 1):
            print(f"[Serper] Searching: {query}")
            try:
                resp = requests.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=(5, 30),
                )

                if resp.status_code == 429 or resp.status_code >= 500:
                    raise _RetryableSearchError(f"Server Error: {resp.status_code}")
                if resp.status_code in {401, 403}:
                    raise PermissionError(
                        f"Serper auth failed ({resp.status_code}): check SERPER_API_KEY"
                    )
                if resp.status_code == 422:
                    raise ValueError(f"URL Unprocessable by Serper: {query}")
                if resp.status_code >= 400:
                    raise _RetryableSearchError(
                        f"Serper request failed: {resp.status_code} {resp.text[:200]}"
                    )

                data = resp.json()
                if isinstance(data, dict) and data.get("error"):
                    raise _RetryableSearchError(f"Serper error: {data['error']}")
                if "organic" not in data:
                    raise _RetryableSearchError("Error, No results found for query")
                return data
            except (ValueError, PermissionError):
                raise
            except (requests.RequestException, _RetryableSearchError) as exc:
                last_error = exc
                if attempt == _MAX_SEARCH_RETRIES:
                    break
                print(f"Serper error, retry {attempt}/{_MAX_SEARCH_RETRIES}: {exc}")
                time.sleep(self._retry_delay(attempt))
            except Exception as exc:
                last_error = exc
                break

        if last_error is None:
            last_error = RuntimeError("Unknown search failure")
        raise last_error

    def _format_results(self, query: str, results: dict) -> str:
        web_snippets: list[str] = []
        for idx, page in enumerate(results["organic"], 1):
            date_published = (
                "\nDate published: " + str(page["date"]) if "date" in page else ""
            )
            source = "\nSource: " + str(page["source"]) if "source" in page else ""
            snippet = "\n" + str(page["snippet"]) if "snippet" in page else ""
            title = str(page.get("title", "No Title"))
            link = str(page.get("link", ""))
            redacted = f"{idx}. [{title}]({link}){date_published}{source}\n{snippet}"
            redacted = redacted.replace("Your browser can't play this video.", "")
            web_snippets.append(redacted)

        return (
            f"A Google search for '{query}' found {len(web_snippets)} results:\n\n"
            "## Web Results\n"
            + "\n\n".join(web_snippets)
        )

    def _single_search(self, query: str) -> str:
        try:
            results = self._request_results(query)
            if "organic" not in results:
                return (
                    f"No results found for query: '{query}'. "
                    "Try with a more general query."
                )
            return self._format_results(query, results)
        except ValueError:
            print("Http Error. Retry next time.")
            return "Http Error. Retry next time."
        except Exception as exc:
            print(f"Serper Error: {type(exc).__name__}: {exc}")
            return f"Search Error: {type(exc).__name__}: {exc}"

    @classmethod
    def _clean_repaired_query(cls, query: str) -> str:
        query = query.strip()
        query = query.strip("[],")
        query = cls._strip_wrapping_quotes(query)
        query = query.strip("[],")
        query = query.strip(" \t\r\n\"'")
        return query.strip()

    @classmethod
    def _clean_single_query_string(cls, query: str) -> str:
        query = str(query or "").strip()
        if len(query) >= 2 and query[0] == "[" and query[-1] == "]":
            inner = query[1:-1].strip()
            if inner and not cls._repair_query_batch_string(inner):
                query = inner

        query = query.strip().strip(",")
        if (
            len(query) >= 2
            and query[0] == query[-1]
            and query[0] in {"'", '"'}
            and query.count(query[0]) % 2 == 1
        ):
            query = query[:-1].strip()
        else:
            query = cls._strip_wrapping_quotes(query)
        for quote in ("'", '"'):
            if query.startswith(quote) and query.count(quote) == 1:
                query = query[1:].strip()
            if query.endswith(quote) and query.count(quote) == 1:
                query = query[:-1].strip()
        query = query.strip().strip(",")
        return query.strip()

    @classmethod
    def _split_quoted_query_batch(cls, text: str) -> list[str]:
        normalized = text.strip().replace('\\"', '"').replace("\\'", "'")
        body = normalized.strip().strip("[]").strip()
        if not cls._QUOTED_QUERY_DELIMITER_RE.search(body):
            return []

        parts = [
            cls._clean_single_query_string(part)
            for part in cls._QUOTED_QUERY_DELIMITER_RE.split(body)
        ]
        return [part for part in parts if part]

    @classmethod
    def _split_trailing_quoted_query_batch(cls, text: str) -> list[str]:
        normalized = text.strip().replace('\\"', '"').replace("\\'", "'")
        body = normalized.strip().strip("[]").strip()
        if not cls._TRAILING_QUOTED_QUERY_DELIMITER_RE.search(body):
            return []

        parts = [
            cls._clean_single_query_string(part)
            for part in cls._TRAILING_QUOTED_QUERY_DELIMITER_RE.split(body)
        ]
        return [part for part in parts if part]

    @classmethod
    def _split_newline_query_batch(cls, text: str) -> list[str]:
        body = text.strip().strip("[]").strip()
        if "\n" not in body:
            return []

        parts = []
        for line in body.splitlines():
            line = cls._clean_repaired_query(line)
            if not line:
                continue
            line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            line = cls._clean_repaired_query(line)
            if line:
                parts.append(line)
        if not 2 <= len(parts) <= 8:
            return []
        if not all(cls._looks_like_query_clause(part) for part in parts):
            return []
        return parts

    @classmethod
    def _looks_like_query_clause(cls, text: str) -> bool:
        text = cls._clean_repaired_query(text)
        if len(text) < 16:
            return False
        if re.search(r"https?://", text):
            return False
        return len(re.findall(r"\S+", text)) >= 3

    @classmethod
    def _split_plain_query_batch(cls, text: str) -> list[str]:
        body = text.strip().strip("[]").strip()
        if "," not in body:
            return []

        parts = [cls._clean_repaired_query(part) for part in body.split(",")]
        parts = [part for part in parts if part]
        if not 2 <= len(parts) <= 8:
            return []
        if len(parts) == 2 and not all(len(part) >= 24 for part in parts):
            return []
        if not all(cls._looks_like_query_clause(part) for part in parts):
            return []
        return parts

    @classmethod
    def _repair_query_batch_string(cls, value: str) -> list[str]:
        text = (value or "").strip()
        if not text:
            return []

        for splitter in (
            cls._split_quoted_query_batch,
            cls._split_trailing_quoted_query_batch,
            cls._split_newline_query_batch,
            cls._split_plain_query_batch,
        ):
            repaired = splitter(text)
            if len(repaired) > 1:
                return repaired
        return []

    @classmethod
    def _coerce_queries(cls, query) -> list[str]:
        queries = cls.coerce_str_list(query, field_name="query")
        if isinstance(query, str) and len(queries) == 1:
            repaired = cls._repair_query_batch_string(query)
            if len(repaired) > 1:
                return repaired
            cleaned = cls._clean_single_query_string(queries[0])
            return [cleaned] if cleaned else []
        return [cls._clean_single_query_string(q) for q in queries if cls._clean_single_query_string(q)]

    def execute(self, query) -> str:
        try:
            queries = self._coerce_queries(query)
        except ValueError as exc:
            return f"[SearchTool] {exc}"

        results_map: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_query = {
                executor.submit(self._single_search, q): q for q in queries
            }
            for future in as_completed(future_to_query):
                q = future_to_query[future]
                try:
                    results_map[q] = future.result()
                except Exception as exc:
                    results_map[q] = f"Search Error: {type(exc).__name__}: {exc}"

        return "\n=======\n".join(results_map.get(q, "Error") for q in queries)
