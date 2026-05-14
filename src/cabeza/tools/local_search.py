"""LocalSearchTool — offline corpus search over a local FAISS index and a remote vLLM embedding API.

Features:
    - FAISS index files (``corpus.index``) and corpus text (``.arrow`` files)
      are loaded lazily from disk in a thread-safe manner.
    - Query vectorization is delegated to a remote vLLM embedding API; no
      embedding model is loaded locally.
    - Results are returned as URLs; ``LocalVisitTool`` recovers the full
      document text via ``get_text_by_url()``.

Constructor parameters:
    index_path:    Directory holding the local FAISS index
                   (``corpus.index`` + ``corpus_lookup.pkl``).
    corpus_path:   Directory of HuggingFace ``.arrow`` files for the corpus.
    embed_api_url: vLLM embedding API base URL
                   (e.g. ``http://localhost:8002/v1``).
    embed_model:   Embedding model name (e.g. ``Qwen3-Embedding-8B``).
"""

import pickle
import threading
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import faiss
import numpy as np
import tiktoken
from datasets import concatenate_datasets, load_dataset
from openai import OpenAI

from cabeza.tools.base import BaseTool

_LOCAL_SEARCH_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "Array of query strings. Include multiple complementary search queries in a single call."
        },
    },
    "required": ["query"],
}

_LOCAL_SEARCH_DESCRIPTION = (
    "Performs batched web searches: supply an array 'query'; the tool retrieves the top 5 results for each query in one call."
)

# Qwen3-Embedding query instruction prefix
_QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages "
    "that answer the query\nQuery: "
)


class LocalSearchTool(BaseTool):
    """Search tool backed by a local FAISS corpus index + remote vllm embedding API.

    Architecture:
    - FAISS index and corpus texts are loaded from local disk (index_path / corpus_path).
    - Query embeddings are obtained by calling a vllm-compatible OpenAI embedding API,
      so the embedding model does NOT need to be loaded locally.

    Args:
        index_path:         Directory containing corpus.index and corpus_lookup.pkl.
        corpus_path:        Directory containing corpus .arrow files (HuggingFace Dataset).
        embed_api_url:      Base URL of the vllm embedding API (e.g. http://localhost:8002/v1).
        embed_model:        Model name registered on vllm (e.g. "Qwen3-Embedding-8B").
        embed_api_key:      API key (usually "EMPTY" for local vllm).
        query_prefix:       Instruction prefix prepended to each query before embedding.
        top_k:              Number of results to return per query.
        snippet_max_tokens: Maximum tokens for the displayed snippet.
        get_text_by_url:    Look up the full text of any corpus URL for LocalVisitTool.
    """

    def __init__(
        self,
        index_path: str,
        corpus_path: str,
        embed_api_url: str,
        embed_model: str,
        embed_api_key: str = "EMPTY",
        query_prefix: str = _QUERY_PREFIX,
        top_k: int = 5,
        snippet_max_tokens: int = 512,
    ):
        super().__init__(
            name="search",
            description=_LOCAL_SEARCH_DESCRIPTION,
            parameters=_LOCAL_SEARCH_PARAMETERS,
        )
        self.index_path = Path(index_path)
        self.corpus_path = corpus_path
        self.top_k = top_k
        self.snippet_max_tokens = snippet_max_tokens
        self.query_prefix = query_prefix

        self._embed_client = OpenAI(api_key=embed_api_key, base_url=embed_api_url)
        self._embed_model = embed_model
        self._encoding = tiktoken.get_encoding("cl100k_base")

        # Lazy-loaded resources
        self._index: Optional[faiss.Index] = None
        self._lookup: Optional[list] = None   # [(corpus_docid, url), ...] by FAISS position
        self._corpus_docid_to_text: Optional[dict] = None
        self._url_to_text: Optional[dict] = None   # full corpus URL → text map
        self._loaded = False
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------------
    # lazy loading
    # ------------------------------------------------------------------

    def _load(self):
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            print("[LocalSearchTool] Loading FAISS index and corpus ...")

            self._index = faiss.read_index(str(self.index_path / "corpus.index"))
            print(f"  FAISS index: {self._index.ntotal} vectors")

            with open(self.index_path / "corpus_lookup.pkl", "rb") as f:
                self._lookup = pickle.load(f)  # [(docid, url), ...]

            arrow_files = sorted(Path(self.corpus_path).glob("*.arrow"))
            if not arrow_files:
                raise FileNotFoundError(f"No .arrow files found in {self.corpus_path}")
            datasets = [
                load_dataset("arrow", data_files=str(f), split="train")
                for f in arrow_files
            ]
            full_dataset = concatenate_datasets(datasets)
            self._corpus_docid_to_text = {row["docid"]: row["text"] for row in full_dataset}
            self._url_to_text = {row["url"]: row["text"] for row in full_dataset}
            print(f"  Corpus loaded: {len(self._corpus_docid_to_text)} documents")

            self._loaded = True
            print("[LocalSearchTool] Ready.")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _embed_queries(self, queries: list[str]) -> np.ndarray:
        prefixed = [self.query_prefix + q for q in queries]
        resp = self._embed_client.embeddings.create(input=prefixed, model=self._embed_model)
        vecs = np.array([item.embedding for item in resp.data], dtype="float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1, norms)

    def _truncate_snippet(self, text: str) -> str:
        tokens = self._encoding.encode(text)
        if len(tokens) <= self.snippet_max_tokens:
            return text
        return self._encoding.decode(tokens[: self.snippet_max_tokens])

    def _domain(self, url: str) -> str:
        return urlparse(url).netloc or url

    def get_text_by_url(self, url: str) -> str | None:
        """Return the full document text for ``url``, covering the whole corpus (not just the latest search hits)."""
        return self._url_to_text.get(url) if self._url_to_text else None

    # ------------------------------------------------------------------
    # execute
    # ------------------------------------------------------------------

    def _search_single(self, query: str, q_emb: np.ndarray) -> str:
        """Run a FAISS lookup for one query embedding and format the results."""
        scores, indices = self._index.search(q_emb, self.top_k)

        lines = [f'Search results for: "{query}"\n']
        seen: set[str] = set()

        for rank in range(self.top_k):
            idx = int(indices[0][rank])
            if idx < 0:
                continue
            corpus_docid, url = self._lookup[idx]
            if corpus_docid in seen:
                continue
            seen.add(corpus_docid)

            full_text = self._corpus_docid_to_text.get(corpus_docid, "")
            snippet = self._truncate_snippet(full_text)

            lines.append(
                f"[{len(lines)}] URL: {url}\n"
                f"    Source: {self._domain(url)}\n"
                f"    Snippet: {snippet}\n"
            )

        if len(lines) == 1:
            return f"[LocalSearchTool] No results for query: {query!r}"
        return "\n".join(lines)

    def execute(self, query) -> str:
        self._load()

        try:
            queries = self.coerce_str_list(query, field_name="query")
        except ValueError as e:
            return f"[LocalSearchTool] {e}"

        try:
            # Batch every query into a single embedding API call.
            q_embs = self._embed_queries(queries)
        except Exception as e:
            return f"[LocalSearchTool] Embedding API error: {e}"

        results = [
            self._search_single(q, q_embs[i : i + 1])
            for i, q in enumerate(queries)
        ]
        return "\n=======\n".join(results)
