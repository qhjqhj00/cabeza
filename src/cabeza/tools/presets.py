"""Common tool bundles for agentic-search workloads.

``build_toolset`` accepts a preset name and a kwargs dict; unrecognized names
or extra tools can be combined freely by the caller.
"""

from __future__ import annotations

from typing import Optional

from cabeza.tools.base import BaseTool


def web_tools(
    *,
    search_api_key: str,
    visit_api_key: str,
    visit_llm_api_key: str,
    visit_llm_base_url: str,
    visit_llm_model: str,
    visit_max_content_tokens: int = 95_000,
    jina_timeout: float = 180.0,
) -> list[BaseTool]:
    """Online search + visit tools (SerpAPI proxy + Jina-style fetch)."""
    from cabeza.tools.search import SearchTool
    from cabeza.tools.visit import VisitTool

    return [
        SearchTool(api_key=search_api_key),
        VisitTool(
            jina_api_key=visit_api_key,
            llm_api_key=visit_llm_api_key,
            llm_base_url=visit_llm_base_url,
            llm_model=visit_llm_model,
            max_content_tokens=int(visit_max_content_tokens),
            jina_timeout=float(jina_timeout),
        ),
    ]


def corpus_tools(
    *,
    index_path: str,
    corpus_path: str,
    embed_api_url: str,
    embed_model: str,
    embed_api_key: str = "EMPTY",
    top_k: int = 5,
    snippet_max_tokens: int = 2048,
    visit_llm_api_key: str = "EMPTY",
    visit_llm_base_url: str,
    visit_llm_model: str,
) -> list[BaseTool]:
    """FAISS-backed local corpus search + visit."""
    from cabeza.tools.local_search import LocalSearchTool
    from cabeza.tools.local_visit import LocalVisitTool

    search = LocalSearchTool(
        index_path=index_path,
        corpus_path=corpus_path,
        embed_api_url=embed_api_url,
        embed_model=embed_model,
        embed_api_key=embed_api_key,
        top_k=int(top_k),
        snippet_max_tokens=int(snippet_max_tokens),
    )
    visit = LocalVisitTool(
        llm_api_key=visit_llm_api_key,
        llm_base_url=visit_llm_base_url,
        llm_model=visit_llm_model,
        search_tool=search,
    )
    return [search, visit]


def extras(
    *,
    include_scholar: bool = False,
    include_python: bool = False,
    scholar_api_key: Optional[str] = None,
) -> list[BaseTool]:
    """Optional add-ons (scholar search, code interpreter)."""
    out: list[BaseTool] = []
    if include_scholar:
        from cabeza.tools.scholar import GoogleScholarTool

        out.append(GoogleScholarTool(api_key=scholar_api_key or None))
    if include_python:
        from cabeza.tools.python_exec import PythonTool

        out.append(PythonTool())
    return out


def build_toolset(preset: Optional[str], /, **kwargs) -> list[BaseTool]:
    """Build a tool list from a preset name and kwargs.

    Recognized presets:

    - ``None`` or ``"none"`` → empty list (caller supplies tools manually).
    - ``"web"``              → ``web_tools(**kwargs)`` plus optional scholar/python.
    - ``"corpus"``           → ``corpus_tools(**kwargs)`` plus optional scholar/python.
    - ``"hle"``              → ``web_tools(**kwargs)`` + scholar + python.

    Any keyword starting with ``include_`` is forwarded to ``extras()``.
    """
    if preset is None or str(preset).lower() == "none":
        return []

    name = str(preset).lower()
    extras_kwargs = {
        "include_scholar": bool(kwargs.pop("include_scholar", False)),
        "include_python": bool(kwargs.pop("include_python", False)),
        "scholar_api_key": kwargs.pop("scholar_api_key", None),
    }
    if name == "hle":
        extras_kwargs["include_scholar"] = True
        extras_kwargs["include_python"] = True

    if name in {"web", "hle"}:
        tools = web_tools(**kwargs)
    elif name == "corpus":
        tools = corpus_tools(**kwargs)
    else:
        raise ValueError(
            f"Unknown tool preset {preset!r}. Use 'web', 'corpus', 'hle', or None."
        )
    tools.extend(extras(**extras_kwargs))
    return tools
