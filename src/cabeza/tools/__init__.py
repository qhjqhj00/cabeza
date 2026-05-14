"""Tool primitives and presets.

Concrete tool classes are loaded lazily so optional heavy deps
(``faiss``, ``datasets``, ``numpy``) are only imported when actually used.
"""

from __future__ import annotations

from cabeza.tools.base import BaseTool
from cabeza.tools.presets import build_toolset

__all__ = [
    "BaseTool",
    "build_toolset",
    # Lazy attribute access:
    "GoogleScholarTool",
    "LocalSearchTool",
    "LocalVisitTool",
    "PythonTool",
    "SearchTool",
    "VisitTool",
]


def __getattr__(name: str):
    if name == "GoogleScholarTool":
        from cabeza.tools.scholar import GoogleScholarTool
        return GoogleScholarTool
    if name == "LocalSearchTool":
        from cabeza.tools.local_search import LocalSearchTool
        return LocalSearchTool
    if name == "LocalVisitTool":
        from cabeza.tools.local_visit import LocalVisitTool
        return LocalVisitTool
    if name == "PythonTool":
        from cabeza.tools.python_exec import PythonTool
        return PythonTool
    if name == "SearchTool":
        from cabeza.tools.search import SearchTool
        return SearchTool
    if name == "VisitTool":
        from cabeza.tools.visit import VisitTool
        return VisitTool
    raise AttributeError(name)
