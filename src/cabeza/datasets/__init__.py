"""Dataset interface.

A ``Dataset`` is anything iterable that yields ``Example`` records (a typed
dict with at least a ``question`` key). Built-in loaders:

- ``JSONLDataset(path)`` — local JSONL file.
- ``HFDataset(repo, split=...)`` — HuggingFace ``datasets`` wrapper (optional).
- ``load(name)``                 — registry of named datasets.
"""

from cabeza.datasets.base import Dataset
from cabeza.datasets.jsonl import JSONLDataset
from cabeza.datasets.registry import (
    builtin_specs,
    load,
    register,
)

__all__ = [
    "Dataset",
    "JSONLDataset",
    "builtin_specs",
    "load",
    "register",
]
