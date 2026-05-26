"""JSONL-backed datasets."""

from __future__ import annotations

import json
import os
from typing import Iterator, Optional

from cabeza.datasets.base import normalize_examples
from cabeza.types import Example


class JSONLDataset:
    """Load a dataset from a JSONL file.

    Each line must be a JSON object with at least a ``question`` field. The
    optional ``id``, ``golden_answers``/``answer``, ``language`` keys are
    normalized; any other keys are preserved under ``metadata``.

    ``limit`` keeps only the first N rows (the typical ``eval_num`` usage:
    "evaluate the first 100 items"). ``None`` or non-positive means no limit.
    """

    def __init__(
        self,
        path: str,
        *,
        name: Optional[str] = None,
        limit: Optional[int] = None,
        spec: object | None = None,
    ) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"JSONL dataset not found: {path}")
        self.path = path
        self.name = name or os.path.splitext(os.path.basename(path))[0]
        self.limit = int(limit) if limit and int(limit) > 0 else None
        self.spec = spec
        self._items: list[Example] = list(self._load(path, limit=self.limit))

    @staticmethod
    def _load(path: str, *, limit: Optional[int] = None) -> Iterator[Example]:
        raw: list[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                raw.append(json.loads(line))
                if limit is not None and len(raw) >= limit:
                    break
        yield from normalize_examples(raw)

    def __iter__(self) -> Iterator[Example]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Example:
        return self._items[idx]
