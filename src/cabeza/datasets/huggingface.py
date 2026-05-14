"""HuggingFace ``datasets`` wrapper (optional dependency)."""

from __future__ import annotations

from typing import Iterator, Optional

from cabeza.datasets.base import normalize_examples
from cabeza.types import Example


class HFDataset:
    """Load a dataset via the HuggingFace ``datasets`` library.

    Requires the ``cabeza[hf]`` extra. ``question_key`` and ``answer_key`` let
    you map non-standard column names onto the cabeza ``Example`` shape.
    """

    def __init__(
        self,
        repo: str,
        *,
        split: str = "test",
        name: Optional[str] = None,
        question_key: str = "question",
        answer_key: Optional[str] = "answer",
        id_key: Optional[str] = "id",
        language_key: Optional[str] = None,
        config: Optional[str] = None,
    ) -> None:
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised via extra
            raise ImportError(
                "HFDataset requires the 'datasets' package. Install with `pip install cabeza[hf]`."
            ) from exc

        ds = load_dataset(repo, name=config, split=split) if config else load_dataset(repo, split=split)
        self.repo = repo
        self.split = split
        self.name = name or f"{repo}:{split}"
        records: list[dict] = []
        for idx, row in enumerate(ds):
            normalized: dict = {"question": row[question_key]}
            if id_key and id_key in row:
                normalized["id"] = row[id_key]
            else:
                normalized["id"] = str(idx)
            if answer_key and answer_key in row:
                normalized["golden_answers"] = row[answer_key]
            if language_key and language_key in row:
                normalized["language"] = row[language_key]
            for k, v in row.items():
                if k in {question_key, answer_key, id_key, language_key}:
                    continue
                normalized.setdefault("metadata", {})[k] = v
            records.append(normalized)
        self._items: list[Example] = list(normalize_examples(records))

    def __iter__(self) -> Iterator[Example]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)
