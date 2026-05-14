"""Dataset protocol."""

from __future__ import annotations

from typing import Iterable, Iterator, Protocol, runtime_checkable

from cabeza.types import Example


@runtime_checkable
class Dataset(Protocol):
    """Minimal dataset interface.

    A dataset is a named, sized, iterable collection of ``Example`` dicts.
    Implementations may stream from disk, an HTTP API, or HuggingFace.
    """

    name: str

    def __iter__(self) -> Iterator[Example]: ...

    def __len__(self) -> int: ...


def normalize_examples(items: Iterable[dict]) -> Iterator[Example]:
    """Yield ``Example`` records, filling in defaults if keys are missing."""
    for idx, item in enumerate(items):
        if "question" not in item:
            raise ValueError(f"Example {idx} is missing 'question': {item!r}")
        example: Example = {
            "question": str(item["question"]),
        }
        if "id" in item:
            example["id"] = str(item["id"])
        else:
            example["id"] = str(idx)
        if "golden_answers" in item:
            example["golden_answers"] = item["golden_answers"]
        elif "answer" in item:
            example["golden_answers"] = item["answer"]
        if "language" in item:
            example["language"] = item["language"]
        # carry-through metadata
        meta = {
            k: v
            for k, v in item.items()
            if k not in {"id", "question", "golden_answers", "answer", "language"}
        }
        if meta:
            example["metadata"] = meta
        yield example
