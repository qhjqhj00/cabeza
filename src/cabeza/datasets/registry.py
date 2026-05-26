"""Named dataset registry.

Bundled evaluation files (BrowseComp, BrowseCompZH, BrowseCompPlus,
WideSearch, HLE, DeepSearchQA, GISA) are registered by their canonical short names. Additional
datasets can be registered programmatically with ``register``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

from cabeza.datasets.base import Dataset
from cabeza.datasets.jsonl import JSONLDataset


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    description: str
    default_path: str  # relative to the data root (or absolute)
    answer_format: str = "free"  # "browsecomp" / "hle" / "gisa" / "free"
    search_mode: str = "online"  # "online" / "corpus"


_DATASET_ALIASES = {
    "bc": "bc",
    "browsecomp": "bc",
    "bc_zh": "bc_zh",
    "bc-zh": "bc_zh",
    "browsecomp_zh": "bc_zh",
    "bcp": "bcp",
    "browsecomp_plus": "bcp",
    "browsecomp-plus": "bcp",
    "ws": "ws",
    "widesearch": "ws",
    "hle": "hle",
    "humanitys_last_exam": "hle",
    "dsqa": "dsqa",
    "deepsearchqa": "dsqa",
    "deepsearch_qa": "dsqa",
    "gisa": "gisa",
}


_SPECS: dict[str, DatasetSpec] = {
    "bc": DatasetSpec(
        name="bc",
        description="BrowseComp",
        default_path="bc/eval.jsonl",
        answer_format="browsecomp",
        search_mode="online",
    ),
    "bc_zh": DatasetSpec(
        name="bc_zh",
        description="BrowseComp-ZH",
        default_path="bc_zh/eval.jsonl",
        answer_format="browsecomp",
        search_mode="online",
    ),
    "bcp": DatasetSpec(
        name="bcp",
        description="BrowseCompPlus (local corpus)",
        default_path="bcp/eval.jsonl",
        answer_format="browsecomp",
        search_mode="corpus",
    ),
    "ws": DatasetSpec(
        name="ws",
        description="WideSearch",
        default_path="ws/eval.jsonl",
        answer_format="free",
        search_mode="online",
    ),
    "hle": DatasetSpec(
        name="hle",
        description="Humanity's Last Exam",
        default_path="hle/eval.jsonl",
        answer_format="hle",
        search_mode="online",
    ),
    "dsqa": DatasetSpec(
        name="dsqa",
        description="DeepSearchQA",
        default_path="dsqa/eval.jsonl",
        answer_format="free",
        search_mode="online",
    ),
    "gisa": DatasetSpec(
        name="gisa",
        description="GISA",
        default_path="gisa/eval.jsonl",
        answer_format="gisa",
        search_mode="online",
    ),
}


_USER_LOADERS: dict[str, Callable[..., Dataset]] = {}


def builtin_specs() -> list[DatasetSpec]:
    return [_SPECS[name] for name in sorted(_SPECS)]


def register(name: str, loader: Callable[..., Dataset]) -> None:
    """Register a custom dataset loader under ``name``."""
    _USER_LOADERS[name] = loader


def get_spec(name: str) -> DatasetSpec:
    canonical = _DATASET_ALIASES.get(name, name)
    if canonical not in _SPECS:
        available = ", ".join(sorted(_DATASET_ALIASES))
        raise ValueError(f"Unknown dataset {name!r}. Available: {available}")
    return _SPECS[canonical]


def _data_root() -> Optional[str]:
    """Resolve the directory holding ``<dataset>/eval.jsonl`` files.

    Lookup order:
      1. ``$CABEZA_DATA_ROOT`` if set.
      2. ``data/`` at the cabeza repo root (the typical editable-install case).
    """
    root = os.environ.get("CABEZA_DATA_ROOT")
    if root and os.path.isdir(root):
        return root

    here = os.path.dirname(os.path.abspath(__file__))
    # cabeza/src/cabeza/datasets → repo root → repo/data
    bundled = os.path.abspath(os.path.join(here, "..", "..", "..", "data"))
    if os.path.isdir(bundled):
        return bundled
    return None


def resolve_default_path(spec: DatasetSpec) -> Optional[str]:
    if os.path.isabs(spec.default_path):
        return spec.default_path
    root = _data_root()
    if root is None:
        return None
    return os.path.join(root, spec.default_path)


def load(
    name: str,
    *,
    path: Optional[str] = None,
    limit: Optional[int] = None,
    **kwargs,
) -> Dataset:
    """Load a named dataset.

    Lookup order:

    1. A user loader previously registered via ``register(name, ...)``.
    2. The built-in spec registry — uses ``path`` if provided, otherwise the
       spec's default path resolved against ``$CABEZA_DATA_ROOT`` or the
       bundled ``cabeza/data/`` layout.

    ``limit`` keeps only the first N examples (i.e. the ``eval_num`` knob:
    ``load("bc", limit=100)`` evaluates the first 100 rows of the full set).
    """
    if name in _USER_LOADERS:
        loader = _USER_LOADERS[name]
        forward = dict(kwargs)
        if limit is not None:
            forward["limit"] = limit
        return loader(path=path, **forward)

    spec = get_spec(name)
    resolved = path or resolve_default_path(spec)
    if not resolved:
        raise FileNotFoundError(
            f"No path supplied for dataset {name!r}; set CABEZA_DATA_ROOT or pass path=..."
        )
    return JSONLDataset(resolved, name=spec.name, limit=limit, spec=spec)
