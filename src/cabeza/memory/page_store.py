"""Page stores for exploration memory.

A ``PageStore`` is a thin mutable list of page dicts shared across the
``PageMemoryStrategy``, the ``MemoryTool``, and (in swarm mode) the
``Summarizer`` for cross-agent recall.

``SharedPageStore`` adds thread-safety and per-owner pruning so multiple
agents working on the same question can each contribute pages without
clobbering each other.
"""

from __future__ import annotations

import threading
from typing import Optional


def _render_page_summary(summary: str) -> str:
    text = (summary or "").strip()
    if not text:
        return "- Explored a new direction"
    if "\n" not in text:
        return f"- {text}"
    return text


def _render_page_block(page: dict) -> str:
    bullets = [
        (item or "").strip()
        for item in page.get("bullets", [])
        if (item or "").strip()
    ]
    if not bullets:
        return ""
    rendered = "\n\n".join(_render_page_summary(b) for b in bullets)
    return f"Page {page['page_num']}:\n{rendered}"


class PageStore:
    """Single-owner page store.

    The page dict format is::

        {
            "page_num":   int,
            "bullets":    list[str],
            "raw_content": str,
            "token_count": int,
            "entry_count": int,
            "owner":      str,  # tag identifying who created this page (optional)
            "flush_round": int,
            "chunk_index": int,
        }
    """

    def __init__(self, *, max_pages: Optional[int] = None) -> None:
        self._pages: list[dict] = []
        self._max_pages = None if max_pages is None else max(0, int(max_pages))

    # ---- store API (matches what PageMemoryStrategy expects) --------------

    def append(self, page_payloads: list[dict]) -> list[int]:
        page_nums: list[int] = []
        for payload in page_payloads:
            page_num = len(self._pages) + 1
            self._pages.append(self._normalize_payload(payload, page_num))
            page_nums.append(page_num)
        return page_nums

    def render(self, *, owner: str = "") -> str:
        # owner is unused for single-owner stores; included for API symmetry.
        _ = owner
        parts = []
        visible = self._pages if self._max_pages is None else self._pages[-self._max_pages:]
        for page in visible:
            rendered = _render_page_block(page)
            if rendered:
                parts.append(rendered)
        return "\n\n".join(parts)

    # ---- snapshots for tooling --------------------------------------------

    def snapshot(self) -> list[dict]:
        return [self._copy_page(p) for p in self._pages]

    def __len__(self) -> int:
        return len(self._pages)

    # ---- internals --------------------------------------------------------

    @staticmethod
    def _normalize_payload(payload: dict, page_num: int) -> dict:
        summary = (payload.get("summary") or "").strip() or "Explored a new direction"
        raw = (payload.get("raw_content") or "").strip() or "[Empty window]"
        return {
            "page_num": page_num,
            "bullets": [summary],
            "raw_content": raw,
            "token_count": int(payload.get("token_count", 0)),
            "entry_count": int(payload.get("entry_count", 0)),
            "owner": payload.get("owner", "") or "",
            "flush_round": int(payload.get("flush_round", 0)),
            "chunk_index": int(payload.get("chunk_index", 0)),
        }

    @staticmethod
    def _copy_page(page: dict) -> dict:
        copied = dict(page)
        copied["bullets"] = list(page.get("bullets", []))
        return copied


class SharedPageStore(PageStore):
    """Thread-safe page store with per-owner pruning.

    When ``max_pages_per_owner`` is set, oldest pages from any owner that
    overflow the per-owner limit are dropped on append. The rendered output
    surfaces the viewer's own pages first followed by sections for each
    teammate.
    """

    def __init__(self, *, max_pages_per_owner: Optional[int] = None) -> None:
        super().__init__(max_pages=None)
        self._max_per_owner = (
            None if max_pages_per_owner is None else max(1, int(max_pages_per_owner))
        )
        self._next_page_num = 1
        self._lock = threading.Lock()

    # ---- overrides ---------------------------------------------------------

    def append(self, page_payloads: list[dict]) -> list[int]:
        with self._lock:
            page_nums: list[int] = []
            for payload in page_payloads:
                page_num = self._next_page_num
                self._next_page_num += 1
                self._pages.append(self._normalize_payload(payload, page_num))
                page_nums.append(page_num)
            dropped = self._prune_locked()
            if dropped:
                dropped_nums = ", ".join(str(p["page_num"]) for p in dropped)
                print(
                    f"[SharedPageStore] pruned page(s) [{dropped_nums}] because "
                    f"max_pages_per_owner={self._max_per_owner}."
                )
            retained = {p["page_num"] for p in self._pages}
            return [n for n in page_nums if n in retained]

    def render(self, *, owner: str = "") -> str:
        with self._lock:
            return self._render_for_viewer_locked(owner)

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [self._copy_page(p) for p in self._pages]

    def __len__(self) -> int:
        with self._lock:
            return len(self._pages)

    # ---- internals --------------------------------------------------------

    def _prune_locked(self) -> list[dict]:
        if self._max_per_owner is None:
            return []
        keep_counts: dict[object, int] = {}
        drop_indices: set[int] = set()
        for idx in range(len(self._pages) - 1, -1, -1):
            owner = self._pages[idx].get("owner", "") or ""
            keep_counts[owner] = keep_counts.get(owner, 0) + 1
            if keep_counts[owner] > self._max_per_owner:
                drop_indices.add(idx)
        if not drop_indices:
            return []
        dropped = [self._copy_page(p) for i, p in enumerate(self._pages) if i in drop_indices]
        self._pages = [p for i, p in enumerate(self._pages) if i not in drop_indices]
        return dropped

    def _render_for_viewer_locked(self, viewer: str) -> str:
        by_owner: dict[str, list[dict]] = {}
        order: list[str] = []
        for page in self._pages:
            owner = (page.get("owner") or "").strip() or "Unknown"
            if owner not in by_owner:
                by_owner[owner] = []
                order.append(owner)
            by_owner[owner].append(page)

        parts: list[str] = []
        viewer = (viewer or "").strip()

        if viewer and viewer in by_owner:
            own_rendered = [r for r in (_render_page_block(p) for p in by_owner[viewer]) if r]
            if own_rendered:
                parts.append("## Your Explorations\n" + "\n\n".join(own_rendered))

        for owner in order:
            if owner == viewer:
                continue
            rendered = [r for r in (_render_page_block(p) for p in by_owner[owner]) if r]
            if not rendered:
                continue
            heading = f"## Teammate {owner} Explorations" if viewer else f"## {owner} Explorations"
            parts.append(heading + "\n" + "\n\n".join(rendered))

        return "\n\n".join(parts)
