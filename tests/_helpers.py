"""Shared bootstrap for live cabeza tests.

Loads cabeza/.env, exposes the endpoint constants used by every probe, and
provides small factory helpers so each test file stays focused on the
behavior under test rather than wiring.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


_load_env(ROOT / ".env")


SERPER = os.environ.get("SERPER_API_KEY", "")
JINA = os.environ.get("JINA_API_KEY", "")
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

QWEN_URL = os.environ.get("QWEN_BASE_URL", "http://localhost:8001/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "Qwen3.6-35B-A3B-FP8")
QWEN_KEY = os.environ.get("QWEN_API_KEY", "EMPTY")

DEEPSEEK_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")


def short(text: Any, limit: int = 240) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit] + " …"


def search_tool():
    """Construct a Serper search tool, or raise RuntimeError if no key."""
    if not SERPER:
        raise RuntimeError("SERPER_API_KEY missing — set it in cabeza/.env")
    from cabeza.tools.search import SearchTool

    return SearchTool(api_key=SERPER)


def visit_tool():
    """Construct a Jina visit tool using Qwen as the LLM extractor."""
    if not JINA:
        raise RuntimeError("JINA_API_KEY missing — set it in cabeza/.env")
    from cabeza.tools.visit import VisitTool

    return VisitTool(
        jina_api_key=JINA,
        llm_api_key=QWEN_KEY,
        llm_base_url=QWEN_URL,
        llm_model=QWEN_MODEL,
        max_content_tokens=4000,
        jina_timeout=90.0,
    )


def memory_config(**overrides):
    """MemoryConfig pointing at DeepSeek as the auxiliary summarizer."""
    from cabeza.agent import MemoryConfig

    defaults = dict(
        kind="page",
        summarizer_model=DEEPSEEK_MODEL,
        summarizer_base_url=DEEPSEEK_URL,
        summarizer_api_key=DEEPSEEK_KEY,
        summarizer_temperature=0.3,
        summarizer_max_response_tokens=2048,
        page_max_tokens=3000,
        memory_tool_max_pages=5,
    )
    defaults.update(overrides)
    return MemoryConfig(**defaults)


def qwen_agent_kwargs(**overrides) -> dict:
    """Common Qwen-Agent kwargs used by every strategy/team test."""
    kw = dict(
        model=QWEN_MODEL,
        base_url=QWEN_URL,
        api_key=QWEN_KEY,
        family="qwen",
        max_steps=6,
        max_time_seconds=300,
        context_window_tokens=64_000,
        context_management_tokens=4_000,
        temperature=0.0,
        timeout=180.0,
        # The summary strategy needs a separate summarizer LLM endpoint;
        # default it to DeepSeek so any strategy can pick it up.
        summary_model=DEEPSEEK_MODEL,
        summary_base_url=DEEPSEEK_URL,
        summary_api_key=DEEPSEEK_KEY,
    )
    kw.update(overrides)
    return kw


def run_probes(probes: dict[str, Callable[[], tuple[bool, str]]], argv: list[str]) -> int:
    """Generic probe-suite runner used by each test file's __main__.

    With no args: run every probe in registration order.
    With one or more names: run only those probes, in the given order.
    Unknown names cause an error listing the available probes.
    """
    if argv:
        unknown = [name for name in argv if name not in probes]
        if unknown:
            available = ", ".join(probes)
            print(f"[!] unknown probe(s): {unknown}. Available: {available}")
            return 2
        names = list(argv)
    else:
        names = list(probes)

    results: list[tuple[str, bool, str, float]] = []
    for name in names:
        print(f"\n=== {name} ===")
        started = time.time()
        try:
            ok, detail = probes[name]()
        except Exception as exc:
            traceback.print_exc()
            ok, detail = False, repr(exc)
        elapsed = time.time() - started
        icon = "✓" if ok else "✗"
        print(f"[{icon}] {name} ({elapsed:.1f}s): {short(detail, 240)}")
        results.append((name, ok, detail, elapsed))

    print("\n" + "=" * 64)
    print("summary")
    print("=" * 64)
    for name, ok, detail, elapsed in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:18s} {elapsed:6.1f}s  {short(detail, 110)}")

    failed = [r for r in results if not r[1]]
    return 0 if not failed else 1
