"""Live smoke test for cabeza.

Loads .env from the cabeza directory, then runs four independent probes:

  1. Local Qwen vLLM endpoint via SearchTool-less bare Agent (sanity LLM call).
  2. SearchTool against the SERPER proxy.
  3. VisitTool against the Jina proxy (LLM extractor uses local Qwen).
  4. DeepSeekAgent against the DeepSeek API (one-shot).

Each probe is wrapped in try/except so the script always finishes and prints
a summary table.
"""

from __future__ import annotations

import os
import sys
import textwrap
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


# ---- Load .env -------------------------------------------------------------


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
DEEPSEEK = os.environ.get("DEEPSEEK_API_KEY", "")
QWEN_URL = os.environ.get("QWEN_BASE_URL", "http://localhost:8001/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "Qwen3.6-35B-A3B-FP8")
QWEN_KEY = os.environ.get("QWEN_API_KEY", "EMPTY")


# ---- Probe results ---------------------------------------------------------


_RESULTS: list[tuple[str, str, str]] = []


def _record(name: str, ok: bool, detail: str) -> None:
    _RESULTS.append((name, "PASS" if ok else "FAIL", detail))
    icon = "✓" if ok else "✗"
    print(f"\n[{icon}] {name}: {detail}")


def _short(text: str, limit: int = 240) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit] + " …"


# ---- Probe 1: Qwen via plain OpenAI client (cheapest sanity check) ---------


def probe_qwen_basic() -> None:
    print("\n=== probe 1: local Qwen LLM call ===")
    try:
        from openai import OpenAI

        client = OpenAI(api_key=QWEN_KEY, base_url=QWEN_URL, timeout=60.0)
        resp = client.chat.completions.create(
            model=QWEN_MODEL,
            messages=[
                {"role": "system", "content": "Reply with a single short word."},
                {"role": "user", "content": "Reply: pong"},
            ],
            max_tokens=16,
            temperature=0.0,
        )
        content = resp.choices[0].message.content or ""
        _record("qwen_basic", True, _short(content))
    except Exception as exc:
        traceback.print_exc()
        _record("qwen_basic", False, repr(exc))


# ---- Probe 2: SearchTool via SERPER proxy ----------------------------------


def probe_search() -> None:
    print("\n=== probe 2: SearchTool (SERPER) ===")
    if not SERPER:
        _record("search", False, "SERPER_API_KEY missing")
        return
    try:
        from cabeza.tools.search import SearchTool

        tool = SearchTool(api_key=SERPER)
        result = tool.execute(query="ICML 2024 best paper award")
        sample = _short(result, 360)
        ok = bool(result) and not result.lower().startswith("error")
        _record("search", ok, sample)
    except Exception as exc:
        traceback.print_exc()
        _record("search", False, repr(exc))


# ---- Probe 3: VisitTool via Jina + local Qwen ------------------------------


def probe_visit() -> None:
    print("\n=== probe 3: VisitTool (Jina + Qwen extractor) ===")
    if not JINA:
        _record("visit", False, "JINA_API_KEY missing")
        return
    try:
        from cabeza.tools.visit import VisitTool

        tool = VisitTool(
            jina_api_key=JINA,
            llm_api_key=QWEN_KEY,
            llm_base_url=QWEN_URL,
            llm_model=QWEN_MODEL,
            max_content_tokens=4000,
            jina_timeout=60.0,
        )
        result = tool.execute(
            url="https://en.wikipedia.org/wiki/ICML",
            goal="When and where was the first ICML conference held?",
        )
        sample = _short(result, 360)
        ok = bool(result) and not result.lower().startswith("error")
        _record("visit", ok, sample)
    except Exception as exc:
        traceback.print_exc()
        _record("visit", False, repr(exc))


# ---- Probe 4: DeepSeek family adapter --------------------------------------


def probe_deepseek() -> None:
    print("\n=== probe 4: DeepSeekAgent (one-shot, no tools) ===")
    if not DEEPSEEK:
        _record("deepseek", False, "DEEPSEEK_API_KEY missing")
        return
    try:
        from cabeza import Agent

        agent = Agent(
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            api_key=DEEPSEEK,
            family="deepseek",
            tools=[],
            max_steps=2,
            context_window_tokens=8000,
            temperature=0.0,
            timeout=60.0,
            enable_thinking=False,
        )
        result = agent.run("Reply with the single word 'pong' and nothing else.")
        _record("deepseek", bool(result), _short(result, 240))
    except Exception as exc:
        traceback.print_exc()
        _record("deepseek", False, repr(exc))


# ---- Bonus: Qwen Agent with the search tool (one tool round) ----------------


def probe_qwen_agent_with_search() -> None:
    print("\n=== probe 5: QwenAgent + SearchTool (one tool round) ===")
    if not SERPER:
        _record("qwen_agent_search", False, "SERPER_API_KEY missing")
        return
    try:
        from cabeza import Agent
        from cabeza.tools.search import SearchTool

        agent = Agent(
            model=QWEN_MODEL,
            base_url=QWEN_URL,
            api_key=QWEN_KEY,
            family="qwen",
            tools=[SearchTool(api_key=SERPER)],
            max_steps=4,
            max_time_seconds=120,
            context_window_tokens=32000,
            temperature=0.0,
            timeout=120.0,
        )
        result = agent.run(
            "Use the search tool once to find the year ICML was first held, then "
            "answer with just the year."
        )
        _record("qwen_agent_search", bool(result), _short(result, 240))
    except Exception as exc:
        traceback.print_exc()
        _record("qwen_agent_search", False, repr(exc))


# ---- Run -------------------------------------------------------------------


def main() -> int:
    started = time.time()
    print(f"cabeza root: {ROOT}")
    print(f"QWEN: {QWEN_MODEL} @ {QWEN_URL}")
    print(f"SERPER configured: {bool(SERPER)}")
    print(f"JINA   configured: {bool(JINA)}")
    print(f"DEEPSEEK configured: {bool(DEEPSEEK)}")

    probe_qwen_basic()
    probe_search()
    probe_visit()
    probe_deepseek()
    probe_qwen_agent_with_search()

    print("\n" + "=" * 60)
    print(f"summary ({time.time() - started:.1f}s elapsed)")
    print("=" * 60)
    for name, status, detail in _RESULTS:
        print(f"  [{status}] {name}: {detail[:180]}")
    failed = [r for r in _RESULTS if r[1] != "PASS"]
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
