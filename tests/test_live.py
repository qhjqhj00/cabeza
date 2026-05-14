"""Live integration probes for cabeza tools and LLM endpoints.

Reads cabeza/.env, then exercises each piece independently. Each test is a
function that returns (ok, detail). Run with::

    python -m cabeza.tests.test_live          # all probes
    python -m cabeza.tests.test_live search   # one probe by name

Probes:
    qwen      — local Qwen vLLM endpoint, bare openai client call
    serper    — SearchTool against https://google.serper.dev/search
    scholar   — GoogleScholarTool against https://google.serper.dev/scholar
    jina_raw  — Raw Jina Reader fetch (no LLM extraction)
    visit     — VisitTool (Jina + Qwen-extractor) end-to-end
    deepseek  — DeepSeek v4-flash via DeepSeekAgent (one-shot)
    qwen_tool — QwenAgent + SearchTool, single tool round
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
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


def _short(text, limit: int = 300) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit] + " …"


# ---- individual probes -----------------------------------------------------


def probe_qwen() -> tuple[bool, str]:
    from openai import OpenAI

    client = OpenAI(api_key=QWEN_KEY, base_url=QWEN_URL, timeout=60.0)
    resp = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[
            {"role": "system", "content": "Reply with a single short word."},
            {"role": "user", "content": "Reply: pong"},
        ],
        max_tokens=32,
        temperature=0.0,
    )
    content = (resp.choices[0].message.content or "").strip()
    ok = "pong" in content.lower()
    return ok, _short(content)


def probe_serper() -> tuple[bool, str]:
    if not SERPER:
        return False, "SERPER_API_KEY missing"
    from cabeza.tools.search import SearchTool

    tool = SearchTool(api_key=SERPER)
    result = tool.execute(query=["ICML conference first year"])
    ok = bool(result) and "found" in result.lower() and "results" in result.lower()
    return ok, _short(result, 360)


def probe_scholar() -> tuple[bool, str]:
    if not SERPER:
        return False, "SERPER_API_KEY missing"
    from cabeza.tools.scholar import GoogleScholarTool

    tool = GoogleScholarTool(api_key=SERPER)
    result = tool.execute(query=["transformer architecture attention is all you need"])
    ok = bool(result) and ("Scholar" in result or "publicationInfo" in result)
    return ok, _short(result, 360)


def probe_jina_raw() -> tuple[bool, str]:
    if not JINA:
        return False, "JINA_API_KEY missing"
    import requests

    url = "https://r.jina.ai/https://en.wikipedia.org/wiki/ICML"
    headers = {"Authorization": f"Bearer {JINA}", "Accept": "text/plain"}
    resp = requests.get(url, headers=headers, timeout=(10, 90))
    text = resp.text or ""
    ok = resp.status_code == 200 and len(text) > 500 and "ICML" in text
    return ok, f"status={resp.status_code} bytes={len(text)} head={_short(text, 200)}"


def probe_visit() -> tuple[bool, str]:
    if not JINA:
        return False, "JINA_API_KEY missing"
    from cabeza.tools.visit import VisitTool

    tool = VisitTool(
        jina_api_key=JINA,
        llm_api_key=QWEN_KEY,
        llm_base_url=QWEN_URL,
        llm_model=QWEN_MODEL,
        max_content_tokens=6000,
        jina_timeout=90.0,
    )
    result = tool.execute(
        url="https://en.wikipedia.org/wiki/ICML",
        goal="When and where was the first ICML conference held?",
    )
    ok = bool(result) and "useful information" in result.lower()
    return ok, _short(result, 360)


def probe_deepseek() -> tuple[bool, str]:
    if not DEEPSEEK_KEY:
        return False, "DEEPSEEK_API_KEY missing"
    from cabeza import Agent

    agent = Agent(
        model=DEEPSEEK_MODEL,
        base_url=DEEPSEEK_URL,
        api_key=DEEPSEEK_KEY,
        family="deepseek",
        tools=[],
        max_steps=2,
        context_window_tokens=8000,
        temperature=0.0,
        timeout=60.0,
        enable_thinking=False,
    )
    out = agent.run("Reply with the single word 'pong' and nothing else.")
    ok = bool(out) and "pong" in out.lower()
    return ok, _short(out)


def probe_qwen_tool() -> tuple[bool, str]:
    if not SERPER:
        return False, "SERPER_API_KEY missing"
    from cabeza import Agent
    from cabeza.tools.search import SearchTool

    agent = Agent(
        model=QWEN_MODEL,
        base_url=QWEN_URL,
        api_key=QWEN_KEY,
        family="qwen",
        tools=[SearchTool(api_key=SERPER)],
        max_steps=4,
        max_time_seconds=180,
        context_window_tokens=32000,
        temperature=0.0,
        timeout=180.0,
    )
    out = agent.run(
        "Use the search tool once to find the year ICML was first held, "
        "then answer with just the year (a 4-digit number)."
    )
    ok = bool(out) and any(year in out for year in ("1980", "1988"))
    return ok, _short(out)


PROBES: dict[str, Callable[[], tuple[bool, str]]] = {
    "qwen": probe_qwen,
    "serper": probe_serper,
    "scholar": probe_scholar,
    "jina_raw": probe_jina_raw,
    "visit": probe_visit,
    "deepseek": probe_deepseek,
    "qwen_tool": probe_qwen_tool,
}


def main(argv: list[str]) -> int:
    if argv and argv[0] in PROBES:
        names = [argv[0]]
    else:
        names = list(PROBES)

    results: list[tuple[str, bool, str, float]] = []
    for name in names:
        print(f"\n=== {name} ===")
        started = time.time()
        try:
            ok, detail = PROBES[name]()
        except Exception as exc:
            traceback.print_exc()
            ok, detail = False, repr(exc)
        elapsed = time.time() - started
        icon = "✓" if ok else "✗"
        print(f"[{icon}] {name} ({elapsed:.1f}s): {detail[:200]}")
        results.append((name, ok, detail, elapsed))

    print("\n" + "=" * 60)
    print("summary")
    print("=" * 60)
    for name, ok, detail, elapsed in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:10s} {elapsed:6.1f}s  {detail[:120]}")

    failed = [r for r in results if not r[1]]
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
