"""End-to-end example: load 3 BrowseComp items, run cabeza, then judge accuracy.

Reads API keys from cabeza/.env (DEEPSEEK_API_KEY / SERPER_API_KEY / JINA_API_KEY)
and runs:

    1. Load the first 3 items of the bundled bc dataset (data/bc/eval.jsonl).
    2. Build an Agent that uses DeepSeek + the Serper SearchTool.
    3. cabeza.runner.evaluate over those 3 questions → results.jsonl.
    4. cabeza.eval.score the results against the gold answers (LLM-as-judge,
       DeepSeek) → summary.json + scored.jsonl.

Run::

    python3 scripts/example_bc.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


# ---- 1. Pick up API keys from cabeza/.env ----------------------------------

def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


_load_env(ROOT / ".env")

DEEPSEEK_KEY = os.environ["DEEPSEEK_API_KEY"]
SERPER_KEY = os.environ.get("SERPER_API_KEY", "")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEEPSEEK_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")


# ---- 2. Build the agent + load the dataset --------------------------------

from cabeza import Agent                                # noqa: E402
from cabeza.datasets import load                        # noqa: E402
from cabeza.runner import evaluate                      # noqa: E402
from cabeza.eval import score                           # noqa: E402

agent_tools = []
if SERPER_KEY:
    from cabeza.tools.search import SearchTool
    agent_tools.append(SearchTool(api_key=SERPER_KEY))

agent = Agent(
    model=DEEPSEEK_MODEL,
    base_url=DEEPSEEK_URL,
    api_key=DEEPSEEK_KEY,
    family="deepseek",
    tools=agent_tools,
    max_steps=4,
    max_time_seconds=180,
    context_window_tokens=32_000,
    temperature=0.0,
    timeout=90.0,
    enable_thinking=False,
    verbose=True,           # stream step/tool/final to stdout
)


# First 3 items of data/bc/eval.jsonl (uses the eval_num knob).
dataset = load("bc", limit=3)
print(f"loaded {len(dataset)} bc items")


# ---- 3. Run the agent over the slice --------------------------------------

out_dir = ROOT / "_example_run"
out_dir.mkdir(exist_ok=True)
results_path = out_dir / "results.jsonl"

summary = evaluate(
    agent,
    dataset,
    out=str(results_path),
    workers=2,
    answer_format="browsecomp",
    log_dir=str(out_dir / "logs"),
)
print("evaluate summary:", json.dumps(summary, ensure_ascii=False))


# ---- 4. Judge accuracy ----------------------------------------------------

scored_path = out_dir / "scored.jsonl"
score_summary_path = out_dir / "score_summary.json"

score(
    benchmark="bc",
    input_files=[str(results_path)],
    judge_model=DEEPSEEK_MODEL,
    judge_base_url=DEEPSEEK_URL,
    judge_api_key=DEEPSEEK_KEY,
    extra_argv=[
        "--scored_output", str(scored_path),
        "--summary_output", str(score_summary_path),
        "--num_workers", "2",
        "--judge_attempts", "2",
    ],
)

print()
print("=== final summary ===")
print(score_summary_path.read_text())
print(f"\nartifacts: {out_dir}")
