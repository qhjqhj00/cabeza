# Cabeza

A configurable inference harness for long-horizon agentic search. Cabeza ships
six agent families (Qwen / GLM / Kimi / DeepSeek / GPT / GPT-OSS), five
pluggable context-management strategies, page-based memory, three multi-agent
topologies, a resumable batch runner, and an LLM-as-a-judge accuracy scorer —
all behind a single `Agent` factory.

> Refactored from the experimental SAM toolkit at
> <https://github.com/namespace-ERI/SAM> into a self-contained inference-only
> harness.

---

## Quickstart

```bash
pip install -e .
```

Cabeza is **provider-agnostic**: any OpenAI-compatible endpoint (hosted or
local vLLM) works for the agent, the search/visit tools, the summarizer, and
the judge — independently. Six families are bundled: `qwen`, `glm`, `kimi`,
`deepseek`, `gpt`, `gpt_oss`.

The minimal **agentic-search** loop is an LLM + a web search tool:

```python
from cabeza import Agent
from cabeza.tools.search import SearchTool

agent = Agent(
    model="deepseek-v4-flash",
    base_url="https://api.deepseek.com/v1",
    api_key="sk-...",                      # your provider key
    family="deepseek",                     # qwen / glm / kimi / deepseek / gpt / gpt_oss
    tools=[SearchTool(api_key="serper-...")],
    max_steps=8,
)

print(agent.run(
    "Which 2017 paper introduced the Transformer architecture? Use web "
    "search to confirm and reply with just the title."
))
# → Attention Is All You Need
```

Swap providers by changing `family` + `model` + `base_url` + `api_key` — the
four knobs are the same shape for every backend:

```python
# Local Qwen via vLLM
agent = Agent(
    model="Qwen3-30B-A3B-Instruct-2507",
    base_url="http://localhost:8001/v1",
    api_key="EMPTY",
    family="qwen",
    tools=[SearchTool(api_key="serper-...")],
)
```

API keys can be passed as explicit kwargs **or** picked up from environment
variables — explicit wins, env is the fallback. Each family / tool reads its
conventional name (`DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, `SERPER_API_KEY`,
`JINA_API_KEY`, …). The full list lives in [`.env.example`](.env.example):

```python
# Equivalent — pick whichever fits your shell setup:
agent = Agent(model="deepseek-v4-flash", base_url="https://api.deepseek.com/v1",
              api_key="sk-...", family="deepseek")

# With DEEPSEEK_API_KEY set in the environment, the kwarg can be dropped:
agent = Agent(model="deepseek-v4-flash", base_url="https://api.deepseek.com/v1",
              family="deepseek")

# Tools follow the same pattern:
SearchTool(api_key="serper-...")    # explicit
SearchTool()                         # reads SERPER_API_KEY
VisitTool(jina_api_key="...", llm_base_url=..., llm_model=...)
VisitTool(llm_base_url=..., llm_model=...)   # reads JINA_API_KEY
```

Cabeza itself does not auto-load `.env` files — populate `os.environ` first
(your shell, `python-dotenv`, the test helpers under `tests/_helpers.py`,
etc.).

---

## Going further

Three orthogonal knobs on top of the minimal `Agent`:

| Knob | Values | What it does |
|---|---|---|
| `context_management` | `None`, `"summary"`, `"recent_k"`, `"discard_tool"`, `"discard_all"`, `"page_memory"` | How the running history is trimmed/compressed when it exceeds the soft budget. |
| `memory`             | `None`, `"page"`                                                                       | Whether the agent gets a long-term page store + a `memory` tool. Auto-on for `page_memory` and `team="fugue"`. |
| `team`               | `None`, `"naive"`, `"swarm"`, `"fugue"`                                                | Single agent vs. three multi-agent topologies. |

Local Qwen vLLM + Serper web search + rolling-summary compression. The
rolling-summary strategy delegates compression to a separate (often cheaper
/ smaller) LLM — any OpenAI-compatible endpoint works:

```python
from cabeza import Agent
from cabeza.tools.search import SearchTool

agent = Agent(
    model="Qwen3-30B-A3B-Instruct-2507",
    base_url="http://localhost:8001/v1",
    api_key="EMPTY",
    family="qwen",
    tools=[SearchTool(api_key="serper-...")],
    context_management="summary",
    context_management_tokens=64_000,
    context_window_tokens=128_000,
    # Compress with a small hosted model — pick whatever fits your budget:
    summary_model="gpt-4o-mini",
    summary_base_url="https://api.openai.com/v1",
    summary_api_key="sk-...",
    max_steps=20,
    timeout=120.0,
)
print(agent.run("Which 2024 paper introduced …?"))
```

A peer team of two agents (mix-and-match families) sharing a page-memory
store. `memory_config` controls the auxiliary summarizer that produces page
bullets — it doesn't have to match either peer's family:

```python
from cabeza.agent import MemoryConfig, TeamConfig

agent = Agent(
    model="Qwen3-30B-A3B-Instruct-2507",
    base_url="http://localhost:8001/v1",
    api_key="EMPTY",
    family="qwen",
    team="fugue",
    team_config=TeamConfig(
        mode="fugue",
        size=2,
        # Optional: each peer can use a different family / endpoint.
        # families=["qwen", "deepseek"],
        # models=["Qwen3-30B-A3B-Instruct-2507", "deepseek-v4-flash"],
        # base_urls=["http://localhost:8001/v1", "https://api.deepseek.com/v1"],
        # api_keys=["EMPTY", "sk-..."],
    ),
    context_management="page_memory",
    memory_config=MemoryConfig(
        summarizer_model="gpt-4o-mini",
        summarizer_base_url="https://api.openai.com/v1",
        summarizer_api_key="sk-...",
    ),
    tools=[SearchTool(api_key="serper-...")],
)
```

**Full configuration reference**: every kwarg accepted by `Agent`, with
defaults and notes, is documented inline in
[`src/cabeza/agent.py`](src/cabeza/agent.py). The dataclasses
`AgentConfig` / `MemoryConfig` / `TeamConfig` are the single source of truth
for available knobs.

---

## Datasets

Bundled full evaluation files live under `data/<name>/eval.jsonl`:

```text
data/
  bc/eval.jsonl       BrowseComp                       (1266 items)
  bc_zh/eval.jsonl    BrowseComp-ZH                     (289 items)
  bcp/eval.jsonl      BrowseCompPlus (local corpus)     (830 items)
  ws/eval.jsonl       WideSearch                        (200 items)
  ws/gold/            WideSearch gold CSV tables        (200 files)
  hle/eval.jsonl      Humanity's Last Exam             (2158 items)
```

```python
from cabeza.datasets import load, JSONLDataset

ds = load("bc")                              # all 1266 items
ds = load("bc", limit=100)                   # first 100 only (--eval-num)
ds = load("bc", path="/path/to/my.jsonl")    # use a different file
ds = JSONLDataset("/path/to/eval.jsonl")     # anything JSONL-shaped

# Override the bundled location entirely:
#   export CABEZA_DATA_ROOT=/somewhere/else
```

Each JSONL row needs at least `question`. `id`, `golden_answers` / `answer`,
`language` are normalized; any extra keys ride along under `metadata`.

---

## Running an evaluation

```python
from cabeza.runner import evaluate

summary = evaluate(
    agent,
    load("bc", limit=100),
    out="results/bc_qwen.jsonl",
    workers=4,
    log_dir="logs/bc_qwen",
)
print(summary)   # {"completed": ..., "skipped": ..., "total": ..., ...}
```

`evaluate` is resumable — re-running picks up where it left off using the row
`id` already present in `out`.

---

## Scoring accuracy

After a run produces `results.jsonl`, use cabeza's LLM-as-a-judge to compute
accuracy. QA-style benchmarks (`bc` / `bc_zh` / `bcp` / `hle`) and WideSearch
(`ws`) are both supported.

```python
from cabeza.eval import score

score(
    benchmark="bc",
    input_files=["results/bc_qwen.jsonl"],
    judge_model="deepseek-v4-flash",
    judge_base_url="https://api.deepseek.com/v1",
    judge_api_key="sk-...",
    extra_argv=[
        "--scored_output", "results/bc_qwen.scored.jsonl",
        "--summary_output", "results/bc_qwen.summary.json",
        "--num_workers", "16",
    ],
)
```

The summary JSON includes `accuracy`, `judged_predictions`,
`available_predictions`, `total_questions`, calibration error, etc. Pass
multiple `input_files=` to get a pass@k aggregate.

---

## CLI

The `cabeza` console script bundles four subcommands:

```bash
cabeza list                     # registered strategies / team modes / datasets

cabeza run "your question" \    # one-off
    --model deepseek-v4-flash --base-url https://api.deepseek.com/v1 \
    --api-key $DEEPSEEK_API_KEY --family deepseek \
    --tools web --search-api-key $SERPER_API_KEY --jina-api-key $JINA_API_KEY

cabeza eval \                   # full benchmark run, resumable
    --dataset bc --eval-num 100 \
    --out results/bc.jsonl --workers 4 \
    --model deepseek-v4-flash --base-url https://api.deepseek.com/v1 \
    --api-key $DEEPSEEK_API_KEY --family deepseek

cabeza score \                  # judge accuracy on a prediction JSONL
    --benchmark bc --input_file results/bc.jsonl \
    --judge_model deepseek-v4-flash --judge_base_url https://api.deepseek.com/v1 \
    --judge_api_key $DEEPSEEK_API_KEY
```

`cabeza run --help` and `cabeza eval --help` print the full flag tables (the
same kwargs as `Agent(...)`, kebab-cased).

---

## End-to-end example

A runnable script that loads 3 BrowseComp items, runs the agent, then scores
accuracy, lives at [`scripts/example_bc.py`](scripts/example_bc.py):

```bash
python3 scripts/example_bc.py
```

It reads keys from `cabeza/.env`, drops artifacts into `_example_run/`, and
prints the final accuracy summary. Use it as a starting template for your own
runs.

---

## Layout

```text
cabeza/
  pyproject.toml
  README.md
  LICENSE
  data/                        Bundled eval JSONLs (see Datasets)
  src/cabeza/
    __init__.py                Agent, Example, RunResult
    agent.py                   Agent factory + AgentConfig / MemoryConfig / TeamConfig
    base.py                    BaseAgent / NativeToolChatAgent / ContextManager / LLMConfig
    types.py                   Example, RunResult, Step, ToolCall
    _budget.py                 Token-budget helpers
    agents/                    Family adapters (qwen, glm, kimi, deepseek, gpt, gpt_oss)
    prompts/
      __init__.py              format_question + default_system_prompt
      _defaults.py             Bundled system prompts per family / benchmark
      _tool_prompts.py         EXTRACTOR_PROMPT for visit tools
    context/                   5 strategies + the registry + hard-window guard
    memory/                    PageStore + SharedPageStore + Summarizer + MemoryTool
    swarm/                     naive / swarm / fugue orchestrators + parallel-tool patch
    tools/                     BaseTool + search / visit / local_search / local_visit / scholar / python_exec + presets
    datasets/                  Dataset protocol + JSONL + HF + builtin registry
    runner/                    evaluate() — parallel, resumable, JSONL-streamed
    eval/                      LLM-as-a-judge scorer (bc / bc_zh / bcp / hle / ws)
    cli.py                     `cabeza` console entry
  scripts/example_bc.py        Runnable end-to-end example
  tests/                       5 test suites (see tests/README.md)
```

---

## Tests

Live test suites covering every public feature:

```bash
python3 tests/run_all.py
```

Recent run: 35/35 probes pass in ~245 s. See
[`tests/README.md`](tests/README.md) for per-suite details, what each probe
asserts, and known flakiness around aggressive context-budget tests.

---

## License

Cabeza is released under the [MIT License](LICENSE).
