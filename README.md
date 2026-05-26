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
    verbose=True,                          # stream step/tool/final to stdout
)

print(agent.run(
    "Which 2017 paper introduced the Transformer architecture? Use web "
    "search to confirm and reply with just the title."
))
# [cabeza] step 1 → search({'query': ['2017 paper Transformer architecture introduced']})
# [Serper]  Searching: 2017 paper Transformer architecture introduced
# [cabeza] step 1 ← search: A Google search for ... found 9 results …
# [cabeza] step 2 ✓ final: "Attention Is All You Need"
# Attention Is All You Need
```

`verbose` is off by default — drop it for a silent run (only tool-side
prints remain). For richer instrumentation, register your own callbacks:

```python
agent.on("after_llm",  lambda state, parsed: ...)
agent.on("after_tool", lambda state, name, args, result: ...)
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

DeepSeek for both the reasoning agent and the rolling-summary compressor,
Serper for web search, and Jina Reader for fetching full pages. With
`context_management="summary"`, whenever the running history exceeds
`context_management_tokens`, the auxiliary LLM rewrites it into a compact
summary tucked into the first user message:

```python
from cabeza import Agent
from cabeza.tools.search import SearchTool
from cabeza.tools.visit import VisitTool

deepseek_url = "https://api.deepseek.com/v1"
deepseek_key = "sk-..."           # or set DEEPSEEK_API_KEY and drop the kwarg

agent = Agent(
    model="deepseek-v4-flash",
    base_url=deepseek_url,
    api_key=deepseek_key,
    family="deepseek",
    tools=[
        SearchTool(api_key="serper-..."),        # or SearchTool() + $SERPER_API_KEY
        VisitTool(                                # or VisitTool() + $JINA_API_KEY
            jina_api_key="jina_...",
            llm_api_key=deepseek_key,             # extractor LLM for visited pages
            llm_base_url=deepseek_url,
            llm_model="deepseek-v4-flash",
        ),
    ],
    context_management="summary",
    context_management_tokens=64_000,
    context_window_tokens=128_000,
    # Same DeepSeek endpoint reused as the rolling-summary compressor:
    summary_model="deepseek-v4-flash",
    summary_base_url=deepseek_url,
    summary_api_key=deepseek_key,
    max_steps=20,
    timeout=120.0,
    verbose=True,
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
defaults and notes, is documented inline in the `Agent.__init__` docstring at
[`src/cabeza/agent.py`](src/cabeza/agent.py). The `MemoryConfig` and
`TeamConfig` dataclasses in that same module cover the memory- and
team-specific knobs.

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
  dsqa/eval.jsonl     DeepSearchQA                      (900 items)
  dsqa/DSQA-full.csv  DeepSearchQA official metadata    (900 rows)
  gisa/eval.jsonl     GISA                              (373 items)
  gisa/encrypted_question.jsonl  GISA official metadata (373 rows)
  gisa/answer/        GISA gold CSV answers             (373 files)
```

```python
from cabeza.datasets import load, JSONLDataset

ds = load("bc")                              # all 1266 items
ds = load("bc", limit=100)                   # first 100 only (--eval-num)
ds = load("dsqa")                            # all 900 DeepSearchQA items
ds = load("gisa")                            # all 373 GISA items
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
accuracy. QA-style benchmarks (`bc` / `bc_zh` / `bcp` / `hle`), WideSearch
(`ws`), DeepSearchQA (`dsqa`), and GISA (`gisa`) are supported.

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

DeepSearchQA uses its official LLM-as-a-judge prompt and reports the official
fully-correct / fully-incorrect / correct-with-excessive plus precision,
recall, and F1 metrics:

```bash
cabeza score \
    --benchmark dsqa --input_file results/dsqa.jsonl \
    --judge_model gemini-2.5-flash \
    --judge_base_url https://your-openai-compatible-endpoint/v1 \
    --judge_api_key $JUDGE_API_KEY
```

GISA uses the official deterministic TSV/CSV scorer and does not require a
judge model. `load("gisa")` / `cabeza eval --dataset gisa` automatically wraps
questions with the expected TSV output instruction.

```bash
cabeza score \
    --benchmark gisa --input_file results/gisa.jsonl \
    --summary_output results/gisa.summary.json
```

---

## CLI

The `cabeza` console script bundles four subcommands.

**Zero-config** — with provider + tool keys in the environment, `cabeza run`
auto-detects everything. `--model` / `--base-url` / `--family` are inferred
from `DEEPSEEK_API_KEY` → `OPENAI_API_KEY` → `QWEN_BASE_URL` (first match
wins), and web tools turn on automatically when `SERPER_API_KEY` +
`JINA_API_KEY` are both set:

```bash
# .env already has DEEPSEEK_API_KEY / SERPER_API_KEY / JINA_API_KEY exported:
cabeza run "Which 2017 paper introduced the Transformer architecture?"
# [cabeza] auto-config: DeepSeek / deepseek-v4-flash @ https://api.deepseek.com/v1 + web tools (Serper + Jina)
# [cabeza] step 1 → search(...)
# ...
```

**Explicit** — pass everything yourself when you don't want auto-detection:

```bash
cabeza list                     # registered strategies / team modes / datasets

cabeza run "your question" \    # one-off  (streams step/tool/final by default)
    --model deepseek-v4-flash --base-url https://api.deepseek.com/v1 \
    --api-key $DEEPSEEK_API_KEY --family deepseek \
    --tools web --search-api-key $SERPER_API_KEY --jina-api-key $JINA_API_KEY

cabeza eval \                   # full benchmark run, resumable (silent by default)
    --dataset bc --eval-num 100 \
    --out results/bc.jsonl --workers 4 \
    --model deepseek-v4-flash --base-url https://api.deepseek.com/v1 \
    --api-key $DEEPSEEK_API_KEY --family deepseek

cabeza eval \                   # GISA full set; prompts request TSV code blocks
    --dataset gisa \
    --out results/gisa.jsonl --workers 4 \
    --model deepseek-v4-flash --base-url https://api.deepseek.com/v1 \
    --api-key $DEEPSEEK_API_KEY --family deepseek

cabeza score \                  # judge accuracy on a prediction JSONL
    --benchmark bc --input_file results/bc.jsonl \
    --judge_model deepseek-v4-flash --judge_base_url https://api.deepseek.com/v1 \
    --judge_api_key $DEEPSEEK_API_KEY

cabeza score \                  # deterministic official GISA scoring
    --benchmark gisa --input_file results/gisa.jsonl
```

`--verbose` / `--no-verbose` overrides the per-subcommand default (`run`
streams, `eval` is silent). `cabeza run --help` / `cabeza eval --help` print
the full flag tables (the same kwargs as `Agent(...)`, kebab-cased).

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
    agent.py                   Agent factory + MemoryConfig / TeamConfig
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
    eval/                      Scorers (bc / bc_zh / bcp / hle / ws / dsqa / gisa)
    cli.py                     `cabeza` console entry
  scripts/example_bc.py        Runnable end-to-end example
  tests/                       5 test suites (see tests/README.md)
```

---

## Tests

Six suites covering every public feature — strategies (unit + live), tools +
LLM smoke, team modes, the runner + CLI, and the eval scorer:

```bash
python3 tests/run_all.py            # all suites
python3 tests/test_strategies_unit.py   # one suite (no network)
```

See [`tests/README.md`](tests/README.md) for per-suite details, what each
probe asserts, and known flakiness around aggressive context-budget tests.

---

## License

Cabeza is released under the [MIT License](LICENSE).
