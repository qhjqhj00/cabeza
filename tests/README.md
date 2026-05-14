# cabeza tests

Six suites cover the public surface area. All live tests read API keys from
`cabeza/.env` (`SERPER_API_KEY`, `JINA_API_KEY`, `DEEPSEEK_API_KEY`) and expect
the local Qwen vLLM endpoint at `http://localhost:8001/v1` (override with
`QWEN_BASE_URL` / `QWEN_MODEL`).

## Suites

| File | What it covers | Network |
|---|---|---|
| `test_strategies_unit.py` | All 5 context managers + registry — pure unit tests on synthetic message lists, fake summarizer. | none |
| `test_live.py` | Per-piece smoke: local Qwen, Serper search, Serper scholar, raw Jina fetch, full VisitTool, DeepSeek adapter, Qwen+SearchTool round-trip. | yes |
| `test_strategies_live.py` | End-to-end smoke for each of the 5 strategies, DeepSeek-driven agent + Serper, with a tiny soft budget so every strategy must fire. | yes |
| `test_teams.py` | naive / swarm / fugue orchestrators end-to-end on DeepSeek. fugue additionally uses DeepSeek as the auxiliary summarizer model. | yes |
| `test_runner.py` | JSONL + builtin + user-registered datasets, the `limit` knob, parallel + resumable `evaluate()`, plus the `cabeza` CLI (`list` / `run` / `eval`). | partial |
| `test_eval.py` | LLM-as-a-judge accuracy scoring (`cabeza.eval.score` + the `cabeza score` CLI) against the bundled `bc` set. | yes |

## Running

```bash
# Run every suite (~5 min on a warm cache):
python3 tests/run_all.py

# Run one suite:
python3 tests/test_strategies_unit.py
python3 tests/test_strategies_live.py
python3 tests/test_teams.py
python3 tests/test_runner.py
python3 tests/test_live.py
python3 tests/test_eval.py

# Run a subset of probes inside one suite:
python3 tests/test_strategies_unit.py summary_fires page_memory_fires
python3 tests/test_teams.py fugue
python3 tests/test_runner.py cli_list cli_run cli_eval
```

Each suite prints per-probe `[✓]/[✗]` lines plus a final `[PASS]/[FAIL]`
summary table. Exit code is `0` iff every probe passed.

## Notes on flakiness

- `test_strategies_live.discard_all`, `summary`, `page_memory` use a
  deliberately tiny soft budget (800 tokens) to *force* the strategy to
  fire after the first tool turn. With that budget, the agent often cannot
  retain enough state to converge on a final answer — the strategy itself
  is verified to work (`fired=True`), but the prediction may be the harness
  fallback `"No answer found."`. These probes still pass because the
  assertion is "strategy fired AND any non-empty string returned."
- The local Qwen vLLM endpoint can be slow under sustained load (>180 s
  timeouts have been observed). DeepSeek is preferred for tests that need
  many sequential rounds.
