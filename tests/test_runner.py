"""Tests for cabeza.datasets, cabeza.runner.evaluate, and the CLI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _helpers import (  # noqa: E402
    DEEPSEEK_KEY,
    DEEPSEEK_MODEL,
    DEEPSEEK_URL,
    ROOT,
    SRC,
    run_probes,
    short,
)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def probe_jsonl_loader() -> tuple[bool, str]:
    from cabeza.datasets import JSONLDataset

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "tiny.jsonl"
        rows = [
            {"id": "q1", "question": "What is 2+2?", "golden_answers": "4"},
            {"id": "q2", "question": "Capital of France?", "answer": "Paris"},
            {"id": "q3", "question": "Boiling point of water (°C)?"},
        ]
        path.write_text("\n".join(json.dumps(r) for r in rows))
        ds = JSONLDataset(str(path))
        items = list(ds)

    ok = (
        len(ds) == 3
        and items[0]["id"] == "q1"
        and items[0]["question"] == "What is 2+2?"
        and items[0]["golden_answers"] == "4"
        # "answer" key normalized to "golden_answers":
        and items[1].get("golden_answers") == "Paris"
        and "golden_answers" not in items[2]
    )
    return ok, f"len={len(ds)} keys={list(items[0].keys())} q2.golden={items[1].get('golden_answers')}"


def probe_builtin_registry() -> tuple[bool, str]:
    from cabeza.datasets import builtin_specs, load
    from cabeza.datasets.registry import get_spec

    specs = builtin_specs()
    names = {s.name for s in specs}
    expected = {"bc", "bc_zh", "bcp", "ws", "hle", "dsqa", "gisa"}
    spec = get_spec("browsecomp")  # alias
    dsqa = get_spec("deepsearchqa")
    gisa = get_spec("gisa")
    gisa_ds = load("gisa", limit=1)
    ok = (
        names == expected
        and spec.name == "bc"
        and dsqa.name == "dsqa"
        and gisa.name == "gisa"
        and gisa.answer_format == "gisa"
        and getattr(getattr(gisa_ds, "spec", None), "answer_format", None) == "gisa"
    )
    return ok, (
        f"names={sorted(names)} alias_resolved={spec.name} "
        f"dsqa={dsqa.name} gisa_format={gisa.answer_format}"
    )


def probe_limit_param() -> tuple[bool, str]:
    """eval_num / limit parameter on JSONLDataset + load()."""
    from cabeza.datasets import JSONLDataset, load

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "many.jsonl"
        rows = [{"id": str(i), "question": f"q{i}?"} for i in range(20)]
        path.write_text("\n".join(json.dumps(r) for r in rows))

        full = JSONLDataset(str(path))
        five = JSONLDataset(str(path), limit=5)
        zero_is_full = JSONLDataset(str(path), limit=0)

    # ``load(..., limit=N)`` over a real built-in dataset:
    bc_full = load("bc")
    bc_three = load("bc", limit=3)
    dsqa_two = load("dsqa", limit=2)
    gisa_two = load("gisa", limit=2)

    ok = (
        len(full) == 20
        and len(five) == 5
        and len(zero_is_full) == 20
        and [x["id"] for x in five] == ["0", "1", "2", "3", "4"]
        and len(bc_full) > 3
        and len(bc_three) == 3
        and len(dsqa_two) == 2
        and len(gisa_two) == 2
        and "golden_answers" in dsqa_two[0]
        and "golden_answers" in gisa_two[0]
    )
    return ok, (
        f"full={len(full)} five={len(five)} ids={[x['id'] for x in five]} "
        f"bc_full={len(bc_full)} bc_three={len(bc_three)} "
        f"dsqa_two={len(dsqa_two)} gisa_two={len(gisa_two)}"
    )


def probe_user_registry() -> tuple[bool, str]:
    """Custom dataset registered via cabeza.datasets.register."""
    from cabeza.datasets import JSONLDataset, load, register

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "u.jsonl"
        path.write_text(json.dumps({"id": "x", "question": "hello?"}) + "\n")
        register("my_set", lambda path=None, _p=str(path): JSONLDataset(path or _p))
        ds = load("my_set")
        items = list(ds)

    ok = len(items) == 1 and items[0]["question"] == "hello?"
    return ok, f"len={len(items)} first={items[0]}"


# ---------------------------------------------------------------------------
# Runner: evaluate() + resume
# ---------------------------------------------------------------------------


def probe_evaluate_basic() -> tuple[bool, str]:
    """Two-item dataset, DeepSeek agent, no tools — verifies the runner pipeline."""
    if not DEEPSEEK_KEY:
        return False, "DEEPSEEK_API_KEY missing"
    from cabeza import Agent
    from cabeza.datasets import JSONLDataset
    from cabeza.runner import evaluate

    with tempfile.TemporaryDirectory() as d:
        ds_path = Path(d) / "ds.jsonl"
        rows = [
            {"id": "a", "question": "Reply with the single word 'one'.", "golden_answers": "one"},
            {"id": "b", "question": "Reply with the single word 'two'.", "golden_answers": "two"},
        ]
        ds_path.write_text("\n".join(json.dumps(r) for r in rows))
        ds = JSONLDataset(str(ds_path))

        agent = Agent(
            model=DEEPSEEK_MODEL,
            base_url=DEEPSEEK_URL,
            api_key=DEEPSEEK_KEY,
            family="deepseek",
            tools=[],
            max_steps=2,
            context_window_tokens=8_000,
            temperature=0.0,
            timeout=60.0,
            enable_thinking=False,
        )

        out_path = Path(d) / "results.jsonl"
        summary = evaluate(agent, ds, out=str(out_path), workers=2)

        # Read back the results:
        records = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]

    by_id = {r["id"]: r for r in records}
    ok = (
        summary["completed"] == 2
        and summary["skipped"] == 0
        and summary["total"] == 2
        and "one" in (by_id["a"]["prediction"] or "").lower()
        and "two" in (by_id["b"]["prediction"] or "").lower()
    )
    return ok, (
        f"summary={summary} "
        f"a={short(by_id['a']['prediction'], 30)} "
        f"b={short(by_id['b']['prediction'], 30)}"
    )


def probe_evaluate_resume() -> tuple[bool, str]:
    """Re-running evaluate with existing results skips completed items."""
    if not DEEPSEEK_KEY:
        return False, "DEEPSEEK_API_KEY missing"
    from cabeza import Agent
    from cabeza.datasets import JSONLDataset
    from cabeza.runner import evaluate

    with tempfile.TemporaryDirectory() as d:
        ds_path = Path(d) / "ds.jsonl"
        rows = [
            {"id": "x", "question": "Reply 'A'.", "golden_answers": "A"},
            {"id": "y", "question": "Reply 'B'.", "golden_answers": "B"},
        ]
        ds_path.write_text("\n".join(json.dumps(r) for r in rows))
        ds = JSONLDataset(str(ds_path))

        agent = Agent(
            model=DEEPSEEK_MODEL,
            base_url=DEEPSEEK_URL,
            api_key=DEEPSEEK_KEY,
            family="deepseek",
            tools=[],
            max_steps=2,
            temperature=0.0,
            timeout=60.0,
            enable_thinking=False,
        )

        out_path = Path(d) / "results.jsonl"
        # Pre-populate one already-completed record:
        out_path.write_text(json.dumps({"id": "x", "prediction": "A", "termination": "stub"}) + "\n")

        summary = evaluate(agent, ds, out=str(out_path), workers=1, resume=True)
        records = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]

    ok = (
        summary["skipped"] == 1
        and summary["completed"] == 1
        and len(records) == 2
        and any(r["id"] == "x" and r["termination"] == "stub" for r in records)
    )
    return ok, f"summary={summary} records_ids={[r.get('id') for r in records]}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_cli(args: list[str], *, env_extra: dict | None = None) -> tuple[int, str, str]:
    cmd = [sys.executable, "-m", "cabeza.cli", *args]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SRC}{os.pathsep}{env.get('PYTHONPATH', '')}"
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    return proc.returncode, proc.stdout, proc.stderr


def probe_cli_list() -> tuple[bool, str]:
    code, out, err = _run_cli(["list"])
    ok = (
        code == 0
        and "context strategies" in out
        and "team modes" in out
        and "datasets" in out
        and "page_memory" in out
    )
    return ok, f"code={code} stdout_lines={len(out.splitlines())} stderr={short(err, 60)}"


def probe_cli_run() -> tuple[bool, str]:
    if not DEEPSEEK_KEY:
        return False, "DEEPSEEK_API_KEY missing"
    code, out, err = _run_cli(
        [
            "run",
            "Reply with the single word 'pong'.",
            "--model", DEEPSEEK_MODEL,
            "--base-url", DEEPSEEK_URL,
            "--api-key", DEEPSEEK_KEY,
            "--family", "deepseek",
            "--max-steps", "2",
            "--context-window-tokens", "8000",
            "--timeout", "60",
        ]
    )
    ok = code == 0 and "pong" in out.lower() and "prediction" in out
    return ok, f"code={code} stdout={short(out, 120)} stderr={short(err, 80)}"


def probe_cli_eval() -> tuple[bool, str]:
    if not DEEPSEEK_KEY:
        return False, "DEEPSEEK_API_KEY missing"
    with tempfile.TemporaryDirectory() as d:
        ds_path = Path(d) / "ds.jsonl"
        ds_path.write_text(
            json.dumps({"id": "1", "question": "Reply 'cli-pong'.", "golden_answers": "cli-pong"}) + "\n"
        )
        # Register a temp dataset via the public registry, then drive CLI by --dataset-path:
        out_path = Path(d) / "results.jsonl"
        code, out, err = _run_cli(
            [
                "eval",
                "--dataset", "bc",  # name doesn't matter; we override path
                "--dataset-path", str(ds_path),
                "--out", str(out_path),
                "--workers", "1",
                "--model", DEEPSEEK_MODEL,
                "--base-url", DEEPSEEK_URL,
                "--api-key", DEEPSEEK_KEY,
                "--family", "deepseek",
                "--max-steps", "2",
                "--context-window-tokens", "8000",
                "--timeout", "60",
            ]
        )
        records = (
            [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
            if out_path.exists()
            else []
        )

    ok = code == 0 and len(records) == 1 and "cli-pong" in (records[0].get("prediction") or "").lower()
    return ok, (
        f"code={code} records={len(records)} "
        f"first_pred={short((records[0].get('prediction') if records else ''), 60)} "
        f"stderr={short(err, 80)}"
    )


PROBES = {
    "jsonl_loader": probe_jsonl_loader,
    "builtin_registry": probe_builtin_registry,
    "limit_param": probe_limit_param,
    "user_registry": probe_user_registry,
    "evaluate_basic": probe_evaluate_basic,
    "evaluate_resume": probe_evaluate_resume,
    "cli_list": probe_cli_list,
    "cli_run": probe_cli_run,
    "cli_eval": probe_cli_eval,
}


def main(argv: list[str]) -> int:
    return run_probes(PROBES, argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
