"""Live smoke tests for cabeza.eval (LLM-as-a-judge accuracy scoring).

Probes:
    bc_two_items   — synthetic 2-row prediction file scored against the
                     bundled bc eval.jsonl; checks summary JSON has a
                     pass-rate field and at least one ``correct=yes`` row.
    cli_score      — same call via the ``cabeza score`` CLI subcommand.
    gisa_perfect   — deterministic GISA scorer over predictions that point
                     directly at the bundled gold CSV files.

DeepSeek is used as the judge throughout.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _helpers import DEEPSEEK_KEY, DEEPSEEK_MODEL, DEEPSEEK_URL, ROOT, SRC, run_probes, short  # noqa: E402


def _write_predictions(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows))


def _read_summary(d: Path) -> dict | None:
    candidates = list(d.glob("*_summary.json")) + list(d.glob("summary*.json"))
    if not candidates:
        return None
    return json.loads(candidates[0].read_text())


def _bc_first_two() -> list[dict]:
    """Pull the first 2 items from cabeza/data/bc/eval.jsonl."""
    bc_path = ROOT / "data" / "bc" / "eval.jsonl"
    rows: list[dict] = []
    with bc_path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= 2:
                break
    return rows


# ---- probe: programmatic ---------------------------------------------------


def probe_bc_two_items() -> tuple[bool, str]:
    if not DEEPSEEK_KEY:
        return False, "DEEPSEEK_API_KEY missing"
    from cabeza.eval import score

    items = _bc_first_two()
    if len(items) < 2:
        return False, "bc dataset has <2 items"

    with tempfile.TemporaryDirectory() as d:
        dpath = Path(d)
        pred_path = dpath / "preds.jsonl"
        # First row: prediction matches the gold answer → judge should say yes.
        # Second row: prediction is a clearly wrong placeholder → judge should say no.
        preds = [
            {
                "id": items[0]["id"],
                "question": items[0]["question"],
                "prediction": str(items[0]["golden_answers"]),
                "response": str(items[0]["golden_answers"]),
            },
            {
                "id": items[1]["id"],
                "question": items[1]["question"],
                "prediction": "definitely-not-the-answer",
                "response": "definitely-not-the-answer",
            },
        ]
        _write_predictions(pred_path, preds)

        summary_out = dpath / "summary.json"
        scored_out = dpath / "scored.jsonl"
        score(
            benchmark="bc",
            input_files=[str(pred_path)],
            judge_model=DEEPSEEK_MODEL,
            judge_base_url=DEEPSEEK_URL,
            judge_api_key=DEEPSEEK_KEY,
            extra_argv=[
                "--scored_output", str(scored_out),
                "--summary_output", str(summary_out),
                "--num_workers", "2",
                "--request_timeout_s", "60",
                "--judge_attempts", "2",
            ],
        )

        if not summary_out.exists() or not scored_out.exists():
            return False, "summary or scored file not produced"

        summary = json.loads(summary_out.read_text())
        scored = [json.loads(line) for line in scored_out.read_text().splitlines() if line.strip()]

    # Scored JSONL covers the full eval set (1266 bc items); the rows we care
    # about are the ones whose id matches what we supplied predictions for.
    pred_ids = {p["id"] for p in preds}
    judged = [r for r in scored if r.get("id") in pred_ids and r.get("judge_mode")]
    truthy_by_id = {
        r["id"]: bool(r.get("correct")) for r in judged
    }
    judged_predictions = int(summary.get("judged_predictions") or 0)
    accuracy = summary.get("accuracy")
    ok = (
        len(judged) == 2
        and judged_predictions == 2
        and truthy_by_id.get(preds[0]["id"]) is True   # matching answer → correct
        and truthy_by_id.get(preds[1]["id"]) is False  # placeholder → incorrect
        and accuracy is not None
    )
    return ok, (
        f"judged={len(judged)}/{judged_predictions} "
        f"correct_by_id={truthy_by_id} accuracy={accuracy}"
    )


# ---- probe: CLI ------------------------------------------------------------


def probe_cli_score() -> tuple[bool, str]:
    if not DEEPSEEK_KEY:
        return False, "DEEPSEEK_API_KEY missing"

    items = _bc_first_two()
    with tempfile.TemporaryDirectory() as d:
        dpath = Path(d)
        pred_path = dpath / "preds.jsonl"
        preds = [
            {
                "id": items[0]["id"],
                "question": items[0]["question"],
                "prediction": str(items[0]["golden_answers"]),
            },
        ]
        _write_predictions(pred_path, preds)
        summary_out = dpath / "summary.json"

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{SRC}{os.pathsep}{env.get('PYTHONPATH', '')}"
        cmd = [
            sys.executable, "-m", "cabeza.cli", "score",
            "--benchmark", "bc",
            "--input_file", str(pred_path),
            "--summary_output", str(summary_out),
            "--judge_model", DEEPSEEK_MODEL,
            "--judge_base_url", DEEPSEEK_URL,
            "--judge_api_key", DEEPSEEK_KEY,
            "--num_workers", "2",
            "--request_timeout_s", "60",
            "--judge_attempts", "2",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=180)
        # Read the summary while the tempdir is still alive.
        summary = json.loads(summary_out.read_text()) if summary_out.exists() else {}

    ok_code = proc.returncode == 0
    ok = ok_code and bool(summary) and any(k in summary for k in ("accuracy", "pass_rate"))
    return ok, (
        f"code={proc.returncode} summary_present={bool(summary)} "
        f"keys={sorted(summary)[:5]} stderr_tail={short(proc.stderr.splitlines()[-1] if proc.stderr else '', 60)}"
    )


def probe_gisa_perfect() -> tuple[bool, str]:
    from cabeza.eval import score

    gisa_question_path = ROOT / "data" / "gisa" / "encrypted_question.jsonl"
    answer_dir = ROOT / "data" / "gisa" / "answer"
    with tempfile.TemporaryDirectory() as d:
        dpath = Path(d)
        pred_path = dpath / "gisa_preds.jsonl"
        scored_out = dpath / "gisa_scored.jsonl"
        summary_out = dpath / "gisa_summary.json"

        rows: list[dict] = []
        with gisa_question_path.open(encoding="utf-8-sig") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                qid = str(obj["id"])
                rows.append({"id": qid, "prediction": str(answer_dir / f"{qid}.csv")})
        _write_predictions(pred_path, rows)

        score(
            benchmark="gisa",
            input_files=[str(pred_path)],
            extra_argv=[
                "--scored_output", str(scored_out),
                "--summary_output", str(summary_out),
            ],
        )
        summary = json.loads(summary_out.read_text()) if summary_out.exists() else {}

    ok = (
        summary.get("total_questions") == 373
        and summary.get("available_predictions") == 373
        and float(summary.get("overall_global_em", 0.0)) == 1.0
        and summary.get("table", {}).get("num_samples") == 253
    )
    return ok, (
        f"total={summary.get('total_questions')} available={summary.get('available_predictions')} "
        f"global_em={summary.get('overall_global_em')} table={summary.get('table', {}).get('num_samples')}"
    )


PROBES = {
    "bc_two_items": probe_bc_two_items,
    "cli_score": probe_cli_score,
    "gisa_perfect": probe_gisa_perfect,
}


def main(argv: list[str]) -> int:
    selected = argv or list(PROBES)
    needs_judge = {"bc_two_items", "cli_score"}
    if not DEEPSEEK_KEY and any(name in needs_judge for name in selected):
        print("[!] DEEPSEEK_API_KEY required.")
        return 2
    return run_probes(PROBES, argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
