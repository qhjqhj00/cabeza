#!/usr/bin/env python3
"""Official GISA evaluator adapted to cabeza prediction files.

This is a local port of `RUC-NLPIR/GISA/eval_script/run_evaluation.py`.
The scoring logic is deterministic and intentionally mirrors the official
implementation:
- parse predictions as TSV from a fenced block, or from the raw response
- normalize numbers / strings with the official normalizer
- evaluate by GISA `answer_type`: item, set, list, table
- aggregate `overall_global_em` and per-type means

cabeza adaptations are limited to input/output plumbing: predictions can be a
JSONL/JSON result file instead of one `{qid}.json` file per question.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import io
import json
import math
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


from cabeza.datasets.registry import get_spec, resolve_default_path


def _default_gisa_eval_path() -> str:
    return resolve_default_path(get_spec("gisa")) or ""


def _default_gisa_question_path() -> str:
    eval_path = _default_gisa_eval_path()
    if not eval_path:
        return ""
    return str(Path(eval_path).with_name("encrypted_question.jsonl"))


def _default_gisa_gt_dir() -> str:
    eval_path = _default_gisa_eval_path()
    if not eval_path:
        return ""
    return str(Path(eval_path).parent / "answer")


DEFAULT_EVAL_DATA_PATH = _default_gisa_question_path()
DEFAULT_CONVERTED_EVAL_DATA_PATH = _default_gisa_eval_path()
DEFAULT_GT_DIR = _default_gisa_gt_dir()


@dataclass
class GISAQuestion:
    id: str
    answer_type: str
    question: str = ""
    question_type: str = ""
    topic: str = ""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_id(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def sort_key_id(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def last_assistant_message(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages or []):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            return content if isinstance(content, str) else str(content)
    return ""


def prediction_text_from_row(row: dict[str, Any]) -> str:
    for key in ("response", "prediction", "model_answer", "final_answer"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    messages = row.get("messages")
    if isinstance(messages, list):
        return last_assistant_message(messages)
    return ""


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if path.is_dir():
        predictions: dict[str, dict[str, Any]] = {}
        for file in sorted(path.iterdir()):
            if file.name.startswith("_"):
                continue
            if file.suffix.lower() == ".json":
                payload = json.loads(file.read_text(encoding="utf-8"))
                qid = file.stem
                response = prediction_text_from_row(payload)
                predictions[qid] = {"id": qid, "prediction": response, "raw": payload}
            elif file.suffix.lower() == ".csv":
                predictions[file.stem] = {"id": file.stem, "prediction": str(file), "raw": {}}
        return predictions

    if path.suffix.lower() == ".jsonl":
        predictions = {}
        for row in read_jsonl(path):
            qid = normalize_id(row.get("id") or row.get("instance_id"))
            if not qid:
                continue
            predictions[qid] = {
                "id": qid,
                "prediction": prediction_text_from_row(row),
                "raw": row,
            }
        return predictions

    payload = json.loads(path.read_text(encoding="utf-8"))
    predictions = {}
    if isinstance(payload, list):
        for row in payload:
            qid = normalize_id(row.get("id") or row.get("instance_id"))
            if qid:
                predictions[qid] = {
                    "id": qid,
                    "prediction": prediction_text_from_row(row),
                    "raw": row,
                }
        return predictions

    if isinstance(payload, dict):
        for key, value in payload.items():
            qid = normalize_id(key)
            if isinstance(value, dict):
                response = prediction_text_from_row(value)
                raw = value
            else:
                response = str(value)
                raw = {"prediction": response}
            predictions[qid] = {"id": qid, "prediction": response, "raw": raw}
        return predictions

    raise ValueError(f"Unsupported prediction payload type: {type(payload).__name__}")


def load_questions(eval_data_path: Path, converted_eval_path: Path | None = None) -> dict[str, GISAQuestion]:
    metadata: dict[str, dict[str, Any]] = {}
    if eval_data_path.exists():
        for row in read_jsonl(eval_data_path):
            qid = normalize_id(row.get("id"))
            if qid:
                metadata[qid] = row

    questions: dict[str, GISAQuestion] = {}
    if metadata:
        for qid, row in metadata.items():
            questions[qid] = GISAQuestion(
                id=qid,
                answer_type=str(row.get("answer_type", "table") or "table"),
                question=str(row.get("question", "")),
                question_type=str(row.get("question_type", "")),
                topic=str(row.get("topic", "")),
            )

    if converted_eval_path and converted_eval_path.exists():
        for row in read_jsonl(converted_eval_path):
            qid = normalize_id(row.get("id"))
            if not qid:
                continue
            existing = questions.get(qid)
            questions[qid] = GISAQuestion(
                id=qid,
                answer_type=existing.answer_type if existing else str(row.get("answer_type", "table") or "table"),
                question=str(row.get("question", "")) or (existing.question if existing else ""),
                question_type=existing.question_type if existing else "",
                topic=existing.topic if existing else "",
            )
    return dict(sorted(questions.items(), key=lambda item: sort_key_id(item[0])))


class SimpleEvaluator:
    def _normalize_val(self, val: str | int | float) -> str:
        val_str = str(val).strip()
        if not val_str or val_str.lower() in ["nan", "none", "null"]:
            return ""

        clean_num = val_str.replace(",", "").replace("$", "")
        is_percent = False
        if clean_num.endswith("%"):
            is_percent = True
            clean_num = clean_num[:-1]

        try:
            f_val = float(clean_num)
            if is_percent:
                f_val /= 100.0

            if f_val.is_integer():
                return str(int(f_val))
            formatted = "{:.6f}".format(f_val).rstrip("0").rstrip(".")
            return formatted if formatted else "0"
        except ValueError:
            pass

        return val_str.lower().replace(" ", "").replace("*", "").replace("\n", "")

    def _extract_model_output(self, model_output: str) -> Optional[pd.DataFrame]:
        pattern = r"```(?:tsv)?\s*(.*?)```"
        match = re.search(pattern, model_output, re.DOTALL)

        raw_content = match.group(1) if match else model_output
        try:
            raw_content = "\n".join(line for line in raw_content.split("\n") if line.strip())
            if not raw_content:
                return None
            output = pd.read_csv(io.StringIO(raw_content), sep="\t")
            output.columns = [str(col).strip().lower().replace(" ", "") for col in output.columns]
            output = output.map(self._normalize_val)
        except Exception:
            output = None
        return output

    def load_ground_truth(self, file_path: str, question_type: str = "table") -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"GT file not found: {file_path}")
        header = None if question_type != "table" else "infer"

        try:
            df = pd.read_csv(file_path, header=header)
        except Exception as exc:
            if "codec" in str(exc):
                df = pd.read_csv(file_path, header=header, encoding="gbk")
            else:
                raise

        df.columns = [str(col).strip().lower().replace(" ", "") for col in df.columns]
        df = df.map(self._normalize_val)
        return df

    def _calculate_f1(self, tp: int, n_pred: int, n_gt: int) -> tuple[float, float, float]:
        precision = tp / n_pred if n_pred > 0 else 0.0
        recall = tp / n_gt if n_gt > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return precision, recall, f1

    def flatten_table(self, df: pd.DataFrame) -> list[tuple[str, str]]:
        items = []
        for col in df.columns:
            values = df[col]
            for val in values:
                items.append((col, val))
        return items

    def evaluate_item(self, pred_df: pd.DataFrame | None, gt_df: pd.DataFrame) -> dict[str, float]:
        if pred_df is None or pred_df.empty:
            return {"item_em": 0}
        pred_item = "".join(pred_df.iloc[0, :].tolist())
        gt_item = "".join(gt_df.iloc[0, :].tolist())
        return {"item_em": 1 if pred_item == gt_item else 0}

    def evaluate_set(self, pred_df: pd.DataFrame | None, gt_df: pd.DataFrame) -> dict[str, float]:
        if pred_df is None or pred_df.empty:
            return {"set_precision": 0.0, "set_recall": 0.0, "set_f1": 0.0}

        pred_set = set(pred_df.iloc[:, -1].tolist())
        gt_set = set(gt_df.iloc[:, -1].tolist())
        tp = len(pred_set.intersection(gt_set))
        precision, recall, f1 = self._calculate_f1(tp, len(pred_set), len(gt_set))
        return {"set_precision": precision, "set_recall": recall, "set_f1": f1}

    def evaluate_list(self, pred_df: pd.DataFrame | None, gt_df: pd.DataFrame) -> dict[str, float]:
        if pred_df is None or pred_df.empty:
            return {"list_content_f1": 0.0, "list_order_score": 0.0}

        pred_list = pred_df.iloc[:, -1].tolist()
        gt_list = gt_df.iloc[:, -1].tolist()

        gt_counter = Counter(gt_list)
        pred_counter = Counter(pred_list)
        num_common = sum((gt_counter & pred_counter).values())

        precision = num_common / len(pred_list) if pred_list else 0.0
        recall = num_common / len(gt_list) if gt_list else 0.0
        content_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        order_score = difflib.SequenceMatcher(None, gt_list, pred_list).ratio()
        return {"list_content_f1": round(content_f1, 4), "list_order_score": round(order_score, 4)}

    def evaluate_table(self, pred_df: pd.DataFrame | None, gt_df: pd.DataFrame) -> dict[str, float]:
        default = {
            "table_row_f1": 0.0,
            "table_row_precision": 0.0,
            "table_row_recall": 0.0,
            "table_item_f1": 0.0,
            "table_item_precision": 0.0,
            "table_item_recall": 0.0,
        }
        if pred_df is None or pred_df.empty:
            return default.copy()

        common_cols = [col for col in gt_df.columns if col in pred_df.columns]
        if not common_cols:
            row_precision, row_recall, row_f1 = 0.0, 0.0, 0.0
        else:
            pred_rows = set(tuple(row) for row in pred_df[common_cols].fillna("__NAN__").astype(str).to_numpy())
            gt_rows = set(tuple(row) for row in gt_df[common_cols].fillna("__NAN__").astype(str).to_numpy())
            tp_rows = len(pred_rows.intersection(gt_rows))
            row_precision, row_recall, row_f1 = self._calculate_f1(tp_rows, len(pred_rows), len(gt_rows))

        pred_counter = Counter(self.flatten_table(pred_df))
        gt_counter = Counter(self.flatten_table(gt_df))
        tp_items = sum((pred_counter & gt_counter).values())

        n_pred_items = sum(pred_counter.values())
        n_gt_items = sum(gt_counter.values())
        item_precision, item_recall, item_f1 = self._calculate_f1(tp_items, n_pred_items, n_gt_items)

        return {
            "table_row_f1": row_f1,
            "table_row_precision": row_precision,
            "table_row_recall": row_recall,
            "table_item_f1": item_f1,
            "table_item_precision": item_precision,
            "table_item_recall": item_recall,
        }

    def evaluate_one(self, prediction: str, gt_path: str, question_type: str, qid: str | None = None) -> dict[str, Any]:
        if prediction.endswith(".csv"):
            pred_df = self.load_ground_truth(prediction, question_type=question_type.lower())
        else:
            pred_df = self._extract_model_output(prediction)

        gt_df = self.load_ground_truth(gt_path, question_type=question_type.lower())

        q_type = question_type.lower()
        if q_type == "item":
            metrics = self.evaluate_item(pred_df, gt_df)
        elif q_type == "set":
            metrics = self.evaluate_set(pred_df, gt_df)
        elif q_type == "list":
            metrics = self.evaluate_list(pred_df, gt_df)
        elif q_type == "table":
            metrics = self.evaluate_table(pred_df, gt_df)
        else:
            metrics = self.evaluate_item(pred_df, gt_df)

        if pred_df is not None:
            if q_type != "set":
                metrics["global_em"] = int(np.array_equal(pred_df.to_numpy(), gt_df.to_numpy()))
            else:
                pred_set = set(pred_df.iloc[:, 0].tolist())
                gt_set = set(gt_df.iloc[:, 0].tolist())
                metrics["global_em"] = int(pred_set == gt_set)
        else:
            metrics["global_em"] = 0

        metrics["question_type"] = question_type
        return metrics

    def gather_results(self, score_list: list[dict[str, Any]]) -> dict[str, Any]:
        df = pd.DataFrame(score_list)
        overall_em = float(df["global_em"].mean()) if not df.empty else 0.0
        type_report = df.groupby("question_type").mean(numeric_only=True).round(4)
        detail_score_dict = type_report.to_dict(orient="index")
        count_by_type = df["question_type"].value_counts().to_dict() if not df.empty else {}

        summary: dict[str, Any] = {"overall_global_em": overall_em}
        for question_type in count_by_type:
            type_result = {
                "num_samples": int(count_by_type[question_type]),
                **{
                    f"overall_{key}": round(float(value), 4)
                    for key, value in detail_score_dict[question_type].items()
                    if not pd.isna(value)
                },
            }
            summary[str(question_type)] = type_result
        return summary


def evaluate_predictions(
    questions: dict[str, GISAQuestion],
    predictions: dict[str, dict[str, Any]],
    gt_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evaluator = SimpleEvaluator()
    scored_rows: list[dict[str, Any]] = []
    metrics_list: list[dict[str, Any]] = []

    for qid, question in questions.items():
        prediction_item = predictions.get(qid)
        prediction = prediction_item.get("prediction", "") if prediction_item else ""
        gt_path = gt_dir / f"{qid}.csv"
        row: dict[str, Any] = {
            "id": qid,
            "question": question.question,
            "answer_type": question.answer_type,
            "question_type": question.question_type,
            "topic": question.topic,
            "prediction": prediction,
        }
        if not prediction_item:
            metrics = evaluator.evaluate_one("", str(gt_path), question.answer_type, qid=qid)
            row["msg"] = "missing prediction"
        else:
            raw = prediction_item.get("raw") or {}
            if raw.get("error"):
                row["msg"] = f"inference error: {raw['error']}"
            metrics = evaluator.evaluate_one(prediction, str(gt_path), question.answer_type, qid=qid)

        row.update(metrics)
        scored_rows.append(row)
        metrics_list.append(metrics)

    summary = evaluator.gather_results(metrics_list)
    return scored_rows, summary


def derive_output_paths(input_path: Path, args: argparse.Namespace) -> tuple[Path, Path]:
    stem = input_path.name if input_path.is_dir() else input_path.stem
    scored_output = (
        Path(args.scored_output)
        if args.scored_output
        else input_path.with_name(f"{stem}__gisa_scored.jsonl")
    )
    summary_output = (
        Path(args.summary_output)
        if args.summary_output
        else input_path.with_name(f"{stem}__gisa_summary.json")
    )
    return scored_output, summary_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GISA predictions with the official deterministic scorer.")
    parser.add_argument("--input_file", required=True, help="Prediction JSONL/JSON file or official per-qid directory.")
    parser.add_argument(
        "--eval_data_path",
        default=DEFAULT_EVAL_DATA_PATH,
        help="GISA question JSONL with answer_type metadata.",
    )
    parser.add_argument(
        "--converted_eval_data_path",
        default=DEFAULT_CONVERTED_EVAL_DATA_PATH,
        help="Optional decrypted cabeza JSONL used only to include question text in outputs.",
    )
    parser.add_argument("--gt_dir_path", "--gold_dir", default=DEFAULT_GT_DIR, help="Directory of answer/{qid}.csv files.")
    parser.add_argument("--scored_output", default="", help="Optional scored JSONL output path.")
    parser.add_argument("--summary_output", default="", help="Optional summary JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    questions = load_questions(
        Path(args.eval_data_path),
        converted_eval_path=Path(args.converted_eval_data_path) if args.converted_eval_data_path else None,
    )
    predictions = load_predictions(input_path)
    scored_rows, metrics_summary = evaluate_predictions(
        questions=questions,
        predictions=predictions,
        gt_dir=Path(args.gt_dir_path),
    )

    scored_output, summary_output = derive_output_paths(input_path, args)
    write_jsonl(scored_output, scored_rows)

    available_predictions = sum(1 for qid in questions if qid in predictions)
    summary = {
        "benchmark": "gisa",
        "input_file": str(input_path),
        "eval_data_path": str(Path(args.eval_data_path)),
        "gt_dir_path": str(Path(args.gt_dir_path)),
        "total_questions": len(questions),
        "available_predictions": available_predictions,
        "missing_predictions": len(questions) - available_predictions,
        "scored_output": str(scored_output),
        **metrics_summary,
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Input       : {input_path}")
    print(f"Eval data   : {Path(args.eval_data_path)}")
    print(f"Gold dir    : {Path(args.gt_dir_path)}")
    print(f"Scored rows : {scored_output}")
    print(f"Summary     : {summary_output}")
    print(f"Global EM   : {summary['overall_global_em']:.4f}")


if __name__ == "__main__":
    main()
