#!/usr/bin/env python3
"""Official-style DeepSearchQA evaluator for cabeza.

This ports the public DeepSearchQA starter-code evaluation flow to the local
cabeza result format:
- official LLM-as-judge prompt
- official JSON judgement shape
- per-answer correctness details and excessive-answer detection
- official precision / recall / F1 and fully-correct style aggregation

The official starter notebook uses Gemini 2.5 Flash directly. This local port
keeps the same prompt and scoring logic, but calls an OpenAI-compatible judge
endpoint so it matches the rest of cabeza's evaluator CLIs.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import random
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio


from cabeza.datasets.registry import get_spec, resolve_default_path


def _default_dsqa_eval_path() -> str:
    return resolve_default_path(get_spec("dsqa")) or ""


def _default_dsqa_metadata_csv_path() -> str:
    eval_path = _default_dsqa_eval_path()
    if not eval_path:
        return ""
    return str(Path(eval_path).with_name("DSQA-full.csv"))


DEFAULT_EVAL_DATA_PATH = _default_dsqa_eval_path()
DEFAULT_METADATA_CSV_PATH = _default_dsqa_metadata_csv_path()


DEEPSEARCH_QA_PROMPT = """\
Your task is to evaluate whether a given "AI Response" for a specific "User Prompt" arrived at the correct answer.

**Answer Correctness Task**

*   **Purpose:** Assess whether the AI response provides the correct answer(s) based on the provided "Correct Answer" and "Prompt Type".
*   **Process:**
    *   Identify the "Prompt Type": "<prompt_type>".
    *   Refer to the "Correct Answer": "<answer>".
    *   Based on the "Prompt Type", determine if the "AI Response" contains the expected answer(s).
        *   **'Single Answer'**: Check if the response provides the answer that addresses the user's question. It does not have to match the exact wording of the provided answer.
        *   **'Set Answer'**: Check if the response includes *each* item from the provided ground truth answers. The order might not matter unless specified otherwise. The response might include more answers than the list. Determine the correctness *only* based on the list first and then check if the response includes answers not in the list.
    *   **Explanation:** Provide a brief explanation justifying your assessment of answer correctness, referencing specific parts of the AI response and the correct answer.
    *   **Correctness Details:** Provide a dictionary, one key for each expected answer part, and value is a boolean indicating whether each expected answer part was found.
        *   For 'Set Answer', this will be a list of attributes, one for each item/part in the "Correct Answer". Each key will be a string indicating the expected answer part, and the value will be a boolean indicating whether that part was found in the response.
    *   **Excessive Answers:** Provide a list of strings, each indicating an excessive answer part. If the response provides answers that are **not** in the "Correct Answer" list, add these answers as excessive answers. Return an empty list when there's no excessive answers in the response.


**Output Format:**

Your evaluation *must* be structured as a nested JSON dictionary with the following top-level keys: `"Answer Correctness"`. Please return NULL if any of "Prompt", "AI Response" or "Correct Answer" is empty.
The value for `"Answer Correctness"` should be a dictionary containing `"Explanation"` (a string), `"Correctness Details"` (a dictionary where each key is the expected correct answer, and the value is a boolean indicating whether the response contains the correct answer), and `"Excessive Answers"` (a list of strings indicating the excessive answers).

Make sure you return a valid JSON string. Pay special attention to quotes, commas and special characters in the JSON string. Make sure to escape all special characters and quotes in the JSON string.

"""


GRADER_RATING_OUTPUT_EXAMPLE = r"""**Example (Partial):**

"```json
{{
  "Answer Correctness": {{
    "Explanation": "The response correctly identified Belgium and France but also includes an excessive answer, Italy.",
    "Correctness Details": {{
      "Belgium": true,
      "France": true,
    }},
    "Excessive Answers": [ "Italy" ]
  }}
}}
```"

**Now, proceed with the evaluation using the provided User Prompt, AI Response, and Correct Answer.**

User Prompt (Wrapped in <prompt> and </prompt>):
<prompt>
{prompt}
</prompt>
--------------------
**  Correct Answer (Wrapped in <answer> and </answer>):
Prompt Type: {prompt_type}
<answer>
{answer}
</answer>
--------------------
AI assistant response (Wrapped in <response> and </response>):
<response>
{response}
</response>

--------------------
Rating:"""


@dataclass
class DSQAQuestion:
    id: str
    problem: str
    answer: str
    answer_type: str = "Single Answer"
    problem_category: str = ""


@dataclass
class ItemRating:
    original_index: int | None = None
    example_id: str = ""
    query: str = ""
    response: str = ""
    category_type: str | None = None
    expected_correct_answer: str | None = None
    answer_correctness_explanation: str | None = None
    expected_correct_answer_list: list[str] | None = None
    response_wrong_answers_list: list[str] | None = None
    grader_ratings_list: list[bool] | None = None
    empty_model_response: bool = False
    empty_auto_rater_response: bool = False
    invalid_auto_rater_response: bool = False
    rating_response: str = ""
    rating_prompt: str = ""
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProjectRating:
    num_total_ratings: int = 0
    num_empty_model_response: int = 0
    num_invalid_auto_rater_response: int = 0
    num_empty_auto_rater_response: int = 0
    num_valid_ratings: int = 0
    num_answer_correctness_evaluated: int = 0
    pct_w_ci_all_answers_correct: str = ""
    pct_w_ci_fully_incorrect_items: str = ""
    pct_w_ci_correct_with_excessive_answers: str = ""
    pct_empty_model_response: float = 0.0
    pct_invalid_auto_rater_response: float = 0.0
    pct_empty_auto_rater_response: float = 0.0
    precision: str = ""
    recall: str = ""
    f1_score: str = ""
    fully_correct_count: int = 0
    fully_incorrect_count: int = 0
    correct_with_excessive_count: int = 0
    category_breakdown: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sanitize_tag(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return cleaned or "default"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
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


def last_assistant_message(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages or []):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            return content if isinstance(content, str) else str(content)
    return ""


def response_from_prediction_row(row: dict[str, Any]) -> str:
    for key in ("response", "prediction", "model_answer", "final_answer"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    messages = row.get("messages")
    if isinstance(messages, list):
        return last_assistant_message(messages)
    return ""


def compact_prediction_row(raw: dict[str, Any], fallback_id: str = "") -> dict[str, Any]:
    row = dict(raw)
    unique_id = normalize_id(row.get("id") or row.get("instance_id") or fallback_id)
    response = response_from_prediction_row(row)
    compact = {
        "id": unique_id,
        "question": row.get("question", ""),
        "answer": row.get("answer", row.get("golden_answers", "")),
        "prediction": row.get("prediction", response),
        "response": response,
    }
    for key in (
        "termination",
        "token_count",
        "total_process_time",
        "total_rounds",
        "tool_call_rounds",
        "error",
    ):
        if key in row:
            compact[key] = row[key]
    return compact


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        predictions: dict[str, dict[str, Any]] = {}
        for row in read_jsonl(path):
            compact = compact_prediction_row(row)
            if compact["id"]:
                predictions[compact["id"]] = compact
        return predictions

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    predictions: dict[str, dict[str, Any]] = {}
    if isinstance(payload, list):
        for row in payload:
            compact = compact_prediction_row(row)
            if compact["id"]:
                predictions[compact["id"]] = compact
        return predictions

    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, dict):
                compact = compact_prediction_row(value, fallback_id=str(key))
            else:
                compact = compact_prediction_row(
                    {"id": str(key), "response": str(value), "prediction": str(value)},
                    fallback_id=str(key),
                )
            if compact["id"]:
                predictions[compact["id"]] = compact
        return predictions

    raise ValueError(f"Unsupported prediction payload type: {type(payload).__name__}")


def load_metadata_csv(path: Path, id_base: int) -> dict[str, DSQAQuestion]:
    metadata: dict[str, DSQAQuestion] = {}
    if not path.exists():
        return metadata
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            qid = str(idx + id_base)
            metadata[qid] = DSQAQuestion(
                id=qid,
                problem=str(row.get("problem", "")),
                answer=str(row.get("answer", "")),
                answer_type=str(row.get("answer_type", "Single Answer") or "Single Answer"),
                problem_category=str(row.get("problem_category", "")),
            )
    return metadata


def load_questions(eval_data_path: Path, metadata_csv_path: Path, id_base: int) -> list[DSQAQuestion]:
    metadata = load_metadata_csv(metadata_csv_path, id_base=id_base)

    if eval_data_path.suffix.lower() == ".csv":
        questions = list(load_metadata_csv(eval_data_path, id_base=id_base).values())
        questions.sort(key=lambda item: int(item.id) if item.id.isdigit() else item.id)
        return questions

    rows = read_jsonl(eval_data_path)
    questions: list[DSQAQuestion] = []
    for row in rows:
        qid = normalize_id(row.get("id") or row.get("instance_id"))
        if not qid:
            continue
        meta = metadata.get(qid)
        questions.append(
            DSQAQuestion(
                id=qid,
                problem=str(row.get("problem") or row.get("question") or (meta.problem if meta else "")),
                answer=str(row.get("answer") or row.get("golden_answers") or (meta.answer if meta else "")),
                answer_type=str(row.get("answer_type") or (meta.answer_type if meta else "Single Answer")),
                problem_category=str(row.get("problem_category") or (meta.problem_category if meta else "")),
            )
        )
    return questions


def build_grader_prompt(question: DSQAQuestion, response: str) -> str:
    return DEEPSEARCH_QA_PROMPT + GRADER_RATING_OUTPUT_EXAMPLE.format(
        prompt=question.problem.strip(),
        prompt_type=question.answer_type.strip(),
        answer=question.answer.strip(),
        response=response.strip(),
    )


def parse_json_response(text: str) -> Any:
    json_str = text.strip()
    start_marker = "```json"
    start_idx = json_str.find(start_marker)
    if start_idx != -1:
        json_str = json_str[start_idx + len(start_marker) :].strip()
        end_idx = json_str.rfind("```")
        if end_idx != -1:
            json_str = json_str[:end_idx].strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for idx, char in enumerate(json_str):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(json_str[idx:])
        except json.JSONDecodeError:
            continue
        return payload
    return None


def get_answer_correctness_details(json_response: Any) -> dict[str, bool] | None:
    try:
        details = json_response["Answer Correctness"]["Correctness Details"]
    except (KeyError, TypeError):
        return None
    if not isinstance(details, dict):
        return None
    if not all(isinstance(key, str) for key in details.keys()):
        return None
    if not all(isinstance(value, bool) for value in details.values()):
        return None
    return details


def get_excessive_answers(json_response: Any) -> list[str] | None:
    try:
        excessive_answers = json_response["Answer Correctness"]["Excessive Answers"]
    except KeyError:
        return []
    except TypeError:
        return None
    if not isinstance(excessive_answers, list):
        return None
    if not all(isinstance(item, str) for item in excessive_answers):
        return None
    return excessive_answers


def reduce_llm_response_to_item_rating(
    item_rating: ItemRating,
    grader_llm_response_text: str,
    grader_llm_prompt_text: str,
) -> ItemRating:
    item_rating.rating_prompt = grader_llm_prompt_text
    item_rating.rating_response = grader_llm_response_text

    if not item_rating.response:
        item_rating.empty_model_response = True
        item_rating.error_message = "AI response was empty."
        return item_rating

    if not grader_llm_response_text:
        item_rating.empty_auto_rater_response = True
        item_rating.error_message = "Auto-rater response was empty."
        return item_rating

    parsed_json_response = parse_json_response(grader_llm_response_text)
    if not parsed_json_response:
        item_rating.invalid_auto_rater_response = True
        item_rating.error_message = "Invalid JSON response from auto-rater."
        return item_rating

    answer_correctness_node = parsed_json_response.get("Answer Correctness")
    if not isinstance(answer_correctness_node, dict):
        item_rating.invalid_auto_rater_response = True
        item_rating.error_message = "Missing or malformed 'Answer Correctness' node."
        return item_rating

    explanation = answer_correctness_node.get("Explanation")
    if not isinstance(explanation, str):
        item_rating.invalid_auto_rater_response = True
        item_rating.error_message = "Missing or malformed 'Explanation' in Answer Correctness."
        return item_rating
    item_rating.answer_correctness_explanation = explanation

    details = get_answer_correctness_details(parsed_json_response)
    if details is None:
        item_rating.invalid_auto_rater_response = True
        item_rating.error_message = "Invalid 'Correctness Details' in Answer Correctness."
        return item_rating
    item_rating.expected_correct_answer_list = list(details.keys())
    item_rating.grader_ratings_list = list(details.values())

    excessive_answers = get_excessive_answers(parsed_json_response)
    if excessive_answers is None:
        item_rating.invalid_auto_rater_response = True
        item_rating.error_message = "Invalid 'Excessive Answers' in Answer Correctness."
        return item_rating
    if excessive_answers:
        item_rating.response_wrong_answers_list = excessive_answers

    return item_rating


def is_local_base_url(base_url: str) -> bool:
    if not base_url:
        return False
    parsed = urlparse(base_url)
    return (parsed.hostname or "").lower() in {"localhost", "127.0.0.1", "0.0.0.0"}


def resolve_client_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    base_url = args.judge_base_url.strip()
    api_key = args.judge_api_key.strip()
    api_dict_key = (args.api_dict_key or args.judge_model).strip()
    api_dict_path = Path(args.api_dict_path)

    if api_dict_path.exists():
        api_dict = json.loads(api_dict_path.read_text(encoding="utf-8"))
        if api_dict_key in api_dict:
            if not base_url:
                base_url = str(api_dict[api_dict_key].get("base_url", "")).strip()
            if not api_key:
                api_key = str(api_dict[api_dict_key].get("api_key", "")).strip()

    if not api_key and not base_url:
        raise ValueError(
            "Missing judge credentials. Pass --judge_api_key / --judge_base_url, "
            "or use --api_dict_key with --api_dict_path."
        )

    args.resolved_judge_base_url = base_url
    args.resolved_judge_api_key = api_key or "EMPTY"
    args.resolved_judge_is_local = is_local_base_url(base_url)

    kwargs: dict[str, Any] = {
        "api_key": args.resolved_judge_api_key,
        "timeout": args.request_timeout_s,
        "max_retries": args.max_retries,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return kwargs


def extract_message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str) and content.strip():
        return content
    reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
    if reasoning:
        return str(reasoning)
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
            if text:
                chunks.append(str(text))
        return "\n".join(chunks)
    return str(content or "")


def build_completion_kwargs(args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": args.judge_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    if args.resolved_judge_is_local:
        kwargs["max_tokens"] = args.max_completion_tokens
    else:
        kwargs["max_completion_tokens"] = args.max_completion_tokens
    if args.disable_thinking or args.resolved_judge_is_local:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    return kwargs


async def call_judge_once(client: AsyncOpenAI, args: argparse.Namespace, prompt: str) -> str:
    response = await client.chat.completions.create(**build_completion_kwargs(args, prompt))
    return extract_message_text(response.choices[0].message)


async def judge_question(
    client: AsyncOpenAI,
    args: argparse.Namespace,
    question: DSQAQuestion,
    prediction: dict[str, Any],
) -> ItemRating:
    response = prediction.get("response") or prediction.get("prediction") or ""
    item_rating = ItemRating(
        original_index=int(question.id) if question.id.isdigit() else None,
        example_id=question.id,
        query=question.problem,
        response=response,
        category_type=question.problem_category,
        expected_correct_answer=question.answer,
    )
    if not response.strip():
        item_rating.empty_model_response = True
        item_rating.error_message = "AI response was empty."
        return item_rating

    prompt = build_grader_prompt(question, response)
    errors: list[str] = []
    for attempt in range(max(1, args.judge_attempts)):
        try:
            judge_response = await call_judge_once(client, args, prompt)
            return reduce_llm_response_to_item_rating(item_rating, judge_response, prompt)
        except Exception as exc:  # pragma: no cover - provider dependent
            errors.append(f"attempt {attempt + 1}: {type(exc).__name__}: {exc}")
            if attempt + 1 < args.judge_attempts:
                await asyncio.sleep(args.judge_retry_delay_s + random.random())

    item_rating.empty_auto_rater_response = True
    item_rating.error_message = " | ".join(errors)
    item_rating.rating_prompt = prompt
    return item_rating


async def judge_all(
    client: AsyncOpenAI,
    args: argparse.Namespace,
    questions: list[DSQAQuestion],
    predictions: dict[str, dict[str, Any]],
    cached_ratings: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, args.num_workers))

    async def bound(question: DSQAQuestion) -> tuple[str, dict[str, Any]]:
        async with semaphore:
            rating = await judge_question(client, args, question, predictions[question.id])
            return question.id, rating.to_dict()

    pending = [
        question
        for question in questions
        if question.id in predictions and question.id not in cached_ratings
    ]
    if not pending:
        return cached_ratings

    results = await tqdm_asyncio.gather(*[bound(question) for question in pending], desc="Judging")
    for qid, rating in results:
        cached_ratings[qid] = rating
    return cached_ratings


def calculate_ci_str(count: int, total: int, z: float = 1.96) -> str:
    if total == 0:
        return f"N/A ({count}/{total})"
    count = max(0, min(count, total))
    p = count / total
    margin = z * math.sqrt((p * (1.0 - p)) / total)
    result = f"{round(p * 100.0, 2):.2f} +/- {round(margin * 100.0, 2):.2f} ({count}/{total})"
    if total <= 5:
        result += " (CI not robust for n<=5)"
    return result


def calculate_metric(true_positives: int, false_positives: int, false_negatives: int) -> dict[str, float]:
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1_score": f1}


def item_metric_values(rating: ItemRating) -> dict[str, Any]:
    ratings = rating.grader_ratings_list or []
    num_correct = sum(1 for value in ratings if value)
    false_negatives = len(ratings) - num_correct
    false_positives = len(rating.response_wrong_answers_list or [])
    metrics = calculate_metric(num_correct, false_positives, false_negatives)
    all_expected_correct = bool(ratings) and num_correct == len(ratings)
    has_excessive = false_positives > 0

    if all_expected_correct and not has_excessive:
        score = 100.0
    elif all_expected_correct and has_excessive:
        score = 75.0
    elif num_correct > 0 and ratings:
        score = (num_correct / len(ratings)) * 50.0
    else:
        score = 0.0

    return {
        "num_expected_answers": len(ratings),
        "num_correct_answers": num_correct,
        "num_excessive_answers": false_positives,
        "fully_correct": all_expected_correct and not has_excessive,
        "fully_incorrect": bool(ratings) and num_correct == 0,
        "correct_with_excessive": all_expected_correct and has_excessive,
        "score": round(score, 2),
        **metrics,
    }


def item_rating_from_dict(payload: dict[str, Any]) -> ItemRating:
    allowed = {item.name for item in fields(ItemRating)}
    return ItemRating(**{key: value for key, value in payload.items() if key in allowed})


def aggregate_ratings(item_ratings: list[ItemRating]) -> ProjectRating:
    project = ProjectRating(num_total_ratings=len(item_ratings))
    category_stats: dict[str, dict[str, int]] = {}
    per_item_metrics: dict[str, list[float]] = {"precision": [], "recall": [], "f1_score": []}

    evaluated = 0
    all_correct = 0
    fully_incorrect = 0
    correct_with_excessive = 0

    for rating in item_ratings:
        if rating.invalid_auto_rater_response:
            project.num_invalid_auto_rater_response += 1
            continue
        if rating.empty_auto_rater_response:
            project.num_empty_auto_rater_response += 1
            continue
        if rating.empty_model_response:
            project.num_empty_model_response += 1
            continue

        project.num_valid_ratings += 1
        if rating.grader_ratings_list is None:
            continue

        evaluated += 1
        category = rating.category_type or "Unknown"
        category_stats.setdefault(category, {"evaluated": 0, "all_correct": 0})
        category_stats[category]["evaluated"] += 1

        values = item_metric_values(rating)
        for key in per_item_metrics:
            per_item_metrics[key].append(float(values[key]))
        if values["fully_correct"]:
            all_correct += 1
            category_stats[category]["all_correct"] += 1
        if values["fully_incorrect"]:
            fully_incorrect += 1
        if values["correct_with_excessive"]:
            correct_with_excessive += 1

    total = len(item_ratings)
    if total:
        project.pct_empty_model_response = round(project.num_empty_model_response * 100.0 / total, 2)
        project.pct_invalid_auto_rater_response = round(
            project.num_invalid_auto_rater_response * 100.0 / total, 2
        )
        project.pct_empty_auto_rater_response = round(
            project.num_empty_auto_rater_response * 100.0 / total, 2
        )

    if evaluated:
        project.num_answer_correctness_evaluated = evaluated
        project.fully_correct_count = all_correct
        project.fully_incorrect_count = fully_incorrect
        project.correct_with_excessive_count = correct_with_excessive
        project.pct_w_ci_all_answers_correct = calculate_ci_str(all_correct, evaluated)
        project.pct_w_ci_fully_incorrect_items = calculate_ci_str(fully_incorrect, evaluated)
        project.pct_w_ci_correct_with_excessive_answers = calculate_ci_str(
            correct_with_excessive, evaluated
        )
        project.precision = f"{mean(per_item_metrics['precision']):.2%}"
        project.recall = f"{mean(per_item_metrics['recall']):.2%}"
        project.f1_score = f"{mean(per_item_metrics['f1_score']):.2%}"
        for category, stats in category_stats.items():
            category_evaluated = stats["evaluated"]
            project.category_breakdown[category] = {
                "evaluated": category_evaluated,
                "all_correct": stats["all_correct"],
                "accuracy": f"{stats['all_correct'] / category_evaluated:.2%}"
                if category_evaluated
                else "0.00%",
            }

    return project


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_scored_rows(
    questions: list[DSQAQuestion],
    predictions: dict[str, dict[str, Any]],
    cached_ratings: dict[str, dict[str, Any]],
    include_prompts: bool,
) -> tuple[list[dict[str, Any]], list[ItemRating]]:
    rows: list[dict[str, Any]] = []
    ratings: list[ItemRating] = []

    for question in questions:
        prediction = predictions.get(question.id)
        rating_payload = cached_ratings.get(question.id)
        if rating_payload:
            rating = item_rating_from_dict(rating_payload)
        else:
            response = prediction.get("response", "") if prediction else ""
            rating = ItemRating(
                original_index=int(question.id) if question.id.isdigit() else None,
                example_id=question.id,
                query=question.problem,
                response=response,
                category_type=question.problem_category,
                expected_correct_answer=question.answer,
                empty_model_response=True,
                error_message="Missing prediction." if prediction is None else "Unjudged prediction.",
            )
        ratings.append(rating)

        values = item_metric_values(rating)
        row = {
            "id": question.id,
            "question": question.problem,
            "answer": question.answer,
            "answer_type": question.answer_type,
            "problem_category": question.problem_category,
            "prediction": rating.response,
            "score": values["score"],
            "precision": values["precision"],
            "recall": values["recall"],
            "f1_score": values["f1_score"],
            "fully_correct": values["fully_correct"],
            "fully_incorrect": values["fully_incorrect"],
            "correct_with_excessive": values["correct_with_excessive"],
            "num_expected_answers": values["num_expected_answers"],
            "num_correct_answers": values["num_correct_answers"],
            "num_excessive_answers": values["num_excessive_answers"],
            "expected_correct_answer_list": rating.expected_correct_answer_list,
            "response_wrong_answers_list": rating.response_wrong_answers_list,
            "grader_ratings_list": rating.grader_ratings_list,
            "answer_correctness_explanation": rating.answer_correctness_explanation,
            "empty_model_response": rating.empty_model_response,
            "empty_auto_rater_response": rating.empty_auto_rater_response,
            "invalid_auto_rater_response": rating.invalid_auto_rater_response,
            "error_message": rating.error_message,
            "rating_response": rating.rating_response,
        }
        if include_prompts:
            row["rating_prompt"] = rating.rating_prompt
        for key in (
            "termination",
            "token_count",
            "total_process_time",
            "total_rounds",
            "tool_call_rounds",
            "error",
        ):
            if prediction and key in prediction:
                row[key] = prediction[key]
        rows.append(row)

    return rows, ratings


def leaderboard_from_project(project: ProjectRating, model_name: str) -> dict[str, Any]:
    evaluated = project.num_answer_correctness_evaluated
    payload = {
        "model": model_name,
        "fully_correct": project.pct_w_ci_all_answers_correct,
        "fully_incorrect": project.pct_w_ci_fully_incorrect_items,
        "correct_with_excessive": project.pct_w_ci_correct_with_excessive_answers,
        "f1": project.f1_score,
        "precision": project.precision,
        "recall": project.recall,
        "num_evaluated": evaluated,
    }
    if evaluated:
        payload.update(
            {
                "fully_correct_pct": round(project.fully_correct_count / evaluated * 100.0, 1),
                "fully_incorrect_pct": round(project.fully_incorrect_count / evaluated * 100.0, 1),
                "correct_with_excessive_pct": round(
                    project.correct_with_excessive_count / evaluated * 100.0, 1
                ),
                "f1_pct": parse_pct(project.f1_score),
                "precision_pct": parse_pct(project.precision),
                "recall_pct": parse_pct(project.recall),
            }
        )
    return payload


def parse_pct(value: str) -> float:
    try:
        return float(value.rstrip("%"))
    except Exception:
        return 0.0


def derive_output_paths(input_path: Path, args: argparse.Namespace) -> tuple[Path, Path, Path]:
    tag = sanitize_tag(args.output_tag or args.judge_model)
    cache_output = (
        Path(args.cache_output)
        if args.cache_output
        else input_path.with_name(f"judged_{input_path.stem}__dsqa_{tag}.json")
    )
    scored_output = (
        Path(args.scored_output)
        if args.scored_output
        else input_path.with_name(f"{input_path.stem}__dsqa_{tag}_scored.jsonl")
    )
    summary_output = (
        Path(args.summary_output)
        if args.summary_output
        else input_path.with_name(f"{input_path.stem}__dsqa_{tag}_summary.json")
    )
    return cache_output, scored_output, summary_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DeepSearchQA predictions with the official LLM-as-judge scorer."
    )
    parser.add_argument("--input_file", required=True, help="Prediction JSONL/JSON file.")
    parser.add_argument(
        "--eval_data_path",
        default=DEFAULT_EVAL_DATA_PATH,
        help="DeepSearchQA CSV or converted JSONL. Defaults to the raw official CSV.",
    )
    parser.add_argument(
        "--metadata_csv_path",
        default=DEFAULT_METADATA_CSV_PATH,
        help="Raw DSQA-full.csv used to recover answer_type/problem_category for JSONL subsets.",
    )
    parser.add_argument(
        "--id_base",
        type=int,
        default=0,
        help="ID assigned to the first CSV row. cabeza converted data uses 0; some official wrappers use 1.",
    )
    parser.add_argument("--cache_output", default="", help="Optional judge cache JSON path.")
    parser.add_argument("--scored_output", default="", help="Optional scored JSONL path.")
    parser.add_argument("--summary_output", default="", help="Optional summary JSON path.")
    parser.add_argument("--output_tag", default="", help="Optional tag appended to default outputs.")
    parser.add_argument("--include_prompts", action="store_true", help="Include full judge prompts in scored rows.")
    parser.add_argument("--force_rerun", action="store_true", help="Ignore existing judge cache.")
    parser.add_argument("--num_workers", type=int, default=8, help="Async judge concurrency.")
    parser.add_argument("--judge_model", required=True, help="Judge model name.")
    parser.add_argument("--judge_base_url", default="", help="OpenAI-compatible judge base URL.")
    parser.add_argument("--judge_api_key", default="", help="Judge API key.")
    parser.add_argument(
        "--api_dict_path",
        default="data/api_dict.json",
        help="Optional API endpoint config.",
    )
    parser.add_argument("--api_dict_key", default="", help="Key inside --api_dict_path.")
    parser.add_argument("--request_timeout_s", type=float, default=300.0)
    parser.add_argument("--max_retries", type=int, default=1)
    parser.add_argument("--max_completion_tokens", type=int, default=4096)
    parser.add_argument("--judge_attempts", type=int, default=3)
    parser.add_argument("--judge_retry_delay_s", type=float, default=2.0)
    parser.add_argument("--disable_thinking", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_workers < 1:
        raise ValueError("--num_workers must be at least 1.")

    input_path = Path(args.input_file)
    eval_data_path = Path(args.eval_data_path)
    questions = load_questions(
        eval_data_path=eval_data_path,
        metadata_csv_path=Path(args.metadata_csv_path),
        id_base=args.id_base,
    )
    predictions = load_predictions(input_path)
    cache_output, scored_output, summary_output = derive_output_paths(input_path, args)

    if args.force_rerun or not cache_output.exists():
        cached_ratings: dict[str, dict[str, Any]] = {}
    else:
        cached_ratings = json.loads(cache_output.read_text(encoding="utf-8"))

    pending_count = sum(
        1 for question in questions if question.id in predictions and question.id not in cached_ratings
    )
    if pending_count:
        client = AsyncOpenAI(**resolve_client_kwargs(args))
        cached_ratings = asyncio.run(
            judge_all(
                client=client,
                args=args,
                questions=questions,
                predictions=predictions,
                cached_ratings=cached_ratings,
            )
        )

    cache_output.parent.mkdir(parents=True, exist_ok=True)
    cache_output.write_text(
        json.dumps(cached_ratings, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    scored_rows, item_ratings = build_scored_rows(
        questions=questions,
        predictions=predictions,
        cached_ratings=cached_ratings,
        include_prompts=args.include_prompts,
    )
    write_jsonl(scored_output, scored_rows)

    project_rating = aggregate_ratings(item_ratings)
    available_predictions = sum(1 for question in questions if question.id in predictions)
    numeric_scores = [float(row["score"]) for row in scored_rows]
    summary = {
        "benchmark": "deepsearchqa",
        "input_file": str(input_path),
        "eval_data_path": str(eval_data_path),
        "metadata_csv_path": str(Path(args.metadata_csv_path)),
        "judge_model": args.judge_model,
        "judge_base_url": getattr(args, "resolved_judge_base_url", args.judge_base_url),
        "total_questions": len(questions),
        "available_predictions": available_predictions,
        "missing_predictions": len(questions) - available_predictions,
        "average_score": round(mean(numeric_scores), 2),
        "cache_output": str(cache_output),
        "scored_output": str(scored_output),
        "project_rating": project_rating.to_dict(),
        "leaderboard": leaderboard_from_project(project_rating, model_name=args.output_tag or "cabeza"),
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Input       : {input_path}")
    print(f"Eval data   : {eval_data_path}")
    print(f"Judge cache : {cache_output}")
    print(f"Scored rows : {scored_output}")
    print(f"Summary     : {summary_output}")
    print(f"F1          : {project_rating.f1_score or 'N/A'}")
    print(f"Fully corr. : {project_rating.pct_w_ci_all_answers_correct or 'N/A'}")


if __name__ == "__main__":
    main()
