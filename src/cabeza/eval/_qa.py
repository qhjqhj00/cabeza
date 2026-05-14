#!/usr/bin/env python3
"""Unified QA-style evaluator for SAM.

This evaluator covers QA-like benchmarks such as:
- BrowseComp (`bc`)
- BrowseCompZH (`bc_zh`)
- BrowseCompPlus (`bcp`)
- Humanity's Last Exam (`hle`)

Key features:
- One evaluator for both local and non-local OpenAI-compatible judge models
- Judge credentials passed explicitly via CLI args
- Optional credential lookup from `api_dict.json`
- Multiple `--input_file` support with per-file scoring and pass@k aggregation
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import numpy as np
from openai import AsyncOpenAI

try:
    from pydantic import BaseModel
except ImportError:  # pragma: no cover - lightweight fallback for CLI-only use
    class BaseModel:
        model_fields: dict[str, object] = {}

        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            annotations = getattr(cls, "__annotations__", {})
            cls.model_fields = {
                name: getattr(cls, name)
                for name in annotations
                if not name.startswith("_")
            }

        def __init__(self, **data):
            for name in self.model_fields:
                value = data.get(name, getattr(self.__class__, name))
                setattr(self, name, value)

        @classmethod
        def model_validate(cls, data):
            if isinstance(data, cls):
                return data
            if not isinstance(data, dict):
                raise TypeError(f"Expected mapping for {cls.__name__}, got {type(data).__name__}")
            return cls(**data)

        @classmethod
        def model_rebuild(cls, *args, **kwargs):
            return None
from tqdm.asyncio import tqdm_asyncio

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - optional fallback only
    load_dataset = None

from cabeza.datasets.registry import get_spec, resolve_default_path
from cabeza.eval._prompts import (
    JUDGE_PROMPT_BROWSECOMP_OFFICIAL,
    JUDGE_PROMPT_CONFIDENCE,
)


def _bundled_eval_path(name: str) -> str:
    """Default eval JSONL path from the cabeza dataset registry, or empty string."""
    try:
        spec = get_spec(name)
    except ValueError:
        return ""
    return resolve_default_path(spec) or ""


BENCHMARK_ALIASES = {
    "bc": "bc",
    "browsecomp": "bc",
    "browsecomp_en": "bc",
    "browsecomp_en_full": "bc",
    "bc_zh": "bc_zh",
    "bc-zh": "bc_zh",
    "browsecomp_zh": "bc_zh",
    "browsecomp-zh": "bc_zh",
    "browsecompzh": "bc_zh",
    "bcp": "bcp",
    "browsecompplus": "bcp",
    "browsecomp_plus": "bcp",
    "hle": "hle",
}


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    default_eval_data_path: str
    judge_prompt: str


BENCHMARK_CONFIGS = {
    "bc": BenchmarkConfig(
        name="bc",
        default_eval_data_path=_bundled_eval_path("bc"),
        judge_prompt=JUDGE_PROMPT_BROWSECOMP_OFFICIAL,
    ),
    "bc_zh": BenchmarkConfig(
        name="bc_zh",
        default_eval_data_path=_bundled_eval_path("bc_zh"),
        judge_prompt=JUDGE_PROMPT_BROWSECOMP_OFFICIAL,
    ),
    "bcp": BenchmarkConfig(
        name="bcp",
        default_eval_data_path=_bundled_eval_path("bcp"),
        judge_prompt=JUDGE_PROMPT_BROWSECOMP_OFFICIAL,
    ),
    "hle": BenchmarkConfig(
        name="hle",
        default_eval_data_path=_bundled_eval_path("hle"),
        judge_prompt=JUDGE_PROMPT_CONFIDENCE,
    ),
}


JSON_ONLY_SUFFIX = """

Respond with a single JSON object only, with exactly these keys:
{
  "extracted_final_answer": "string",
  "reasoning": "string",
  "correct": "yes or no",
  "confidence": 0,
  "strict": true
}
"""


JSON_SCHEMA_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "extracted_answer",
        "schema": {
            "type": "object",
            "properties": {
                "extracted_final_answer": {"type": "string"},
                "reasoning": {"type": "string"},
                "correct": {"type": "string", "enum": ["yes", "no"]},
                "confidence": {"type": "integer"},
                "strict": {"type": "boolean"},
            },
            "required": [
                "extracted_final_answer",
                "reasoning",
                "correct",
                "confidence",
                "strict",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: Literal[True] = True


ExtractedAnswer.model_rebuild()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SAM QA-style predictions and optionally compute pass@k."
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=sorted(BENCHMARK_ALIASES.keys()),
        help="Benchmark preset used to choose the default eval set and judge prompt.",
    )
    parser.add_argument(
        "--input_file",
        action="append",
        default=[],
        help="Prediction file to evaluate. Pass multiple times for pass@k aggregation.",
    )
    parser.add_argument(
        "--eval_data_path",
        default="",
        help="Override the benchmark eval JSONL path.",
    )
    parser.add_argument(
        "--dataset",
        default="",
        help="Optional HF dataset id used only if --eval_data_path is missing.",
    )
    parser.add_argument(
        "--cache_output",
        default="",
        help="Optional judge cache path. Only valid for a single --input_file.",
    )
    parser.add_argument(
        "--scored_output",
        default="",
        help="Optional scored JSONL output path. Only valid for a single --input_file.",
    )
    parser.add_argument(
        "--summary_output",
        default="",
        help="Optional summary JSON output path. Only valid for a single --input_file.",
    )
    parser.add_argument(
        "--passk_scored_output",
        default="",
        help="Optional merged pass@k scored JSONL path when multiple --input_file are passed.",
    )
    parser.add_argument(
        "--passk_summary_output",
        default="",
        help="Optional merged pass@k summary JSON path when multiple --input_file are passed.",
    )
    parser.add_argument(
        "--output_tag",
        default="",
        help="Optional tag appended to default cache / scored / summary filenames.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Async semaphore size. Match this to your judge rate limit.",
    )
    parser.add_argument(
        "--judge_model",
        required=True,
        help="Judge model name.",
    )
    parser.add_argument(
        "--judge_base_url",
        default="",
        help="Optional OpenAI-compatible base URL for the judge.",
    )
    parser.add_argument(
        "--judge_api_key",
        default="",
        help="Optional judge API key.",
    )
    parser.add_argument(
        "--api_dict_path",
        default="data/api_dict.json",
        help="Optional config file with {model_or_key: {api_key, base_url}}.",
    )
    parser.add_argument(
        "--api_dict_key",
        default="",
        help="Optional key inside --api_dict_path used to resolve credentials.",
    )
    parser.add_argument(
        "--request_timeout_s",
        type=float,
        default=300.0,
        help="Judge request timeout in seconds.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=1,
        help="Max retries for the OpenAI client.",
    )
    parser.add_argument(
        "--max_completion_tokens",
        type=int,
        default=4096,
        help="Max completion tokens for judge calls.",
    )
    parser.add_argument(
        "--judge_attempts",
        type=int,
        default=3,
        help="Total end-to-end judge attempts per question before local fallback.",
    )
    parser.add_argument(
        "--judge_retry_delay_s",
        type=float,
        default=2.0,
        help="Seconds to wait between end-to-end judge retries.",
    )
    parser.add_argument(
        "--disable_local_fallback",
        action="store_true",
        help="Do not emit a conservative local fallback judgement after repeated judge failures.",
    )
    parser.add_argument(
        "--disable_thinking",
        action="store_true",
        help="Force disable_thinking for OpenAI-compatible local judge servers.",
    )
    parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="Ignore existing caches and rejudge all available predictions.",
    )
    return parser.parse_args()


def normalize_benchmark_name(value: str) -> str:
    key = value.strip().lower()
    if key not in BENCHMARK_ALIASES:
        raise ValueError(f"Unsupported benchmark: {value}")
    return BENCHMARK_ALIASES[key]


def sanitize_tag(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "default"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
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


def compact_prediction_row(raw: dict[str, Any], fallback_id: str = "") -> dict[str, Any]:
    row = dict(raw)
    unique_id = normalize_id(row.get("id") or row.get("instance_id") or fallback_id)
    response = ""
    for key in ("response", "prediction", "model_answer", "final_answer"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            response = value
            break

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
        "shared_memory",
        "error",
    ):
        if key in row:
            compact[key] = row[key]
    if "judge_response" in row:
        compact["judge_response"] = row["judge_response"]
    return compact


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = read_jsonl(path)
        predictions: dict[str, dict[str, Any]] = {}
        for row in rows:
            compact = compact_prediction_row(row)
            unique_id = compact["id"]
            if unique_id:
                predictions[unique_id] = compact
        return predictions

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    predictions = {}
    if isinstance(payload, list):
        for row in payload:
            compact = compact_prediction_row(row)
            unique_id = compact["id"]
            if unique_id:
                predictions[unique_id] = compact
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
            unique_id = compact["id"]
            if unique_id:
                predictions[unique_id] = compact
        return predictions

    raise ValueError(f"Unsupported predictions payload type: {type(payload).__name__}")


def load_questions(eval_data_path: Path, dataset_name: str) -> list[dict[str, Any]]:
    if eval_data_path.exists():
        rows = read_jsonl(eval_data_path)
    elif dataset_name:
        if load_dataset is None:
            raise ImportError(
                "datasets is not installed, so --dataset cannot be used without it."
            )
        hf_rows = load_dataset(dataset_name, split="test").to_dict()
        rows = [dict(zip(hf_rows.keys(), values)) for values in zip(*hf_rows.values())]
    else:
        raise FileNotFoundError(
            f"Eval data not found at {eval_data_path}. Pass --dataset for HF loading."
        )

    questions: list[dict[str, Any]] = []
    for row in rows:
        unique_id = normalize_id(row.get("id") or row.get("instance_id"))
        if not unique_id:
            continue
        answer = row.get("answer", row.get("golden_answers", ""))
        questions.append(
            {
                "id": unique_id,
                "question": row.get("question", ""),
                "answer": answer,
            }
        )
    return questions


def is_local_base_url(base_url: str) -> bool:
    if not base_url:
        return False
    parsed = urlparse(base_url)
    hostname = (parsed.hostname or "").lower()
    return hostname in {"localhost", "127.0.0.1", "0.0.0.0"}


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

    client_kwargs: dict[str, Any] = {
        "api_key": args.resolved_judge_api_key,
        "timeout": args.request_timeout_s,
        "max_retries": args.max_retries,
    }
    if base_url:
        client_kwargs["base_url"] = base_url
    return client_kwargs


def answer_to_text(answer: Any) -> str:
    if isinstance(answer, str):
        return answer
    return json.dumps(answer, ensure_ascii=False)


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = strip_code_fences(text)
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for idx, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(cleaned[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("Could not extract a JSON object from judge response.")


def extract_message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        if content.strip():
            return content
        reasoning_content = getattr(message, "reasoning_content", None) or getattr(
            message, "reasoning", None
        )
        if reasoning_content:
            return str(reasoning_content)
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
            else:
                text = getattr(item, "text", None)
            if text:
                chunks.append(str(text))
        if chunks:
            return "\n".join(chunks)
        reasoning_content = getattr(message, "reasoning_content", None) or getattr(
            message, "reasoning", None
        )
        if reasoning_content:
            return str(reasoning_content)
        return ""
    return str(content or "")


def canonicalize_field_name(label: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    aliases = {
        "extracted_final_answer": "extracted_final_answer",
        "final_answer": "extracted_final_answer",
        "model_answer": "extracted_final_answer",
        "answer": "extracted_final_answer",
        "reasoning": "reasoning",
        "correct": "correct",
        "confidence": "confidence",
        "strict": "strict",
    }
    return aliases.get(normalized)


def extract_key_value_payload(text: str) -> dict[str, Any]:
    cleaned = strip_code_fences(text)
    key_pattern = re.compile(
        r"^\s*(?:[-*]\s*)?(?:\*\*|__)?([A-Za-z][A-Za-z _-]{1,40})(?:\*\*|__)?\s*[:：-]\s*(.*)$"
    )
    payload: dict[str, str] = {}
    current_key: str | None = None

    for raw_line in cleaned.splitlines():
        line = raw_line.rstrip()
        match = key_pattern.match(line)
        if match:
            field_name = canonicalize_field_name(match.group(1))
            if field_name is not None:
                current_key = field_name
                payload[current_key] = match.group(2).strip()
                continue

        if current_key is not None:
            existing = payload.get(current_key, "")
            continuation = line.strip()
            if continuation:
                payload[current_key] = (
                    f"{existing}\n{continuation}" if existing else continuation
                )

    if not payload:
        raise ValueError("Could not extract expected key/value fields from judge response.")

    normalized_payload: dict[str, Any] = dict(payload)
    if "strict" in normalized_payload:
        normalized_payload["strict"] = str(normalized_payload["strict"]).strip().lower() in {
            "1",
            "true",
            "yes",
        }
    return normalized_payload


def extract_structured_payload(text: str) -> dict[str, Any]:
    errors: list[str] = []
    for extractor in (extract_json_object, extract_key_value_payload):
        try:
            return extractor(text)
        except ValueError as exc:
            errors.append(str(exc))
    raise ValueError(" | ".join(errors))


def validate_judge_payload(payload: dict[str, Any]) -> ExtractedAnswer:
    normalized = dict(payload)
    normalized.setdefault("extracted_final_answer", "None")
    normalized.setdefault("reasoning", "")
    normalized.setdefault("strict", True)

    correct_value = str(normalized.get("correct", "")).strip().lower()
    normalized["correct"] = "yes" if correct_value in {"yes", "y", "true", "correct"} else "no"

    confidence_value = normalized.get("confidence", 100)
    try:
        if isinstance(confidence_value, str):
            confidence_value = confidence_value.strip().replace("%", "")
        normalized["confidence"] = int(round(float(confidence_value)))
    except (TypeError, ValueError):
        normalized["confidence"] = 100
    normalized["confidence"] = max(0, min(100, normalized["confidence"]))

    return ExtractedAnswer.model_validate(normalized)


def to_judge_response(correct_answer: Any, parsed: ExtractedAnswer) -> dict[str, Any]:
    return {
        "correct_answer": correct_answer,
        "model_answer": parsed.extracted_final_answer,
        "reasoning": parsed.reasoning,
        "correct": parsed.correct,
        "confidence": parsed.confidence,
        "judge_mode": "llm",
    }


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

    if (args.disable_thinking or args.resolved_judge_is_local) and args.resolved_judge_base_url:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    return kwargs


async def parse_with_beta(
    client: AsyncOpenAI, args: argparse.Namespace, prompt: str
) -> ExtractedAnswer:
    if args.resolved_judge_is_local:
        raise ValueError("Local judge detected; skipping beta parse.")

    kwargs = build_completion_kwargs(args, prompt)
    response = await client.beta.chat.completions.parse(
        **kwargs,
        response_format=ExtractedAnswer,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        message_text = extract_message_text(response.choices[0].message)
        if not message_text.strip():
            raise ValueError("Structured parse returned no parsed payload.")
        payload = extract_structured_payload(message_text)
        return validate_judge_payload(payload)
    return parsed


async def parse_with_json_schema(
    client: AsyncOpenAI, args: argparse.Namespace, prompt: str
) -> ExtractedAnswer:
    if args.resolved_judge_is_local:
        raise ValueError("Local judge detected; skipping JSON schema parse.")

    kwargs = build_completion_kwargs(args, prompt + JSON_ONLY_SUFFIX)
    response = await client.chat.completions.create(
        **kwargs,
        response_format=JSON_SCHEMA_RESPONSE_FORMAT,
    )
    message_text = extract_message_text(response.choices[0].message)
    payload = extract_structured_payload(message_text)
    return validate_judge_payload(payload)


async def parse_with_prompt_only(
    client: AsyncOpenAI, args: argparse.Namespace, prompt: str
) -> ExtractedAnswer:
    kwargs = build_completion_kwargs(args, prompt + JSON_ONLY_SUFFIX)
    response = await client.chat.completions.create(**kwargs)
    message_text = extract_message_text(response.choices[0].message)
    payload = extract_structured_payload(message_text)
    return validate_judge_payload(payload)


async def extract_answer_once(
    client: AsyncOpenAI,
    args: argparse.Namespace,
    question: str,
    correct_answer: Any,
    response: str,
) -> tuple[dict[str, Any] | None, str | None]:
    prompt = args.judge_prompt.format(
        question=question,
        correct_answer=answer_to_text(correct_answer),
        response=response,
    )
    errors: list[str] = []
    if args.benchmark_name == "hle":
        # Keep the official HLE flow as the primary path:
        # 1. hosted judge -> beta structured parse
        # 2. local judge -> prompt-only JSON extraction
        # Then preserve our compatibility fallbacks if the primary path fails.
        parse_attempts = (
            (parse_with_prompt_only,)
            if args.resolved_judge_is_local
            else (parse_with_beta, parse_with_json_schema, parse_with_prompt_only)
        )
    else:
        parse_attempts = (
            parse_with_beta,
            parse_with_json_schema,
            parse_with_prompt_only,
        )

    for parser_fn in parse_attempts:
        try:
            parsed = await parser_fn(client, args, prompt)
            return to_judge_response(correct_answer, parsed), None
        except Exception as exc:  # pragma: no cover - network/provider dependent
            errors.append(f"{parser_fn.__name__}: {type(exc).__name__}: {exc}")

    return None, " | ".join(errors)


def unwrap_answer_text(text: str) -> str:
    cleaned = strip_code_fences(text).strip()
    while True:
        updated = cleaned
        wrappers = (
            ("**", "**"),
            ("__", "__"),
            ("`", "`"),
            ("$", "$"),
            ("\\(", "\\)"),
            ("\\[", "\\]"),
        )
        for prefix, suffix in wrappers:
            if updated.startswith(prefix) and updated.endswith(suffix):
                updated = updated[len(prefix) : len(updated) - len(suffix)].strip()
        if updated == cleaned:
            return cleaned
        cleaned = updated


def normalize_answer_for_match(value: Any) -> str:
    text = answer_to_text(value)
    text = unwrap_answer_text(text)
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip("\"'.,;: ")
    return text


def canonicalize_answer_for_match(value: Any) -> str:
    text = normalize_answer_for_match(value).lower()
    text = text.replace(" ", "").replace(",", "")
    return text


def extract_confidence_from_response(response: str) -> int:
    match = re.search(r"confidence\s*[:：-]?\s*(\d+(?:\.\d+)?)\s*%?", response, re.IGNORECASE)
    if not match:
        return 100
    try:
        return max(0, min(100, int(round(float(match.group(1))))))
    except ValueError:
        return 100


def clean_extracted_answer_candidate(candidate: str) -> str:
    cleaned = unwrap_answer_text(candidate)
    cleaned = re.sub(
        r"^(?:final\s+answer|answer|final)\s*[:：-]\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"^(?:the\s+)?answer\s+is\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()
    return cleaned or "None"


def extract_final_answer_from_response(response: str) -> str:
    cleaned = strip_code_fences(response).strip()
    if not cleaned:
        return "None"

    patterns = (
        r"(?:^|\n)\s*(?:final\s+answer|answer)\s*[:：-]\s*(.+)",
        r"(?:^|\n)\s*\*\*(?:final\s+answer|answer)\*\*\s*(?:\n+|[:：-]\s*)(.+)",
        r"(?:^|\n)\s*(?:the\s+)?answer\s+is\s+(.+)",
    )
    for pattern in patterns:
        matches = re.findall(pattern, cleaned, flags=re.IGNORECASE)
        for candidate in reversed(matches):
            answer = clean_extracted_answer_candidate(candidate.splitlines()[0])
            if answer and answer.lower() != "confidence":
                return answer

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    for line in reversed(lines):
        if re.match(r"^confidence\b", line, flags=re.IGNORECASE):
            continue
        if re.match(r"^\*{0,2}explanation\b", line, flags=re.IGNORECASE):
            continue
        answer = clean_extracted_answer_candidate(line)
        if answer:
            return answer

    return "None"


def extract_choice_letter(value: Any) -> str | None:
    text = normalize_answer_for_match(value)
    if not text:
        return None

    direct_match = re.fullmatch(r"\(?([A-Z])\)?", text, flags=re.IGNORECASE)
    if direct_match:
        return direct_match.group(1).upper()

    patterns = (
        r"(?:answer|option|choice)\s*(?:is|=|:)?\s*\(?([A-Z])\)?",
        r"^\(?([A-Z])\)?[.)]?$",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()
    return None


def try_parse_float(value: Any) -> float | None:
    text = normalize_answer_for_match(value)
    if not text:
        return None
    numeric = text.replace(",", "")
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", numeric):
        return None
    try:
        return float(numeric)
    except ValueError:
        return None


def fallback_answers_match(correct_answer: Any, extracted_answer: str) -> bool:
    gold_choice = extract_choice_letter(correct_answer)
    pred_choice = extract_choice_letter(extracted_answer)
    if gold_choice and pred_choice:
        return gold_choice == pred_choice

    gold_number = try_parse_float(correct_answer)
    pred_number = try_parse_float(extracted_answer)
    if gold_number is not None and pred_number is not None:
        tolerance = max(1e-6, 1e-4 * max(abs(gold_number), 1.0))
        return abs(gold_number - pred_number) <= tolerance

    return canonicalize_answer_for_match(correct_answer) == canonicalize_answer_for_match(
        extracted_answer
    )


def build_fallback_judge_response(
    correct_answer: Any,
    response: str,
    error: str,
    attempts_used: int,
) -> dict[str, Any]:
    extracted_answer = extract_final_answer_from_response(response)
    is_match = fallback_answers_match(correct_answer, extracted_answer)
    match_text = "matched" if is_match else "did not match"
    return {
        "correct_answer": correct_answer,
        "model_answer": extracted_answer,
        "reasoning": (
            "LLM judge parsing failed after "
            f"{attempts_used} attempt(s); local fallback extracted the final answer and "
            f"{match_text} the normalized reference answer with conservative exact-match logic."
        ),
        "correct": "yes" if is_match else "no",
        "confidence": extract_confidence_from_response(response),
        "judge_mode": "fallback_exact_match",
        "judge_error": error,
    }


def should_use_local_fallback(error: str) -> bool:
    parse_markers = (
        "Structured parse returned no parsed payload.",
        "Could not extract a JSON object from judge response.",
        "Could not extract expected key/value fields from judge response.",
        "Local judge detected; skipping",
    )
    return any(marker in error for marker in parse_markers)


async def extract_answer(
    client: AsyncOpenAI,
    args: argparse.Namespace,
    question: str,
    correct_answer: Any,
    response: str,
) -> tuple[dict[str, Any] | None, str | None]:
    attempt_errors: list[str] = []
    total_attempts = max(1, args.judge_attempts)

    for attempt_idx in range(total_attempts):
        content, error = await extract_answer_once(
            client=client,
            args=args,
            question=question,
            correct_answer=correct_answer,
            response=response,
        )
        if content is not None:
            return content, None

        if error:
            attempt_errors.append(f"attempt {attempt_idx + 1}/{total_attempts}: {error}")

        should_retry = attempt_idx + 1 < total_attempts
        if should_retry and args.judge_retry_delay_s > 0:
            await asyncio.sleep(args.judge_retry_delay_s)

    combined_error = " || ".join(attempt_errors)
    if args.disable_local_fallback or not should_use_local_fallback(combined_error):
        return None, combined_error

    fallback_response = build_fallback_judge_response(
        correct_answer=correct_answer,
        response=response,
        error=combined_error,
        attempts_used=total_attempts,
    )
    return fallback_response, combined_error


async def add_judge_response(
    client: AsyncOpenAI,
    args: argparse.Namespace,
    question: dict[str, Any],
    predictions: dict[str, dict[str, Any]],
    judged_predictions: dict[str, dict[str, Any]],
) -> tuple[str | None, dict[str, Any] | None]:
    unique_id = question["id"]

    cached_prediction = judged_predictions.get(unique_id)
    if cached_prediction and "judge_response" in cached_prediction:
        return unique_id, cached_prediction

    prediction = copy.deepcopy(predictions[unique_id])
    prediction["id"] = unique_id
    prediction["question"] = prediction.get("question") or question["question"]
    prediction["answer"] = question["answer"]

    response = prediction.get("response") or prediction.get("prediction") or ""
    if not isinstance(response, str) or not response.strip():
        return None, None

    content, error = await extract_answer(
        client=client,
        args=args,
        question=question["question"],
        correct_answer=question["answer"],
        response=response,
    )
    if content is None:
        if error:
            print(f"Judge failed for {unique_id}: {error}")
        return None, None

    if error and content.get("judge_mode") != "llm":
        print(f"Judge fallback used for {unique_id}: {error}")

    prediction["judge_response"] = content
    return unique_id, prediction


async def judge_all_responses(
    client: AsyncOpenAI,
    args: argparse.Namespace,
    questions: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    judged_predictions: dict[str, dict[str, Any]],
) -> list[tuple[str | None, dict[str, Any] | None]]:
    semaphore = asyncio.Semaphore(max(1, args.num_workers))

    async def bound_func(question: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
        async with semaphore:
            return await add_judge_response(
                client=client,
                args=args,
                question=question,
                predictions=predictions,
                judged_predictions=judged_predictions,
            )

    tasks = [bound_func(question) for question in questions]
    return await tqdm_asyncio.gather(*tasks, desc="Judging")


def calib_err(confidence: np.ndarray, correct: np.ndarray, p: str = "2", beta: int = 100) -> float:
    if len(confidence) == 0:
        return 0.0

    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    num_bins = max(1, math.ceil(len(confidence) / beta))
    bins = [[i * beta, min((i + 1) * beta, len(confidence))] for i in range(num_bins)]

    cerr = 0.0
    total_examples = len(confidence)
    for start, end in bins:
        bin_confidence = confidence[start:end]
        bin_correct = correct[start:end]
        num_examples_in_bin = len(bin_confidence)

        if num_examples_in_bin > 0:
            difference = abs(float(np.nanmean(bin_confidence)) - float(np.nanmean(bin_correct)))

            if p == "2":
                cerr += num_examples_in_bin / total_examples * (difference**2)
            elif p == "1":
                cerr += num_examples_in_bin / total_examples * difference
            elif p in {"infty", "infinity", "max"}:
                cerr = max(cerr, difference)
            else:
                raise AssertionError("p must be '1', '2', or 'infty'")

    if p == "2":
        cerr = math.sqrt(cerr)

    return float(cerr)


def build_scored_rows(
    questions: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    judged_predictions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    scored_rows: list[dict[str, Any]] = []

    for question in questions:
        unique_id = question["id"]
        prediction = judged_predictions.get(unique_id) or predictions.get(unique_id)
        base_row = {
            "id": unique_id,
            "question": question["question"],
            "answer": question["answer"],
            "prediction": "",
            "model_answer": "",
            "reasoning": "",
            "confidence": 0,
            "correct": False,
            "judgement": "MissingPrediction",
        }

        if prediction is None:
            scored_rows.append(base_row)
            continue

        base_row["prediction"] = prediction.get("prediction") or prediction.get("response") or ""
        for key in (
            "termination",
            "token_count",
            "total_process_time",
            "total_rounds",
            "tool_call_rounds",
        ):
            if key in prediction:
                base_row[key] = prediction[key]

        judge_response = prediction.get("judge_response")
        if not judge_response:
            base_row["judgement"] = "Unjudged"
            scored_rows.append(base_row)
            continue

        is_correct = str(judge_response.get("correct", "")).lower() == "yes"
        base_row.update(
            {
                "model_answer": judge_response.get("model_answer", ""),
                "reasoning": judge_response.get("reasoning", ""),
                "confidence": int(judge_response.get("confidence", 100)),
                "correct": is_correct,
                "judgement": "Correct" if is_correct else "Incorrect",
            }
        )
        if "judge_mode" in judge_response:
            base_row["judge_mode"] = judge_response["judge_mode"]
        if "judge_error" in judge_response:
            base_row["judge_error"] = judge_response["judge_error"]
        scored_rows.append(base_row)

    return scored_rows


def compute_metrics_from_scored_rows(rows: list[dict[str, Any]], total_questions: int) -> dict[str, Any]:
    judged_rows = [
        row for row in rows if row.get("judgement") in {"Correct", "Incorrect"}
    ]
    correct = np.array([bool(row.get("correct")) for row in judged_rows], dtype=bool)
    confidence_values = np.array(
        [float(row.get("confidence", 100)) for row in judged_rows],
        dtype=float,
    )
    confidence = confidence_values / 100 if len(confidence_values) else np.array([])

    accuracy = round(100 * int(correct.sum()) / total_questions, 2) if total_questions else 0.0
    confidence_half_width = (
        round(1.96 * math.sqrt(accuracy * (100 - accuracy) / total_questions), 2)
        if total_questions
        else 0.0
    )
    calibration_error = (
        100 * round(calib_err(confidence, correct, p="2", beta=100), 2)
        if len(confidence)
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "confidence_half_width": confidence_half_width,
        "calibration_error": calibration_error,
        "judged_predictions": len(judged_rows),
    }


def derive_output_paths(
    input_path: Path,
    args: argparse.Namespace,
) -> tuple[Path, Path, Path]:
    tag = sanitize_tag(args.output_tag or args.judge_model)
    cache_output = (
        Path(args.cache_output)
        if args.cache_output
        else input_path.with_name(f"judged_{input_path.stem}__{tag}.json")
    )
    scored_output = (
        Path(args.scored_output)
        if args.scored_output
        else input_path.with_name(f"{input_path.stem}__{tag}_scored.jsonl")
    )
    summary_output = (
        Path(args.summary_output)
        if args.summary_output
        else input_path.with_name(f"{input_path.stem}__{tag}_summary.json")
    )
    return cache_output, scored_output, summary_output


def choose_best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            1 if row.get("correct") else 0,
            0 if row.get("judgement") in {"Correct", "Incorrect"} else -1,
            int(row.get("confidence", 0) or 0),
        ),
        reverse=True,
    )
    return sorted_rows[0]


def derive_passk_output_paths(
    input_paths: list[Path],
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    if args.passk_scored_output:
        scored_output = Path(args.passk_scored_output)
    else:
        common_prefix = Path(input_paths[0]).stem
        if len(input_paths) > 1:
            common_prefix = sanitize_tag(Path(input_paths[0]).stem)
        tag = sanitize_tag(args.output_tag or args.judge_model)
        scored_output = input_paths[0].with_name(
            f"{common_prefix}__{tag}_pass@{len(input_paths)}_scored.jsonl"
        )

    if args.passk_summary_output:
        summary_output = Path(args.passk_summary_output)
    else:
        common_prefix = sanitize_tag(Path(input_paths[0]).stem)
        tag = sanitize_tag(args.output_tag or args.judge_model)
        summary_output = input_paths[0].with_name(
            f"{common_prefix}__{tag}_pass@{len(input_paths)}_summary.json"
        )

    return scored_output, summary_output


def evaluate_single_input(
    input_path: Path,
    args: argparse.Namespace,
    questions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    cache_output, scored_output, summary_output = derive_output_paths(input_path, args)
    predictions = load_predictions(input_path)
    total_questions = len(questions)

    if args.force_rerun or not cache_output.exists():
        judged_predictions: dict[str, dict[str, Any]] = {}
    else:
        judged_predictions = json.loads(cache_output.read_text(encoding="utf-8"))

    pending_questions = [
        question
        for question in questions
        if question["id"] in predictions
        and "judge_response" not in judged_predictions.get(question["id"], {})
    ]

    if pending_questions:
        client = AsyncOpenAI(**resolve_client_kwargs(args))
        results = asyncio.run(
            judge_all_responses(
                client=client,
                args=args,
                questions=pending_questions,
                predictions=predictions,
                judged_predictions=judged_predictions,
            )
        )
        for unique_id, prediction in results:
            if unique_id is not None and prediction is not None:
                judged_predictions[unique_id] = prediction

    cache_output.parent.mkdir(parents=True, exist_ok=True)
    cache_output.write_text(
        json.dumps(judged_predictions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    scored_rows = build_scored_rows(
        questions=questions,
        predictions=predictions,
        judged_predictions=judged_predictions,
    )
    write_jsonl(scored_output, scored_rows)

    predicted_question_ids = {question["id"] for question in questions if question["id"] in predictions}
    metrics = compute_metrics_from_scored_rows(scored_rows, total_questions=total_questions)
    summary = {
        "benchmark": args.benchmark_name,
        "input_file": str(input_path),
        "eval_data_path": str(args.eval_data_path_resolved) if args.eval_data_path_resolved.exists() else "",
        "judge_model": args.judge_model,
        "judge_base_url": args.resolved_judge_base_url,
        "total_questions": total_questions,
        "available_predictions": len(predicted_question_ids),
        "missing_predictions": total_questions - len(predicted_question_ids),
        "cache_output": str(cache_output),
        "scored_output": str(scored_output),
        **metrics,
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Input       : {input_path}")
    print(f"Judge cache : {cache_output}")
    print(f"Scored rows : {scored_output}")
    print(f"Summary     : {summary_output}")
    print(
        f"Accuracy    : {summary['accuracy']}% "
        f"({summary['judged_predictions']} judged / {summary['total_questions']} total)"
    )

    return scored_rows, summary, {
        "cache_output": cache_output,
        "scored_output": scored_output,
        "summary_output": summary_output,
    }


def build_passk_outputs(
    questions: list[dict[str, Any]],
    input_paths: list[Path],
    scored_rows_list: list[list[dict[str, Any]]],
    per_file_summaries: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_file_lookup = [
        {normalize_id(row.get("id")): row for row in scored_rows}
        for scored_rows in scored_rows_list
    ]

    merged_rows: list[dict[str, Any]] = []
    for question in questions:
        unique_id = question["id"]
        source_rows: list[dict[str, Any]] = []
        source_results: list[dict[str, Any]] = []

        for input_path, row_map in zip(input_paths, per_file_lookup):
            row = row_map.get(unique_id)
            if row is not None:
                source_rows.append(row)
                source_results.append(
                    {
                        "file": str(input_path),
                        "judgement": row.get("judgement"),
                        "correct": bool(row.get("correct")),
                        "confidence": int(row.get("confidence", 0) or 0),
                        "prediction": row.get("prediction", ""),
                        "model_answer": row.get("model_answer", ""),
                    }
                )
            else:
                source_results.append(
                    {
                        "file": str(input_path),
                        "judgement": "MissingPrediction",
                        "correct": False,
                        "confidence": 0,
                        "prediction": "",
                        "model_answer": "",
                    }
                )

        best_row = choose_best_row(source_rows)
        is_correct = any(bool(row.get("correct")) for row in source_rows)
        merged_row = {
            "id": unique_id,
            "question": question["question"],
            "answer": question["answer"],
            "prediction": best_row.get("prediction", ""),
            "model_answer": best_row.get("model_answer", ""),
            "reasoning": best_row.get("reasoning", ""),
            "confidence": int(best_row.get("confidence", 0) or 0),
            "correct": is_correct,
            "judgement": "Correct"
            if is_correct
            else best_row.get("judgement", "MissingPrediction"),
            "best_from": best_row.get("__source_file", ""),
            "source_results": source_results,
        }
        if "judge_mode" in best_row:
            merged_row["judge_mode"] = best_row["judge_mode"]
        if "judge_error" in best_row:
            merged_row["judge_error"] = best_row["judge_error"]
        merged_rows.append(merged_row)

    metrics = compute_metrics_from_scored_rows(merged_rows, total_questions=len(questions))
    per_file_brief = [
        {
            "input_file": summary["input_file"],
            "accuracy": summary["accuracy"],
            "judged_predictions": summary["judged_predictions"],
            "available_predictions": summary["available_predictions"],
        }
        for summary in per_file_summaries
    ]
    summary = {
        "benchmark": args.benchmark_name,
        "judge_model": args.judge_model,
        "judge_base_url": args.resolved_judge_base_url,
        "num_files": len(input_paths),
        "input_files": [str(path) for path in input_paths],
        "total_questions": len(questions),
        "per_file": per_file_brief,
        f"pass@{len(input_paths)}": metrics["accuracy"],
        **metrics,
    }
    return merged_rows, summary


def main() -> None:
    args = parse_args()
    if args.num_workers < 1:
        raise ValueError("--num_workers must be at least 1.")
    if args.judge_attempts < 1:
        raise ValueError("--judge_attempts must be at least 1.")
    if args.judge_retry_delay_s < 0:
        raise ValueError("--judge_retry_delay_s must be non-negative.")
    if not args.input_file:
        raise ValueError("At least one --input_file is required.")
    if len(args.input_file) > 1 and any([args.cache_output, args.scored_output, args.summary_output]):
        raise ValueError(
            "--cache_output/--scored_output/--summary_output only support a single --input_file."
        )

    args.benchmark_name = normalize_benchmark_name(args.benchmark)
    benchmark_config = BENCHMARK_CONFIGS[args.benchmark_name]
    args.judge_prompt = benchmark_config.judge_prompt
    args.eval_data_path_resolved = Path(
        args.eval_data_path or benchmark_config.default_eval_data_path
    )
    resolve_client_kwargs(args)

    questions = load_questions(
        eval_data_path=args.eval_data_path_resolved,
        dataset_name=args.dataset,
    )
    input_paths = [Path(path) for path in args.input_file]

    scored_rows_list: list[list[dict[str, Any]]] = []
    per_file_summaries: list[dict[str, Any]] = []
    for input_path in input_paths:
        scored_rows, summary, output_paths = evaluate_single_input(
            input_path=input_path,
            args=args,
            questions=questions,
        )
        source_file = str(input_path)
        for row in scored_rows:
            row["__source_file"] = source_file
        scored_rows_list.append(scored_rows)
        per_file_summaries.append(summary)

    if len(input_paths) > 1:
        passk_scored_output, passk_summary_output = derive_passk_output_paths(input_paths, args)
        merged_rows, merged_summary = build_passk_outputs(
            questions=questions,
            input_paths=input_paths,
            scored_rows_list=scored_rows_list,
            per_file_summaries=per_file_summaries,
            args=args,
        )
        for row in merged_rows:
            row.pop("__source_file", None)
        write_jsonl(passk_scored_output, merged_rows)
        passk_summary_output.parent.mkdir(parents=True, exist_ok=True)
        passk_summary_output.write_text(
            json.dumps(merged_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Pass@{len(input_paths)} rows : {passk_scored_output}")
        print(f"Pass@{len(input_paths)} summary : {passk_summary_output}")
        print(
            f"Pass@{len(input_paths)} : "
            f"{merged_summary[f'pass@{len(input_paths)}']}%"
        )


if __name__ == "__main__":
    main()
