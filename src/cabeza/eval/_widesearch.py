#!/usr/bin/env python3
"""Official-style WideSearch evaluator for SAM.

This script ports the public WideSearch evaluation logic from the official
ByteDance-Seed/WideSearch repository to the local SAM result format.

Key behaviors aligned with the official implementation:
- Load per-instance structured evaluation rules from `golden_answers`
- Load the gold CSV table for each instance
- Extract a single Markdown table from model predictions
- Normalize column names and optionally align them with an LLM
- Score rows/items using the official preprocessing / metric rules
- Report `score`, `precision_by_row`, `recall_by_row`, `f1_by_row`,
  `precision_by_item`, `recall_by_item`, and `f1_by_item`

Only WideSearch uses this evaluator. BrowseComp / BrowseCompPlus continue to use
the existing judge pipeline unchanged.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import threading
import traceback
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlparse

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

try:
    import dateparser
except ImportError:  # pragma: no cover - optional dependency
    dateparser = None


DEFAULT_GOLD_BASE_URLS = (
    "https://hf-mirror.com/datasets/ByteDance-Seed/WideSearch/resolve/main/widesearch_gold",
    "https://huggingface.co/datasets/ByteDance-Seed/WideSearch/resolve/main/widesearch_gold",
)
DEFAULT_JUDGE_MODEL = "Qwen3-32B"

# Resolve bundled WideSearch eval/gold paths against the cabeza dataset registry.
from cabeza.datasets.registry import get_spec, resolve_default_path as _resolve_path


def _default_ws_eval_path() -> str:
    spec = get_spec("ws")
    return _resolve_path(spec) or ""


def _default_ws_gold_dir() -> str:
    eval_path = _default_ws_eval_path()
    if not eval_path:
        return ""
    return str(Path(eval_path).parent / "gold")


def sanitize_tag(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "default"


def parse_date_like(value: Any):
    text = str(value).strip()
    if not text:
        return None
    if dateparser is not None:
        return dateparser.parse(text, settings={"PREFER_DAY_OF_MONTH": "first"})
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def norm_column(col: str) -> str:
    return col.strip().lower().replace(" ", "")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_json_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return json.loads(value)
    raise TypeError(f"Unsupported JSON payload type: {type(value)}")


def last_assistant_message(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages or []):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            return str(content)
    return ""


@dataclass
class WideSearchQuery:
    instance_id: str
    question: str
    evaluation: dict[str, Any]
    language: str = ""


@dataclass
class WideSearchResponse:
    instance_id: str
    response: str

    def extract_dataframe(self) -> Optional[pd.DataFrame]:
        response_df: Optional[pd.DataFrame] = None

        markdown_blocks = re.findall(
            r"```(?:markdown)?\s*(.*?)```",
            self.response,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if not markdown_blocks:
            pipe_positions = [m.start() for m in re.finditer(r"\|", self.response)]
            if len(pipe_positions) >= 4:
                first_pipe = pipe_positions[0]
                last_pipe = pipe_positions[-1]
                start = self.response.rfind("\n", 0, first_pipe)
                start = 0 if start == -1 else start
                end = self.response.find("\n", last_pipe)
                end = len(self.response) if end == -1 else end
                table_candidate = self.response[start:end]
                markdown_blocks = re.findall(r"((?:\|.*\n?)+)", table_candidate)

        if not markdown_blocks:
            return None

        markdown_str = markdown_blocks[0].strip()
        lines = [line.strip() for line in markdown_str.splitlines() if line.strip()]
        if not lines:
            return None

        cleaned_lines: list[str] = []
        for idx, line in enumerate(lines):
            if "|" not in line:
                continue
            if idx > 0 and set(line.strip()).issubset(set("|- :")):
                continue
            cleaned_lines.append("|".join(part.strip() for part in line.split("|")))

        if not cleaned_lines:
            return None

        try:
            response_df = pd.read_csv(StringIO("\n".join(cleaned_lines)), sep="|")
        except Exception:
            return None

        response_df = response_df.loc[:, ~response_df.columns.str.startswith("Unnamed")]
        return response_df


@dataclass
class EvaluationResult:
    instance_id: str
    score: float = 0.0
    precision_by_row: float = 0.0
    recall_by_row: float = 0.0
    f1_by_row: float = 0.0
    precision_by_item: float = 0.0
    recall_by_item: float = 0.0
    f1_by_item: float = 0.0
    msg: str = ""


class JudgeClient:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        timeout: float = 120.0,
        max_tokens: int = 4096,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.base_url = base_url
        self.disable_thinking = "localhost" in base_url or "127.0.0.1" in base_url
        self._thread_local = threading.local()

    def _get_client(self) -> OpenAI:
        client = getattr(self._thread_local, "client", None)
        if client is None:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self._thread_local.client = client
        return client

    def complete(self, prompt: str) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "timeout": self.timeout,
        }
        if self.disable_thinking:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        response = self._get_client().chat.completions.create(
            **kwargs,
        )
        message = response.choices[0].message
        content = (
            message.content
            or getattr(message, "reasoning_content", None)
            or getattr(message, "reasoning", None)
            or ""
        )
        return content or ""


def resolve_judge_config(
    api_dict_path: Path,
    judge_model: str,
    judge_base_url: str,
    judge_api_key: str,
) -> tuple[str, str]:
    base_url = judge_base_url
    api_key = judge_api_key

    if api_dict_path.exists():
        api_dict = json.loads(api_dict_path.read_text(encoding="utf-8"))
        if not base_url and judge_model in api_dict:
            base_url = api_dict[judge_model].get("base_url", "")
        if not api_key and judge_model in api_dict:
            api_key = api_dict[judge_model].get("api_key", "")

    if not base_url:
        raise ValueError(
            f"Judge model '{judge_model}' is missing a base URL. Pass --judge_base_url "
            f"or add it to {api_dict_path}."
        )
    if not api_key:
        api_key = "EMPTY"

    return base_url, api_key


class GoldAnswerStore:
    def __init__(
        self,
        gold_dir: Path,
        base_urls: tuple[str, ...],
        request_timeout_s: float,
    ) -> None:
        self.gold_dir = gold_dir
        self.base_urls = base_urls
        self.request_timeout_s = request_timeout_s
        self.gold_dir.mkdir(parents=True, exist_ok=True)

    def get_csv_path(self, instance_id: str) -> Path:
        local_path = self.gold_dir / f"{instance_id}.csv"
        if local_path.exists():
            return local_path

        errors: list[str] = []
        for base_url in self.base_urls:
            url = f"{base_url.rstrip('/')}/{instance_id}.csv"
            try:
                request = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"},
                )
                with urllib.request.urlopen(request, timeout=self.request_timeout_s) as r:
                    content = r.read()
                local_path.write_bytes(content)
                return local_path
            except Exception as exc:  # pragma: no cover - network dependent
                errors.append(f"{url}: {exc}")

        joined = "\n".join(errors)
        raise FileNotFoundError(
            f"Unable to fetch gold CSV for {instance_id}. Tried:\n{joined}"
        )

    def load_answer(
        self,
        instance_id: str,
        required_columns: list[str],
    ) -> pd.DataFrame:
        csv_path = self.get_csv_path(instance_id)
        answer_df = pd.read_csv(csv_path, encoding="utf-8-sig")
        answer_df.columns = [norm_column(col) for col in answer_df.columns]
        missing = [col for col in required_columns if col not in answer_df.columns]
        if missing:
            raise ValueError(
                f"Gold CSV {csv_path} missing required columns {missing}; "
                f"available={answer_df.columns.tolist()}"
            )
        return answer_df[required_columns].copy()


preprocess_function_registry: dict[str, Callable[[str], str]] = {}
metric_function_registry: dict[str, Callable[..., tuple[float, str]]] = {}


def register_preprocess_function(func: Callable[[str], str]) -> Callable[[str], str]:
    preprocess_function_registry[func.__name__] = func
    return func


def register_metric_function(
    func: Callable[..., tuple[float, str]]
) -> Callable[..., tuple[float, str]]:
    metric_function_registry[func.__name__] = func
    return func


@register_preprocess_function
def extract_number(content: str) -> str:
    numbers = re.findall(
        r"[-+]?\d*\.\d+%?|[-+]?\d+\.?\d*%?",
        str(content).replace(",", ""),
    )
    if not numbers:
        return "NULL"
    return numbers[0]


@register_preprocess_function
def norm_str(content: str) -> str:
    return str(content).lower().strip().replace(" ", "").replace("*", "")


@register_preprocess_function
def norm_date(content: str) -> str:
    parsed = parse_date_like(content)
    if parsed is None:
        return str(content)
    return parsed.strftime("%Y-%m-%d")


@register_metric_function
def exact_match(response: str, target: str) -> tuple[float, str]:
    if response.lower() == target.lower():
        return 1.0, f"exact match, response: {response}, target: {target}"
    return 0.0, f"exact not match, response: {response}, target: {target}"


@register_metric_function
def url_match(response: str, target: str) -> tuple[float, str]:
    pattern = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    response_urls = [urlparse(url).netloc for url in pattern.findall(response)]
    target_urls = [urlparse(url).netloc for url in pattern.findall(target)]
    if set(response_urls) == set(target_urls):
        return 1.0, f"url match, response: {response}, target: {target}"
    return 0.0, f"url not match, response: {response}, target: {target}"


@register_metric_function
def in_match(response: str, target: str) -> tuple[float, str]:
    if response in target:
        return 1.0, f"response in target, response: {response}, target: {target}"
    return 0.0, f"response not in target, response: {response}, target: {target}"


@register_metric_function
def number_near(response: str, target: str, criterion: float) -> tuple[float, str]:
    def parse_number(value: str) -> Optional[float]:
        if "%" in value:
            cleaned = value.replace("%", "")
            try:
                return float(cleaned) / 100.0
            except (TypeError, ValueError):
                return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    response_num = parse_number(response)
    target_num = parse_number(target)
    if response_num is None or target_num is None:
        if response_num is None and target_num is None and response == target:
            return 1.0, f"number equal, response: {response}, target: {target}"
        return 0.0, f"number not convertible, response: {response}, target: {target}"

    if abs(response_num - target_num) <= abs(target_num) * criterion:
        return (
            1.0,
            f"number near in range {criterion * 100}%, "
            f"response: {response_num}, target: {target_num}",
        )
    return 0.0, f"number not near, response: {response_num}, target: {target_num}"


@register_metric_function
def date_near(response: str, target: str) -> tuple[float, str]:
    response_date = parse_date_like(response)
    target_date = parse_date_like(target)
    if response_date is None or target_date is None:
        if response_date is None and target_date is None:
            return 1.0, f"date near, response: {response}, target: {target}"
        return 0.0, f"date not convertible, response: {response}, target: {target}"

    if abs((response_date - target_date).days) <= 31:
        return 1.0, f"date near, response: {response_date}, target: {target_date}"
    return 0.0, f"date not near, response: {response_date}, target: {target_date}"


PRIMARY_KEY_PREPROCESS_PROMPT = """Your task is to align two vocabularies. The inputs are the vocabulary to be aligned and the reference vocabulary respectively. Note that you need to perform semantic alignment (not positional alignment). If two strings are exactly the same, they must correspond to each other. These two strings are supposed to represent the same entity, with differences only in the expression forms and formats.

The vocabulary to be aligned is as follows:
{response}

The reference vocabulary is as follows:
{reference}

The alignment rules are as follows:
List the values in the vocabulary to be aligned one by one. If there is a value in the reference vocabulary that has the same meaning as this value, `transform` should be represented as the value from the reference vocabulary; otherwise, `transform` should be represented as the original value from the vocabulary to be aligned.

Note that `origin` must be taken from the vocabulary to be aligned keeping the original format, and `transform` must be taken from the reference vocabulary.

Please output the alignment results in the following format:
```json
{{
    "origin_str1": "transform_str1",
    "origin_str2": "transform_str2"
}}
```
"""


EVAL_COLUMN_PROMPT = """You are an expert in grading answers. Your task is to score the responses to a certain question. Below, you will be provided with a set of standard answers, a set of responses to be graded, and specific grading criteria.

Each answer and each response has an idx. Please score each pair of answers and responses in this set according to the following methods:
1. The score can only be 0 or 1.
2. A score of 1 indicates the response satisfies the criterion for that target.
3. Judge each idx independently.
4. Return ONLY the final results in Markdown JSON:
```json
{{
    "idx_0": 1,
    "idx_1": 0
}}
```

====== criterion-start ======
{criterion}
====== criterion-end ======

====== response-start ======
{response}
====== response-end ======
"""


def parse_markdown_json(completion: str) -> Optional[dict[str, Any]]:
    fenced = re.findall(r"```json\s*(.*?)```", completion, re.DOTALL | re.IGNORECASE)
    candidates = list(reversed(fenced))
    stripped = completion.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)

    for json_str in candidates:
        try:
            loaded = json.loads(json_str)
        except Exception:
            continue
        if isinstance(loaded, dict):
            return loaded
    return None


def normalize_binary_score(value: Any) -> int:
    if isinstance(value, dict):
        if "score" in value:
            return normalize_binary_score(value["score"])
        if "result" in value:
            return normalize_binary_score(value["result"])
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and int(value) in (0, 1):
        return int(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "correct"}:
            return 1
        if lowered in {"0", "false", "no", "incorrect"}:
            return 0
    return 0


def primary_key_preprocess(
    response: list[str],
    reference: list[str],
    judge_client: JudgeClient,
) -> dict[str, str]:
    prompt = PRIMARY_KEY_PREPROCESS_PROMPT.format(
        response=response,
        reference=reference,
    )
    result = judge_client.complete(prompt)
    mapping = parse_markdown_json(result)
    if not mapping:
        return {}
    return {str(k): str(v) for k, v in mapping.items()}


def llm_judge_column(
    response: list[str],
    target: list[str],
    criterion: str,
    judge_client: JudgeClient,
) -> tuple[list[int], list[str]]:
    payload = {
        f"idx_{idx}": {"response": resp, "target": tar}
        for idx, (resp, tar) in enumerate(zip(response, target))
    }
    prompt = EVAL_COLUMN_PROMPT.format(
        criterion=criterion,
        response=json.dumps(payload, ensure_ascii=False, indent=2),
    )
    result = judge_client.complete(prompt)
    score_dict = parse_markdown_json(result) or {}
    scores = [normalize_binary_score(score_dict.get(f"idx_{idx}", 0)) for idx in range(len(response))]
    msgs = [result] * len(response)
    return scores, msgs


def preprocess_call(content: str, preprocess_func_name: str) -> str:
    preprocess_func = preprocess_function_registry[preprocess_func_name]
    return preprocess_func(content)


def metric_call(
    response: str,
    target: str,
    criterion: Any,
    metric_func_name: str,
) -> tuple[float, str]:
    metric_func = metric_function_registry[metric_func_name]
    if metric_func_name in {"llm_judge", "number_near"}:
        return metric_func(response, target, criterion)
    return metric_func(response, target)


def calc_f1(precision: float, recall: float) -> float:
    epsilon = 1e-9
    if precision + recall <= epsilon:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def load_queries(eval_data_path: Path) -> dict[str, WideSearchQuery]:
    queries: dict[str, WideSearchQuery] = {}
    for row in read_jsonl(eval_data_path):
        instance_id = str(row.get("id") or row.get("instance_id") or "")
        if not instance_id:
            continue
        evaluation = ensure_json_obj(
            row.get("golden_answers") or row.get("evaluation") or row.get("answer")
        )
        queries[instance_id] = WideSearchQuery(
            instance_id=instance_id,
            question=str(row.get("question") or row.get("query") or ""),
            evaluation=evaluation,
            language=str(row.get("language") or ""),
        )
    return queries


def build_response_from_result(item: dict[str, Any]) -> WideSearchResponse:
    instance_id = str(item.get("id") or item.get("instance_id") or "")
    prediction = item.get("prediction")
    if isinstance(prediction, str) and prediction.strip():
        response_text = prediction
    else:
        response_text = last_assistant_message(item.get("messages") or [])
    return WideSearchResponse(instance_id=instance_id, response=response_text)


def evaluate_single_query(
    query: WideSearchQuery,
    result_item: dict[str, Any],
    gold_store: GoldAnswerStore,
    judge_client: JudgeClient,
    detail_csv_path: Optional[Path] = None,
) -> EvaluationResult:
    if result_item.get("error"):
        return EvaluationResult(
            instance_id=query.instance_id,
            msg=f"inference error: {result_item['error']}",
        )

    response = build_response_from_result(result_item)
    if not response.response.strip():
        return EvaluationResult(
            instance_id=query.instance_id,
            msg="empty prediction",
        )

    try:
        required_columns = [norm_column(col) for col in query.evaluation["required"]]
        unique_columns = [norm_column(col) for col in query.evaluation["unique_columns"]]
        answer_df = gold_store.load_answer(query.instance_id, required_columns)
        answer_df.columns = [norm_column(col) for col in answer_df.columns]

        response_df = response.extract_dataframe()
        if response_df is None:
            return EvaluationResult(instance_id=query.instance_id, msg="response_df is None")

        response_df.columns = [norm_column(col) for col in response_df.columns]

        if set(required_columns) != set(response_df.columns):
            column_map = primary_key_preprocess(
                response=response_df.columns.tolist(),
                reference=required_columns,
                judge_client=judge_client,
            )
            if column_map:
                response_df.rename(columns=column_map, inplace=True)

        if set(required_columns) != set(response_df.columns):
            return EvaluationResult(
                instance_id=query.instance_id,
                msg=(
                    f"required_columns {required_columns} != "
                    f"response_df {response_df.columns.tolist()}"
                ),
            )

        answer_df = answer_df[required_columns].copy()
        response_df = response_df[required_columns].copy()

        for col in required_columns:
            try:
                answer_type = answer_df[col].dtype
                response_type = response_df[col].dtype
            except Exception:
                answer_type = None
                response_type = None

            if (response_type == float and answer_type == int) or (
                response_type == int and answer_type == float
            ):
                if response_type == int:
                    response_df[col] = response_df[col].astype(float)
                elif answer_type == int:
                    answer_df[col] = answer_df[col].astype(float)

            answer_df[col] = answer_df[col].astype(str)
            response_df[col] = response_df[col].astype(str)

        response_df.drop_duplicates(subset=unique_columns, inplace=True)
        answer_df.drop_duplicates(subset=unique_columns, inplace=True)

        eval_pipeline = {
            norm_column(col): value
            for col, value in query.evaluation["eval_pipeline"].items()
        }

        for col in unique_columns:
            item = eval_pipeline.get(col)
            if not item:
                continue
            metric_names = item.get("metric", [])
            if "llm_judge" in metric_names or "exact_match" in metric_names:
                primary_key_map = primary_key_preprocess(
                    response_df[col].tolist(),
                    answer_df[col].tolist(),
                    judge_client=judge_client,
                )
                if primary_key_map:
                    response_df[f"{col}_before_map"] = response_df[col]
                    response_df[col] = response_df[col].apply(
                        lambda x: primary_key_map.get(x, x)
                    )

        for col, item in eval_pipeline.items():
            preprocess_names = item.get("preprocess", [])
            for preprocess_func_name in preprocess_names:
                response_df[col] = response_df[col].apply(
                    lambda x: preprocess_call(x, preprocess_func_name)
                )
                answer_df[col] = answer_df[col].apply(
                    lambda x: preprocess_call(x, preprocess_func_name)
                )

        score = 0.0
        if answer_df.shape == response_df.shape:
            gt_sorted = answer_df.sort_values(by=required_columns).reset_index(drop=True)
            pred_sorted = response_df.sort_values(by=required_columns).reset_index(drop=True)
            if gt_sorted.equals(pred_sorted):
                score = 1.0

        df_inner = pd.merge(
            answer_df,
            response_df,
            on=unique_columns,
            how="inner",
            suffixes=("_query", "_response"),
        )

        answer_outer = answer_df.copy()
        answer_outer["exist_flag_gt"] = 1
        response_outer = response_df.copy()
        response_outer["exist_flag_response"] = 1

        df_outer = pd.merge(
            answer_outer,
            response_outer,
            on=unique_columns,
            how="outer",
            suffixes=("_query", "_response"),
        )
        df_outer_wo_inner = df_outer[
            df_outer["exist_flag_gt"].isna() | df_outer["exist_flag_response"].isna()
        ]

        df_inner_score = pd.DataFrame(index=df_inner.index)
        df_inner_msg = pd.DataFrame(index=df_inner.index)

        for col in required_columns:
            if col in unique_columns:
                df_inner_score[f"{col}_exact_match"] = 1.0
                df_inner_msg[f"{col}_exact_match_eval_msg"] = "key_match"
                continue

            item = eval_pipeline[col]
            metric_names = item.get("metric", [])
            criterion = item.get("criterion")
            for metric_name in metric_names:
                if metric_name == "llm_judge":
                    score_list, msg_list = llm_judge_column(
                        df_inner[f"{col}_response"].tolist(),
                        df_inner[f"{col}_query"].tolist(),
                        criterion,
                        judge_client=judge_client,
                    )
                    metric_info_series = pd.Series(
                        list(zip(score_list, msg_list)),
                        index=df_inner.index,
                    )
                else:
                    metric_info_series = df_inner.apply(
                        lambda x: metric_call(
                            x[f"{col}_response"],
                            x[f"{col}_query"],
                            criterion,
                            metric_name,
                        ),
                        axis=1,
                    )

                df_inner_score[f"{col}_{metric_name}"] = metric_info_series.apply(lambda x: x[0])
                df_inner_msg[f"{col}_{metric_name}_eval_msg"] = metric_info_series.apply(
                    lambda x: x[1]
                )

        if detail_csv_path is not None:
            detail_csv_path.parent.mkdir(parents=True, exist_ok=True)
            result_df = pd.concat([df_inner, df_inner_score, df_inner_msg], axis=1)
            result_df = pd.concat([result_df, df_outer_wo_inner], axis=0)
            result_columns = result_df.columns.tolist()
            key_cols = (
                unique_columns
                + [f"{col}_before_map" for col in unique_columns]
                + ["exist_flag_gt", "exist_flag_response"]
            )
            cols1 = sorted([col for col in result_columns if col in key_cols])
            cols2 = sorted([col for col in result_columns if col not in key_cols])
            result_df = result_df[cols1 + cols2]
            result_df.to_csv(detail_csv_path, index=False, encoding="utf-8-sig")

        row_scores = df_inner_score.min(axis=1) if not df_inner_score.empty else pd.Series(dtype=float)
        tp_by_row = float(row_scores.sum()) if not row_scores.empty else 0.0
        tp_by_item = float(df_inner_score.sum().sum()) if not df_inner_score.empty else 0.0

        num_pred_rows = len(response_df)
        num_gt_rows = len(answer_df)
        num_pred_items = num_pred_rows * len(required_columns)
        num_gt_items = num_gt_rows * len(required_columns)

        precision_by_row = tp_by_row / num_pred_rows if num_pred_rows > 0 else 0.0
        recall_by_row = tp_by_row / num_gt_rows if num_gt_rows > 0 else 0.0
        precision_by_item = tp_by_item / num_pred_items if num_pred_items > 0 else 0.0
        recall_by_item = tp_by_item / num_gt_items if num_gt_items > 0 else 0.0
        f1_by_row = calc_f1(precision_by_row, recall_by_row)
        f1_by_item = calc_f1(precision_by_item, recall_by_item)

        msg = df_inner_score.to_string()
        if (
            precision_by_item == recall_by_item == f1_by_item == 1.0
            and precision_by_row == recall_by_row == f1_by_row == 1.0
        ):
            msg += "\nAll items match perfectly."
            score = 1.0

        return EvaluationResult(
            instance_id=query.instance_id,
            score=score,
            precision_by_row=precision_by_row,
            recall_by_row=recall_by_row,
            f1_by_row=f1_by_row,
            precision_by_item=precision_by_item,
            recall_by_item=recall_by_item,
            f1_by_item=f1_by_item,
            msg=msg,
        )
    except Exception:
        return EvaluationResult(
            instance_id=query.instance_id,
            msg=f"evaluator error:\n{traceback.format_exc()}",
        )


def summarize(results: list[EvaluationResult]) -> dict[str, Any]:
    metric_names = [
        "score",
        "precision_by_row",
        "recall_by_row",
        "f1_by_row",
        "precision_by_item",
        "recall_by_item",
        "f1_by_item",
    ]

    summary: dict[str, Any] = {
        "num_instances": len(results),
        "num_errors": sum(
            1
            for item in results
            if item.msg
            and (
                "error" in item.msg.lower()
                or "response_df is none" in item.msg.lower()
                or "empty prediction" in item.msg.lower()
                or "required_columns" in item.msg.lower()
            )
        ),
    }
    for name in metric_names:
        values = [getattr(item, name) for item in results]
        avg_value = float(sum(values) / len(values)) if values else 0.0
        summary[name] = {
            "avg_n": avg_value,
            "max_n": avg_value,
            "min_n": avg_value,
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate WideSearch predictions with the official-style table scorer."
    )
    parser.add_argument(
        "--input_file",
        action="append",
        default=[],
        help="Path to the prediction JSONL. Pass multiple times for best-of-k aggregation.",
    )
    parser.add_argument(
        "--eval_data_path",
        default=_default_ws_eval_path(),
        help="Path to WideSearch eval JSONL containing questions and golden_answers.",
    )
    parser.add_argument(
        "--gold_dir",
        default=_default_ws_gold_dir(),
        help="Directory containing WideSearch gold CSV files.",
    )
    parser.add_argument(
        "--gold_base_url",
        action="append",
        default=[],
        help=(
            "Optional base URL for downloading gold CSV files. Can be passed multiple times. "
            "Defaults to hf-mirror first, then Hugging Face."
        ),
    )
    parser.add_argument(
        "--judge_model",
        default=DEFAULT_JUDGE_MODEL,
        help="OpenAI-compatible judge model used for column alignment and llm_judge.",
    )
    parser.add_argument(
        "--judge_base_url",
        default="",
        help="Override the judge model base URL.",
    )
    parser.add_argument(
        "--judge_api_key",
        default="",
        help="Override the judge model API key.",
    )
    parser.add_argument(
        "--api_dict_path",
        default="data/api_dict.json",
        help="Path to API endpoint definitions.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of evaluation worker threads.",
    )
    parser.add_argument(
        "--request_timeout_s",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds for judge and gold downloads.",
    )
    parser.add_argument(
        "--detail_dir",
        default="",
        help="Optional directory for per-instance detailed CSV / JSON outputs.",
    )
    parser.add_argument(
        "--scored_output",
        default="",
        help="Optional JSONL output path for per-instance summary scores.",
    )
    parser.add_argument(
        "--summary_output",
        default="",
        help="Optional JSON output path for aggregate metrics.",
    )
    parser.add_argument(
        "--passk_scored_output",
        default="",
        help="Optional merged best-of-k JSONL path when multiple --input_file are passed.",
    )
    parser.add_argument(
        "--passk_summary_output",
        default="",
        help="Optional merged best-of-k summary JSON path when multiple --input_file are passed.",
    )
    parser.add_argument(
        "--output_tag",
        default="",
        help="Optional tag appended to default output filenames.",
    )
    return parser.parse_args()


def derive_single_output_paths(
    input_path: Path,
    args: argparse.Namespace,
) -> tuple[Path, Path, Path]:
    tag = sanitize_tag(args.output_tag or args.judge_model)
    detail_dir = (
        Path(args.detail_dir)
        if args.detail_dir
        else input_path.with_name(f"{input_path.stem}__{tag}_ws_details")
    )
    scored_output = (
        Path(args.scored_output)
        if args.scored_output
        else input_path.with_name(f"{input_path.stem}__{tag}_ws_scored.jsonl")
    )
    summary_output = (
        Path(args.summary_output)
        if args.summary_output
        else input_path.with_name(f"{input_path.stem}__{tag}_ws_summary.json")
    )
    return detail_dir, scored_output, summary_output


def derive_passk_output_paths(
    input_paths: list[Path],
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    common_prefix = sanitize_tag(input_paths[0].stem)
    tag = sanitize_tag(args.output_tag or args.judge_model)
    scored_output = (
        Path(args.passk_scored_output)
        if args.passk_scored_output
        else input_paths[0].with_name(f"{common_prefix}__{tag}_ws_pass@{len(input_paths)}_scored.jsonl")
    )
    summary_output = (
        Path(args.passk_summary_output)
        if args.passk_summary_output
        else input_paths[0].with_name(f"{common_prefix}__{tag}_ws_pass@{len(input_paths)}_summary.json")
    )
    return scored_output, summary_output


def evaluate_single_input(
    input_path: Path,
    queries: dict[str, WideSearchQuery],
    gold_store: GoldAnswerStore,
    judge_client: JudgeClient,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    detail_dir, scored_output, summary_output = derive_single_output_paths(input_path, args)
    prediction_rows = read_jsonl(input_path)

    def evaluate_row(item: dict[str, Any]) -> dict[str, Any]:
        instance_id = str(item.get("id") or item.get("instance_id") or "")
        if not instance_id:
            result = EvaluationResult(instance_id="", msg="missing instance_id")
            return asdict(result)

        query = queries.get(instance_id)
        if query is None:
            fallback_eval = item.get("answer") or item.get("golden_answers") or item.get("evaluation")
            if not fallback_eval:
                result = EvaluationResult(
                    instance_id=instance_id,
                    msg=f"instance_id {instance_id} not found in {args.eval_data_path}",
                )
                return asdict(result)
            query = WideSearchQuery(
                instance_id=instance_id,
                question=str(item.get("question") or ""),
                evaluation=ensure_json_obj(fallback_eval),
            )

        detail_csv_path = detail_dir / f"{instance_id}.csv"
        eval_result = evaluate_single_query(
            query=query,
            result_item=item,
            gold_store=gold_store,
            judge_client=judge_client,
            detail_csv_path=detail_csv_path,
        )
        detail_json_path = detail_dir / f"{instance_id}.json"
        detail_json_path.parent.mkdir(parents=True, exist_ok=True)
        detail_json_path.write_text(
            json.dumps(asdict(eval_result), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return asdict(eval_result)

    scored_rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(evaluate_row, row) for row in prediction_rows]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"Evaluating {input_path.name}",
        ):
            scored_rows.append(future.result())

    scored_rows.sort(key=lambda x: x.get("instance_id", ""))
    write_jsonl(scored_output, scored_rows)

    summary = summarize([EvaluationResult(**row) for row in scored_rows])
    summary.update(
        {
            "input_file": str(input_path),
            "judge_model": args.judge_model,
            "judge_base_url": judge_client.base_url,
            "scored_output": str(scored_output),
            "detail_dir": str(detail_dir),
        }
    )
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Input       : {input_path}")
    print(f"Scored rows : {scored_output}")
    print(f"Detail dir  : {detail_dir}")
    print(f"Summary     : {summary_output}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    return scored_rows, summary


def choose_best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return max(
        rows,
        key=lambda row: (
            float(row.get("score", 0.0) or 0.0),
            float(row.get("f1_by_row", 0.0) or 0.0),
            float(row.get("f1_by_item", 0.0) or 0.0),
            float(row.get("recall_by_row", 0.0) or 0.0),
            float(row.get("precision_by_row", 0.0) or 0.0),
        ),
    )


def build_passk_outputs(
    queries: dict[str, WideSearchQuery],
    input_paths: list[Path],
    per_file_rows: list[list[dict[str, Any]]],
    per_file_summaries: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row_maps = [
        {str(row.get("instance_id") or ""): row for row in rows if row.get("instance_id")}
        for rows in per_file_rows
    ]

    merged_rows: list[dict[str, Any]] = []
    for instance_id in queries.keys():
        source_rows: list[dict[str, Any]] = []
        source_results: list[dict[str, Any]] = []
        for input_path, row_map in zip(input_paths, row_maps):
            row = row_map.get(instance_id)
            if row is not None:
                enriched = dict(row)
                enriched["__source_file"] = str(input_path)
                source_rows.append(enriched)
                source_results.append(
                    {
                        "file": str(input_path),
                        "score": float(row.get("score", 0.0) or 0.0),
                        "f1_by_row": float(row.get("f1_by_row", 0.0) or 0.0),
                        "f1_by_item": float(row.get("f1_by_item", 0.0) or 0.0),
                        "msg": row.get("msg", ""),
                    }
                )
            else:
                source_results.append(
                    {
                        "file": str(input_path),
                        "score": 0.0,
                        "f1_by_row": 0.0,
                        "f1_by_item": 0.0,
                        "msg": "missing prediction",
                    }
                )

        best_row = choose_best_row(source_rows)
        merged_row = {
            "instance_id": instance_id,
            "score": float(best_row.get("score", 0.0) or 0.0),
            "precision_by_row": float(best_row.get("precision_by_row", 0.0) or 0.0),
            "recall_by_row": float(best_row.get("recall_by_row", 0.0) or 0.0),
            "f1_by_row": float(best_row.get("f1_by_row", 0.0) or 0.0),
            "precision_by_item": float(best_row.get("precision_by_item", 0.0) or 0.0),
            "recall_by_item": float(best_row.get("recall_by_item", 0.0) or 0.0),
            "f1_by_item": float(best_row.get("f1_by_item", 0.0) or 0.0),
            "msg": best_row.get("msg", "missing prediction"),
            "best_from": best_row.get("__source_file", ""),
            "source_results": source_results,
        }
        merged_rows.append(merged_row)

    merged_summary = summarize(
        [
            EvaluationResult(
                instance_id=row["instance_id"],
                score=row["score"],
                precision_by_row=row["precision_by_row"],
                recall_by_row=row["recall_by_row"],
                f1_by_row=row["f1_by_row"],
                precision_by_item=row["precision_by_item"],
                recall_by_item=row["recall_by_item"],
                f1_by_item=row["f1_by_item"],
                msg=row["msg"],
            )
            for row in merged_rows
        ]
    )
    merged_summary.update(
        {
            "num_files": len(input_paths),
            "input_files": [str(path) for path in input_paths],
            "judge_model": args.judge_model,
            "per_file": [
                {
                    "input_file": summary.get("input_file", ""),
                    "score": summary.get("score", {}).get("avg_n", 0.0),
                    "f1_by_row": summary.get("f1_by_row", {}).get("avg_n", 0.0),
                    "f1_by_item": summary.get("f1_by_item", {}).get("avg_n", 0.0),
                }
                for summary in per_file_summaries
            ],
        }
    )
    merged_summary[f"pass@{len(input_paths)}"] = round(
        float(merged_summary.get("score", {}).get("avg_n", 0.0)) * 100,
        2,
    )
    return merged_rows, merged_summary


def main() -> None:
    args = parse_args()
    if not args.input_file:
        raise ValueError("At least one --input_file is required.")
    if len(args.input_file) > 1 and any([args.detail_dir, args.scored_output, args.summary_output]):
        raise ValueError(
            "--detail_dir/--scored_output/--summary_output only support a single --input_file."
        )

    eval_data_path = Path(args.eval_data_path)
    api_dict_path = Path(args.api_dict_path)

    judge_base_url, judge_api_key = resolve_judge_config(
        api_dict_path=api_dict_path,
        judge_model=args.judge_model,
        judge_base_url=args.judge_base_url,
        judge_api_key=args.judge_api_key,
    )
    judge_client = JudgeClient(
        model=args.judge_model,
        base_url=judge_base_url,
        api_key=judge_api_key,
        timeout=args.request_timeout_s,
    )
    gold_store = GoldAnswerStore(
        gold_dir=Path(args.gold_dir),
        base_urls=tuple(args.gold_base_url) if args.gold_base_url else DEFAULT_GOLD_BASE_URLS,
        request_timeout_s=args.request_timeout_s,
    )

    queries = load_queries(eval_data_path)
    input_paths = [Path(path) for path in args.input_file]
    per_file_rows: list[list[dict[str, Any]]] = []
    per_file_summaries: list[dict[str, Any]] = []

    for input_path in input_paths:
        scored_rows, summary = evaluate_single_input(
            input_path=input_path,
            queries=queries,
            gold_store=gold_store,
            judge_client=judge_client,
            args=args,
        )
        per_file_rows.append(scored_rows)
        per_file_summaries.append(summary)

    if len(input_paths) > 1:
        passk_scored_output, passk_summary_output = derive_passk_output_paths(input_paths, args)
        merged_rows, merged_summary = build_passk_outputs(
            queries=queries,
            input_paths=input_paths,
            per_file_rows=per_file_rows,
            per_file_summaries=per_file_summaries,
            args=args,
        )
        write_jsonl(passk_scored_output, merged_rows)
        passk_summary_output.parent.mkdir(parents=True, exist_ok=True)
        passk_summary_output.write_text(
            json.dumps(merged_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Pass@{len(input_paths)} rows : {passk_scored_output}")
        print(f"Pass@{len(input_paths)} summary : {passk_summary_output}")
        print(f"Pass@{len(input_paths)} : {merged_summary[f'pass@{len(input_paths)}']}%")


if __name__ == "__main__":
    main()
