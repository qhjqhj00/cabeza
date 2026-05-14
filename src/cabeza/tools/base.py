import ast
import json
import re
from abc import ABC, abstractmethod
from typing import Optional


class BaseTool(ABC):
    name: str = ""
    description: str = ""
    parameters: dict = {}

    def __init__(self, name: str, description: str, parameters: dict):
        self.name = name
        self.description = description
        self.parameters = parameters

    @abstractmethod
    def execute(self, **kwargs) -> str:
        raise NotImplementedError

    def to_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_response_schema(self, *, strict: Optional[bool] = None) -> dict:
        schema = {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
        if strict is None:
            strict = getattr(self, "strict", None)
        if strict is not None:
            schema["strict"] = strict
        return schema

    @staticmethod
    def _strip_wrapping_quotes(text: str) -> str:
        text = text.strip()
        while len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
            text = text[1:-1].strip()
        return text

    @classmethod
    def coerce_str_list(
        cls,
        value,
        *,
        field_name: str = "value",
        extract_urls: bool = False,
    ) -> list[str]:
        if isinstance(value, (list, tuple, set)):
            items = list(value)
        elif isinstance(value, str):
            text = value.strip()
            if not text:
                raise ValueError(f"Invalid {field_name} format: expected string or array.")

            parsed = None
            for loader in (json.loads, ast.literal_eval):
                try:
                    parsed = loader(text)
                    break
                except Exception:
                    continue

            if isinstance(parsed, (list, tuple, set)):
                items = list(parsed)
            elif isinstance(parsed, str):
                items = [parsed]
            elif extract_urls:
                items = re.findall(r"https?://[^\s'\"<>\],]+", text)
                if not items:
                    items = [text]
            else:
                items = [text]
        else:
            raise ValueError(f"Invalid {field_name} format: expected string or array.")

        normalized: list[str] = []
        for item in items:
            if item is None:
                continue
            if not isinstance(item, str):
                item = str(item)
            item = cls._strip_wrapping_quotes(item)
            if item.lower().startswith("url:"):
                item = item.split(":", 1)[1].strip()
            if extract_urls:
                item = item.rstrip(",]")
            if item:
                normalized.append(item)

        if not normalized:
            raise ValueError(f"Invalid {field_name} format: expected at least one non-empty string.")
        return normalized
