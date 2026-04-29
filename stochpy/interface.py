from __future__ import annotations

import json
import re
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError

from .exceptions import ContractException

T = TypeVar("T", bound="BaseModel")


def _extract_json(raw: str) -> str:
    """Extract the first {...} or [...] block from raw model output."""
    # try to find a JSON object
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return match.group(0)
    # fall back to array
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        return match.group(0)
    # last resort — treat entire string as JSON
    return raw.strip()


def _build_schema_hint(schema: dict) -> str:
    """Return a concise instruction telling the model what JSON to emit."""
    return (
        "Respond ONLY with valid JSON matching this schema. "
        "No prose before or after.\n"
        f"Schema: {json.dumps(schema, indent=2)}"
    )


def interface(cls: Type[T]) -> Type[T]:
    """
    Decorator — marks a Pydantic BaseModel as a StochPy interface contract.

    Adds:
      cls.parse_llm_output(raw: str) -> instance
      cls.__stochpy_schema__         -> JSON schema dict
      cls.__stochpy_schema_hint__    -> prompt instruction string
    """
    if not issubclass(cls, BaseModel):
        raise TypeError(
            f"@interface requires a Pydantic BaseModel subclass, got {cls}"
        )

    schema = cls.model_json_schema()

    @classmethod  # type: ignore[misc]
    def parse_llm_output(cls_inner: Type[T], raw: str) -> T:
        json_str = _extract_json(raw)
        try:
            return cls_inner.model_validate_json(json_str)
        except (ValidationError, json.JSONDecodeError) as exc:
            raise ContractException(
                f"Output did not match interface {cls_inner.__name__}: {exc}",
                raw_output=raw,
            ) from exc

    cls.parse_llm_output = parse_llm_output  # type: ignore[attr-defined]
    cls.__stochpy_schema__ = schema  # type: ignore[attr-defined]
    cls.__stochpy_schema_hint__ = _build_schema_hint(schema)  # type: ignore[attr-defined]

    return cls


def is_interface(cls: Any) -> bool:
    """Return True if cls was decorated with @interface."""
    return isinstance(cls, type) and hasattr(cls, "__stochpy_schema__")
