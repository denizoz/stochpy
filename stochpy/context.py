from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, get_type_hints

from .exceptions import (
    ConfidenceException,
    ContractException,
    EvalAssertException,
    ModelException,
    ModelTimeoutException,
)
from .interface import is_interface
from .prompt_registry import Prompt
from .registry import ModelEntry, ModelRegistry, _AutoSelector, _FederatedSelector


def _resolve_model(model: Any, context_key: str) -> ModelEntry:
    """Resolve whatever was passed as model= to a concrete ModelEntry."""
    if isinstance(model, _AutoSelector):
        return model.resolve(context_key)
    if isinstance(model, ModelEntry):
        return model
    if isinstance(model, str):
        # bare litellm name
        return ModelEntry(name=model, litellm_name=model)
    raise TypeError(
        f"model= must be a ModelEntry, _AutoSelector, or litellm string, "
        f"got {type(model)}"
    )


def _inject_schema_hint(
    messages: list[dict], output_type: Any
) -> list[dict]:
    """Append the output schema instruction to the system message (or add one)."""
    if not is_interface(output_type):
        return messages

    hint = output_type.__stochpy_schema_hint__

    result = list(messages)
    if result and result[0]["role"] == "system":
        result[0] = {
            "role":    "system",
            "content": result[0]["content"] + "\n\n" + hint,
        }
    else:
        result.insert(0, {"role": "system", "content": hint})
    return result


def _flatten_arguments(bound: inspect.BoundArguments) -> dict[str, Any]:
    """Convert bound arguments to a flat dict for prompt rendering."""
    result: dict[str, Any] = {}
    for k, v in bound.arguments.items():
        result[k] = v
    return result


def _call_model(litellm_name: str, messages: list[dict]) -> str:
    """Call model via LiteLLM and return raw string content."""
    try:
        import litellm  # imported lazily — not required at import time
        response = litellm.completion(model=litellm_name, messages=messages)
        return response.choices[0].message.content
    except ImportError as exc:
        raise ImportError(
            "litellm is required to call models. "
            "Install it with: pip install litellm"
        ) from exc
    except Exception as exc:
        # wrap provider errors
        msg = str(exc).lower()
        if "timeout" in msg:
            raise ModelTimeoutException(str(exc)) from exc
        raise ModelException(str(exc)) from exc


def _record(model: ModelEntry, registry: Any, context_key: str, success: bool) -> None:
    if isinstance(registry, ModelRegistry):
        registry.record(context_key, model.name, success)


def context(
    model: Any,
    prompt: Prompt | str,
    eval: list[Callable] | None = None,
    threshold: float = 0.90,
    retries: int = 0,
    _registry: ModelRegistry | None = None,
) -> Callable:
    """
    Decorator factory.  Wraps a function stub into a stochastic LLM call.

    Args:
        model:     ModelEntry, _AutoSelector, or bare litellm string
        prompt:    Prompt from PromptRegistry, or a plain string template
        eval:      list of assert lambdas — each receives the parsed output
        threshold: minimum confidence score (if output has a .confidence field)
        retries:   number of additional attempts on eval/contract failure
        _registry: ModelRegistry instance for tracking (injected automatically
                   when model comes from a registry)
    """
    eval_fns: list[Callable] = eval or []

    # normalise a raw string prompt to a minimal Prompt object
    if isinstance(prompt, str):
        from .prompt_registry import Prompt as P

        _raw = prompt

        class _InlinePrompt(P):
            pass

        prompt = P(
            key="inline",
            version="inline",
            system=None,
            user=_raw,
            examples=[],
        )

    # infer registry from _AutoSelector if not supplied
    inferred_registry: ModelRegistry | None = _registry
    if inferred_registry is None and isinstance(model, _AutoSelector):
        inferred_registry = model.registry

    def decorator(fn: Callable) -> Callable:
        hints       = get_type_hints(fn)
        output_type = hints.get("return")
        sig         = inspect.signature(fn)
        context_key = fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            flat = _flatten_arguments(bound)

            resolved = _resolve_model(model, context_key)
            messages = prompt.render(**flat)

            if output_type is not None:
                messages = _inject_schema_hint(messages, output_type)

            last_exc: Exception | None = None

            for attempt in range(retries + 1):
                try:
                    raw    = _call_model(resolved.litellm_name, messages)
                    parsed = output_type.parse_llm_output(raw) \
                             if is_interface(output_type) else raw

                    # run eval assertions
                    for fn_eval in eval_fns:
                        if not fn_eval(parsed):
                            raise EvalAssertException(fn_eval, parsed)

                    # confidence gate
                    if is_interface(output_type) and hasattr(parsed, "confidence"):
                        if parsed.confidence < threshold:
                            raise ConfidenceException(parsed.confidence, threshold)

                    _record(resolved, inferred_registry, context_key, success=True)
                    return parsed

                except (ContractException, EvalAssertException, ConfidenceException) as exc:
                    last_exc = exc
                    _record(resolved, inferred_registry, context_key, success=False)
                    if attempt < retries:
                        continue
                    raise

            raise last_exc  # should be unreachable, but satisfies type checkers

        return wrapper

    return decorator
