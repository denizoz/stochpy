from __future__ import annotations

import functools
import inspect
import json
import time
from typing import Any, Callable, get_type_hints

from .exceptions import (
    AgentTimeoutException,
    EvalAssertException,
    MaxStepsException,
    ModelException,
    ObjectiveException,
)
from .context import _call_model, _resolve_model
from .interface import is_interface
from .registry import ModelEntry, ModelRegistry, _AutoSelector


def _describe_tools(tools: list[Callable]) -> str:
    """Build a JSON-schema-like description of available tools for the agent prompt."""
    descriptions = []
    for fn in tools:
        hints   = get_type_hints(fn)
        sig     = inspect.signature(fn)
        params  = {
            k: str(v.annotation.__name__) if v.annotation != inspect.Parameter.empty
               else "any"
            for k, v in sig.parameters.items()
        }
        ret     = hints.get("return", None)
        ret_str = ret.__name__ if ret and hasattr(ret, "__name__") else str(ret)
        descriptions.append({
            "name":        fn.__name__,
            "description": (fn.__doc__ or "").strip(),
            "parameters":  params,
            "returns":     ret_str,
        })
    return json.dumps(descriptions, indent=2)


def _build_agent_prompt(
    state:             dict,
    tool_descriptions: str,
    output_type:       Any,
    step_history:      list[dict],
) -> list[dict]:
    output_schema = (
        output_type.__stochpy_schema_hint__
        if is_interface(output_type) else ""
    )

    history_text = ""
    if step_history:
        history_text = "\n\nSteps taken so far:\n" + json.dumps(
            step_history, indent=2, default=str
        )

    user_content = f"""You are an autonomous agent. Decide the next action.

Available tools:
{tool_descriptions}

Current state:
{json.dumps({k: str(v) for k, v in state.items() if not k.startswith('_')}, indent=2)}
{history_text}

Choose ONE of:
1. Call a tool:  {{"action": "tool", "tool": "<name>", "args": {{...}}}}
2. Return final: {{"action": "return", "output": {{...}} }}

{output_schema}

Respond with a single JSON object only."""

    return [{"role": "user", "content": user_content}]


def _parse_decision(raw: str) -> dict:
    """Extract action/tool/args from model output."""
    import re
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"Agent response did not contain JSON: {raw!r}")
    return json.loads(match.group(0))


def _find_tool(tools: list[Callable], name: str) -> Callable:
    for fn in tools:
        if fn.__name__ == name:
            return fn
    available = [fn.__name__ for fn in tools]
    raise ValueError(f"Agent selected unknown tool {name!r}. Available: {available}")


def agent(
    tools:      list[Callable],
    objective:  list[Callable] | None = None,
    model:      Any = None,
    threshold:  float = 0.85,
    max_steps:  int   = 20,
    timeout:    float = 120.0,
    step_eval:  list[Callable] | None = None,
    _registry:  ModelRegistry | None  = None,
) -> Callable:
    """
    Decorator factory.  Turns a function stub into a self-directing agent loop.

    The agent:
      1. Receives the function's arguments as initial state
      2. Asks the model which tool to call next
      3. Executes the tool, runs step_eval
      4. Updates state and repeats until objective is met or max_steps reached

    Args:
        tools:      callables the agent can invoke
        objective:  list of assert lambdas run against the final output
        model:      ModelEntry, _AutoSelector, or litellm string
        threshold:  minimum confidence on final output (if output has .confidence)
        max_steps:  hard cap on iterations
        timeout:    wall-clock seconds before AgentTimeoutException
        step_eval:  assert lambdas run after each tool call
    """
    objective_fns: list[Callable] = objective or []
    step_eval_fns: list[Callable] = step_eval  or []

    # infer registry from _AutoSelector
    inferred_registry: ModelRegistry | None = _registry
    if inferred_registry is None and isinstance(model, _AutoSelector):
        inferred_registry = model.registry

    def decorator(fn: Callable) -> Callable:
        hints       = get_type_hints(fn)
        output_type = hints.get("return")
        sig         = inspect.signature(fn)
        context_key = fn.__qualname__

        tool_descriptions = _describe_tools(tools)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            state: dict = dict(bound.arguments)
            step_history: list[dict] = []

            resolved = _resolve_model(model, context_key) if model else None

            start = time.monotonic()

            for step_num in range(max_steps):
                if time.monotonic() - start > timeout:
                    raise AgentTimeoutException(timeout, step_num)

                messages = _build_agent_prompt(
                    state, tool_descriptions, output_type, step_history
                )

                raw      = _call_model(resolved.litellm_name, messages)
                decision = _parse_decision(raw)

                action = decision.get("action")

                # ── agent wants to return ────────────────────────────────────
                if action == "return":
                    raw_output = decision.get("output", {})
                    if is_interface(output_type):
                        result = output_type.parse_llm_output(
                            json.dumps(raw_output)
                        )
                    else:
                        result = raw_output

                    # check objectives
                    for obj_fn in objective_fns:
                        if not obj_fn(result):
                            raise ObjectiveException(
                                f"Objective not met: {getattr(obj_fn, '__name__', repr(obj_fn))}",
                                steps_taken=step_num,
                            )

                    return result

                # ── agent wants to call a tool ───────────────────────────────
                if action == "tool":
                    tool_name = decision.get("tool")
                    tool_args = decision.get("args", {})

                    tool_fn     = _find_tool(tools, tool_name)
                    tool_result = tool_fn(**tool_args)

                    # step eval
                    for step_fn in step_eval_fns:
                        if not step_fn(tool_result):
                            raise EvalAssertException(step_fn, tool_result)

                    # update state and history
                    state[tool_name + "_result"] = tool_result
                    step_history.append({
                        "step":   step_num,
                        "tool":   tool_name,
                        "args":   tool_args,
                        "result": str(tool_result)[:300],  # truncate for prompt
                    })
                    continue

                raise ModelException(
                    f"Agent returned unrecognised action {action!r}"
                )

            raise MaxStepsException(max_steps)

        return wrapper

    return decorator
