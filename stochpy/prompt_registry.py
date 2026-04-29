from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jinja2
import yaml


@dataclass
class Prompt:
    key:      str
    version:  str
    system:   str | None
    user:     str
    examples: list[dict] = field(default_factory=list)

    def render(self, **kwargs: Any) -> list[dict[str, str]]:
        """
        Render Jinja2 templates with kwargs and build an OpenAI-style
        messages list.  Nested objects are accessible via dot notation
        inside templates: {{ intent.segments }}.
        """
        env = jinja2.Environment(undefined=jinja2.Undefined)

        # flatten nested objects so Jinja2 can reach them
        flat = _flatten_kwargs(kwargs)

        messages: list[dict[str, str]] = []

        if self.system:
            messages.append({
                "role":    "system",
                "content": env.from_string(self.system).render(**flat),
            })

        for ex in self.examples:
            messages.append({"role": "user",      "content": str(ex.get("input", ""))})
            messages.append({"role": "assistant",  "content": str(ex.get("output", ""))})

        messages.append({
            "role":    "user",
            "content": env.from_string(self.user).render(**flat),
        })

        return messages


class PromptRegistry:
    """
    Loads prompts from YAML files.

    Key format:  "module.submodule.section"
    File mapping: prompts/<module>/<submodule>.yaml  →  section key

    Example:
        prompts.load("borusan.pl_graph.analyze")
        reads   prompts/borusan/pl_graph.yaml  →  analyze:
    """

    def __init__(self, root: str = "prompts/"):
        self.root   = Path(root)
        self._cache: dict[str, dict] = {}

    def load(self, key: str, version: str = "latest") -> Prompt:
        parts   = key.split(".")
        section = parts[-1]

        if len(parts) == 2:
            # "shared.time_filter" → shared/time_filter.yaml, section "time_filter"
            yaml_path = self.root / parts[0] / (parts[1] + ".yaml")
        else:
            # "borusan.pl_graph.analyze" → prompts/borusan/pl_graph.yaml
            rel_path  = Path(*parts[:-1]).with_suffix(".yaml")
            yaml_path = self.root / rel_path

        raw_file = self._read_yaml(yaml_path)

        if section not in raw_file:
            raise KeyError(
                f"Section '{section}' not found in {yaml_path}. "
                f"Available: {list(raw_file.keys())}"
            )

        data: dict = dict(raw_file[section])   # shallow copy — don't mutate cache

        # compose partials — append each fragment's user template
        for partial_key in data.get("partials", []):
            fragment = self.load(partial_key)
            data["user"] = data["user"] + "\n\n" + fragment.user

        # version filtering (future: pick specific version from list)
        # for now a single section = a single version
        resolved_version = str(data.get("version", "unversioned"))

        return Prompt(
            key      = key,
            version  = resolved_version,
            system   = data.get("system"),
            user     = data["user"],
            examples = data.get("examples", []),
        )

    def reload(self) -> None:
        """Clear YAML cache — picks up file edits without restart."""
        self._cache.clear()

    def _read_yaml(self, path: Path) -> dict:
        key = str(path)
        if key not in self._cache:
            if not path.exists():
                raise FileNotFoundError(f"Prompt file not found: {path}")
            with open(path, encoding="utf-8") as fh:
                self._cache[key] = yaml.safe_load(fh) or {}
        return self._cache[key]


def _flatten_kwargs(kwargs: dict, prefix: str = "") -> dict:
    """
    Recursively flatten nested objects/dicts so Jinja2 can reference
    {{ intent_segments }} from intent.segments, etc.
    Also keeps original objects so {{ intent.segments }} works directly
    via Jinja2's attribute access.
    """
    result = dict(kwargs)
    for k, v in kwargs.items():
        if hasattr(v, "__dict__") or hasattr(v, "model_fields"):
            # Pydantic model or dataclass — expose as-is for dot access
            result[k] = v
        elif isinstance(v, dict):
            for sub_k, sub_v in v.items():
                result[f"{k}_{sub_k}"] = sub_v
    return result


# module-level singleton — override root in application entry point
prompts = PromptRegistry()
