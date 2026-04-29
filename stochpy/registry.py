from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

STRATEGIES = ("cost_optimized", "quality_optimized", "latency_optimized", "balanced")


@dataclass
class ModelEntry:
    name:         str          # human label e.g. "quality"
    litellm_name: str          # e.g. "anthropic/claude-sonnet-4-6"
    cost_per_1k:  float = 0.0  # approximate USD per 1k tokens (for cost strategy)
    avg_latency:  float = 1.0  # seconds, seed value


class _AutoSelector:
    """Sentinel passed as model= to @context; resolved at call time."""

    def __init__(self, registry: "ModelRegistry", strategy: str = "balanced"):
        self.registry = registry
        self.strategy = strategy

    def resolve(self, context_key: str) -> ModelEntry:
        return self.registry.best_for(context_key, self.strategy)

    def __repr__(self) -> str:
        return f"auto(strategy={self.strategy})"


class _FederatedSelector:
    """Runs all models, merges results (consensus / best-score)."""

    def __init__(self, registry: "ModelRegistry", merge: str = "consensus"):
        self.registry = registry
        self.merge    = merge

    def resolve_all(self) -> list[ModelEntry]:
        return list(self.registry.models.values())

    def __repr__(self) -> str:
        return f"federated(merge={self.merge})"


class ModelRegistry:
    """
    Central store for model entries + runtime tracking.

    Usage:
        registry = ModelRegistry(
            fast    = "openai/gpt-4o-mini",
            balanced= "openai/gpt-4o",
            quality = "anthropic/claude-sonnet-4-6",
            tracking= True,
        )

        # fixed model
        @context(model=registry.quality, ...)

        # auto-select per call
        @context(model=registry.auto, ...)

        # cost-optimised auto
        @context(model=registry.auto_cost, ...)
    """

    def __init__(self, tracking: bool = True, **kwargs: str):
        self.tracking = tracking
        self.models:  dict[str, ModelEntry] = {}
        # context_key → model_name → list[bool]  (True = eval passed)
        self._scores: dict[str, dict[str, list[bool]]] = {}

        for name, litellm_name in kwargs.items():
            if name == "tracking":
                continue
            self.models[name] = ModelEntry(name=name, litellm_name=litellm_name)

        # sentinels
        self.auto          = _AutoSelector(self, strategy="balanced")
        self.auto_cost     = _AutoSelector(self, strategy="cost_optimized")
        self.auto_quality  = _AutoSelector(self, strategy="quality_optimized")
        self.federated     = _FederatedSelector(self)

    # ── attribute access: registry.quality → ModelEntry ──────────────────────

    def __getattr__(self, name: str) -> ModelEntry:
        if name.startswith("_") or name in ("models", "tracking"):
            raise AttributeError(name)
        try:
            return self.models[name]
        except KeyError:
            raise AttributeError(
                f"No model named '{name}' in registry. "
                f"Available: {list(self.models.keys())}"
            )

    # ── tracking ──────────────────────────────────────────────────────────────

    def record(self, context_key: str, model_name: str, success: bool) -> None:
        if not self.tracking:
            return
        bucket = self._scores.setdefault(context_key, {})
        bucket.setdefault(model_name, []).append(success)
        # keep last 200 results per slot to avoid unbounded growth
        if len(bucket[model_name]) > 200:
            bucket[model_name] = bucket[model_name][-200:]

    def success_rate(self, context_key: str, model_name: str) -> float:
        history = self._scores.get(context_key, {}).get(model_name, [])
        return sum(history) / len(history) if history else 1.0  # optimistic default

    # ── auto-selection ────────────────────────────────────────────────────────

    def best_for(self, context_key: str, strategy: str = "balanced") -> ModelEntry:
        if not self.models:
            raise RuntimeError("ModelRegistry has no models registered.")

        candidates = list(self.models.values())

        # filter out models with consistently poor eval performance
        candidates = [
            m for m in candidates
            if self.success_rate(context_key, m.name) >= 0.60
        ] or list(self.models.values())  # fallback: use all if all are poor

        if strategy == "cost_optimized":
            candidates.sort(key=lambda m: m.cost_per_1k)
        elif strategy == "quality_optimized":
            candidates.sort(key=lambda m: -self.success_rate(context_key, m.name))
        elif strategy == "latency_optimized":
            candidates.sort(key=lambda m: m.avg_latency)
        # balanced: no sort — use registration order

        return candidates[0]

    def stats(self) -> dict:
        """Return a snapshot of eval scores for inspection / logging."""
        return {
            context_key: {
                model_name: {
                    "calls":        len(results),
                    "success_rate": round(sum(results) / len(results), 3)
                                    if results else None,
                }
                for model_name, results in models.items()
            }
            for context_key, models in self._scores.items()
        }
