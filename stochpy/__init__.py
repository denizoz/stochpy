"""
StochPy — composable deterministic + stochastic Python functions.

Quick start:
    from pydantic import BaseModel
    from stochpy import context, interface, ModelRegistry, PromptRegistry

    registry = ModelRegistry(
        quality = "anthropic/claude-sonnet-4-6",
        fast    = "openai/gpt-4o-mini",
    )

    @interface
    class SpamCheck(BaseModel):
        is_spam:    bool
        confidence: float
        reason:     str

    @context(
        model     = registry.quality,
        prompt    = "Is this email spam? Email: {{ email }}",
        eval      = [lambda o: 0.0 <= o.confidence <= 1.0],
        threshold = 0.80,
    )
    def check_spam(email: str) -> SpamCheck:
        ...
"""

from .agent           import agent
from .context         import context
from .exceptions      import (
    AgentTimeoutException,
    ApprovalRejectedException,
    ConfidenceException,
    ContractException,
    EscalationException,
    EscalationTimeoutException,
    EvalAssertException,
    EvalScoreException,
    MaxStepsException,
    ModelException,
    ModelTimeoutException,
    ModelUnavailableException,
    ObjectiveException,
    StochPyException,
)
from .human           import ApprovalGate, EscalationResult, escalate, role
from .interface       import interface, is_interface
from .parallel        import ParallelResult, parallel, parallel_map
from .prompt_registry import Prompt, PromptRegistry, prompts
from .registry        import ModelEntry, ModelRegistry

__all__ = [
    # decorators
    "context",
    "interface",
    "agent",
    # registries
    "ModelRegistry",
    "ModelEntry",
    "PromptRegistry",
    "Prompt",
    "prompts",
    # human loop
    "escalate",
    "ApprovalGate",
    "EscalationResult",
    "role",
    # parallel
    "parallel",
    "parallel_map",
    "ParallelResult",
    # helpers
    "is_interface",
    # exceptions
    "StochPyException",
    "ContractException",
    "EvalAssertException",
    "EvalScoreException",
    "ConfidenceException",
    "ModelException",
    "ModelTimeoutException",
    "ModelUnavailableException",
    "ObjectiveException",
    "MaxStepsException",
    "AgentTimeoutException",
    "EscalationException",
    "EscalationTimeoutException",
    "ApprovalRejectedException",
]

__version__ = "0.1.0"
