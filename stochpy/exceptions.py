from __future__ import annotations


class StochPyException(Exception):
    pass


# ── Contract failures ────────────────────────────────────────────────────────

class ContractException(StochPyException):
    def __init__(self, message: str, raw_output: str | None = None):
        super().__init__(message)
        self.raw_output = raw_output


class EvalAssertException(ContractException):
    def __init__(self, failed_assertion, output):
        name = getattr(failed_assertion, "__name__", repr(failed_assertion))
        super().__init__(f"Eval assertion failed: {name}")
        self.failed_assertion = failed_assertion
        self.output = output


class EvalScoreException(ContractException):
    def __init__(self, score: float, threshold: float):
        super().__init__(f"Eval score {score:.3f} below threshold {threshold:.3f}")
        self.score = score
        self.threshold = threshold


class ConfidenceException(ContractException):
    def __init__(self, confidence: float, threshold: float):
        super().__init__(
            f"Model confidence {confidence:.3f} below threshold {threshold:.3f}"
        )
        self.confidence = confidence
        self.threshold = threshold


# ── Model failures ───────────────────────────────────────────────────────────

class ModelException(StochPyException):
    pass


class ModelTimeoutException(ModelException):
    pass


class ModelUnavailableException(ModelException):
    pass


# ── Agent failures ───────────────────────────────────────────────────────────

class ObjectiveException(StochPyException):
    def __init__(self, message: str, steps_taken: int = 0):
        super().__init__(message)
        self.steps_taken = steps_taken


class MaxStepsException(ObjectiveException):
    def __init__(self, max_steps: int):
        super().__init__(f"Agent exceeded {max_steps} steps", max_steps)
        self.max_steps = max_steps


class AgentTimeoutException(ObjectiveException):
    def __init__(self, timeout: float, steps_taken: int):
        super().__init__(f"Agent timed out after {timeout}s", steps_taken)
        self.timeout = timeout


# ── Human loop ───────────────────────────────────────────────────────────────

class EscalationException(StochPyException):
    def __init__(self, message: str, role: str | None = None):
        super().__init__(message)
        self.role = role


class EscalationTimeoutException(EscalationException):
    pass


class ApprovalRejectedException(StochPyException):
    def __init__(self, role: str, reason: str | None = None):
        super().__init__(f"Approval rejected by {role}")
        self.role = role
        self.reason = reason
