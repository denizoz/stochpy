from stochpy.exceptions import (
    AgentTimeoutException,
    ApprovalRejectedException,
    ConfidenceException,
    ContractException,
    EvalAssertException,
    EvalScoreException,
    MaxStepsException,
    ModelException,
    ModelTimeoutException,
    ObjectiveException,
    StochPyException,
)


def test_hierarchy():
    assert issubclass(ContractException,    StochPyException)
    assert issubclass(EvalAssertException,  ContractException)
    assert issubclass(EvalScoreException,   ContractException)
    assert issubclass(ConfidenceException,  ContractException)
    assert issubclass(ModelTimeoutException, ModelException)
    assert issubclass(MaxStepsException,    ObjectiveException)
    assert issubclass(AgentTimeoutException, ObjectiveException)


def test_contract_exception_stores_raw():
    exc = ContractException("bad output", raw_output="raw text")
    assert exc.raw_output == "raw text"
    assert "bad output" in str(exc)


def test_confidence_exception_stores_values():
    exc = ConfidenceException(confidence=0.5, threshold=0.9)
    assert exc.confidence == 0.5
    assert exc.threshold  == 0.9
    assert "0.500" in str(exc)


def test_eval_score_exception():
    exc = EvalScoreException(score=0.7, threshold=0.85)
    assert exc.score     == 0.7
    assert exc.threshold == 0.85


def test_max_steps_exception():
    exc = MaxStepsException(max_steps=20)
    assert exc.max_steps    == 20
    assert exc.steps_taken  == 20


def test_approval_rejected():
    exc = ApprovalRejectedException(role="CFO", reason="not approved")
    assert exc.role   == "CFO"
    assert exc.reason == "not approved"
