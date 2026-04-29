import json
import pytest
from pydantic import BaseModel

from stochpy.agent      import agent
from stochpy.exceptions import EvalAssertException, MaxStepsException, ObjectiveException
from stochpy.interface  import interface
from stochpy.testing    import MockModel


@interface
class ComplaintResolution(BaseModel):
    resolved: bool
    method:   str
    response: str


# ── dummy tools ──────────────────────────────────────────────────────────────

def check_spam(email: str) -> dict:
    """Check if email is spam."""
    return {"is_spam": False}


def search_faq(query: str) -> dict:
    """Search FAQ for answer."""
    return {"answered": True, "answer": "Please restart the device."}


# ── helpers ───────────────────────────────────────────────────────────────────

def _tool_decision(tool: str, **args) -> str:
    return json.dumps({"action": "tool", "tool": tool, "args": args})


def _return_decision(output: dict) -> str:
    return json.dumps({"action": "return", "output": output})


# ── tests ─────────────────────────────────────────────────────────────────────

def test_agent_resolves_via_faq():
    """Agent calls check_spam then search_faq then returns."""
    call_count = {"n": 0}
    responses  = [
        _tool_decision("check_spam", email="user@example.com"),
        _tool_decision("search_faq", query="device not working"),
        _return_decision({"resolved": True, "method": "faq",
                          "response": "Please restart the device."}),
    ]

    mock = MockModel()
    original = mock._get_response

    def seq_responses(messages):
        idx = min(call_count["n"], len(responses) - 1)
        call_count["n"] += 1
        return responses[idx]

    mock._get_response = seq_responses

    @agent(
        tools     = [check_spam, search_faq],
        model     = "openai/gpt-4o",
        objective = [lambda o: o.resolved == True],
        max_steps = 10,
    )
    def handle_complaint(email: str) -> ComplaintResolution:
        ...

    with mock.patch():
        result = handle_complaint(email="user@example.com")

    assert isinstance(result, ComplaintResolution)
    assert result.resolved == True
    assert result.method   == "faq"


def test_agent_raises_max_steps():
    """Agent that never returns should raise MaxStepsException."""
    mock = MockModel().default(
        _tool_decision("check_spam", email="x@x.com")
    )

    @agent(
        tools     = [check_spam],
        model     = "openai/gpt-4o",
        max_steps = 3,
    )
    def looping_agent(email: str) -> ComplaintResolution:
        ...

    with mock.patch():
        with pytest.raises(MaxStepsException):
            looping_agent(email="x@x.com")


def test_agent_objective_not_met_raises():
    """Agent returns output that fails objective check."""
    mock = MockModel().default(
        _return_decision({"resolved": False, "method": "none", "response": ""})
    )

    @agent(
        tools     = [check_spam],
        model     = "openai/gpt-4o",
        objective = [lambda o: o.resolved == True],
        max_steps = 5,
    )
    def strict_agent(email: str) -> ComplaintResolution:
        ...

    with mock.patch():
        with pytest.raises(ObjectiveException):
            strict_agent(email="x@x.com")


def test_step_eval_fails_raises():
    """step_eval assertion on tool result raises EvalAssertException."""
    call_count = {"n": 0}
    responses  = [
        _tool_decision("check_spam", email="x@x.com"),
        _return_decision({"resolved": True, "method": "faq", "response": "ok"}),
    ]

    mock = MockModel()

    def seq(messages):
        idx = min(call_count["n"], len(responses) - 1)
        call_count["n"] += 1
        return responses[idx]

    mock._get_response = seq

    @agent(
        tools     = [check_spam],
        model     = "openai/gpt-4o",
        step_eval = [lambda result: result.get("is_spam") is not None
                                    and result["is_spam"] == True],  # always fails
        max_steps = 5,
    )
    def picky_agent(email: str) -> ComplaintResolution:
        ...

    with mock.patch():
        with pytest.raises(EvalAssertException):
            picky_agent(email="x@x.com")
