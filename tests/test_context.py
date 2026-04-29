import pytest
from pydantic import BaseModel

from stochpy.context    import context
from stochpy.exceptions import ConfidenceException, ContractException, EvalAssertException
from stochpy.interface  import interface
from stochpy.registry   import ModelRegistry
from stochpy.testing    import MockModel


@interface
class SpamCheck(BaseModel):
    is_spam:    bool
    confidence: float
    reason:     str


SPAM_JSON     = '{"is_spam": true,  "confidence": 0.98, "reason": "promotional"}'
HAM_JSON      = '{"is_spam": false, "confidence": 0.95, "reason": "personal email"}'
LOW_CONF_JSON = '{"is_spam": false, "confidence": 0.40, "reason": "unclear"}'


def make_check_spam(model="openai/gpt-4o", eval_fns=None, threshold=0.80, retries=0):
    @context(
        model     = model,
        prompt    = "Is this email spam? From: {{ sender }} Body: {{ body }}",
        eval      = eval_fns or [],
        threshold = threshold,
        retries   = retries,
    )
    def check_spam(sender: str, body: str) -> SpamCheck:
        ...
    return check_spam


# ── happy path ────────────────────────────────────────────────────────────────

def test_returns_typed_output():
    mock = MockModel().default(HAM_JSON)
    with mock.patch():
        check_spam = make_check_spam()
        result = check_spam(sender="alice@example.com", body="Hello")
    assert isinstance(result, SpamCheck)
    assert result.is_spam    == False
    assert result.confidence == 0.95


def test_spam_detected():
    mock = MockModel().default(SPAM_JSON)
    with mock.patch():
        check_spam = make_check_spam()
        result = check_spam(sender="promo@spam.com", body="Buy now!!!")
    assert result.is_spam == True


# ── eval ──────────────────────────────────────────────────────────────────────

def test_eval_assertion_passes():
    mock = MockModel().default(HAM_JSON)
    with mock.patch():
        check_spam = make_check_spam(eval_fns=[lambda o: o.confidence >= 0.80])
        result = check_spam(sender="x", body="y")
    assert result.confidence >= 0.80


def test_eval_assertion_fails_raises():
    mock = MockModel().default(HAM_JSON)
    with mock.patch():
        check_spam = make_check_spam(eval_fns=[lambda o: o.confidence >= 0.99])
        with pytest.raises(EvalAssertException):
            check_spam(sender="x", body="y")


# ── confidence ────────────────────────────────────────────────────────────────

def test_low_confidence_raises():
    mock = MockModel().default(LOW_CONF_JSON)
    with mock.patch():
        check_spam = make_check_spam(threshold=0.80)
        with pytest.raises(ConfidenceException) as exc_info:
            check_spam(sender="x", body="y")
    assert exc_info.value.confidence == 0.40


# ── retries ───────────────────────────────────────────────────────────────────

def test_retry_succeeds_on_second_attempt():
    mock = MockModel()
    # first call returns low confidence, second returns good
    call_count = {"n": 0}
    original_get = mock._get_response

    def patched_get(messages):
        call_count["n"] += 1
        return LOW_CONF_JSON if call_count["n"] == 1 else HAM_JSON

    mock._get_response = patched_get

    with mock.patch():
        check_spam = make_check_spam(threshold=0.80, retries=1)
        result = check_spam(sender="x", body="y")
    assert result.confidence == 0.95
    assert call_count["n"]   == 2


# ── model registry integration ────────────────────────────────────────────────

def test_auto_model_selection():
    registry = ModelRegistry(quality="openai/gpt-4o", fast="openai/gpt-4o-mini")
    mock     = MockModel().default(HAM_JSON)

    @context(model=registry.auto, prompt="Is this spam? {{ text }}")
    def classify(text: str) -> SpamCheck:
        ...

    with mock.patch():
        result = classify(text="hello world")
    assert isinstance(result, SpamCheck)


# ── contract failure ──────────────────────────────────────────────────────────

def test_bad_json_raises_contract_exception():
    mock = MockModel().default("Not JSON at all")
    with mock.patch():
        check_spam = make_check_spam()
        with pytest.raises(ContractException):
            check_spam(sender="x", body="y")
