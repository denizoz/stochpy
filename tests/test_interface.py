import pytest
from pydantic import BaseModel

from stochpy.exceptions import ContractException
from stochpy.interface  import interface, is_interface


@interface
class SpamCheck(BaseModel):
    is_spam:    bool
    confidence: float
    reason:     str


def test_is_interface():
    assert is_interface(SpamCheck)


def test_schema_attached():
    assert hasattr(SpamCheck, "__stochpy_schema__")
    assert "is_spam" in SpamCheck.__stochpy_schema__["properties"]


def test_schema_hint_attached():
    assert hasattr(SpamCheck, "__stochpy_schema_hint__")
    assert "JSON" in SpamCheck.__stochpy_schema_hint__


def test_parse_clean_json():
    raw    = '{"is_spam": false, "confidence": 0.95, "reason": "looks fine"}'
    result = SpamCheck.parse_llm_output(raw)
    assert result.is_spam    == False
    assert result.confidence == 0.95
    assert result.reason     == "looks fine"


def test_parse_json_wrapped_in_prose():
    raw = 'Here is the result: {"is_spam": true, "confidence": 0.99, "reason": "spam"} — done.'
    result = SpamCheck.parse_llm_output(raw)
    assert result.is_spam == True


def test_parse_invalid_raises_contract_exception():
    raw = '{"is_spam": "not_a_bool", "confidence": "bad"}'
    with pytest.raises(ContractException):
        SpamCheck.parse_llm_output(raw)


def test_parse_no_json_raises_contract_exception():
    with pytest.raises(ContractException):
        SpamCheck.parse_llm_output("This is not JSON at all.")


def test_non_basemodel_raises():
    with pytest.raises(TypeError):
        @interface
        class NotAModel:
            pass
