from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch


class MockModel:
    """
    Test harness for StochPy contexts and agents.

    Bypasses LiteLLM entirely — returns canned JSON responses while still
    exercising the full interface validation, eval, and exception pipeline.

    Usage:
        mock = MockModel()
        mock.when("check_spam", '{"is_spam": false, "confidence": 0.95, "reason": "ok"}')

        with mock.patch():
            result = check_spam(my_email)
            assert result.is_spam == False
    """

    def __init__(self):
        self._responses: dict[str, list[str]] = {}   # key → [response, ...]
        self._call_counts: dict[str, int]     = {}
        self._default: str | None = None

    def when(
        self,
        context: str,
        returns: str | dict,
        *,
        times: int = 0,   # 0 = unlimited
    ) -> "MockModel":
        """
        Register a canned response for a context (matched by function name substring).

        Args:
            context: substring matched against the prompt content or function name
            returns: JSON string or dict to return as model output
            times:   how many times this response fires (0 = unlimited)
        """
        payload = json.dumps(returns) if isinstance(returns, dict) else returns
        self._responses[context] = [payload] * (times if times > 0 else 9999)
        return self

    def default(self, returns: str | dict) -> "MockModel":
        """Fallback response when no specific match is found."""
        self._default = json.dumps(returns) if isinstance(returns, dict) else returns
        return self

    def _get_response(self, messages: list[dict]) -> str:
        """Find the best matching canned response for the given messages."""
        # build a search string from all message content
        content = " ".join(m.get("content", "") for m in messages).lower()

        for key, responses in self._responses.items():
            if key.lower() in content and responses:
                self._call_counts[key] = self._call_counts.get(key, 0) + 1
                return responses[min(self._call_counts[key] - 1, len(responses) - 1)]

        if self._default is not None:
            return self._default

        raise ValueError(
            f"MockModel has no response registered for messages. "
            f"Registered keys: {list(self._responses.keys())}\n"
            f"Message content (first 300 chars): {content[:300]}"
        )

    def _make_litellm_response(self, content: str):
        """Build a mock object shaped like a litellm.ModelResponse."""
        choice  = MagicMock()
        choice.message.content = content
        response = MagicMock()
        response.choices = [choice]
        return response

    @contextmanager
    def patch(self):
        """
        Context manager — patches _call_model for the duration of the block.

        Example:
            with mock.patch():
                result = my_context(...)
        """
        def _mock_call_model(litellm_name: str, messages: list) -> str:
            return self._get_response(messages)

        with patch("stochpy.context._call_model", _mock_call_model):
            with patch("stochpy.agent._call_model", _mock_call_model):
                yield self

    def call_count(self, context: str) -> int:
        """Return how many times a registered response was served."""
        return self._call_counts.get(context, 0)

    def assert_called(self, context: str, times: int | None = None) -> None:
        count = self.call_count(context)
        if times is not None:
            assert count == times, (
                f"Expected '{context}' to be called {times} time(s), got {count}"
            )
        else:
            assert count > 0, f"Expected '{context}' to be called at least once"
