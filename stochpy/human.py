from __future__ import annotations

import asyncio
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Type

from .exceptions import ApprovalRejectedException, EscalationTimeoutException
from .interface import is_interface


# ── Role helper ──────────────────────────────────────────────────────────────

class _RoleNamespace:
    """Allows role.cfo_analyst syntax as a plain string carrier."""

    def __getattr__(self, name: str) -> str:
        return name.replace("_", " ").title()


role = _RoleNamespace()


# ── Ticket transport protocol ─────────────────────────────────────────────────
# Default implementation is in-process (useful for testing and local runs).
# Replace by subclassing TicketTransport and passing to configure_transport().

class TicketTransport:
    """Base transport — override to integrate Slack, email, Teams, etc."""

    async def create(
        self,
        ticket_id:  str,
        to:         str,
        message:    str,
        context:    Any,
        requires_approval: bool = False,
    ) -> None:
        """Send the ticket to the human.  Default: print to stdout."""
        print(
            f"\n[StochPy Escalation]\n"
            f"  Ticket:  {ticket_id}\n"
            f"  To:      {to}\n"
            f"  Message: {message}\n"
        )

    async def poll(self, ticket_id: str) -> dict | None:
        """
        Return resolved ticket dict or None if still pending.
        Dict shape: {"output": Any, "resolved_by": str, "approved": bool | None}
        """
        return None  # in-process transport never auto-resolves

    async def resolve(self, ticket_id: str, output: Any, resolved_by: str) -> None:
        """Mark a ticket as resolved (used by test harness / mock transport)."""


_transport: TicketTransport = TicketTransport()
_pending: dict[str, asyncio.Event] = {}
_resolved: dict[str, dict] = {}


def configure_transport(t: TicketTransport) -> None:
    """Replace the global ticket transport."""
    global _transport
    _transport = t


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_timeout(spec: str) -> float:
    """Convert '4h', '30m', '10s' to seconds."""
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([hms]?)", spec.strip())
    if not m:
        raise ValueError(f"Invalid timeout spec: {spec!r}")
    val, unit = float(m.group(1)), m.group(2)
    return val * {"h": 3600, "m": 60, "s": 1, "": 1}[unit]


async def _wait_for_ticket(ticket_id: str, timeout_secs: float) -> dict:
    event = asyncio.Event()
    _pending[ticket_id] = event
    try:
        await asyncio.wait_for(event.wait(), timeout=timeout_secs)
    except asyncio.TimeoutError:
        raise EscalationTimeoutException(
            f"Escalation ticket {ticket_id} timed out after {timeout_secs}s",
        )
    finally:
        _pending.pop(ticket_id, None)
    return _resolved.get(ticket_id, {})


async def _resolve_ticket(ticket_id: str, output: Any, resolved_by: str,
                          approved: bool | None = None) -> None:
    """Called by transport implementations or test harness to close a ticket."""
    _resolved[ticket_id] = {
        "output":      output,
        "resolved_by": resolved_by,
        "approved":    approved,
    }
    if ticket_id in _pending:
        _pending[ticket_id].set()


# ── Public API ────────────────────────────────────────────────────────────────

@dataclass
class EscalationResult:
    ticket_id:   str
    resolved_by: str
    output:      Any
    timestamp:   datetime = field(default_factory=datetime.utcnow)


async def escalate(
    to:       str,
    message:  str,
    context:  Any            = None,
    timeout:  str            = "4h",
    blocking: bool           = True,
    returns:  Type | None    = None,
) -> EscalationResult | None:
    """
    Escalate to a human role.

    Args:
        to:       role identifier — use role.some_name or a plain string
        message:  human-readable description of what needs resolution
        context:  optional payload attached to the ticket
        timeout:  how long to wait for a response, e.g. "4h", "30m"
        blocking: if False, fire-and-forget (returns None immediately)
        returns:  optional @interface type — validates human response against it

    Returns:
        EscalationResult if blocking=True, else None
    """
    ticket_id = str(uuid.uuid4())[:8]

    await _transport.create(
        ticket_id = ticket_id,
        to        = to,
        message   = message,
        context   = context,
    )

    if not blocking:
        asyncio.create_task(
            _wait_for_ticket(ticket_id, _parse_timeout(timeout))
        )
        return None

    timeout_secs = _parse_timeout(timeout)
    resolved     = await _wait_for_ticket(ticket_id, timeout_secs)

    output = resolved.get("output")
    if returns is not None and output is not None and is_interface(returns):
        output = returns.parse_llm_output(str(output))

    return EscalationResult(
        ticket_id   = ticket_id,
        resolved_by = resolved.get("resolved_by", to),
        output      = output,
    )


# ── Approval gate ─────────────────────────────────────────────────────────────

class ApprovalGate:
    """
    Async context manager — pauses execution until a human approves.

    Usage:
        async with ApprovalGate(required=role.cfo, message="Review Q3 report"):
            send_report(report)   # only runs after approval
    """

    def __init__(
        self,
        required: str,
        timeout:  str = "24h",
        message:  str = "",
    ):
        self.required  = required
        self.timeout   = timeout
        self.message   = message
        self._ticket_id: str | None = None

    async def __aenter__(self) -> "ApprovalGate":
        self._ticket_id = str(uuid.uuid4())[:8]
        await _transport.create(
            ticket_id          = self._ticket_id,
            to                 = self.required,
            message            = self.message or f"Approval required",
            context            = None,
            requires_approval  = True,
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        timeout_secs = _parse_timeout(self.timeout)
        resolved     = await _wait_for_ticket(self._ticket_id, timeout_secs)

        if not resolved.get("approved", False):
            raise ApprovalRejectedException(
                role   = self.required,
                reason = resolved.get("output"),
            )


# ── Test / mock helper ────────────────────────────────────────────────────────

async def mock_resolve(ticket_id: str, output: Any = None,
                       resolved_by: str = "mock",
                       approved: bool = True) -> None:
    """Resolve a pending ticket in tests without going through transport."""
    await _resolve_ticket(ticket_id, output, resolved_by, approved)
