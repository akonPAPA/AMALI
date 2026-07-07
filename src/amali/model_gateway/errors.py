"""Model Gateway errors and refusal reason codes.

Refusals are typed and carry a closed :class:`GatewayReason` code so that a
denied model call is machine-distinguishable from a validation error, and so
the reason can be recorded in the audit ledger without leaking any request
payload.
"""

from __future__ import annotations

from enum import Enum

from amali.core.errors import AMALIError

__all__ = [
    "GatewayReason",
    "ModelGatewayError",
    "ModelGatewayRefused",
]


class GatewayReason(str, Enum):
    """Why the gateway refused (or accepted) an invocation."""

    OK = "OK"
    MANIFEST_NOT_FOUND = "MANIFEST_NOT_FOUND"
    MANIFEST_INVALID = "MANIFEST_INVALID"
    MODEL_RETIRED = "MODEL_RETIRED"
    PROMPT_VERSION_MISSING = "PROMPT_VERSION_MISSING"
    OFFLINE_HOSTED_BLOCKED = "OFFLINE_HOSTED_BLOCKED"
    DATA_CLASS_FORBIDDEN = "DATA_CLASS_FORBIDDEN"
    RAW_SECRET_BLOCKED = "RAW_SECRET_BLOCKED"
    CONTEXT_OVERFLOW = "CONTEXT_OVERFLOW"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"


class ModelGatewayError(AMALIError):
    """Base class for all Model Gateway errors."""


class ModelGatewayRefused(ModelGatewayError):
    """Raised when the gateway refuses to invoke a model.

    The refusal carries a :class:`GatewayReason` and, once the refusal has
    been audited, the ``audit_event_id`` of the recorded deny event.
    """

    def __init__(
        self,
        reason: GatewayReason,
        message: str,
        *,
        audit_event_id: str | None = None,
    ) -> None:
        super().__init__(f"{reason.value}: {message}")
        self.reason = reason
        self.audit_event_id = audit_event_id
