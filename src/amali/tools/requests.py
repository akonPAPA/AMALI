"""Tool-invocation request and scope models.

A :class:`ToolInvocationRequest` is *untrusted input* crossing a trust
boundary: it may come from a user, an agent, a model, or a test harness.
Validation here is therefore a security control, not a convenience. In
particular:

* authority fields cannot be lowered through ``metadata`` — the effective
  risk is read only from the typed ``task_risk`` field;
* ``metadata`` may not carry raw secret material;
* the requested scope is explicit and bounded (lists, not wildcards).
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from amali.policy.enums import DataSensitivity, RequestSource, RiskLevel
from amali.policy.scopes import GrantedScope, RequestedScope
from amali.security.redaction import contains_secret

__all__ = ["RequestedScope", "GrantedScope", "ToolInvocationRequest"]

# Primitive types permitted inside request metadata. Anything else (nested
# callables, objects, etc.) is rejected so metadata stays inert and safe to
# redact and audit.
_SAFE_PRIMITIVES = (str, int, float, bool, type(None))


class ToolInvocationRequest(BaseModel):
    """A request to invoke a tool operation, prior to any decision."""

    model_config = {"extra": "forbid"}

    task_id: str
    actor_ref: str
    actor_roles: list[str] = Field(default_factory=list)
    tool_id: str
    operation: str
    requested_scope: RequestedScope = Field(default_factory=RequestedScope)
    task_risk: RiskLevel
    data_sensitivity: DataSensitivity
    reason: str
    request_source: RequestSource
    approval_ref: str | None = None
    ttl_seconds: int | None = None
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("task_id", "actor_ref", "tool_id", "operation", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("required field must not be empty")
        return value

    @field_validator("ttl_seconds")
    @classmethod
    def _ttl_non_negative(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("ttl_seconds must not be negative")
        return value

    @field_validator("metadata")
    @classmethod
    def _metadata_is_safe(
        cls, value: dict[str, object]
    ) -> dict[str, object]:
        # Reject non-primitive values: metadata must stay inert.
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("metadata keys must be strings")
            if not isinstance(item, _SAFE_PRIMITIVES):
                raise ValueError(
                    "metadata values must be safe primitives "
                    "(str, int, float, bool, None)"
                )
        # Reject obvious raw secrets rather than silently redacting them.
        if contains_secret(value):
            raise ValueError("metadata must not contain secret-like material")
        return value
