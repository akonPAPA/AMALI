"""Typed permission-decision objects.

The compiler returns exactly one of four discriminated result types. Each is
immutable-by-convention and carries the full evidence of *why* it was
produced: reason codes, the policy and manifest versions in force, the audit
event id, and timestamps. None of them ever carries raw request payload or
secret material — only structural facts and closed-vocabulary codes.

The discriminator is the ``decision`` field (a :class:`DecisionType`).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Union

from pydantic import BaseModel, Field

from amali.policy.enums import DataSensitivity, DecisionType, RiskLevel
from amali.policy.reason_codes import ReasonCode
from amali.policy.scopes import GrantedScope, RequestedScope

__all__ = [
    "PermissionGrant",
    "PermissionDeny",
    "PermissionReviewRequired",
    "InvalidPermissionRequest",
    "PermissionDecision",
]


class _DecisionBase(BaseModel):
    """Fields common to every decision."""

    model_config = {"extra": "forbid"}

    decision_id: str
    audit_event_id: str
    reason_codes: list[ReasonCode] = Field(default_factory=list)
    created_at: datetime

    def is_grant(self) -> bool:
        return False


class PermissionGrant(_DecisionBase):
    """A least-privilege grant the executor boundary may later consume."""

    decision: Literal[DecisionType.GRANT] = DecisionType.GRANT
    task_id: str
    actor_ref: str
    tool_id: str
    operation: str
    granted_scope: GrantedScope
    risk_level: RiskLevel
    data_sensitivity: DataSensitivity
    ttl_seconds: int
    expires_at: datetime
    policy_version: str
    manifest_version: str

    def is_grant(self) -> bool:
        return True


class PermissionDeny(_DecisionBase):
    """An explicit denial. The default outcome of the kernel."""

    decision: Literal[DecisionType.DENY] = DecisionType.DENY
    task_id: str
    actor_ref: str
    tool_id: str
    operation: str
    denied_scope: RequestedScope
    risk_level: RiskLevel
    data_sensitivity: DataSensitivity
    policy_version: str
    # Null when the tool is unknown (no manifest could be resolved).
    manifest_version: str | None = None


class PermissionReviewRequired(_DecisionBase):
    """A decision deferred to owner review; not a grant."""

    decision: Literal[DecisionType.REVIEW_REQUIRED] = (
        DecisionType.REVIEW_REQUIRED
    )
    task_id: str
    actor_ref: str
    tool_id: str
    operation: str
    requested_scope: RequestedScope
    risk_level: RiskLevel
    data_sensitivity: DataSensitivity
    required_approval_type: str
    policy_version: str
    manifest_version: str


class InvalidPermissionRequest(_DecisionBase):
    """The request itself was malformed; nothing was decided about a tool."""

    decision: Literal[DecisionType.INVALID_REQUEST] = (
        DecisionType.INVALID_REQUEST
    )
    task_id: str | None = None
    actor_ref: str | None = None
    tool_id: str | None = None
    validation_errors: list[str] = Field(default_factory=list)
    policy_version: str | None = None


PermissionDecision = Union[
    PermissionGrant,
    PermissionDeny,
    PermissionReviewRequired,
    InvalidPermissionRequest,
]
