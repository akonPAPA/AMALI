"""Minimal owner-approval model and registry.

This stage does **not** implement an owner UI or any real approval workflow.
It implements only the type and the verification boundary needed to prove a
single security property: an approval is trusted only if it can be resolved,
unexpired, in a local registry and its scope actually covers the request.

A raw string such as ``"approved"`` is never proof of anything — it will not
resolve in the registry and so will never authorize a grant.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from amali.policy.enums import RiskLevel, risk_rank

__all__ = ["ApprovalScope", "ApprovalRef", "ApprovalRegistry"]


class ApprovalScope(BaseModel):
    """The bounded scope an approval authorizes."""

    model_config = {"extra": "forbid"}

    tool_id: str
    operation: str
    paths: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    max_risk: RiskLevel


class ApprovalRef(BaseModel):
    """A recorded owner approval. Trusted only via the registry."""

    model_config = {"extra": "forbid"}

    approval_id: str
    actor_ref: str
    scope: ApprovalScope
    expires_at: datetime
    approved_by: str
    reason: str

    def covers(
        self,
        *,
        tool_id: str,
        operation: str,
        paths: list[str],
        domains: list[str],
        effective_risk: RiskLevel,
        now: datetime,
    ) -> bool:
        """Return True iff this approval authorizes the given request facts.

        Every dimension must match: not expired, same tool and operation,
        requested paths/domains a subset of the approved set, and the
        effective risk at or below the approved ceiling. Any mismatch means
        the approval is out of scope and must not authorize a grant.
        """
        if now >= self.expires_at:
            return False
        if self.scope.tool_id != tool_id:
            return False
        if self.scope.operation != operation:
            return False
        if not set(paths).issubset(set(self.scope.paths)):
            return False
        if not set(domains).issubset(set(self.scope.domains)):
            return False
        if risk_rank(effective_risk) > risk_rank(self.scope.max_risk):
            return False
        return True


class ApprovalRegistry:
    """Local, in-memory registry of owner approvals.

    Approvals must be added here explicitly (e.g. by a future owner-governed
    flow or by a test). An ``approval_ref`` string on a request is only a
    *lookup key*; it confers no authority by itself.
    """

    def __init__(self) -> None:
        self._approvals: dict[str, ApprovalRef] = {}

    def add(self, approval: ApprovalRef) -> None:
        self._approvals[approval.approval_id] = approval

    def get(self, approval_id: str | None) -> ApprovalRef | None:
        if not approval_id:
            return None
        return self._approvals.get(approval_id)
