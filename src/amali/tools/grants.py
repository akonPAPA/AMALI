"""Grant-consumption checks.

A :class:`~amali.policy.decisions.PermissionGrant` is the *only* thing that
may authorize a tool execution. This module centralizes the question "does
this grant authorize this exact request, right now?" so the executor
boundary and any future executor share one definition of validity.

The checks are strict and structural: tool, operation, expiry, and scope
must all match. A grant for one path/operation can never be replayed for a
different one.
"""

from __future__ import annotations

from datetime import datetime, timezone

from amali.policy.decisions import PermissionGrant
from amali.policy.reason_codes import ReasonCode
from amali.tools.requests import ToolInvocationRequest

__all__ = ["GrantCheck", "check_grant"]


class GrantCheck:
    """Result of validating a grant against a request."""

    __slots__ = ("ok", "reason")

    def __init__(self, ok: bool, reason: ReasonCode | None) -> None:
        self.ok = ok
        self.reason = reason

    def __bool__(self) -> bool:
        return self.ok


def _norm_path(path: str) -> str:
    norm = path.replace("\\", "/").strip()
    while norm.startswith("./"):
        norm = norm[2:]
    return norm.rstrip("/")


def check_grant(
    request: ToolInvocationRequest,
    grant: PermissionGrant | None,
    now: datetime | None = None,
) -> GrantCheck:
    """Return whether ``grant`` authorizes ``request`` at time ``now``."""
    now = now or datetime.now(timezone.utc)

    # Only a real PermissionGrant authorizes anything. A Deny / Review /
    # Invalid object (or None, or anything else) is refused outright, so a
    # non-grant decision can never be coerced into an execution path.
    if not isinstance(grant, PermissionGrant):
        return GrantCheck(False, ReasonCode.APPROVAL_MISSING)

    if now >= grant.expires_at:
        return GrantCheck(False, ReasonCode.TTL_EXCEEDS_POLICY)

    if grant.tool_id != request.tool_id:
        return GrantCheck(False, ReasonCode.APPROVAL_OUT_OF_SCOPE)

    if grant.operation != request.operation:
        return GrantCheck(False, ReasonCode.APPROVAL_OUT_OF_SCOPE)

    granted = grant.granted_scope
    req = request.requested_scope

    # The request may never escalate beyond what was granted.
    if req.write and not granted.write:
        return GrantCheck(False, ReasonCode.WRITE_SCOPE_DENIED)
    if req.network and not granted.network:
        return GrantCheck(False, ReasonCode.NETWORK_DENIED)
    if req.secret_access and not granted.secret_access:
        return GrantCheck(False, ReasonCode.SECRET_ACCESS_DENIED)

    granted_paths = {_norm_path(p) for p in granted.paths}
    for path in req.paths:
        if _norm_path(path) not in granted_paths:
            return GrantCheck(False, ReasonCode.FILESYSTEM_SCOPE_DENIED)

    granted_domains = set(granted.domains)
    for domain in req.domains:
        if domain not in granted_domains:
            return GrantCheck(False, ReasonCode.DOMAIN_NOT_ALLOWLISTED)

    return GrantCheck(True, None)
