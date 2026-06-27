"""The executor boundary — a refusal gate, not an executor.

AMALI-IGA-02 does **not** execute tools. It implements only the boundary
that a *future* sandbox/tool executor must sit behind. The single guarantee
proven here is structural and testable:

    No PermissionGrant -> no execution path is ever reached.

``execute_tool`` refuses (raises :class:`PermissionRequiredError`) unless a
grant is present, unexpired, and an exact match for the request's tool,
operation, and scope. When a valid grant *is* present, it still does not run
any shell, network, or filesystem command: it returns an inert
"boundary only" result. There is deliberately no code path here that spawns
a process, opens a socket, or writes a file.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from amali.policy.decisions import PermissionGrant
from amali.policy.errors import PermissionRequiredError
from amali.tools.grants import check_grant
from amali.tools.requests import ToolInvocationRequest

__all__ = ["BOUNDARY_RESULT", "execute_tool"]

BOUNDARY_RESULT = "execution not implemented in AMALI-IGA-02 (boundary only)"


def execute_tool(
    request: ToolInvocationRequest,
    grant: PermissionGrant | None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Refuse unless a valid grant authorizes this exact request.

    Raises :class:`PermissionRequiredError` if there is no grant, the grant
    is expired, or it does not match the request. Otherwise returns an inert
    boundary result. It never performs real execution.
    """
    result = check_grant(request, grant, now)
    if not result.ok:
        reason = result.reason.value if result.reason else "NO_GRANT"
        # Message carries only structural facts and a reason code — never
        # request payload or secret material.
        raise PermissionRequiredError(
            f"execution refused for tool={request.tool_id!r} "
            f"operation={request.operation!r}: {reason}"
        )

    return {
        "executed": False,
        "status": "boundary_only",
        "detail": BOUNDARY_RESULT,
        "tool_id": request.tool_id,
        "operation": request.operation,
        "grant_decision_id": grant.decision_id if grant else None,
    }
