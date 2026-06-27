"""Audit emission for permission decisions.

This module does not introduce a second ledger. It reuses the existing
hash-chained :class:`~amali.audit.ledger.AuditLedger` from IGA-00/01 and
records each TRPC decision as one chained event. Every field placed into the
event metadata is passed through :func:`~amali.security.redaction.redact`
first, so no token, key, or other secret-shaped value can enter the audit
trail.

The function returns the ``audit_id`` of the appended event, which the
caller stamps onto the decision object as ``audit_event_id`` — binding the
decision and its audit record together.
"""

from __future__ import annotations

from typing import Any

from amali.audit.ledger import AuditLedger
from amali.policy.enums import DataSensitivity, DecisionType, RiskLevel
from amali.policy.reason_codes import ReasonCode
from amali.security.redaction import redact

__all__ = ["emit_permission_audit", "PERMISSION_EVENT_TYPE"]

PERMISSION_EVENT_TYPE = "trpc.permission_decision"

_OUTCOME_BY_DECISION = {
    DecisionType.GRANT: "allow",
    DecisionType.DENY: "deny",
    DecisionType.REVIEW_REQUIRED: "review",
    DecisionType.INVALID_REQUEST: "invalid",
}


def emit_permission_audit(
    ledger: AuditLedger,
    *,
    decision_type: DecisionType,
    decision_id: str,
    task_id: str | None,
    actor_ref: str | None,
    tool_id: str | None,
    operation: str | None,
    reason_codes: list[ReasonCode],
    risk_level: RiskLevel | None,
    data_sensitivity: DataSensitivity | None,
    policy_version: str | None,
    manifest_version: str | None,
    redacted_payload: dict[str, Any] | None = None,
) -> str:
    """Append a redacted permission-decision event and return its audit_id.

    All metadata is redacted before it is written, so this function is safe
    to call with summaries derived from untrusted request material.
    """
    outcome = _OUTCOME_BY_DECISION[decision_type]
    metadata: dict[str, Any] = {
        "component": "trpc",
        "decision_id": decision_id,
        "decision": decision_type.value,
        "outcome": outcome,
        "tool_id": tool_id,
        "operation": operation,
        "reason_codes": [code.value for code in reason_codes],
        "risk_level": risk_level.value if risk_level is not None else None,
        "data_sensitivity": (
            data_sensitivity.value if data_sensitivity is not None else None
        ),
        "policy_version": policy_version,
        "manifest_version": manifest_version,
        "redacted_payload": redact(redacted_payload or {}),
    }
    event = ledger.append(
        event_type=PERMISSION_EVENT_TYPE,
        actor=actor_ref or "unknown",
        action=operation or "unknown",
        target=tool_id or "unknown",
        metadata=redact(metadata),
    )
    return event.audit_id
