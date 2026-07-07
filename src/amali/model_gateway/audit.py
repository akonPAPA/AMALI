"""Audit emission for model-gateway invocations.

Reuses the hash-chained :class:`~amali.audit.ledger.AuditLedger`. Every field
is passed through :func:`~amali.security.redaction.redact` before it is
written, and only *hashes and identifiers* — never the raw prompt or context —
are ever placed in the event. Both allowed invocations and refusals are
recorded, satisfying the non-negotiable law ("Audit records every critical
event").
"""

from __future__ import annotations

from typing import Any

from amali.audit.ledger import AuditLedger
from amali.model_gateway.errors import GatewayReason
from amali.policy.enums import DataSensitivity
from amali.security.redaction import redact

__all__ = ["emit_model_invocation_audit", "MODEL_INVOCATION_EVENT_TYPE"]

MODEL_INVOCATION_EVENT_TYPE = "model_gateway.invocation"


def emit_model_invocation_audit(
    ledger: AuditLedger,
    *,
    outcome: str,
    reason: GatewayReason,
    trace_id: str,
    task_id: str,
    actor_ref: str,
    model_id: str | None,
    model_version: str | None,
    prompt_manifest_id: str | None,
    prompt_version: str | None,
    prompt_sha256: str | None,
    data_sensitivity: DataSensitivity | None,
    context_redacted: bool | None,
    total_tokens: int | None,
    budget_max_tokens: int | None,
) -> str:
    """Append a redacted model-invocation event and return its audit_id.

    ``outcome`` is ``"allow"`` or ``"deny"``. No prompt text is passed here —
    only its ``prompt_sha256`` — so the audit trail cannot leak prompts.
    """
    metadata: dict[str, Any] = {
        "component": "model_gateway",
        "outcome": outcome,
        "reason": reason.value,
        "trace_id": trace_id,
        "model_id": model_id,
        "model_version": model_version,
        "prompt_manifest_id": prompt_manifest_id,
        "prompt_version": prompt_version,
        "prompt_sha256": prompt_sha256,
        "data_sensitivity": (
            data_sensitivity.value if data_sensitivity is not None else None
        ),
        "context_redacted": context_redacted,
        "total_tokens": total_tokens,
        "budget_max_tokens": budget_max_tokens,
    }
    event = ledger.append(
        event_type=MODEL_INVOCATION_EVENT_TYPE,
        actor=actor_ref or "unknown",
        action=outcome,
        target=model_id or "unknown",
        metadata=redact(metadata),
    )
    return event.audit_id
