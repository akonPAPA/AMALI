"""The Data Wall operation matrix and decision engine (WP1).

Deterministic, default-block admission control for data moving into a flow:

* the matrix maps ``(DataFlow, DataSensitivity) -> WallOutcome``;
* any pair not in the matrix is **BLOCK** (default-deny);
* ``secret`` data is blocked in every flow, always;
* holdout-tagged items are blocked from ``training`` and ``benchmark``
  unconditionally (holdout isolation beats the matrix);
* REDACT outcomes return a redacted copy — the original is never mutated;
* every decision is appended to the hash-chained audit ledger.

The wall never trusts payload content: a payload that *claims* to be public
is still redaction-scanned, and secret-shaped values force a BLOCK even in
an ALLOW cell.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from amali.audit.ledger import AuditLedger
from amali.core.ids import new_decision_id
from amali.policy.enums import DataSensitivity
from amali.security.redaction import contains_secret, redact

__all__ = ["DataFlow", "WallOutcome", "WallDecision", "DataWall"]

WALL_EVENT_TYPE = "data_wall.decision"


class DataFlow(str, Enum):
    """The destination flow the data is trying to enter."""

    RUNTIME = "runtime"
    EVAL = "eval"
    TRAINING = "training"
    BENCHMARK = "benchmark"
    ARTIFACT = "artifact"


class WallOutcome(str, Enum):
    """What the wall did with the data."""

    ALLOW = "ALLOW"
    REDACT = "REDACT"
    BLOCK = "BLOCK"


# The operation matrix. Any (flow, sensitivity) pair absent here is BLOCK.
# secret is deliberately absent from every flow: no cell can allow it.
_MATRIX: dict[tuple[DataFlow, DataSensitivity], WallOutcome] = {
    # runtime: work happens here; internal data flows, sensitive is redacted.
    (DataFlow.RUNTIME, DataSensitivity.PUBLIC): WallOutcome.ALLOW,
    (DataFlow.RUNTIME, DataSensitivity.INTERNAL): WallOutcome.ALLOW,
    (DataFlow.RUNTIME, DataSensitivity.CONFIDENTIAL): WallOutcome.REDACT,
    (DataFlow.RUNTIME, DataSensitivity.PERSONAL): WallOutcome.REDACT,
    # eval: items must be shareable with graders; redact non-public.
    (DataFlow.EVAL, DataSensitivity.PUBLIC): WallOutcome.ALLOW,
    (DataFlow.EVAL, DataSensitivity.INTERNAL): WallOutcome.REDACT,
    (DataFlow.EVAL, DataSensitivity.CONFIDENTIAL): WallOutcome.REDACT,
    # training: only public/internal, redacted; personal/confidential never.
    (DataFlow.TRAINING, DataSensitivity.PUBLIC): WallOutcome.ALLOW,
    (DataFlow.TRAINING, DataSensitivity.INTERNAL): WallOutcome.REDACT,
    # benchmark: mirrors eval.
    (DataFlow.BENCHMARK, DataSensitivity.PUBLIC): WallOutcome.ALLOW,
    (DataFlow.BENCHMARK, DataSensitivity.INTERNAL): WallOutcome.REDACT,
    # artifact: reports leave the runtime; redact anything non-public.
    (DataFlow.ARTIFACT, DataSensitivity.PUBLIC): WallOutcome.ALLOW,
    (DataFlow.ARTIFACT, DataSensitivity.INTERNAL): WallOutcome.REDACT,
    (DataFlow.ARTIFACT, DataSensitivity.CONFIDENTIAL): WallOutcome.REDACT,
}


class WallDecision(BaseModel):
    """The audited result of one wall check."""

    model_config = {"extra": "forbid"}

    decision_id: str
    audit_event_id: str
    flow: DataFlow
    sensitivity: DataSensitivity
    outcome: WallOutcome
    reasons: list[str] = Field(default_factory=list)
    # Present only for ALLOW/REDACT; None on BLOCK so blocked content
    # cannot leak through the decision object.
    payload: dict[str, Any] | None = None


class DataWall:
    """Deterministic, audited data admission control."""

    def __init__(self, ledger: AuditLedger) -> None:
        self._ledger = ledger

    def check(
        self,
        *,
        flow: DataFlow,
        sensitivity: DataSensitivity,
        payload: dict[str, Any],
        is_holdout: bool = False,
    ) -> WallDecision:
        """Decide whether ``payload`` may enter ``flow``.

        Order of authority: holdout isolation > secret hard-block >
        matrix cell > default BLOCK.
        """
        reasons: list[str] = []

        # 1. Holdout isolation is absolute for training/benchmark.
        if is_holdout and flow in (DataFlow.TRAINING, DataFlow.BENCHMARK):
            outcome = WallOutcome.BLOCK
            reasons.append("HOLDOUT_ISOLATION")
        # 2. secret sensitivity never passes any flow.
        elif sensitivity is DataSensitivity.SECRET:
            outcome = WallOutcome.BLOCK
            reasons.append("SECRET_ALWAYS_BLOCKED")
        else:
            outcome = _MATRIX.get((flow, sensitivity), WallOutcome.BLOCK)
            if (flow, sensitivity) not in _MATRIX:
                reasons.append("DEFAULT_BLOCK_NOT_IN_MATRIX")
            else:
                reasons.append(f"MATRIX_{outcome.value}")

        # 3. Content veto: secret-shaped values force BLOCK even in an
        #    ALLOW cell; in a REDACT cell redaction handles them.
        if outcome is WallOutcome.ALLOW and contains_secret(payload):
            outcome = WallOutcome.BLOCK
            reasons.append("SECRET_SHAPED_CONTENT_DETECTED")

        out_payload: dict[str, Any] | None
        if outcome is WallOutcome.ALLOW:
            out_payload = dict(payload)
        elif outcome is WallOutcome.REDACT:
            out_payload = redact(payload)
            reasons.append("PAYLOAD_REDACTED")
        else:
            out_payload = None

        decision_id = new_decision_id()
        event = self._ledger.append(
            event_type=WALL_EVENT_TYPE,
            actor="data_wall",
            action=f"admit_{flow.value}",
            target=decision_id,
            metadata=redact(
                {
                    "flow": flow.value,
                    "sensitivity": sensitivity.value,
                    "outcome": outcome.value,
                    "reasons": list(reasons),
                    "is_holdout": is_holdout,
                    # never the payload itself; only its key names.
                    "payload_keys": sorted(payload.keys()),
                }
            ),
        )
        return WallDecision(
            decision_id=decision_id,
            audit_event_id=event.audit_id,
            flow=flow,
            sensitivity=sensitivity,
            outcome=outcome,
            reasons=reasons,
            payload=out_payload,
        )
