"""RICARDO-EWA — Reproducible Invariant-Checked Evidence-Weighted Arbitration.

Turns claim verdicts + risk context into one final response status. Hard
gates run first and can never be overridden by any score:

    HG1: any CONTRADICTED claim                    -> NEEDS_REVIEW
    HG2: secret/raw-secret exposure flag           -> BLOCKED
    HG3: no claims at all                          -> UNKNOWN
    HG4: every claim INSUFFICIENT_EVIDENCE         -> UNKNOWN
    HG5: any UNSUPPORTED claim                     -> never SUCCESS
         (all-supported-but-some-unsupported mix   -> PARTIAL)

Score (recorded, informational — used only *below* the gates):

    ewa_score = mean(support_i) over verified claims, in [0, 1]

AMALI-UGE (Uncertainty-Gated Escalation) then applies risk gating on the
uncertainty ``u = 1 - ewa_score``:

    risk >= L3 and u > 0.3   -> escalate to NEEDS_REVIEW
    risk >= L4               -> always NEEDS_REVIEW unless BLOCKED
    otherwise                -> keep the arbitration status

Invariant proven by tests: an unsupported factual claim can never produce
SUCCESS, no matter the score.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from amali.policy.enums import RiskLevel, risk_at_least
from amali.verifier.verify import ClaimVerdict, ClaimVerdictStatus

__all__ = ["ResponseStatus", "ArbitrationResult", "arbitrate", "uge_escalate"]


class ResponseStatus(str, Enum):
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    UNKNOWN = "UNKNOWN"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    BLOCKED = "BLOCKED"


class ArbitrationResult(BaseModel):
    """The final, gate-checked arbitration outcome."""

    model_config = {"extra": "forbid"}

    status: ResponseStatus
    ewa_score: float
    hard_gates_fired: list[str] = Field(default_factory=list)
    supported: int = 0
    unsupported: int = 0
    contradicted: int = 0
    insufficient: int = 0


def _ewa_score(verdicts: list[ClaimVerdict]) -> float:
    if not verdicts:
        return 0.0
    return round(sum(v.support_score for v in verdicts) / len(verdicts), 6)


def arbitrate(
    verdicts: list[ClaimVerdict],
    *,
    secret_exposure_flag: bool = False,
) -> ArbitrationResult:
    """Apply the hard gates, then the score. Gates always win."""
    counts = {
        ClaimVerdictStatus.SUPPORTED: 0,
        ClaimVerdictStatus.UNSUPPORTED: 0,
        ClaimVerdictStatus.CONTRADICTED: 0,
        ClaimVerdictStatus.INSUFFICIENT_EVIDENCE: 0,
    }
    for verdict in verdicts:
        counts[verdict.status] += 1

    score = _ewa_score(verdicts)
    gates: list[str] = []

    # HG2 first: secret exposure blocks everything else outright.
    if secret_exposure_flag:
        gates.append("HG2_SECRET_EXPOSURE")
        status = ResponseStatus.BLOCKED
    elif counts[ClaimVerdictStatus.CONTRADICTED] > 0:
        gates.append("HG1_CONTRADICTED_CLAIM")
        status = ResponseStatus.NEEDS_REVIEW
    elif not verdicts:
        gates.append("HG3_NO_CLAIMS")
        status = ResponseStatus.UNKNOWN
    elif counts[ClaimVerdictStatus.INSUFFICIENT_EVIDENCE] == len(verdicts):
        gates.append("HG4_NO_EVIDENCE")
        status = ResponseStatus.UNKNOWN
    elif counts[ClaimVerdictStatus.UNSUPPORTED] > 0 or (
        counts[ClaimVerdictStatus.INSUFFICIENT_EVIDENCE] > 0
    ):
        # HG5: unsupported claims can never be SUCCESS, whatever the score.
        gates.append("HG5_UNSUPPORTED_NEVER_SUCCESS")
        status = (
            ResponseStatus.PARTIAL
            if counts[ClaimVerdictStatus.SUPPORTED] > 0
            else ResponseStatus.UNKNOWN
        )
    else:
        status = ResponseStatus.SUCCESS

    return ArbitrationResult(
        status=status,
        ewa_score=score,
        hard_gates_fired=gates,
        supported=counts[ClaimVerdictStatus.SUPPORTED],
        unsupported=counts[ClaimVerdictStatus.UNSUPPORTED],
        contradicted=counts[ClaimVerdictStatus.CONTRADICTED],
        insufficient=counts[ClaimVerdictStatus.INSUFFICIENT_EVIDENCE],
    )


def uge_escalate(
    result: ArbitrationResult,
    *,
    task_risk: RiskLevel,
) -> ArbitrationResult:
    """AMALI-UGE: escalate on uncertainty x risk. Never de-escalates.

    BLOCKED stays BLOCKED. High-risk tasks with uncertain evidence go to
    NEEDS_REVIEW; L4+ always requires review unless already blocked.
    """
    if result.status is ResponseStatus.BLOCKED:
        return result

    uncertainty = round(1.0 - result.ewa_score, 6)
    escalate = False
    if risk_at_least(task_risk, RiskLevel.L4):
        escalate = True
        gate = "UGE_L4_ALWAYS_REVIEW"
    elif risk_at_least(task_risk, RiskLevel.L3) and uncertainty > 0.3:
        escalate = True
        gate = "UGE_HIGH_RISK_UNCERTAIN"

    if not escalate or result.status is ResponseStatus.NEEDS_REVIEW:
        return result

    return ArbitrationResult(
        status=ResponseStatus.NEEDS_REVIEW,
        ewa_score=result.ewa_score,
        hard_gates_fired=[*result.hard_gates_fired, gate],
        supported=result.supported,
        unsupported=result.unsupported,
        contradicted=result.contradicted,
        insufficient=result.insufficient,
    )
