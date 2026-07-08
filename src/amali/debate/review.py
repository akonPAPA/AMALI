"""Structured debate and review (WP6).

Deterministic, rule-based reviewers examine a completed pipeline pass and
raise typed objections. No model argues here — each reviewer is a fixed
check, so the same inputs always produce the same objections:

* **proposer** frames the proposal (never objects);
* **skeptic** objects to every unsupported or contradicted claim;
* **security** objects to injection-suspicious evidence in the citation set
  and to any secret-exposure flag;
* **cost** objects when the activation plan was infeasible or degraded to a
  fallback;
* **data_wall** objects when any wall BLOCK occurred in the pass;
* **benchmark** objects when a comparison claim lacks a stored baseline.

An objection is *resolved* only when the pipeline's final status already
accounts for it (e.g. an unsupported-claim objection is resolved by a
PARTIAL/NEEDS_REVIEW status — the system did not claim SUCCESS). Anything
unresolved marks the record ``needs_owner``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from amali.arbiter.ewa import ArbitrationResult, ResponseStatus
from amali.retrieval.hsgr import ScoredDoc
from amali.verifier.verify import ClaimVerdict, ClaimVerdictStatus

__all__ = ["Objection", "DebateRecord", "run_debate"]

_NON_SUCCESS = {
    ResponseStatus.PARTIAL,
    ResponseStatus.UNKNOWN,
    ResponseStatus.NEEDS_REVIEW,
    ResponseStatus.BLOCKED,
}


class Objection(BaseModel):
    """One typed reviewer objection and its resolution state."""

    model_config = {"extra": "forbid"}

    reviewer: str
    code: str
    detail: str
    severity: str  # info | warn | high
    resolved: bool
    resolution: str | None = None


class DebateRecord(BaseModel):
    """The full deterministic debate outcome."""

    model_config = {"extra": "forbid"}

    objections: list[Objection] = Field(default_factory=list)
    objection_count: int = 0
    resolved_count: int = 0
    resolution_rate: float = 1.0
    needs_owner: bool = False


def run_debate(
    *,
    verdicts: list[ClaimVerdict],
    arbitration: ArbitrationResult,
    citations: list[ScoredDoc],
    wall_blocks: int = 0,
    activation_feasible: bool = True,
    activation_fallback: bool = False,
    external_baseline_present: bool = False,
    comparison_claimed: bool = False,
) -> DebateRecord:
    """Run every reviewer and compute the resolution state."""
    objections: list[Objection] = []
    status_accounts = arbitration.status in _NON_SUCCESS

    # skeptic: every unsupported/contradicted claim is an objection.
    for verdict in verdicts:
        if verdict.status in (
            ClaimVerdictStatus.UNSUPPORTED,
            ClaimVerdictStatus.CONTRADICTED,
        ):
            resolved = status_accounts
            objections.append(
                Objection(
                    reviewer="skeptic",
                    code=f"CLAIM_{verdict.status.value}",
                    detail=f"claim[{verdict.claim_index}] is "
                    f"{verdict.status.value.lower()}",
                    severity="high",
                    resolved=resolved,
                    resolution=(
                        f"final status {arbitration.status.value} does not "
                        "assert success"
                        if resolved
                        else None
                    ),
                )
            )

    # security: injection-suspicious citations and secret gates.
    for hit in citations:
        if hit.injection_suspicious:
            objections.append(
                Objection(
                    reviewer="security",
                    code="INJECTION_SUSPICIOUS_EVIDENCE",
                    detail=f"cited doc {hit.doc_id} carries injection-shaped "
                    "text; treated as data and penalized",
                    severity="warn",
                    resolved=True,
                    resolution="HSGR labeled and penalized the document; "
                    "text was never executed",
                )
            )
    if "HG2_SECRET_EXPOSURE" in arbitration.hard_gates_fired:
        objections.append(
            Objection(
                reviewer="security",
                code="SECRET_EXPOSURE_GATE",
                detail="secret exposure flag fired",
                severity="high",
                resolved=arbitration.status is ResponseStatus.BLOCKED,
                resolution="EWA hard gate blocked the response"
                if arbitration.status is ResponseStatus.BLOCKED
                else None,
            )
        )

    # cost: activation plan quality.
    if not activation_feasible:
        objections.append(
            Objection(
                reviewer="cost",
                code="ACTIVATION_INFEASIBLE",
                detail="no valid activation plan fit the budget",
                severity="high",
                resolved=status_accounts,
                resolution="pipeline did not claim success"
                if status_accounts
                else None,
            )
        )
    elif activation_fallback:
        objections.append(
            Objection(
                reviewer="cost",
                code="ACTIVATION_FALLBACK",
                detail="degraded fallback pack was activated",
                severity="warn",
                resolved=True,
                resolution="fallback is explicit and capability gaps are "
                "recorded in the activation report",
            )
        )

    # data_wall: blocked admissions in this pass.
    if wall_blocks > 0:
        objections.append(
            Objection(
                reviewer="data_wall",
                code="WALL_BLOCKS_OCCURRED",
                detail=f"{wall_blocks} data-wall BLOCK decision(s) in pass",
                severity="warn",
                resolved=True,
                resolution="blocked data never entered the flow; decisions "
                "are audited",
            )
        )

    # benchmark: comparison honesty.
    if comparison_claimed and not external_baseline_present:
        objections.append(
            Objection(
                reviewer="benchmark",
                code="COMPARISON_WITHOUT_BASELINE",
                detail="an external comparison was requested but no stored "
                "baseline exists",
                severity="high",
                resolved=False,
                resolution=None,
            )
        )

    resolved = sum(1 for o in objections if o.resolved)
    count = len(objections)
    return DebateRecord(
        objections=objections,
        objection_count=count,
        resolved_count=resolved,
        resolution_rate=round(resolved / count, 6) if count else 1.0,
        needs_owner=any(not o.resolved for o in objections),
    )
