"""The Gate A runtime orchestrator (WP8).

Wires the canonical AMALI path over the real components — nothing mocked:

    TaskRequest -> Data Wall -> TRPC permission -> HER-MoE route
    -> DWAC activation plan -> HSGR retrieval -> deterministic answer
    -> claim extraction -> verification -> EWA arbitration -> UGE
    -> structured debate -> failure-to-eval -> audit trail

Every step appends a hash-chained audit event and is recorded in the trace.
Hard failures (wall block, permission deny, no route) short-circuit the
pipeline with an honest terminal status — there is no fail-open branch.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from amali.arbiter.ewa import (
    ArbitrationResult,
    ResponseStatus,
    arbitrate,
    uge_escalate,
)
from amali.audit.ledger import AuditLedger
from amali.activation.planner import (
    ActivationPlan,
    ActivationRequest,
    DWACPlanner,
)
from amali.contracts.task import TaskHandle, TaskRequest, TaskStatus
from amali.data_wall.wall import DataFlow, DataWall, WallDecision, WallOutcome
from amali.debate.review import DebateRecord, run_debate
from amali.eval.fec import FailureTrace, FECResult, compile_failures
from amali.policy.decisions import PermissionDecision, PermissionGrant
from amali.policy.registry import ManifestRegistry
from amali.policy.trpc import compile_permission
from amali.retrieval.hsgr import (
    EvidenceDoc,
    HSGRRetriever,
    RetrievalResult,
)
from amali.router.decision import RouteDecision
from amali.router.enums import RouteDecisionStatus
from amali.router.profile import TaskProfile
from amali.router.router import HERMoERouter
from amali.runtime.answerer import AnswerCandidate, DeterministicAnswerer
from amali.security.redaction import redact
from amali.tools.requests import RequestedScope, ToolInvocationRequest
from amali.verifier.claims import extract_claims
from amali.verifier.verify import ClaimVerdict, ClaimVerdictStatus, verify_claims

__all__ = ["RuntimeOrchestrator", "GateRunResult", "TraceStep"]

RUNTIME_EVENT_TYPE = "runtime.step"

_STATUS_MAP = {
    ResponseStatus.SUCCESS: TaskStatus.SUCCESS,
    ResponseStatus.PARTIAL: TaskStatus.PARTIAL,
    ResponseStatus.UNKNOWN: TaskStatus.UNKNOWN,
    ResponseStatus.NEEDS_REVIEW: TaskStatus.NEEDS_REVIEW,
    ResponseStatus.BLOCKED: TaskStatus.BLOCKED,
}


class TraceStep(BaseModel):
    """One recorded pipeline step."""

    model_config = {"extra": "forbid"}

    step: int
    component: str
    outcome: str
    ref_id: str | None = None
    audit_event_id: str | None = None


class GateRunResult(BaseModel):
    """Everything one Gate A pass produced, for artifacts and assertions."""

    model_config = {"extra": "forbid"}

    handle: TaskHandle
    status: ResponseStatus
    answer_text: str = ""
    trace: list[TraceStep] = Field(default_factory=list)
    wall_decision: WallDecision | None = None
    permission_decision: Any = None  # one of the four TRPC decision types
    route_decision: RouteDecision | None = None
    activation_plan: ActivationPlan | None = None
    retrieval: RetrievalResult | None = None
    candidate: AnswerCandidate | None = None
    verdicts: list[ClaimVerdict] = Field(default_factory=list)
    arbitration: ArbitrationResult | None = None
    debate: DebateRecord | None = None
    fec: FECResult | None = None


class RuntimeOrchestrator:
    """Deterministic Gate A pipeline over injected, real components."""

    def __init__(
        self,
        *,
        ledger: AuditLedger,
        registry: ManifestRegistry,
        router: HERMoERouter,
        planner: DWACPlanner,
        corpus: list[EvidenceDoc],
    ) -> None:
        self._ledger = ledger
        self._registry = registry
        self._router = router
        self._planner = planner
        self._corpus = list(corpus)
        self._retriever = HSGRRetriever(self._corpus)
        self._answerer = DeterministicAnswerer()

    # -- helpers -----------------------------------------------------------

    def _audit_step(
        self,
        handle: TaskHandle,
        component: str,
        outcome: str,
        detail: dict[str, Any] | None = None,
    ) -> str:
        event = self._ledger.append(
            event_type=RUNTIME_EVENT_TYPE,
            actor=handle.actor_id,
            action=component,
            target=handle.task_id,
            metadata=redact({"outcome": outcome, **(detail or {})}),
        )
        handle.audit_event_refs.append(event.audit_id)
        return event.audit_id

    # -- pipeline ----------------------------------------------------------

    def run_task(self, request: TaskRequest) -> GateRunResult:
        """Run one task through the full Gate A path."""
        handle = request.mint_handle()
        trace: list[TraceStep] = []
        failures: list[FailureTrace] = []
        step = 0

        def record(component: str, outcome: str, ref_id: str | None = None,
                   detail: dict[str, Any] | None = None) -> None:
            nonlocal step
            step += 1
            audit_id = self._audit_step(handle, component, outcome, detail)
            trace.append(
                TraceStep(
                    step=step,
                    component=component,
                    outcome=outcome,
                    ref_id=ref_id,
                    audit_event_id=audit_id,
                )
            )

        record("contracts", "TASK_ACCEPTED", handle.task_id)

        # 1. Data Wall ------------------------------------------------------
        wall = DataWall(self._ledger)
        wall_decision = wall.check(
            flow=DataFlow.RUNTIME,
            sensitivity=request.data_sensitivity,
            payload={"query": request.payload.query},
        )
        record(
            "data_wall",
            wall_decision.outcome.value,
            wall_decision.decision_id,
        )
        if wall_decision.outcome is WallOutcome.BLOCK:
            failures.append(
                FailureTrace(
                    task_id=handle.task_id,
                    component="data_wall",
                    failure_kind="WALL_BLOCK",
                    query=request.payload.query,
                    observed="payload blocked at runtime flow",
                )
            )
            handle.status = TaskStatus.BLOCKED
            return GateRunResult(
                handle=handle,
                status=ResponseStatus.BLOCKED,
                trace=trace,
                wall_decision=wall_decision,
                fec=compile_failures(failures),
            )
        query = (wall_decision.payload or {}).get(
            "query", request.payload.query
        )

        # 2. TRPC permission for evidence access ----------------------------
        tool_request = ToolInvocationRequest(
            task_id=handle.task_id,
            actor_ref=handle.actor_id,
            tool_id="local_readonly_filesystem",
            operation="read_file",
            requested_scope=RequestedScope(paths=["./docs"]),
            task_risk=request.task_risk,
            data_sensitivity=request.data_sensitivity,
            reason="retrieve local evidence for task",
            request_source="agent",
        )
        permission: PermissionDecision = compile_permission(
            tool_request, registry=self._registry, ledger=self._ledger
        )
        record(
            "trpc",
            permission.decision.value,
            permission.decision_id,
        )
        if not isinstance(permission, PermissionGrant):
            failures.append(
                FailureTrace(
                    task_id=handle.task_id,
                    component="trpc",
                    failure_kind=f"PERMISSION_{permission.decision.value}",
                    query=query,
                    observed="evidence access was not granted",
                )
            )
            status = (
                ResponseStatus.NEEDS_REVIEW
                if permission.decision.value == "REVIEW_REQUIRED"
                else ResponseStatus.BLOCKED
            )
            handle.status = _STATUS_MAP[status]
            return GateRunResult(
                handle=handle,
                status=status,
                trace=trace,
                wall_decision=wall_decision,
                permission_decision=permission,
                fec=compile_failures(failures),
            )

        # 3. HER-MoE routing -------------------------------------------------
        profile = TaskProfile(
            task_id=handle.task_id,
            task_risk=request.task_risk,
            data_sensitivity=request.data_sensitivity,
            estimated_tokens=max(1, len(query) // 4),
            offline_only=True,
            evidence_required=True,
        )
        route_decision = self._router.route_task(profile)
        record(
            "her_moe_router",
            route_decision.status.value,
            route_decision.route_decision_id,
        )
        if route_decision.status is not RouteDecisionStatus.SELECTED:
            failures.append(
                FailureTrace(
                    task_id=handle.task_id,
                    component="her_moe_router",
                    failure_kind=f"ROUTE_{route_decision.status.value}",
                    query=query,
                    observed="no route selected",
                )
            )
            status = (
                ResponseStatus.NEEDS_REVIEW
                if route_decision.status
                is RouteDecisionStatus.NEEDS_HUMAN_REVIEW
                else ResponseStatus.BLOCKED
            )
            handle.status = _STATUS_MAP[status]
            return GateRunResult(
                handle=handle,
                status=status,
                trace=trace,
                wall_decision=wall_decision,
                permission_decision=permission,
                route_decision=route_decision,
                fec=compile_failures(failures),
            )

        # 4. DWAC activation plan ---------------------------------------------
        plan = self._planner.plan(
            ActivationRequest(
                required_capabilities=["answer_from_evidence"],
                budget_mb=4096,
            )
        )
        record(
            "dwac",
            "FEASIBLE" if plan.feasible else "INFEASIBLE",
            ",".join(plan.selected_pack_ids) or None,
        )

        # 5. HSGR retrieval ---------------------------------------------------
        retrieval = self._retriever.retrieve(query)
        record(
            "hsgr_retrieval",
            f"RANKED_{len(retrieval.ranked)}",
            retrieval.top().doc_id if retrieval.top() else None,
        )

        # 6. Deterministic answer ----------------------------------------------
        candidate = self._answerer.answer(query, self._corpus)
        record(
            "answerer",
            "ANSWERED" if candidate.answered else "NO_ANSWER",
        )

        # 7. Verify ------------------------------------------------------------
        claims = extract_claims(candidate.text) if candidate.answered else []
        verdicts = verify_claims(claims, self._corpus)
        record("verifier", f"VERDICTS_{len(verdicts)}")

        # 8. EWA + UGE ----------------------------------------------------------
        arbitration = arbitrate(verdicts)
        arbitration = uge_escalate(arbitration, task_risk=request.task_risk)
        record("ewa_arbiter", arbitration.status.value)

        # 9. Debate --------------------------------------------------------------
        debate = run_debate(
            verdicts=verdicts,
            arbitration=arbitration,
            citations=retrieval.ranked,
            wall_blocks=0,
            activation_feasible=plan.feasible,
            activation_fallback=plan.fallback_used,
        )
        record(
            "debate",
            "NEEDS_OWNER" if debate.needs_owner else "RESOLVED",
        )

        # 10. Failure-to-eval ------------------------------------------------------
        for verdict in verdicts:
            if verdict.status in (
                ClaimVerdictStatus.UNSUPPORTED,
                ClaimVerdictStatus.CONTRADICTED,
            ):
                failures.append(
                    FailureTrace(
                        task_id=handle.task_id,
                        component="verifier",
                        failure_kind=f"CLAIM_{verdict.status.value}",
                        query=query,
                        observed=verdict.claim_text,
                    )
                )
        if not candidate.answered:
            failures.append(
                FailureTrace(
                    task_id=handle.task_id,
                    component="answerer",
                    failure_kind="NO_EVIDENCE_ANSWER",
                    query=query,
                    observed="no evidence sentence covered the query",
                )
            )
        fec = compile_failures(failures)
        record("fec", f"EVAL_ITEMS_{fec.unique_items}")

        handle.status = _STATUS_MAP[arbitration.status]
        record("runtime", handle.status.value)

        return GateRunResult(
            handle=handle,
            status=arbitration.status,
            answer_text=candidate.text,
            trace=trace,
            wall_decision=wall_decision,
            permission_decision=permission,
            route_decision=route_decision,
            activation_plan=plan,
            retrieval=retrieval,
            candidate=candidate,
            verdicts=verdicts,
            arbitration=arbitration,
            debate=debate,
            fec=fec,
        )
