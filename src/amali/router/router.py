"""HER-MoE System Router v0 (IGA-03, §6.9, §8).

Hierarchical Evidence-Risk Mixture-of-Experts at the *system* level: it
routes tasks over bounded workflows, not tokens. v0 is deliberately
deterministic and configuration-driven — no neural gating, no learned
weights, no model consulted. That is the IGA-03 contract: prove the routing
*architecture* (hard constraints -> scoring -> ranked decision -> audit)
before any learning is layered on.

Decision procedure:

1. every route is checked against the §8.2 hard constraints; violators are
   excluded with recorded reasons and can never be selected;
2. survivors are scored deterministically (§8.3) and ranked;
3. if no route survives -> ``NO_ROUTE`` (the Safe Fallback Manager's cue);
   if the winner demands human review without an approval ->
   ``NEEDS_HUMAN_REVIEW``; otherwise -> ``SELECTED``;
4. the full deliberation is written to the hash-chained audit ledger.

The router owns route selection only. It does not execute anything, does not
own truth, and cannot grant tool permissions (that stays with TRPC).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from amali.audit.ledger import AuditLedger
from amali.core.ids import new_route_id
from amali.model_gateway.manifest import ModelManifest
from amali.router.constraints import check_hard_constraints
from amali.router.decision import ExcludedRoute, RouteDecision
from amali.router.enums import ModelHealth, RouteDecisionStatus
from amali.router.profile import TaskProfile
from amali.router.route import Route
from amali.router.scoring import RouterWeights, score_route
from amali.security.redaction import redact

__all__ = ["HERMoERouter", "ROUTE_DECISION_EVENT_TYPE"]

ROUTE_DECISION_EVENT_TYPE = "her_moe.route_decision"


class HERMoERouter:
    """Deterministic, policy-first route selection over authored routes."""

    def __init__(
        self,
        *,
        routes: Sequence[Route],
        models: Mapping[str, ModelManifest],
        ledger: AuditLedger,
        weights: RouterWeights | None = None,
    ) -> None:
        route_ids = [r.route_id for r in routes]
        if len(set(route_ids)) != len(route_ids):
            raise ValueError("route_ids must be unique")
        self._routes = list(routes)
        self._models = dict(models)
        self._ledger = ledger
        self._weights = weights or RouterWeights()

    def route_task(
        self,
        profile: TaskProfile,
        *,
        health: Mapping[str, ModelHealth] | None = None,
        frozen: frozenset[str] | None = None,
    ) -> RouteDecision:
        """Run constraints, score survivors, rank, decide, audit."""
        health = dict(health or {})
        frozen = frozen or frozenset()

        excluded: list[ExcludedRoute] = []
        eligible: list[Route] = []
        for route in self._routes:
            # Defensive re-validation: a route mutated after construction
            # must not influence a decision.
            route.assert_valid()
            reasons = check_hard_constraints(
                route,
                profile,
                models=self._models,
                health=health,
                frozen=frozen,
            )
            if reasons:
                excluded.append(
                    ExcludedRoute(route_id=route.route_id, reasons=reasons)
                )
            else:
                eligible.append(route)

        scores = [
            score_route(
                route,
                profile,
                models=self._models,
                health=health,
                weights=self._weights,
            )
            for route in eligible
        ]
        # Deterministic ranking: score desc, then route_id asc as the
        # tie-breaker so equal scores never rank nondeterministically.
        ranked = sorted(scores, key=lambda s: (-s.total, s.route_id))
        ranked_ids = [s.route_id for s in ranked]

        if not ranked_ids:
            status = RouteDecisionStatus.NO_ROUTE
            selected: str | None = None
        else:
            selected = ranked_ids[0]
            winner = next(r for r in eligible if r.route_id == selected)
            needs_human = winner.required_gates.human_review
            if needs_human and not profile.approval_ref:
                status = RouteDecisionStatus.NEEDS_HUMAN_REVIEW
            else:
                status = RouteDecisionStatus.SELECTED

        decision = RouteDecision(
            route_decision_id=new_route_id(),
            task_id=profile.task_id,
            status=status,
            selected_route_id=selected,
            ranked_route_ids=ranked_ids,
            scores=ranked,
            excluded=excluded,
            weights_version=self._weights.version,
            approval_ref=profile.approval_ref,
        )
        audit_id = self._emit_audit(decision, profile)
        return decision.model_copy(update={"audit_event_id": audit_id})

    def _emit_audit(
        self, decision: RouteDecision, profile: TaskProfile
    ) -> str:
        """Record the full deliberation. Feature metadata only — the router
        never sees task text, so none can leak."""
        metadata: dict[str, Any] = {
            "component": "her_moe_router",
            "route_decision_id": decision.route_decision_id,
            "status": decision.status.value,
            "selected_route_id": decision.selected_route_id,
            "ranked_route_ids": decision.ranked_route_ids,
            "excluded": [
                {
                    "route_id": e.route_id,
                    "reasons": [r.value for r in e.reasons],
                }
                for e in decision.excluded
            ],
            "weights_version": decision.weights_version,
            "task_risk": profile.task_risk.value,
            "data_sensitivity": profile.data_sensitivity.value,
            "offline_only": profile.offline_only,
        }
        event = self._ledger.append(
            event_type=ROUTE_DECISION_EVENT_TYPE,
            actor="her_moe_router",
            action=decision.status.value,
            target=decision.selected_route_id or "none",
            metadata=redact(metadata),
        )
        return event.audit_id
