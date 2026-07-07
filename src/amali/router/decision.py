"""RouteDecision — the full, auditable record of one routing choice.

The decision records not just the winner but the entire deliberation:
which routes were excluded and *why* (every §8.2 reason), how every eligible
route scored (full component breakdown), and the ranked fallback chain. The
Safe Fallback Manager (§6.36) consumes ``ranked_route_ids`` later; the
Evaluation Orchestrator consumes the breakdowns to compute §8.4 metrics
(route_correct@1, regret_vs_oracle_route, ...).
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from amali.router.enums import RouteDecisionStatus, RouteExclusionReason
from amali.router.scoring import ScoreBreakdown

__all__ = ["ExcludedRoute", "RouteDecision"]


class ExcludedRoute(BaseModel):
    """A route removed by hard constraints, with every reason."""

    model_config = {"extra": "forbid"}

    route_id: str
    reasons: list[RouteExclusionReason]


class RouteDecision(BaseModel):
    """The immutable outcome of one HER-MoE routing pass."""

    model_config = {"extra": "forbid"}

    route_decision_id: str
    task_id: str
    status: RouteDecisionStatus
    selected_route_id: str | None
    ranked_route_ids: list[str] = Field(default_factory=list)
    scores: list[ScoreBreakdown] = Field(default_factory=list)
    excluded: list[ExcludedRoute] = Field(default_factory=list)
    weights_version: str
    approval_ref: str | None = None
    audit_event_id: str | None = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
