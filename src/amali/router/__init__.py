"""HER-MoE Local Routing Kernel (IGA-03, §6.9, §8).

Deterministic, configuration-driven routing of tasks over bounded workflow
routes: hard constraints first (§8.2, violations are structural exclusions),
then non-learned weighted scoring (§8.3), then a fully-audited
:class:`RouteDecision` carrying the ranked fallback chain and every
exclusion reason. No neural MoE, no learned weights, no task text.
"""

from __future__ import annotations

from amali.router.constraints import check_hard_constraints
from amali.router.decision import ExcludedRoute, RouteDecision
from amali.router.enums import (
    ModelHealth,
    RouteDecisionStatus,
    RouteExclusionReason,
)
from amali.router.profile import TaskProfile
from amali.router.route import RequiredGates, Route, RouteBudget
from amali.router.router import ROUTE_DECISION_EVENT_TYPE, HERMoERouter
from amali.router.scoring import RouterWeights, ScoreBreakdown, score_route

__all__ = [
    "check_hard_constraints",
    "ExcludedRoute",
    "RouteDecision",
    "ModelHealth",
    "RouteDecisionStatus",
    "RouteExclusionReason",
    "TaskProfile",
    "RequiredGates",
    "Route",
    "RouteBudget",
    "ROUTE_DECISION_EVENT_TYPE",
    "HERMoERouter",
    "RouterWeights",
    "ScoreBreakdown",
    "score_route",
]
