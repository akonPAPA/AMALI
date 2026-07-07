"""Closed vocabularies for the HER-MoE routing kernel (IGA-03, §8).

Routing decisions reason only over these closed enums — a route or task
profile cannot smuggle in an unknown status, health state, or exclusion
reason the scorer has not accounted for.
"""

from __future__ import annotations

from enum import Enum

__all__ = ["RouteDecisionStatus", "ModelHealth", "RouteExclusionReason"]


class RouteDecisionStatus(str, Enum):
    """Outcome of one routing decision."""

    SELECTED = "SELECTED"
    NEEDS_HUMAN_REVIEW = "NEEDS_HUMAN_REVIEW"
    NO_ROUTE = "NO_ROUTE"


class ModelHealth(str, Enum):
    """Runtime health signal for a model (§8.2 ModelHealth constraint)."""

    HEALTHY = "healthy"
    DEGRADED_ALLOWED = "degraded_allowed"
    UNHEALTHY = "unhealthy"


class RouteExclusionReason(str, Enum):
    """Why a candidate route was excluded by a hard constraint (§8.2).

    Hard constraints run *before* scoring: an excluded route can never be
    selected no matter how well it would have scored. This is how the
    required violation metrics (owner/tool/privacy violation rate = 0) are
    enforced structurally rather than statistically.
    """

    RISK_ABOVE_ROUTE_LIMIT = "RISK_ABOVE_ROUTE_LIMIT"
    TOOL_RISK_ABOVE_ALLOWED = "TOOL_RISK_ABOVE_ALLOWED"
    DATA_CLASS_FORBIDDEN = "DATA_CLASS_FORBIDDEN"
    PRIVACY_LEAK = "PRIVACY_LEAK"
    OWNER_FROZEN = "OWNER_FROZEN"
    MODEL_UNHEALTHY = "MODEL_UNHEALTHY"
    MODEL_UNKNOWN = "MODEL_UNKNOWN"
    MODEL_RETIRED = "MODEL_RETIRED"
    CAPABILITY_UNSUPPORTED = "CAPABILITY_UNSUPPORTED"
    CONTEXT_WINDOW_EXCEEDED = "CONTEXT_WINDOW_EXCEEDED"
