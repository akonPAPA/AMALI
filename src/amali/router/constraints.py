"""Hard constraints — run before scoring, never overridden by score (§8.2).

Each check returns the reasons a route is ineligible for a task. A route
with any reason is excluded from scoring entirely. This ordering is the
structural enforcement of the §8.4 zero-violation metrics: a route that
would violate owner policy, tool policy, or privacy policy is never even a
candidate, so its violation rate is 0 by construction, not by luck.
"""

from __future__ import annotations

from collections.abc import Mapping

from amali.model_gateway.enums import ModelProvider, PromotionStage
from amali.model_gateway.manifest import ModelManifest
from amali.router.enums import ModelHealth, RouteExclusionReason
from amali.router.profile import TaskProfile
from amali.router.route import Route
from amali.policy.enums import risk_rank

__all__ = ["check_hard_constraints"]


def check_hard_constraints(
    route: Route,
    profile: TaskProfile,
    *,
    models: Mapping[str, ModelManifest],
    health: Mapping[str, ModelHealth],
    frozen: frozenset[str],
) -> list[RouteExclusionReason]:
    """Return every §8.2 constraint this route violates for this task.

    An empty list means the route is eligible for scoring. Reasons are
    accumulated (not short-circuited) so the decision record can show *all*
    grounds for exclusion — useful for audit and for route authoring.
    """
    reasons: list[RouteExclusionReason] = []

    # OwnerFreeze(component) = false — the owner can freeze a route or any
    # model it uses, and a frozen component excludes the whole route.
    if route.route_id in frozen or any(m in frozen for m in route.models):
        reasons.append(RouteExclusionReason.OWNER_FROZEN)

    # Policy(r, x) = allow — the task's risk must be within the ceiling this
    # route was authored for.
    if risk_rank(profile.task_risk) > risk_rank(route.max_task_risk):
        reasons.append(RouteExclusionReason.RISK_ABOVE_ROUTE_LIMIT)

    # ToolRisk(r) <= allowed risk for this task.
    if risk_rank(route.tool_risk) > risk_rank(profile.max_allowed_tool_risk):
        reasons.append(RouteExclusionReason.TOOL_RISK_ABOVE_ALLOWED)

    route_manifests: list[ModelManifest] = []
    for model_id in route.models:
        manifest = models.get(model_id)
        if manifest is None:
            reasons.append(RouteExclusionReason.MODEL_UNKNOWN)
            continue
        route_manifests.append(manifest)

        if manifest.promotion_stage is PromotionStage.RETIRED:
            reasons.append(RouteExclusionReason.MODEL_RETIRED)

        # ModelHealth = healthy or degraded_allowed.
        model_health = health.get(model_id, ModelHealth.HEALTHY)
        if model_health is ModelHealth.UNHEALTHY:
            reasons.append(RouteExclusionReason.MODEL_UNHEALTHY)

        # PrivacyLeak(r) = 0 — every model must be allowed to touch data of
        # this sensitivity.
        if not manifest.data_class_allowed(profile.data_sensitivity):
            reasons.append(RouteExclusionReason.DATA_CLASS_FORBIDDEN)

        # OfflineOnly(x) -> hosted_provider = false.
        if profile.offline_only and manifest.provider is ModelProvider.HOSTED:
            reasons.append(RouteExclusionReason.PRIVACY_LEAK)

    if route_manifests:
        # RequiredModality(x) supported — every required capability must be
        # provided by at least one model in the route.
        provided = {cap for m in route_manifests for cap in m.capabilities}
        if any(cap not in provided for cap in profile.required_capabilities):
            reasons.append(RouteExclusionReason.CAPABILITY_UNSUPPORTED)

        # ContextWindow(x) <= model_limit or a compression/retrieval plan
        # exists.
        min_window = min(m.context_window for m in route_manifests)
        if profile.estimated_tokens > min_window and not (
            route.has_compression_plan
        ):
            reasons.append(RouteExclusionReason.CONTEXT_WINDOW_EXCEEDED)

    # Deduplicate while preserving first-seen order.
    seen: set[RouteExclusionReason] = set()
    unique: list[RouteExclusionReason] = []
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            unique.append(reason)
    return unique
