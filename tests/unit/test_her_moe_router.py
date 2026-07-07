"""HER-MoE routing kernel (IGA-03) — determinism, constraints, audit.

The central invariant proven here is structural zero-violation (§8.4): a
route excluded by any hard constraint can never be selected, regardless of
score. Plus: deterministic ranking, human-review gating, NO_ROUTE fallback
signal, audit-chain integrity, and route-authoring invariants.
"""

from __future__ import annotations

import pytest

from amali.audit.ledger import AuditLedger
from amali.model_gateway import (
    Capability,
    ModelManifest,
    ModelProvider,
    PrivacyMode,
    PromotionStage,
)
from amali.policy.enums import DataSensitivity, RiskLevel
from amali.router import (
    HERMoERouter,
    ModelHealth,
    RequiredGates,
    Route,
    RouteBudget,
    RouteDecisionStatus,
    RouteExclusionReason,
    RouterWeights,
    TaskProfile,
)


# --------------------------------------------------------------------------
# Fixtures / factories
# --------------------------------------------------------------------------


def local_model(**overrides) -> ModelManifest:
    defaults = dict(
        model_id="local_small",
        version="1.0.0",
        provider=ModelProvider.LOCAL,
        model_type="generator",
        allowed_data_classes=[DataSensitivity.PUBLIC, DataSensitivity.INTERNAL],
        context_window=8192,
        max_tokens=1024,
        capabilities=[Capability.TEXT, Capability.CODE],
        privacy_mode=PrivacyMode.LOCAL_ONLY,
        license="open-weights-local",
        promotion_stage=PromotionStage.EVAL,
    )
    defaults.update(overrides)
    return ModelManifest(**defaults)


def hosted_model(**overrides) -> ModelManifest:
    defaults = dict(
        model_id="hosted_big",
        version="1.0.0",
        provider=ModelProvider.HOSTED,
        model_type="generator",
        allowed_data_classes=[DataSensitivity.PUBLIC, DataSensitivity.INTERNAL],
        context_window=200000,
        max_tokens=8192,
        capabilities=[
            Capability.TEXT,
            Capability.CODE,
            Capability.TOOL_REASONING,
        ],
        privacy_mode=PrivacyMode.HOSTED_ALLOWED,
        license="vendor-hosted",
        promotion_stage=PromotionStage.PRODUCTION,
        owner_approval_ref="approval_owner_1",
        eval_report_refs=["eval_report_1"],
    )
    defaults.update(overrides)
    return ModelManifest(**defaults)


def budget(**overrides) -> RouteBudget:
    defaults = dict(
        max_steps=10,
        max_tool_calls=5,
        max_tokens=8192,
        max_wall_clock_seconds=120,
        max_cost_units=10.0,
    )
    defaults.update(overrides)
    return RouteBudget(**defaults)


def gates(**overrides) -> RequiredGates:
    defaults = dict(
        verifier=True, critic=False, security_review=False, human_review=False
    )
    defaults.update(overrides)
    return RequiredGates(**defaults)


def make_route(**overrides) -> Route:
    defaults = dict(
        route_id="route_local_default",
        agents=["planner", "executor", "verifier"],
        models=["local_small"],
        budget=budget(),
        required_gates=gates(),
        max_task_risk=RiskLevel.L2,
        tool_risk=RiskLevel.L1,
    )
    defaults.update(overrides)
    return Route(**defaults)


def make_profile(**overrides) -> TaskProfile:
    defaults = dict(
        task_id="task_fixed_1",
        task_risk=RiskLevel.L1,
        data_sensitivity=DataSensitivity.PUBLIC,
        required_capabilities=[Capability.TEXT],
        estimated_tokens=1000,
        max_allowed_tool_risk=RiskLevel.L2,
    )
    defaults.update(overrides)
    return TaskProfile(**defaults)


def build_router(routes, *, models=None, weights=None):
    ledger = AuditLedger()
    if models is None:
        models = {m.model_id: m for m in (local_model(), hosted_model())}
    router = HERMoERouter(
        routes=routes, models=models, ledger=ledger, weights=weights
    )
    return router, ledger


# --------------------------------------------------------------------------
# Selection and determinism
# --------------------------------------------------------------------------


def test_selects_highest_scoring_eligible_route():
    low = make_route(route_id="route_low", declared_quality=0.2)
    high = make_route(route_id="route_high", declared_quality=0.9)
    router, _ = build_router([low, high])
    decision = router.route_task(make_profile())
    assert decision.status is RouteDecisionStatus.SELECTED
    assert decision.selected_route_id == "route_high"
    assert decision.ranked_route_ids == ["route_high", "route_low"]


def test_routing_is_deterministic():
    routes = [
        make_route(route_id=f"route_{i}", declared_quality=0.1 * i)
        for i in range(1, 6)
    ]
    router, _ = build_router(routes)
    first = router.route_task(make_profile())
    for _ in range(5):
        again = router.route_task(make_profile())
        assert again.selected_route_id == first.selected_route_id
        assert again.ranked_route_ids == first.ranked_route_ids


def test_equal_scores_tie_break_on_route_id():
    a = make_route(route_id="route_b")
    b = make_route(route_id="route_a")
    router, _ = build_router([a, b])
    decision = router.route_task(make_profile())
    assert decision.selected_route_id == "route_a"  # lexicographic tie-break


def test_no_route_when_all_excluded():
    route = make_route(max_task_risk=RiskLevel.L0)
    router, _ = build_router([route])
    decision = router.route_task(make_profile(task_risk=RiskLevel.L2))
    assert decision.status is RouteDecisionStatus.NO_ROUTE
    assert decision.selected_route_id is None
    assert decision.excluded[0].reasons == [
        RouteExclusionReason.RISK_ABOVE_ROUTE_LIMIT
    ]


# --------------------------------------------------------------------------
# Human review gating
# --------------------------------------------------------------------------


def test_human_review_route_without_approval_needs_review():
    route = make_route(required_gates=gates(human_review=True))
    router, _ = build_router([route])
    decision = router.route_task(make_profile())
    assert decision.status is RouteDecisionStatus.NEEDS_HUMAN_REVIEW
    assert decision.selected_route_id == route.route_id


def test_human_review_route_with_approval_selected():
    route = make_route(required_gates=gates(human_review=True))
    router, _ = build_router([route])
    decision = router.route_task(make_profile(approval_ref="approval_1"))
    assert decision.status is RouteDecisionStatus.SELECTED


# --------------------------------------------------------------------------
# Hard constraints — structural zero-violation
# --------------------------------------------------------------------------


def test_offline_only_excludes_hosted_route():
    local = make_route(route_id="route_local")
    hosted = make_route(
        route_id="route_hosted", models=["hosted_big"], declared_quality=1.0
    )
    router, _ = build_router([local, hosted])
    decision = router.route_task(make_profile(offline_only=True))
    # The hosted route scores higher on quality but is structurally out.
    assert decision.selected_route_id == "route_local"
    hosted_exclusion = next(
        e for e in decision.excluded if e.route_id == "route_hosted"
    )
    assert RouteExclusionReason.PRIVACY_LEAK in hosted_exclusion.reasons


def test_forbidden_data_class_excludes_route():
    router, _ = build_router([make_route()])
    decision = router.route_task(
        make_profile(data_sensitivity=DataSensitivity.CONFIDENTIAL)
    )
    assert decision.status is RouteDecisionStatus.NO_ROUTE
    assert RouteExclusionReason.DATA_CLASS_FORBIDDEN in (
        decision.excluded[0].reasons
    )


def test_tool_risk_above_allowed_excluded():
    route = make_route(
        tool_risk=RiskLevel.L3,
        max_task_risk=RiskLevel.L3,
        required_gates=gates(verifier=True),
    )
    router, _ = build_router([route])
    decision = router.route_task(
        make_profile(max_allowed_tool_risk=RiskLevel.L1)
    )
    assert decision.status is RouteDecisionStatus.NO_ROUTE
    assert RouteExclusionReason.TOOL_RISK_ABOVE_ALLOWED in (
        decision.excluded[0].reasons
    )


def test_owner_freeze_excludes_route_and_model():
    frozen_route = make_route(route_id="route_frozen", declared_quality=1.0)
    frozen_model_route = make_route(
        route_id="route_frozen_model", models=["hosted_big"],
        declared_quality=1.0,
    )
    survivor = make_route(route_id="route_ok", declared_quality=0.1)
    router, _ = build_router([frozen_route, frozen_model_route, survivor])
    decision = router.route_task(
        make_profile(),
        frozen=frozenset({"route_frozen", "hosted_big"}),
    )
    assert decision.selected_route_id == "route_ok"
    assert {e.route_id for e in decision.excluded} == {
        "route_frozen",
        "route_frozen_model",
    }
    for e in decision.excluded:
        assert RouteExclusionReason.OWNER_FROZEN in e.reasons


def test_unhealthy_model_excludes_route():
    router, _ = build_router([make_route()])
    decision = router.route_task(
        make_profile(), health={"local_small": ModelHealth.UNHEALTHY}
    )
    assert decision.status is RouteDecisionStatus.NO_ROUTE
    assert RouteExclusionReason.MODEL_UNHEALTHY in decision.excluded[0].reasons


def test_degraded_model_allowed_but_ranked_down():
    healthy = make_route(route_id="route_healthy", models=["hosted_big"])
    degraded = make_route(route_id="route_degraded")
    router, _ = build_router([healthy, degraded])
    decision = router.route_task(
        make_profile(), health={"local_small": ModelHealth.DEGRADED_ALLOWED}
    )
    # Degraded is still eligible (not excluded)...
    assert not decision.excluded
    degraded_score = next(
        s for s in decision.scores if s.route_id == "route_degraded"
    )
    # ...but its availability component is halved.
    assert degraded_score.availability == 0.5


def test_unknown_model_excludes_route():
    route = make_route(models=["ghost_model"])
    router, _ = build_router([route])
    decision = router.route_task(make_profile())
    assert RouteExclusionReason.MODEL_UNKNOWN in decision.excluded[0].reasons


def test_retired_model_excludes_route():
    retired = local_model(
        model_id="retired_model", promotion_stage=PromotionStage.RETIRED
    )
    route = make_route(models=["retired_model"])
    router, _ = build_router(
        [route], models={"retired_model": retired}
    )
    decision = router.route_task(make_profile())
    assert RouteExclusionReason.MODEL_RETIRED in decision.excluded[0].reasons


def test_missing_capability_excludes_route():
    router, _ = build_router([make_route()])
    decision = router.route_task(
        make_profile(required_capabilities=[Capability.VISION])
    )
    assert RouteExclusionReason.CAPABILITY_UNSUPPORTED in (
        decision.excluded[0].reasons
    )


def test_context_overflow_excludes_unless_compression_plan():
    plain = make_route(route_id="route_plain")
    compressed = make_route(
        route_id="route_compressed", has_compression_plan=True
    )
    router, _ = build_router([plain, compressed])
    decision = router.route_task(make_profile(estimated_tokens=50000))
    assert decision.selected_route_id == "route_compressed"
    plain_exclusion = next(
        e for e in decision.excluded if e.route_id == "route_plain"
    )
    assert RouteExclusionReason.CONTEXT_WINDOW_EXCEEDED in (
        plain_exclusion.reasons
    )


def test_selected_route_is_never_an_excluded_route():
    # Sweep a mixed pool across profiles; the winner must never appear in
    # the excluded list — the structural zero-violation invariant.
    routes = [
        make_route(route_id="r_local", declared_quality=0.3),
        make_route(
            route_id="r_hosted", models=["hosted_big"], declared_quality=1.0
        ),
        make_route(
            route_id="r_risky",
            tool_risk=RiskLevel.L3,
            max_task_risk=RiskLevel.L3,
            declared_quality=1.0,
        ),
    ]
    router, _ = build_router(routes)
    profiles = [
        make_profile(),
        make_profile(offline_only=True),
        make_profile(task_risk=RiskLevel.L2),
        make_profile(max_allowed_tool_risk=RiskLevel.L0),
        make_profile(data_sensitivity=DataSensitivity.INTERNAL),
    ]
    for profile in profiles:
        decision = router.route_task(profile)
        excluded_ids = {e.route_id for e in decision.excluded}
        if decision.selected_route_id is not None:
            assert decision.selected_route_id not in excluded_ids


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------


def test_every_decision_is_audited_and_chain_verifies():
    router, ledger = build_router([make_route()])
    d1 = router.route_task(make_profile())
    d2 = router.route_task(make_profile(offline_only=True))
    events = ledger.events()
    assert len(events) == 2
    assert d1.audit_event_id == events[0].audit_id
    assert d2.audit_event_id == events[1].audit_id
    assert events[0].event_type == "her_moe.route_decision"
    assert events[0].metadata["selected_route_id"] == d1.selected_route_id
    assert ledger.verify_chain() is True


def test_audit_records_exclusion_reasons():
    router, ledger = build_router([make_route(max_task_risk=RiskLevel.L0)])
    router.route_task(make_profile(task_risk=RiskLevel.L2))
    metadata = ledger.events()[0].metadata
    assert metadata["status"] == "NO_ROUTE"
    assert metadata["excluded"][0]["reasons"] == [
        "RISK_ABOVE_ROUTE_LIMIT"
    ]


# --------------------------------------------------------------------------
# Route authoring invariants
# --------------------------------------------------------------------------


def test_high_risk_route_requires_verifier_gate():
    with pytest.raises(ValueError):
        make_route(
            max_task_risk=RiskLevel.L3,
            required_gates=gates(verifier=False),
        )


def test_route_requires_models_and_agents():
    with pytest.raises(ValueError):
        make_route(models=[])
    with pytest.raises(ValueError):
        make_route(agents=[])


def test_budget_must_be_positive():
    with pytest.raises(ValueError):
        budget(max_steps=0)
    with pytest.raises(ValueError):
        budget(max_cost_units=0)


def test_declared_scores_confined_to_unit_interval():
    with pytest.raises(ValueError):
        make_route(declared_quality=1.5)
    with pytest.raises(ValueError):
        make_route(historical_success=-0.1)


def test_duplicate_route_ids_rejected():
    with pytest.raises(ValueError):
        build_router([make_route(), make_route()])
