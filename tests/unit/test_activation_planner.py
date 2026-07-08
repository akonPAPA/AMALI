"""DWAC: budget feasibility, tie-breaks, fallback, no invalid winner."""

import pytest
from pydantic import ValidationError

from amali.activation import (
    ActivationRequest,
    DWACPlanner,
    PackSpec,
)


def _packs() -> list[PackSpec]:
    return [
        PackSpec(
            pack_id="base_small",
            kind="base_model",
            cost_mb=3000,
            capabilities=["generate"],
        ),
        PackSpec(
            pack_id="base_big",
            kind="base_model",
            cost_mb=6000,
            capabilities=["generate", "reason"],
        ),
        PackSpec(
            pack_id="adapter_code",
            kind="adapter",
            cost_mb=500,
            capabilities=["code"],
        ),
        PackSpec(
            pack_id="expert_math",
            kind="expert_pack",
            cost_mb=800,
            capabilities=["math"],
        ),
    ]


def test_feasible_minimal_plan_within_budget():
    planner = DWACPlanner(_packs())
    plan = planner.plan(
        ActivationRequest(
            required_capabilities=["generate", "code"], budget_mb=4000
        )
    )
    assert plan.feasible is True
    assert plan.fallback_used is False
    assert set(plan.selected_pack_ids) == {"base_small", "adapter_code"}
    assert plan.total_cost_mb == 3500
    assert plan.total_cost_mb <= plan.budget_mb
    assert plan.uncovered_capabilities == []


def test_no_useless_pack_selected():
    planner = DWACPlanner(_packs())
    plan = planner.plan(
        ActivationRequest(required_capabilities=["generate"], budget_mb=10000)
    )
    # Minimality: only one pack needed; nothing extra activated.
    assert plan.selected_pack_ids == ["base_small"]


def test_deterministic_tie_break_on_pack_id():
    packs = [
        PackSpec(pack_id="b_pack", kind="adapter", cost_mb=100,
                 capabilities=["x"]),
        PackSpec(pack_id="a_pack", kind="adapter", cost_mb=100,
                 capabilities=["x"]),
    ]
    planner = DWACPlanner(packs)
    plans = [
        planner.plan(
            ActivationRequest(required_capabilities=["x"], budget_mb=200)
        )
        for _ in range(3)
    ]
    # Same cost -> lexically smaller pack_id always wins.
    assert all(p.selected_pack_ids == ["a_pack"] for p in plans)


def test_fallback_used_when_cover_infeasible():
    planner = DWACPlanner(_packs(), fallback_pack_id="base_small")
    plan = planner.plan(
        ActivationRequest(
            required_capabilities=["generate", "reason"], budget_mb=3200
        )
    )
    # base_big (6000) does not fit; fallback is explicit and marked.
    assert plan.feasible is True
    assert plan.fallback_used is True
    assert plan.selected_pack_ids == ["base_small"]
    assert "reason" in plan.uncovered_capabilities


def test_infeasible_when_nothing_fits():
    planner = DWACPlanner(_packs())
    plan = planner.plan(
        ActivationRequest(required_capabilities=["generate"], budget_mb=100)
    )
    assert plan.feasible is False
    assert plan.selected_pack_ids == []
    assert "BUDGET_INFEASIBLE_NO_VALID_PLAN" in plan.reasons


def test_no_invalid_winner_budget_never_exceeded():
    planner = DWACPlanner(_packs(), fallback_pack_id="base_small")
    for budget in (100, 3000, 3500, 4000, 10000):
        plan = planner.plan(
            ActivationRequest(
                required_capabilities=["generate", "code", "math"],
                budget_mb=budget,
            )
        )
        if plan.selected_pack_ids:
            assert plan.total_cost_mb <= budget
            for pack_id in plan.selected_pack_ids:
                assert pack_id in {p.pack_id for p in _packs()}


def test_invalid_request_rejected():
    with pytest.raises(ValidationError):
        ActivationRequest(required_capabilities=[], budget_mb=100)
    with pytest.raises(ValidationError):
        ActivationRequest(required_capabilities=["x"], budget_mb=0)


def test_unknown_fallback_rejected():
    with pytest.raises(ValueError):
        DWACPlanner(_packs(), fallback_pack_id="ghost")
