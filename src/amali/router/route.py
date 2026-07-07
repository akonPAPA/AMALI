"""The Route object (§8.1) — a bounded workflow, not a model pick.

HER-MoE routes tasks over *workflows*: a route names the agents involved,
the model manifests and tools it may use, its hard budget, and the
verification gates its output must pass. Routes are configuration — they are
authored, versioned and reviewed, never generated at runtime by a model.

Invariants enforced here:

* budgets are strictly positive — an unbounded route cannot exist;
* a route must name at least one model and one agent;
* declared score inputs are confined to [0, 1] so no route can bribe the
  scorer with an out-of-range constant;
* a route whose task-risk ceiling is high (L3+) must require the verifier
  gate — high-risk work without verification is not a valid configuration.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from amali.core.errors import ValidationBoundaryError
from amali.policy.enums import RiskLevel, risk_rank

__all__ = ["RouteBudget", "RequiredGates", "Route"]


class RouteBudget(BaseModel):
    """Hard resource ceilings for one route (§8.1)."""

    model_config = {"extra": "forbid"}

    max_steps: int
    max_tool_calls: int
    max_tokens: int
    max_wall_clock_seconds: int
    max_cost_units: float

    @model_validator(mode="after")
    def _validate(self) -> "RouteBudget":
        for name in (
            "max_steps",
            "max_tool_calls",
            "max_tokens",
            "max_wall_clock_seconds",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.max_cost_units <= 0:
            raise ValueError("max_cost_units must be > 0")
        return self


class RequiredGates(BaseModel):
    """Verification gates a route's output must pass (§8.1)."""

    model_config = {"extra": "forbid"}

    verifier: bool
    critic: bool
    security_review: bool
    human_review: bool


class Route(BaseModel):
    """A versioned, bounded workflow the router may select (§8.1)."""

    model_config = {"extra": "forbid"}

    route_id: str
    agents: list[str]
    models: list[str]
    tools: list[str] = Field(default_factory=list)
    retrieval_policy_id: str | None = None
    memory_policy_id: str | None = None
    eval_plan_id: str | None = None
    budget: RouteBudget
    required_gates: RequiredGates
    # Policy ceilings for this route.
    max_task_risk: RiskLevel
    tool_risk: RiskLevel = RiskLevel.L0
    # Declared, config-owned score inputs in [0, 1]. Per §8.3 these are
    # "config, not truth": tuned only after eval data exists.
    declared_quality: float = 0.5
    declared_latency_score: float = 0.5
    historical_success: float = 0.5
    has_compression_plan: bool = False

    @model_validator(mode="after")
    def _validate(self) -> "Route":
        try:
            self.assert_valid()
        except ValidationBoundaryError as exc:
            raise ValueError(str(exc)) from exc
        return self

    def assert_valid(self) -> None:
        """Re-assert every hard route invariant."""
        if not self.route_id or not self.route_id.strip():
            raise ValidationBoundaryError("route_id must not be empty")
        if not self.agents:
            raise ValidationBoundaryError("agents must be non-empty")
        if not self.models:
            raise ValidationBoundaryError("models must be non-empty")
        for name in (
            "declared_quality",
            "declared_latency_score",
            "historical_success",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValidationBoundaryError(
                    f"{name} must be within [0, 1]"
                )
        # High-risk routes must gate through the verifier.
        if risk_rank(self.max_task_risk) >= risk_rank(RiskLevel.L3) and not (
            self.required_gates.verifier
        ):
            raise ValidationBoundaryError(
                "routes with max_task_risk >= L3 must require the verifier gate"
            )
