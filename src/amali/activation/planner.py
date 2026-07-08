"""AMALI-DWAC activation planning (WP4). Label: CUSTOM_AMALI_COMPOSITION.

Algorithm (deterministic greedy cover, fully re-derivable):

1. validate the request (budget > 0, capabilities non-empty);
2. candidate packs are sorted by ``(cost_mb ascending, pack_id ascending)``
   — the tie-break is lexical and total, so the same inputs always produce
   the same plan;
3. walk the sorted candidates; take a pack only if it covers at least one
   still-uncovered required capability and fits the remaining budget
   (minimality: no pack that adds nothing is ever selected);
4. if all capabilities are covered -> feasible plan;
5. else, if a designated fallback pack fits the budget -> fallback plan
   (explicitly marked; capabilities it lacks are reported as uncovered);
6. else -> infeasible result with reasons. **No invalid winner**: a returned
   plan never exceeds the budget and never contains an unknown pack.

This is resource planning only. Nothing is loaded, no weights are touched,
no model is invoked.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

__all__ = ["PackSpec", "ActivationRequest", "ActivationPlan", "DWACPlanner"]


class PackSpec(BaseModel):
    """One activatable pack (base model, adapter, or expert pack)."""

    model_config = {"extra": "forbid"}

    pack_id: str
    kind: str  # base_model | adapter | expert_pack
    cost_mb: int
    capabilities: list[str] = Field(default_factory=list)

    @field_validator("pack_id")
    @classmethod
    def _id_not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("pack_id must not be empty")
        return value

    @field_validator("kind")
    @classmethod
    def _kind_known(cls, value: str) -> str:
        if value not in {"base_model", "adapter", "expert_pack"}:
            raise ValueError("kind must be base_model|adapter|expert_pack")
        return value

    @field_validator("cost_mb")
    @classmethod
    def _cost_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("cost_mb must be positive")
        return value


class ActivationRequest(BaseModel):
    """What the task needs and what resources it may consume."""

    model_config = {"extra": "forbid"}

    required_capabilities: list[str]
    budget_mb: int

    @field_validator("required_capabilities")
    @classmethod
    def _caps_non_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("required_capabilities must not be empty")
        return value

    @field_validator("budget_mb")
    @classmethod
    def _budget_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("budget_mb must be positive")
        return value


class ActivationPlan(BaseModel):
    """The planned activation set and its resource accounting."""

    model_config = {"extra": "forbid"}

    feasible: bool
    fallback_used: bool = False
    selected_pack_ids: list[str] = Field(default_factory=list)
    total_cost_mb: int = 0
    budget_mb: int = 0
    covered_capabilities: list[str] = Field(default_factory=list)
    uncovered_capabilities: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class DWACPlanner:
    """Deterministic minimal-sufficient activation planning under budget."""

    def __init__(
        self,
        packs: list[PackSpec],
        *,
        fallback_pack_id: str | None = None,
    ) -> None:
        ids = [p.pack_id for p in packs]
        if len(set(ids)) != len(ids):
            raise ValueError("pack_ids must be unique")
        if fallback_pack_id is not None and fallback_pack_id not in ids:
            raise ValueError("fallback_pack_id must reference a known pack")
        self._packs = {p.pack_id: p for p in packs}
        self._fallback_pack_id = fallback_pack_id

    def plan(self, request: ActivationRequest) -> ActivationPlan:
        required = sorted(set(request.required_capabilities))
        budget = request.budget_mb

        # Deterministic candidate order: cheapest first, then pack_id.
        candidates = sorted(
            self._packs.values(), key=lambda p: (p.cost_mb, p.pack_id)
        )

        selected: list[PackSpec] = []
        covered: set[str] = set()
        total = 0
        for pack in candidates:
            gain = (set(pack.capabilities) & set(required)) - covered
            if not gain:
                continue  # minimality: never select a pack that adds nothing
            if total + pack.cost_mb > budget:
                continue  # never exceed budget
            selected.append(pack)
            covered |= gain
            total += pack.cost_mb
            if covered == set(required):
                break

        if covered == set(required):
            return ActivationPlan(
                feasible=True,
                selected_pack_ids=[p.pack_id for p in selected],
                total_cost_mb=total,
                budget_mb=budget,
                covered_capabilities=sorted(covered),
                uncovered_capabilities=[],
                reasons=["GREEDY_COVER_COMPLETE"],
            )

        # Fallback: an explicitly designated degraded pack, if it fits.
        if self._fallback_pack_id is not None:
            fallback = self._packs[self._fallback_pack_id]
            if fallback.cost_mb <= budget:
                fb_covered = set(fallback.capabilities) & set(required)
                return ActivationPlan(
                    feasible=True,
                    fallback_used=True,
                    selected_pack_ids=[fallback.pack_id],
                    total_cost_mb=fallback.cost_mb,
                    budget_mb=budget,
                    covered_capabilities=sorted(fb_covered),
                    uncovered_capabilities=sorted(
                        set(required) - fb_covered
                    ),
                    reasons=["FALLBACK_PACK_ACTIVATED"],
                )

        return ActivationPlan(
            feasible=False,
            selected_pack_ids=[],
            total_cost_mb=0,
            budget_mb=budget,
            covered_capabilities=[],
            uncovered_capabilities=sorted(set(required) - covered),
            reasons=["BUDGET_INFEASIBLE_NO_VALID_PLAN"],
        )
