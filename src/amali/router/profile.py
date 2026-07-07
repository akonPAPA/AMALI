"""TaskProfile — the routing input ``x`` (§8.2/§8.3).

A profile is produced upstream (risk classifier, evidence-need classifier,
ingress validation) and is the only task-side information the router sees.
The router never reads raw task text: routing on classified features, not
free text, is what keeps the routing kernel deterministic and injection-free.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from amali.core.errors import ValidationBoundaryError
from amali.model_gateway.enums import Capability
from amali.policy.enums import DataSensitivity, RiskLevel

__all__ = ["TaskProfile"]


class TaskProfile(BaseModel):
    """Classified task features the router decides over."""

    model_config = {"extra": "forbid"}

    task_id: str
    task_risk: RiskLevel
    data_sensitivity: DataSensitivity
    required_capabilities: list[Capability] = Field(default_factory=list)
    estimated_tokens: int
    offline_only: bool = False
    evidence_required: bool = False
    approval_ref: str | None = None
    # Ceiling from the policy layer: the highest tool risk this task may use.
    max_allowed_tool_risk: RiskLevel = RiskLevel.L1

    @model_validator(mode="after")
    def _validate(self) -> "TaskProfile":
        if not self.task_id or not self.task_id.strip():
            raise ValidationBoundaryError("task_id must not be empty")
        if self.estimated_tokens <= 0:
            raise ValidationBoundaryError("estimated_tokens must be > 0")
        return self
