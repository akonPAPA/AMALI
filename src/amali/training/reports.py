"""Typed reports for the training pipeline (Phase 5/6).

Every report is honest, typed data. There is no field a fake value can
hide in: metrics that were not measured stay ``None``/empty and the
status says why.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["DryRunReport", "TrainingRunReport", "TRAINING_STATUSES"]

TRAINING_STATUSES = (
    "NOT_RUN",
    "DRY_RUN_PASS",
    "TRAINING_STARTED",
    "TRAINING_COMPLETED",
    "TRAINING_FAILED",
    "TRAIN_DEPS_MISSING",
    "MODEL_MISSING",
    "MODEL_NOT_AVAILABLE",
    "DATASET_NOT_READY",
    "CONTAMINATION_FAIL",
    "VRAM_INFEASIBLE",
    "REVISION_NOT_PINNED",
    "NOT_OWNER_APPROVED",
    "OWNER_ACTION_REQUIRED",
)


class DryRunReport(BaseModel):
    """Feasibility gate result. status DRY_RUN_PASS means training may run."""

    model_config = {"extra": "forbid"}

    status: str
    checks: dict[str, str] = Field(default_factory=dict)  # gate -> PASS/FAIL/...
    reasons: list[str] = Field(default_factory=list)
    missing_deps: list[str] = Field(default_factory=list)
    estimated_vram_gb: float | None = None
    vram_budget_gb: float | None = None
    config_hash: str = ""
    dataset_hash: str = ""
    eval_suite_hash: str = ""
    owner_actions: list[str] = Field(default_factory=list)


class TrainingRunReport(BaseModel):
    """What actually happened in one training invocation."""

    model_config = {"extra": "forbid"}

    status: str
    run_id: str = ""
    started_at: str = ""
    ended_at: str = ""
    wall_clock_seconds: float | None = None
    device: str = ""
    gpu_name: str = ""
    peak_vram_gb: float | None = None
    step_count: int = 0
    loss_curve: list[float] = Field(default_factory=list)
    config_hash: str = ""
    dataset_hash: str = ""
    eval_suite_hash: str = ""
    base_model_id: str = ""
    base_model_revision: str = ""
    adapter_dir: str = ""
    adapter_file_hashes: dict[str, str] = Field(default_factory=dict)
    degradations_applied: list[str] = Field(default_factory=list)
    dry_run: DryRunReport | None = None
    risks: list[str] = Field(default_factory=list)
    owner_actions: list[str] = Field(default_factory=list)
