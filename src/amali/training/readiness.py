"""Training readiness structures and gates (WP10).

Honest posture: AMALI is **NOT_READY** to train until every gate passes,
and this stage does not implement training at all. What it implements:

* validated dataset/holdout manifests (versioned, hashed item ids);
* a deterministic contamination check — the intersection of training item
  hashes and holdout item hashes must be empty;
* a readiness report that lists every gate with its true state instead of
  an aspirational summary.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "DatasetManifest",
    "HoldoutManifest",
    "TrainingReadinessReport",
    "check_contamination",
    "training_readiness",
]


class DatasetManifest(BaseModel):
    """A versioned training-dataset declaration (structure only)."""

    model_config = {"extra": "forbid"}

    dataset_id: str
    version: str
    item_hashes: list[str] = Field(default_factory=list)
    source_description: str = ""

    @field_validator("dataset_id", "version")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("field must not be empty")
        return value

    @field_validator("item_hashes")
    @classmethod
    def _hashes_unique(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("item_hashes must be unique")
        return value


class HoldoutManifest(BaseModel):
    """A versioned holdout declaration; isolated by the Data Wall."""

    model_config = {"extra": "forbid"}

    holdout_id: str
    version: str
    item_hashes: list[str] = Field(default_factory=list)

    @field_validator("holdout_id", "version")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("field must not be empty")
        return value


class TrainingReadinessReport(BaseModel):
    """Every readiness gate with its actual state. No aspiration."""

    model_config = {"extra": "forbid"}

    gate_status: str  # READY | NOT_READY
    contamination_free: bool
    contaminated_hashes: list[str] = Field(default_factory=list)
    gates: dict[str, bool] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)


def check_contamination(
    dataset: DatasetManifest, holdout: HoldoutManifest
) -> list[str]:
    """Return the training∩holdout hash intersection (empty = clean)."""
    return sorted(set(dataset.item_hashes) & set(holdout.item_hashes))


def training_readiness(
    dataset: DatasetManifest,
    holdout: HoldoutManifest,
    *,
    eval_suite_exists: bool = False,
    owner_training_approval: bool = False,
) -> TrainingReadinessReport:
    """Compute the honest readiness state. Training needs every gate true."""
    contaminated = check_contamination(dataset, holdout)
    gates = {
        "dataset_manifest_valid": True,  # constructing it validated it
        "holdout_manifest_valid": True,
        "holdout_contamination_free": not contaminated,
        "eval_suite_exists": eval_suite_exists,
        "owner_training_approval": owner_training_approval,
        "training_pipeline_implemented": False,  # honestly: not in scope
    }
    reasons = []
    if contaminated:
        reasons.append(
            f"{len(contaminated)} holdout item(s) leaked into training set"
        )
    if not eval_suite_exists:
        reasons.append("no frozen eval suite registered yet")
    if not owner_training_approval:
        reasons.append("owner has not approved any training run")
    reasons.append(
        "training pipeline is intentionally not implemented in the "
        "Base AI Gate stage"
    )
    return TrainingReadinessReport(
        gate_status="READY" if all(gates.values()) else "NOT_READY",
        contamination_free=not contaminated,
        contaminated_hashes=contaminated,
        gates=gates,
        reasons=reasons,
    )
