"""Base Model Readiness decision (Base Model Readiness layer).

Pure, torch-free aggregation of the readiness evidence into one typed
verdict. BASE_MODEL_READY requires every prerequisite; the first missing
prerequisite names the verdict, so the owner always sees the exact next
blocker instead of a generic NOT_READY.

BASE_MODEL_READY never implies AMALI-FT-v0 exists: the FT verdict is a
separate field and stays AMALI_FT_NOT_READY unless a real training run
was completed, verified, evaluated, safety-regressed, and promoted.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = [
    "BaseModelReadinessInputs",
    "BaseModelReadinessReport",
    "decide_base_model_readiness",
]

STATUS_READY = "BASE_MODEL_READY"
STATUS_NOT_READY = "BASE_MODEL_NOT_READY"
STATUS_OWNER_REVISION = "OWNER_MODEL_REVISION_REQUIRED"
STATUS_MODEL_NOT_AVAILABLE = "MODEL_NOT_AVAILABLE"
STATUS_GATEWAY_FAILED = "MODEL_GATEWAY_FAILED"
STATUS_RAW_EVAL_NOT_RUN = "RAW_EVAL_NOT_RUN"
STATUS_WRAPPED_EVAL_NOT_RUN = "WRAPPED_EVAL_NOT_RUN"
STATUS_SECURITY_FAILED = "SECURITY_VALIDATION_FAILED"
STATUS_CORE_REGRESSION = "CORE_REGRESSION_FAILED"

FT_NOT_READY = "AMALI_FT_NOT_READY"
FT_READY = "AMALI_FT_READY"


class BaseModelReadinessInputs(BaseModel):
    """Statuses gathered from the newest typed artifacts. NOT_RUN when an
    artifact does not exist — absence is evidence of absence here."""

    model_config = {"extra": "forbid", "protected_namespaces": ()}

    gate_a_status: str = "NOT_RUN"  # PASS expected
    core_import_torch_free: bool = False
    revision_pin_status: str = "NOT_RUN"  # REVISION_PINNED expected
    model_availability_status: str = "NOT_RUN"  # AVAILABLE expected
    gateway_smoke_status: str = "NOT_RUN"  # SUCCESS expected
    raw_eval_status: str = "NOT_RUN"  # PASS expected
    wrapped_eval_status: str = "NOT_RUN"  # PASS expected
    comparison_status: str = "NOT_RUN"  # PASS expected
    security_validation_status: str = "NOT_RUN"  # PASS expected
    promotion_decision: str = "NOT_RUN"  # PROMOTED means FT exists


class BaseModelReadinessReport(BaseModel):
    """The single typed readiness verdict plus the full check table."""

    model_config = {"extra": "forbid", "protected_namespaces": ()}

    status: str
    base_model_ready: bool = False
    amali_ft_status: str = FT_NOT_READY
    checks: dict[str, str] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    owner_actions: list[str] = Field(default_factory=list)


def decide_base_model_readiness(
    inputs: BaseModelReadinessInputs,
) -> BaseModelReadinessReport:
    checks = {
        "gate_a": inputs.gate_a_status,
        "core_import_torch_free": "PASS" if inputs.core_import_torch_free else "FAIL",
        "revision_pin": inputs.revision_pin_status,
        "model_availability": inputs.model_availability_status,
        "gateway_smoke": inputs.gateway_smoke_status,
        "raw_base_eval": inputs.raw_eval_status,
        "wrapped_raw_base_eval": inputs.wrapped_eval_status,
        "comparison": inputs.comparison_status,
        "security_validation": inputs.security_validation_status,
    }
    reasons: list[str] = []
    owner_actions: list[str] = []

    # Ordered: the first failing prerequisite names the verdict.
    if inputs.gate_a_status != "PASS" or not inputs.core_import_torch_free:
        status = STATUS_CORE_REGRESSION
        if inputs.gate_a_status != "PASS":
            reasons.append(f"Gate A status: {inputs.gate_a_status}")
            owner_actions.append("python scripts/run_base_ai_gate_demo.py")
        if not inputs.core_import_torch_free:
            reasons.append("core import loads torch (lazy boundary broken)")
    elif inputs.revision_pin_status != "REVISION_PINNED":
        status = STATUS_OWNER_REVISION
        reasons.append(
            f"revision pin status: {inputs.revision_pin_status}; no exact "
            "snapshot commit SHA is pinned"
        )
        owner_actions.append(
            "python scripts/pin_base_model_revision.py --model "
            "Qwen/Qwen2.5-1.5B-Instruct --revision <HF_SNAPSHOT_COMMIT_SHA>"
        )
    elif inputs.model_availability_status != "AVAILABLE":
        status = STATUS_MODEL_NOT_AVAILABLE
        reasons.append(
            f"local model availability: {inputs.model_availability_status}"
        )
        owner_actions.append(
            "python scripts/download_model.py --model "
            "Qwen/Qwen2.5-1.5B-Instruct --revision <HF_SNAPSHOT_COMMIT_SHA>"
        )
    elif inputs.gateway_smoke_status != "SUCCESS":
        status = STATUS_GATEWAY_FAILED
        reasons.append(f"gateway smoke: {inputs.gateway_smoke_status}")
        owner_actions.append(
            "python scripts/run_model_gateway_smoke.py --model "
            "Qwen/Qwen2.5-1.5B-Instruct --revision <HF_SNAPSHOT_COMMIT_SHA> "
            "--local-files-only"
        )
    elif inputs.raw_eval_status != "PASS":
        status = STATUS_RAW_EVAL_NOT_RUN
        reasons.append(f"raw base eval: {inputs.raw_eval_status}")
        owner_actions.append(
            "python scripts/evaluate_model.py --model raw_base --model-id "
            "Qwen/Qwen2.5-1.5B-Instruct --revision <HF_SNAPSHOT_COMMIT_SHA> "
            "--local-files-only"
        )
    elif inputs.wrapped_eval_status != "PASS":
        status = STATUS_WRAPPED_EVAL_NOT_RUN
        reasons.append(f"wrapped raw base eval: {inputs.wrapped_eval_status}")
        owner_actions.append(
            "python scripts/evaluate_model.py --model amali_wrapped_raw_base "
            "--model-id Qwen/Qwen2.5-1.5B-Instruct --revision "
            "<HF_SNAPSHOT_COMMIT_SHA> --local-files-only"
        )
    elif inputs.comparison_status not in ("PASS", "NOT_RUN"):
        # A FAIL here means the eval reports disagree on suite/model/revision.
        status = STATUS_NOT_READY
        reasons.append(f"comparison: {inputs.comparison_status}")
        owner_actions.append("python scripts/compare_models.py")
    elif inputs.comparison_status == "NOT_RUN":
        status = STATUS_NOT_READY
        reasons.append("raw vs wrapped comparison report missing")
        owner_actions.append("python scripts/compare_models.py")
    elif inputs.security_validation_status != "PASS":
        status = STATUS_SECURITY_FAILED
        reasons.append(
            f"security validation: {inputs.security_validation_status}"
        )
        owner_actions.append("python scripts/security_validate_amali_ft_v0.py")
    else:
        status = STATUS_READY

    ft_status = FT_READY if inputs.promotion_decision == "PROMOTED" else FT_NOT_READY
    if ft_status == FT_NOT_READY:
        reasons.append(
            "AMALI-FT-v0 does not exist: promotion decision is "
            f"{inputs.promotion_decision!r} (this is independent of base "
            "model readiness)"
        )

    return BaseModelReadinessReport(
        status=status,
        base_model_ready=status == STATUS_READY,
        amali_ft_status=ft_status,
        checks=checks,
        reasons=reasons,
        owner_actions=owner_actions,
    )
