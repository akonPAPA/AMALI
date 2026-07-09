"""Training environment probe + fallback planning (AMALI-FT-v0 Stage 4).

Pure decision logic over an injected :class:`EnvProbe` snapshot, so the
policy is unit-testable without a GPU. The script
``scripts/probe_training_environment.py`` gathers the real probe (the
only place torch is touched, and only in a subprocess-safe, explicit
owner command) and feeds it here.

Fallback policy, in order of authority:

* required deps missing (torch/transformers/peft/datasets/accelerate)
  -> ``TRAIN_DEPS_MISSING``; nothing is planned.
* CUDA unavailable -> ``CUDA_UNAVAILABLE``; CPU training is **never**
  auto-approved — the plan demands explicit owner approval.
* bitsandbytes unavailable -> QLoRA is impossible; fall back to standard
  LoRA **only** when the VRAM estimate fits, and record the degradation.
  We never fake QLoRA.
* VRAM too small for the preferred model -> fall back to the smaller
  allowlisted base, recorded, never silent.

Every degradation lands in ``training_fallback_plan.json`` with
``requires_owner_approval`` — training must not consume an unapproved
degraded plan.

No torch import at module level. No network.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = [
    "EnvProbe",
    "FallbackDecision",
    "TrainingFallbackPlan",
    "TrainingEnvReport",
    "plan_training_environment",
    "PREFERRED_MODEL",
    "FALLBACK_MODEL",
]

PREFERRED_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
FALLBACK_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Conservative peak-VRAM estimates (GB) for max_seq_len=1024, batch=1,
# grad-accum=8, rank-16 adapters. Deliberately pessimistic: an estimate
# that fits with no headroom is treated as not fitting.
VRAM_ESTIMATES_GB: dict[tuple[str, str], float] = {
    (PREFERRED_MODEL, "qlora"): 4.5,
    (PREFERRED_MODEL, "lora"): 7.2,
    (FALLBACK_MODEL, "qlora"): 2.0,
    (FALLBACK_MODEL, "lora"): 3.2,
}
VRAM_HEADROOM_GB = 0.5

REQUIRED_DEPS = ("torch", "transformers", "peft", "datasets", "accelerate")


class EnvProbe(BaseModel):
    """A snapshot of the machine, gathered by the probe script."""

    model_config = {"extra": "forbid"}

    python_version: str
    torch_installed: bool
    torch_version: str = ""
    cuda_available: bool = False
    gpu_name: str = ""
    vram_total_gb: float = 0.0
    bf16_supported: bool = False
    transformers_installed: bool = False
    peft_installed: bool = False
    datasets_installed: bool = False
    accelerate_installed: bool = False
    bitsandbytes_installed: bool = False
    huggingface_hub_installed: bool = False


class FallbackDecision(BaseModel):
    """One recorded degradation or refusal."""

    model_config = {"extra": "forbid"}

    code: str
    detail: str


class TrainingFallbackPlan(BaseModel):
    """What training is allowed to attempt, and under whose approval."""

    model_config = {"extra": "forbid", "protected_namespaces": ()}

    status: str  # PLANNED | NOT_READY | OWNER_ACTION_REQUIRED
    method: str = ""  # qlora | lora | ""
    model_id: str = ""
    estimated_vram_gb: float = 0.0
    vram_total_gb: float = 0.0
    degraded: bool = False
    requires_owner_approval: bool = False
    owner_approved: bool = False  # owner flips this explicitly, never code
    decisions: list[FallbackDecision] = Field(default_factory=list)
    owner_actions: list[str] = Field(default_factory=list)


class TrainingEnvReport(BaseModel):
    """Typed probe outcome."""

    model_config = {"extra": "forbid"}

    status: str
    probe: EnvProbe
    missing_deps: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


def _missing_deps(probe: EnvProbe) -> list[str]:
    flags = {
        "torch": probe.torch_installed,
        "transformers": probe.transformers_installed,
        "peft": probe.peft_installed,
        "datasets": probe.datasets_installed,
        "accelerate": probe.accelerate_installed,
    }
    return [name for name in REQUIRED_DEPS if not flags[name]]


def _fits(model_id: str, method: str, vram_total_gb: float) -> bool:
    estimate = VRAM_ESTIMATES_GB.get((model_id, method))
    if estimate is None:
        return False
    return estimate + VRAM_HEADROOM_GB <= vram_total_gb


def plan_training_environment(
    probe: EnvProbe,
    *,
    preferred_model: str = PREFERRED_MODEL,
) -> tuple[TrainingEnvReport, TrainingFallbackPlan]:
    """Turn a probe snapshot into a typed status + fallback plan."""
    missing = _missing_deps(probe)
    decisions: list[FallbackDecision] = []
    owner_actions: list[str] = []
    risks: list[str] = []

    if missing:
        owner_actions.append(
            'python -m pip install -e ".[dev,local_llm,train]"'
        )
        return (
            TrainingEnvReport(
                status="TRAIN_DEPS_MISSING",
                probe=probe,
                missing_deps=missing,
                risks=[f"missing training dependencies: {', '.join(missing)}"],
            ),
            TrainingFallbackPlan(
                status="NOT_READY",
                vram_total_gb=probe.vram_total_gb,
                decisions=[
                    FallbackDecision(
                        code="TRAIN_DEPS_MISSING",
                        detail=f"missing: {', '.join(missing)}",
                    )
                ],
                owner_actions=owner_actions,
            ),
        )

    if not probe.cuda_available:
        owner_actions.append(
            "install a CUDA-enabled torch build, or explicitly approve "
            "CPU training (multi-day; never auto-approved)"
        )
        return (
            TrainingEnvReport(
                status="CUDA_UNAVAILABLE",
                probe=probe,
                risks=["no CUDA device; CPU fallback requires owner approval"],
            ),
            TrainingFallbackPlan(
                status="OWNER_ACTION_REQUIRED",
                vram_total_gb=probe.vram_total_gb,
                requires_owner_approval=True,
                decisions=[
                    FallbackDecision(
                        code="CUDA_UNAVAILABLE",
                        detail=(
                            "CPU training is never auto-approved; owner "
                            "must opt in explicitly"
                        ),
                    )
                ],
                owner_actions=owner_actions,
            ),
        )

    method = "qlora"
    degraded = False
    if not probe.bitsandbytes_installed:
        method = "lora"
        degraded = True
        decisions.append(
            FallbackDecision(
                code="BITSANDBYTES_UNAVAILABLE",
                detail=(
                    "QLoRA impossible without bitsandbytes; falling back "
                    "to standard LoRA (no fake quantization)"
                ),
            )
        )

    model_id = preferred_model
    if not _fits(model_id, method, probe.vram_total_gb):
        if _fits(FALLBACK_MODEL, method, probe.vram_total_gb):
            decisions.append(
                FallbackDecision(
                    code="VRAM_MODEL_FALLBACK",
                    detail=(
                        f"{model_id} ({method}) needs "
                        f"{VRAM_ESTIMATES_GB.get((model_id, method), 0.0)} GB "
                        f"+ {VRAM_HEADROOM_GB} GB headroom > "
                        f"{probe.vram_total_gb} GB; falling back to "
                        f"{FALLBACK_MODEL}"
                    ),
                )
            )
            model_id = FALLBACK_MODEL
            degraded = True
        else:
            owner_actions.append(
                "free VRAM or approve a different configuration; no "
                "allowlisted model fits the current device"
            )
            return (
                TrainingEnvReport(
                    status="VRAM_LOW",
                    probe=probe,
                    risks=[
                        f"no (model, {method}) combination fits "
                        f"{probe.vram_total_gb} GB VRAM"
                    ],
                ),
                TrainingFallbackPlan(
                    status="OWNER_ACTION_REQUIRED",
                    method=method,
                    vram_total_gb=probe.vram_total_gb,
                    degraded=degraded,
                    requires_owner_approval=True,
                    decisions=decisions
                    + [
                        FallbackDecision(
                            code="VRAM_INFEASIBLE",
                            detail="no feasible configuration",
                        )
                    ],
                    owner_actions=owner_actions,
                ),
            )

    status = "TRAIN_ENV_PASS" if not degraded else "BITSANDBYTES_UNAVAILABLE"
    if degraded and probe.bitsandbytes_installed:
        # degradation came from VRAM, not bitsandbytes
        status = "VRAM_LOW"
    if degraded:
        risks.append("degraded training plan; owner approval required")

    plan = TrainingFallbackPlan(
        status="PLANNED",
        method=method,
        model_id=model_id,
        estimated_vram_gb=VRAM_ESTIMATES_GB[(model_id, method)],
        vram_total_gb=probe.vram_total_gb,
        degraded=degraded,
        requires_owner_approval=degraded,
        decisions=decisions,
        owner_actions=(
            ["review training_fallback_plan.json and approve the recorded "
             "degradations before training"]
            if degraded
            else []
        ),
    )
    return (
        TrainingEnvReport(status=status, probe=probe, risks=risks),
        plan,
    )
