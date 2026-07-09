"""Owner-controlled base model allowlist (Phase 4).

Only models on this list may be downloaded, trained on, or evaluated.
Promotion additionally requires a **pinned revision** — a floating
``main`` is fine for local experimentation but can never anchor a
reproducible promoted artifact.

This module is pure data + checks: no torch, no network, importable on
the Gate A path.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = [
    "BaseModelEntry",
    "DEFAULT_ALLOWLIST",
    "get_allowlist",
    "find_entry",
    "check_model_allowed",
    "AllowlistDecision",
    "ROLE_TRAIN_EVAL",
    "ROLE_FALLBACK",
    "ROLE_EVAL_ONLY",
]

FLOATING_REVISIONS = ("", "main", "latest", "HEAD")

# Roles a base model can hold. Only train_eval models may anchor training;
# fallback requires explicit owner fallback approval; eval_only never trains.
ROLE_TRAIN_EVAL = "train_eval"
ROLE_FALLBACK = "fallback"
ROLE_EVAL_ONLY = "eval_only"


class BaseModelEntry(BaseModel):
    """One owner-approved base model."""

    model_config = {"extra": "forbid", "protected_namespaces": ()}

    model_id: str
    allowed_revision: str  # pinned commit/tag; "main" means not yet pinned
    role: str = ROLE_TRAIN_EVAL  # train_eval | fallback | eval_only
    license: str
    parameter_count: str
    expected_disk_gb: float
    expected_vram_gb: float
    default_quantization: str
    permitted_for_training: bool
    permitted_for_eval: bool
    owner_approval_required: bool = True
    notes: str = ""


# The owner-approved list, sized for a single RTX 4060 Laptop (8 GB VRAM).
# allowed_revision stays "main" until the owner pins the exact snapshot
# they downloaded; promotion is impossible until then — by design.
DEFAULT_ALLOWLIST: list[BaseModelEntry] = [
    BaseModelEntry(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        allowed_revision="main",
        role=ROLE_TRAIN_EVAL,
        license="Apache-2.0",
        parameter_count="1.5B",
        expected_disk_gb=3.1,
        expected_vram_gb=4.5,
        default_quantization="4bit_nf4",
        permitted_for_training=True,
        permitted_for_eval=True,
        notes=(
            "Default base for AMALI-FT-v0. Owner must pin the downloaded "
            "revision before promotion."
        ),
    ),
    BaseModelEntry(
        model_id="Qwen/Qwen2.5-3B-Instruct",
        allowed_revision="main",
        role=ROLE_EVAL_ONLY,
        license="Qwen Research License",
        parameter_count="3B",
        expected_disk_gb=6.2,
        expected_vram_gb=7.5,
        default_quantization="4bit_nf4",
        permitted_for_training=False,
        permitted_for_eval=True,
        notes=(
            "Optional larger base; training only if a dry-run proves VRAM "
            "feasibility on 8 GB. Eval permitted."
        ),
    ),
    BaseModelEntry(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        allowed_revision="main",
        role=ROLE_FALLBACK,
        license="Apache-2.0",
        parameter_count="0.5B",
        expected_disk_gb=1.0,
        expected_vram_gb=2.0,
        default_quantization="none",
        permitted_for_training=True,
        permitted_for_eval=True,
        notes="Small fallback when the 1.5B is too tight.",
    ),
]


class AllowlistDecision(BaseModel):
    """Typed result of an allowlist check."""

    model_config = {"extra": "forbid", "protected_namespaces": ()}

    model_id: str
    allowed: bool
    revision_pinned: bool
    permitted_for_training: bool = False
    permitted_for_eval: bool = False
    reasons: list[str] = Field(default_factory=list)


def get_allowlist() -> list[BaseModelEntry]:
    return [entry.model_copy(deep=True) for entry in DEFAULT_ALLOWLIST]


def find_entry(model_id: str) -> BaseModelEntry | None:
    for entry in DEFAULT_ALLOWLIST:
        if entry.model_id == model_id:
            return entry.model_copy(deep=True)
    return None


def check_model_allowed(
    model_id: str,
    *,
    revision: str | None = None,
    for_training: bool = False,
    for_promotion: bool = False,
) -> AllowlistDecision:
    """Check a model against the allowlist.

    Promotion (and anything ``for_promotion``) demands a pinned revision
    matching the allowlist entry; a floating revision is refused.
    """
    entry = find_entry(model_id)
    if entry is None:
        return AllowlistDecision(
            model_id=model_id,
            allowed=False,
            revision_pinned=False,
            reasons=["MODEL_NOT_ALLOWLISTED"],
        )

    reasons: list[str] = []
    requested = revision if revision is not None else entry.allowed_revision
    pinned = requested not in FLOATING_REVISIONS

    if for_training and not entry.permitted_for_training:
        reasons.append("NOT_PERMITTED_FOR_TRAINING")
    if for_promotion and not pinned:
        reasons.append("REVISION_NOT_PINNED")
    if (
        for_promotion
        and pinned
        and entry.allowed_revision not in FLOATING_REVISIONS
        and requested != entry.allowed_revision
    ):
        reasons.append("REVISION_DOES_NOT_MATCH_ALLOWLIST_PIN")

    return AllowlistDecision(
        model_id=model_id,
        allowed=not reasons,
        revision_pinned=pinned,
        permitted_for_training=entry.permitted_for_training,
        permitted_for_eval=entry.permitted_for_eval,
        reasons=reasons,
    )
