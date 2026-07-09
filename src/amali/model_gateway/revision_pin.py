"""Base model revision pinning (Base Model Readiness).

An exact Hugging Face snapshot commit SHA is the only revision that can
anchor a promotable model. This module holds the pure decision logic plus
the reader that later stages (gateway smoke, readiness checker) use to
discover the latest owner pin. It is torch-free and network-free.

Pin reports are runtime artifacts under
``artifacts/base_model_readiness/<timestamp>/`` — local evidence, never
committed. The owner re-pins by re-running
``scripts/pin_base_model_revision.py``.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from amali.model_gateway.base_model_allowlist import (
    FLOATING_REVISIONS,
    find_entry,
)

__all__ = [
    "STATUS_REVISION_PINNED",
    "STATUS_OWNER_MODEL_REVISION_REQUIRED",
    "STATUS_MODEL_NOT_ALLOWLISTED",
    "STATUS_REVISION_FLOATING_REFUSED",
    "EXACT_SHA_PATTERN",
    "RevisionPinReport",
    "decide_revision_pin",
    "load_latest_pin",
]

STATUS_REVISION_PINNED = "REVISION_PINNED"
STATUS_OWNER_MODEL_REVISION_REQUIRED = "OWNER_MODEL_REVISION_REQUIRED"
STATUS_MODEL_NOT_ALLOWLISTED = "MODEL_NOT_ALLOWLISTED"
STATUS_REVISION_FLOATING_REFUSED = "REVISION_FLOATING_REFUSED"

# A Hugging Face snapshot commit is a full 40-hex-char SHA-1. Nothing
# shorter or fuzzier is "exact".
EXACT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")

PIN_REPORT_NAME = "base_model_revision_pin_report.json"


class RevisionPinReport(BaseModel):
    """Typed outcome of one pin attempt."""

    model_config = {"extra": "forbid", "protected_namespaces": ()}

    status: str
    model_id: str
    revision: str = ""
    role: str = ""
    license_label: str = ""
    promotion_eligible: bool = False
    permitted_for_training: bool = False
    pinned_at: str = ""
    reasons: list[str] = Field(default_factory=list)
    owner_actions: list[str] = Field(default_factory=list)


def decide_revision_pin(model_id: str, revision: str | None) -> RevisionPinReport:
    """Pure pin decision: allowlisted model + exact SHA, or a typed refusal."""
    entry = find_entry(model_id)
    if entry is None:
        return RevisionPinReport(
            status=STATUS_MODEL_NOT_ALLOWLISTED,
            model_id=model_id,
            reasons=[f"model {model_id!r} is not on the owner allowlist"],
            owner_actions=[
                "approve the model deliberately in "
                "src/amali/model_gateway/base_model_allowlist.py, or pin an "
                "allowlisted model instead"
            ],
        )

    if revision is None or not revision.strip():
        return RevisionPinReport(
            status=STATUS_OWNER_MODEL_REVISION_REQUIRED,
            model_id=model_id,
            role=entry.role,
            license_label=entry.license,
            permitted_for_training=entry.permitted_for_training,
            reasons=["no revision provided; the owner must choose the exact "
                     "Hugging Face snapshot commit SHA"],
            owner_actions=[
                f"choose the exact snapshot commit SHA for {model_id} and "
                "re-run: python scripts/pin_base_model_revision.py --model "
                f"{model_id} --revision <HF_SNAPSHOT_COMMIT_SHA>"
            ],
        )

    candidate = revision.strip()
    if candidate in FLOATING_REVISIONS or not EXACT_SHA_PATTERN.match(candidate):
        return RevisionPinReport(
            status=STATUS_REVISION_FLOATING_REFUSED,
            model_id=model_id,
            revision=candidate,
            role=entry.role,
            license_label=entry.license,
            permitted_for_training=entry.permitted_for_training,
            reasons=[
                f"revision {candidate!r} is floating or not an exact "
                "40-hex snapshot commit SHA; floating revisions are "
                "NON_PROMOTABLE by design"
            ],
            owner_actions=[
                "re-run with the full 40-character snapshot commit SHA "
                "(visible on the model's Hugging Face 'Files' page)"
            ],
        )

    return RevisionPinReport(
        status=STATUS_REVISION_PINNED,
        model_id=model_id,
        revision=candidate,
        role=entry.role,
        license_label=entry.license,
        promotion_eligible=True,
        permitted_for_training=entry.permitted_for_training,
        pinned_at=datetime.now(timezone.utc).isoformat(),
    )


def load_latest_pin(
    artifact_root: Path,
    *,
    model_id: str | None = None,
) -> RevisionPinReport | None:
    """Newest REVISION_PINNED report under the readiness artifact tree.

    Only a successful pin counts; refusals never become an implicit pin.
    """
    if not artifact_root.is_dir():
        return None
    candidates = sorted(artifact_root.glob(f"*/{PIN_REPORT_NAME}"))
    for path in reversed(candidates):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        decisions = data.get("decisions") or []
        if not decisions:
            continue
        report = RevisionPinReport(**decisions[0])
        if report.status != STATUS_REVISION_PINNED:
            continue
        if model_id is not None and report.model_id != model_id:
            continue
        return report
    return None
