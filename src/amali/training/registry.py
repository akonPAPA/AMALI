"""Checkpoint registry for locally trained adapters (Phase 9).

Every trained adapter becomes a :class:`CheckpointManifest` with SHA-256
hashes of every file it contains. The registry lives **outside git**
(default ``D:/AMALI/models/registry/checkpoint_registry.json``) next to
the model files themselves. Integrity is verifiable at any time: a
missing file or a hash mismatch invalidates the checkpoint — such a
checkpoint can never load or promote.

No torch. No network.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from amali.eval.suite import canonical_json

__all__ = [
    "CheckpointManifest",
    "CheckpointIntegrity",
    "DEFAULT_REGISTRY_PATH",
    "hash_file",
    "hash_directory_files",
    "register_checkpoint",
    "load_registry",
    "verify_checkpoint",
]

DEFAULT_REGISTRY_PATH = Path("D:/AMALI/models/registry/checkpoint_registry.json")

SCHEMA_VERSION = "1.0.0"


class CheckpointManifest(BaseModel):
    """One trained adapter candidate, fully pinned."""

    model_config = {"extra": "forbid", "protected_namespaces": ()}

    checkpoint_id: str
    adapter_name: str
    base_model_id: str
    base_model_revision: str
    training_config_hash: str
    dataset_manifest_hash: str
    eval_suite_hash: str
    contamination_report_hash: str = ""
    adapter_dir: str
    file_hashes: dict[str, str]  # relative path -> sha256
    created_at: str
    train_status: str = "NOT_RUN"
    eval_status: str = "NOT_RUN"
    promotion_status: str = "NOT_RUN"
    risks: list[str] = Field(default_factory=list)


class CheckpointIntegrity(BaseModel):
    """Result of verifying one checkpoint against disk."""

    model_config = {"extra": "forbid"}

    checkpoint_id: str
    status: str  # PASS | FAIL
    missing_files: list[str] = Field(default_factory=list)
    hash_mismatches: list[str] = Field(default_factory=list)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_directory_files(directory: Path) -> dict[str, str]:
    """sha256 of every file under ``directory``, keyed by relative path."""
    hashes: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            rel = str(path.relative_to(directory)).replace("\\", "/")
            hashes[rel] = hash_file(path)
    return hashes


def load_registry(registry_path: Path = DEFAULT_REGISTRY_PATH) -> list[CheckpointManifest]:
    if not registry_path.exists():
        return []
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    return [CheckpointManifest(**entry) for entry in data.get("checkpoints", [])]


def _write_registry(
    manifests: list[CheckpointManifest], registry_path: Path
) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoints": [
            m.model_dump(mode="json")
            for m in sorted(manifests, key=lambda m: m.checkpoint_id)
        ],
    }
    registry_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def register_checkpoint(
    manifest: CheckpointManifest,
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> CheckpointManifest:
    """Add or replace a checkpoint entry after verifying it against disk.

    Raises ``ValueError`` when the adapter directory does not match the
    manifest's file hashes — an unverifiable checkpoint is never recorded.
    """
    integrity = verify_checkpoint(manifest)
    if integrity.status != "PASS":
        raise ValueError(
            "checkpoint failed integrity verification: "
            f"missing={integrity.missing_files} "
            f"mismatched={integrity.hash_mismatches}"
        )
    manifests = [
        m
        for m in load_registry(registry_path)
        if m.checkpoint_id != manifest.checkpoint_id
    ]
    manifests.append(manifest)
    _write_registry(manifests, registry_path)
    return manifest


def verify_checkpoint(manifest: CheckpointManifest) -> CheckpointIntegrity:
    """Recompute file hashes on disk against the manifest."""
    adapter_dir = Path(manifest.adapter_dir)
    missing: list[str] = []
    mismatched: list[str] = []
    if not manifest.file_hashes:
        missing.append("<manifest lists no files>")
    for rel, expected in manifest.file_hashes.items():
        path = adapter_dir / rel
        if not path.is_file():
            missing.append(rel)
        elif hash_file(path) != expected:
            mismatched.append(rel)
    return CheckpointIntegrity(
        checkpoint_id=manifest.checkpoint_id,
        status="PASS" if not missing and not mismatched else "FAIL",
        missing_files=missing,
        hash_mismatches=mismatched,
    )


def manifest_hash(manifest: CheckpointManifest) -> str:
    return hashlib.sha256(
        canonical_json(manifest.model_dump(mode="json")).encode("utf-8")
    ).hexdigest()
