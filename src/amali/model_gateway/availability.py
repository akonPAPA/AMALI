"""Local model availability gate (WP5).

Answers one question without ever touching the network: *could* the optional
local base model run here? Three honest outcomes:

* ``DEPS_MISSING`` — torch/transformers are not installed;
* ``SKIPPED_LOCAL_MODEL_NOT_AVAILABLE`` — deps exist but no local weights
  are cached for the model id;
* ``AVAILABLE`` — deps and locally cached weights both present.

The check inspects the local Hugging Face cache directory structure only.
It never downloads, never opens a socket, and never imports torch at
module import time. Gate B consumes this; Gate A never depends on it.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from pydantic import BaseModel, Field

__all__ = [
    "ModelAvailability",
    "check_local_model_availability",
    "DEFAULT_MODEL_ID",
    "SnapshotAvailability",
    "check_model_snapshot",
    "STATUS_AVAILABLE",
    "STATUS_MODEL_NOT_AVAILABLE",
    "STATUS_MODEL_INCOMPLETE",
    "STATUS_REVISION_MISMATCH",
]

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

STATUS_AVAILABLE = "AVAILABLE"
STATUS_SKIPPED = "SKIPPED_LOCAL_MODEL_NOT_AVAILABLE"
STATUS_DEPS_MISSING = "DEPS_MISSING"
STATUS_MODEL_NOT_AVAILABLE = "MODEL_NOT_AVAILABLE"
STATUS_MODEL_INCOMPLETE = "MODEL_INCOMPLETE"
STATUS_REVISION_MISMATCH = "REVISION_MISMATCH"

# Any of these makes a snapshot's tokenizer real.
TOKENIZER_MARKERS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "vocab.json",
    "spiece.model",
)


class ModelAvailability(BaseModel):
    """The result of a local-only availability probe."""

    model_config = {"extra": "forbid", "protected_namespaces": ()}

    model_id: str
    status: str
    torch_installed: bool
    transformers_installed: bool
    weights_cached_locally: bool
    cache_dir_checked: str
    notes: list[str] = Field(default_factory=list)


def _dep_installed(name: str) -> bool:
    # find_spec does not import the package; cheap and side-effect free.
    return importlib.util.find_spec(name) is not None


def _hf_cache_dir() -> Path:
    env = os.environ.get("HF_HOME")
    if env:
        return Path(env) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _snapshot_has_weight_files(snapshot: Path) -> bool:
    """True when the snapshot contains real weight artifacts (not config-only)."""
    if not snapshot.is_dir():
        return False
    if any(snapshot.glob("*.safetensors")):
        return True
    if any(snapshot.glob("pytorch_model*.bin")):
        return True
    if (snapshot / "model.safetensors.index.json").is_file():
        return True
    if (snapshot / "pytorch_model.bin.index.json").is_file():
        return True
    return False


def _weights_cached(model_id: str, cache_dir: Path) -> bool:
    # HF hub layout: models--{org}--{name}/snapshots/<rev>/...
    folder = "models--" + model_id.replace("/", "--")
    snapshots = cache_dir / folder / "snapshots"
    if not snapshots.is_dir():
        return False
    for snapshot in snapshots.iterdir():
        if _snapshot_has_weight_files(snapshot):
            return True
    return False


class SnapshotAvailability(BaseModel):
    """Full availability contract for one model at (optionally) one revision.

    AVAILABLE only when real weight markers AND tokenizer AND config are all
    present in the snapshot that matches the requested revision. Config-only
    or tokenizer-only snapshots are MODEL_NOT_AVAILABLE; weights without
    tokenizer/config are MODEL_INCOMPLETE; a cached snapshot at a different
    revision than requested is REVISION_MISMATCH.
    """

    model_config = {"extra": "forbid", "protected_namespaces": ()}

    model_id: str
    revision_requested: str = ""
    status: str
    snapshot_dir: str = ""
    weight_files_detected: bool = False
    tokenizer_files_detected: bool = False
    config_files_detected: bool = False
    cached_revisions: list[str] = Field(default_factory=list)
    cache_dir_checked: str = ""
    notes: list[str] = Field(default_factory=list)


def _snapshot_has_tokenizer(snapshot: Path) -> bool:
    return any((snapshot / name).is_file() for name in TOKENIZER_MARKERS)


def _snapshot_has_config(snapshot: Path) -> bool:
    return (snapshot / "config.json").is_file()


def _classify_snapshot(model_id: str, revision: str, snapshot: Path) -> SnapshotAvailability:
    weights = _snapshot_has_weight_files(snapshot)
    tokenizer = _snapshot_has_tokenizer(snapshot)
    config = _snapshot_has_config(snapshot)

    if not weights:
        status = STATUS_MODEL_NOT_AVAILABLE
        note = (
            "snapshot has no real weight markers (config-only or "
            "tokenizer-only caches never count as available)"
        )
    elif not (tokenizer and config):
        status = STATUS_MODEL_INCOMPLETE
        missing = [
            name
            for present, name in ((tokenizer, "tokenizer"), (config, "config"))
            if not present
        ]
        note = f"weights present but missing: {missing}"
    else:
        status = STATUS_AVAILABLE
        note = "weights, tokenizer, and config all present"

    return SnapshotAvailability(
        model_id=model_id,
        revision_requested=revision,
        status=status,
        snapshot_dir=str(snapshot),
        weight_files_detected=weights,
        tokenizer_files_detected=tokenizer,
        config_files_detected=config,
        notes=[note],
    )


def check_model_snapshot(
    model_id: str,
    revision: str | None = None,
    *,
    cache_dir: Path | None = None,
) -> SnapshotAvailability:
    """File-inspection-only availability check; never downloads.

    With ``revision`` the exact snapshot directory must exist; with no
    revision the newest snapshot that has weight markers is classified.
    """
    cache = cache_dir if cache_dir is not None else _hf_cache_dir()
    folder = "models--" + model_id.replace("/", "--")
    snapshots = cache / folder / "snapshots"

    if not snapshots.is_dir():
        return SnapshotAvailability(
            model_id=model_id,
            revision_requested=revision or "",
            status=STATUS_MODEL_NOT_AVAILABLE,
            cache_dir_checked=str(cache),
            notes=[
                "no local snapshot cache for this model; owner action: "
                f"python scripts/download_model.py --model {model_id} "
                "--revision <HF_SNAPSHOT_COMMIT_SHA>"
            ],
        )

    cached = sorted(p.name for p in snapshots.iterdir() if p.is_dir())

    if revision:
        target = snapshots / revision
        if not target.is_dir():
            return SnapshotAvailability(
                model_id=model_id,
                revision_requested=revision,
                status=STATUS_REVISION_MISMATCH,
                cached_revisions=cached,
                cache_dir_checked=str(cache),
                notes=[
                    f"requested revision {revision!r} is not cached; "
                    f"cached snapshots: {cached or 'none'}"
                ],
            )
        result = _classify_snapshot(model_id, revision, target)
        return result.model_copy(
            update={"cached_revisions": cached, "cache_dir_checked": str(cache)}
        )

    # No revision requested: report the best cached snapshot honestly, but
    # callers that need promotion-grade certainty must pass the pin.
    best: SnapshotAvailability | None = None
    for name in reversed(cached):
        candidate = _classify_snapshot(model_id, "", snapshots / name)
        if candidate.status == STATUS_AVAILABLE:
            best = candidate
            break
        if best is None:
            best = candidate
    if best is None:
        return SnapshotAvailability(
            model_id=model_id,
            status=STATUS_MODEL_NOT_AVAILABLE,
            cached_revisions=cached,
            cache_dir_checked=str(cache),
            notes=["snapshot directory exists but contains no snapshots"],
        )
    return best.model_copy(
        update={"cached_revisions": cached, "cache_dir_checked": str(cache)}
    )


def check_local_model_availability(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    cache_dir: Path | None = None,
) -> ModelAvailability:
    """Probe local deps + cache. Never downloads; never fails Gate A."""
    torch_ok = _dep_installed("torch")
    transformers_ok = _dep_installed("transformers")
    cache_dir = cache_dir if cache_dir is not None else _hf_cache_dir()
    cached = _weights_cached(model_id, cache_dir)

    notes: list[str] = []
    if not (torch_ok and transformers_ok):
        status = STATUS_DEPS_MISSING
        notes.append(
            "optional local_llm dependencies not installed; "
            "Gate B unavailable, Gate A unaffected"
        )
    elif not cached:
        status = STATUS_SKIPPED
        notes.append(
            "no locally cached weights; run scripts/download_model.py "
            "(owner-initiated) to enable Gate B"
        )
    else:
        status = STATUS_AVAILABLE
        notes.append("local weights found; Gate B smoke may run offline")

    return ModelAvailability(
        model_id=model_id,
        status=status,
        torch_installed=torch_ok,
        transformers_installed=transformers_ok,
        weights_cached_locally=cached,
        cache_dir_checked=str(cache_dir),
        notes=notes,
    )
