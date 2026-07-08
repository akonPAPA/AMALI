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

__all__ = ["ModelAvailability", "check_local_model_availability", "DEFAULT_MODEL_ID"]

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

STATUS_AVAILABLE = "AVAILABLE"
STATUS_SKIPPED = "SKIPPED_LOCAL_MODEL_NOT_AVAILABLE"
STATUS_DEPS_MISSING = "DEPS_MISSING"


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


def _weights_cached(model_id: str, cache_dir: Path) -> bool:
    # HF hub layout: models--{org}--{name}/snapshots/<rev>/...
    folder = "models--" + model_id.replace("/", "--")
    snapshots = cache_dir / folder / "snapshots"
    if not snapshots.is_dir():
        return False
    # A snapshot with at least one file counts as cached weights.
    for snapshot in snapshots.iterdir():
        if snapshot.is_dir() and any(snapshot.iterdir()):
            return True
    return False


def check_local_model_availability(
    model_id: str = DEFAULT_MODEL_ID,
) -> ModelAvailability:
    """Probe local deps + cache. Never downloads; never fails Gate A."""
    torch_ok = _dep_installed("torch")
    transformers_ok = _dep_installed("transformers")
    cache_dir = _hf_cache_dir()
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
