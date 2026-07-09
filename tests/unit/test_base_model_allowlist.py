"""Phase 4: base model allowlist + availability weight markers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from amali.model_gateway.availability import (
    STATUS_AVAILABLE,
    check_local_model_availability,
)
from amali.model_gateway.base_model_allowlist import (
    check_model_allowed,
    find_entry,
    get_allowlist,
)


# --- allowlist ----------------------------------------------------------------


def test_default_model_is_allowlisted_for_training():
    decision = check_model_allowed(
        "Qwen/Qwen2.5-1.5B-Instruct", for_training=True
    )
    assert decision.allowed
    assert decision.permitted_for_training


def test_unknown_model_refused():
    decision = check_model_allowed("evil/backdoored-model")
    assert not decision.allowed
    assert "MODEL_NOT_ALLOWLISTED" in decision.reasons


def test_floating_revision_refused_for_promotion():
    decision = check_model_allowed(
        "Qwen/Qwen2.5-1.5B-Instruct", revision="main", for_promotion=True
    )
    assert not decision.allowed
    assert "REVISION_NOT_PINNED" in decision.reasons


def test_pinned_revision_accepted_for_promotion_when_allowlist_unpinned():
    # Allowlist entry still says "main" (owner has not pinned), so any
    # concrete pin the owner passes is accepted as the pin.
    decision = check_model_allowed(
        "Qwen/Qwen2.5-1.5B-Instruct",
        revision="0123abc0123abc0123abc0123abc0123abc01234",
        for_promotion=True,
    )
    assert decision.allowed
    assert decision.revision_pinned


def test_3b_not_permitted_for_training_by_default():
    decision = check_model_allowed(
        "Qwen/Qwen2.5-3B-Instruct", for_training=True
    )
    assert not decision.allowed
    assert "NOT_PERMITTED_FOR_TRAINING" in decision.reasons


def test_allowlist_returns_copies():
    entries = get_allowlist()
    entries[0].license = "tampered"
    assert find_entry(entries[0].model_id).license != "tampered"


# --- availability weight markers ----------------------------------------------


def _fake_snapshot(tmp_path: Path, model_id: str, files: list[str]) -> Path:
    folder = "models--" + model_id.replace("/", "--")
    snap = tmp_path / folder / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    for name in files:
        (snap / name).write_text("x", encoding="utf-8")
    return tmp_path


def test_config_only_cache_is_not_available(tmp_path):
    cache = _fake_snapshot(
        tmp_path, "org/model-a", ["config.json", "tokenizer.json"]
    )
    result = check_local_model_availability("org/model-a", cache_dir=cache)
    assert result.weights_cached_locally is False
    assert result.status != STATUS_AVAILABLE


def test_safetensors_cache_counts_as_weights(tmp_path):
    cache = _fake_snapshot(
        tmp_path,
        "org/model-b",
        ["config.json", "model-00001-of-00002.safetensors"],
    )
    result = check_local_model_availability("org/model-b", cache_dir=cache)
    assert result.weights_cached_locally is True


def test_pytorch_bin_index_counts_as_weights(tmp_path):
    cache = _fake_snapshot(
        tmp_path, "org/model-c", ["pytorch_model.bin.index.json"]
    )
    result = check_local_model_availability("org/model-c", cache_dir=cache)
    assert result.weights_cached_locally is True


def test_fake_model_never_available(tmp_path):
    result = check_local_model_availability(
        "definitely/not-a-real-model", cache_dir=tmp_path
    )
    assert result.status in (
        "DEPS_MISSING",
        "SKIPPED_LOCAL_MODEL_NOT_AVAILABLE",
    )


def test_allowlist_module_import_is_torch_free():
    code = (
        "import sys; import amali.model_gateway.base_model_allowlist; "
        "import amali.model_gateway.availability; "
        "print('torch_loaded=', 'torch' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    assert "torch_loaded= False" in out.stdout
