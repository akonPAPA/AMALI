"""Phase 9: checkpoint registry — hashes, tamper detection, stability."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from amali.training.registry import (
    CheckpointManifest,
    hash_directory_files,
    load_registry,
    register_checkpoint,
    verify_checkpoint,
)


def _adapter_dir(tmp_path: Path) -> Path:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"weights-bytes")
    (adapter / "adapter_config.json").write_text(
        '{"r": 16}', encoding="utf-8"
    )
    return adapter


def _manifest(adapter: Path, **overrides) -> CheckpointManifest:
    defaults = dict(
        checkpoint_id="ckpt_a",
        adapter_name="amali_ft_v0",
        base_model_id="Qwen/Qwen2.5-1.5B-Instruct",
        base_model_revision="0123abc0123abc0123abc0123abc0123abc01234",
        training_config_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        eval_suite_hash="e" * 64,
        adapter_dir=str(adapter),
        file_hashes=hash_directory_files(adapter),
        created_at="2026-07-09T00:00:00+00:00",
        train_status="TRAINING_COMPLETED",
    )
    defaults.update(overrides)
    return CheckpointManifest(**defaults)


def test_register_valid_checkpoint_roundtrips(tmp_path):
    adapter = _adapter_dir(tmp_path)
    registry = tmp_path / "registry.json"
    register_checkpoint(_manifest(adapter), registry_path=registry)
    loaded = load_registry(registry)
    assert len(loaded) == 1
    assert loaded[0].checkpoint_id == "ckpt_a"
    assert verify_checkpoint(loaded[0]).status == "PASS"


def test_missing_file_detected(tmp_path):
    adapter = _adapter_dir(tmp_path)
    manifest = _manifest(adapter)
    (adapter / "adapter_config.json").unlink()
    integrity = verify_checkpoint(manifest)
    assert integrity.status == "FAIL"
    assert "adapter_config.json" in integrity.missing_files


def test_hash_mismatch_detected(tmp_path):
    adapter = _adapter_dir(tmp_path)
    manifest = _manifest(adapter)
    (adapter / "adapter_model.safetensors").write_bytes(b"tampered")
    integrity = verify_checkpoint(manifest)
    assert integrity.status == "FAIL"
    assert "adapter_model.safetensors" in integrity.hash_mismatches


def test_register_refuses_unverifiable_checkpoint(tmp_path):
    adapter = _adapter_dir(tmp_path)
    manifest = _manifest(adapter)
    (adapter / "adapter_model.safetensors").unlink()
    with pytest.raises(ValueError):
        register_checkpoint(manifest, registry_path=tmp_path / "r.json")
    assert not (tmp_path / "r.json").exists()


def test_empty_file_hashes_invalid(tmp_path):
    adapter = _adapter_dir(tmp_path)
    manifest = _manifest(adapter, file_hashes={})
    assert verify_checkpoint(manifest).status == "FAIL"


def test_registry_file_is_stable_canonical_json(tmp_path):
    adapter = _adapter_dir(tmp_path)
    registry = tmp_path / "registry.json"
    manifest = _manifest(adapter)
    register_checkpoint(manifest, registry_path=registry)
    first = json.loads(registry.read_text(encoding="utf-8"))
    register_checkpoint(manifest, registry_path=registry)
    second = json.loads(registry.read_text(encoding="utf-8"))
    # identical apart from the update timestamp
    first.pop("updated_at")
    second.pop("updated_at")
    assert first == second
    assert len(second["checkpoints"]) == 1


def test_reregistering_replaces_not_duplicates(tmp_path):
    adapter = _adapter_dir(tmp_path)
    registry = tmp_path / "registry.json"
    register_checkpoint(_manifest(adapter), registry_path=registry)
    register_checkpoint(
        _manifest(adapter, train_status="TRAINING_COMPLETED"),
        registry_path=registry,
    )
    assert len(load_registry(registry)) == 1
