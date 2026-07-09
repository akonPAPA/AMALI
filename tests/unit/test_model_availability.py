"""Snapshot availability contract: weights + tokenizer + config + revision."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from amali.model_gateway.availability import (
    STATUS_AVAILABLE,
    STATUS_MODEL_INCOMPLETE,
    STATUS_MODEL_NOT_AVAILABLE,
    STATUS_REVISION_MISMATCH,
    check_model_snapshot,
)

MODEL = "org/test-model"
REV = "a" * 40

FULL_SNAPSHOT = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors",
]


def _cache(tmp_path: Path, revision: str, files: list[str]) -> Path:
    snap = (
        tmp_path
        / ("models--" + MODEL.replace("/", "--"))
        / "snapshots"
        / revision
    )
    snap.mkdir(parents=True)
    for name in files:
        (snap / name).write_text("x", encoding="utf-8")
    return tmp_path


def test_full_snapshot_is_available(tmp_path):
    cache = _cache(tmp_path, REV, FULL_SNAPSHOT)
    result = check_model_snapshot(MODEL, REV, cache_dir=cache)
    assert result.status == STATUS_AVAILABLE
    assert result.weight_files_detected
    assert result.tokenizer_files_detected
    assert result.config_files_detected


def test_config_only_snapshot_not_available(tmp_path):
    cache = _cache(tmp_path, REV, ["config.json"])
    result = check_model_snapshot(MODEL, REV, cache_dir=cache)
    assert result.status == STATUS_MODEL_NOT_AVAILABLE
    assert not result.weight_files_detected


def test_tokenizer_only_snapshot_not_available(tmp_path):
    cache = _cache(tmp_path, REV, ["tokenizer.json", "tokenizer_config.json"])
    result = check_model_snapshot(MODEL, REV, cache_dir=cache)
    assert result.status == STATUS_MODEL_NOT_AVAILABLE


def test_weights_without_tokenizer_is_incomplete(tmp_path):
    cache = _cache(tmp_path, REV, ["config.json", "model.safetensors"])
    result = check_model_snapshot(MODEL, REV, cache_dir=cache)
    assert result.status == STATUS_MODEL_INCOMPLETE
    assert result.weight_files_detected
    assert not result.tokenizer_files_detected


def test_weights_without_config_is_incomplete(tmp_path):
    cache = _cache(tmp_path, REV, ["tokenizer.json", "model.safetensors"])
    result = check_model_snapshot(MODEL, REV, cache_dir=cache)
    assert result.status == STATUS_MODEL_INCOMPLETE
    assert not result.config_files_detected


def test_revision_mismatch_detected(tmp_path):
    cache = _cache(tmp_path, "b" * 40, FULL_SNAPSHOT)
    result = check_model_snapshot(MODEL, REV, cache_dir=cache)
    assert result.status == STATUS_REVISION_MISMATCH
    assert result.cached_revisions == ["b" * 40]


def test_missing_model_not_available(tmp_path):
    result = check_model_snapshot(MODEL, REV, cache_dir=tmp_path)
    assert result.status == STATUS_MODEL_NOT_AVAILABLE


def test_no_revision_reports_best_cached_snapshot(tmp_path):
    cache = _cache(tmp_path, "b" * 40, FULL_SNAPSHOT)
    result = check_model_snapshot(MODEL, None, cache_dir=cache)
    assert result.status == STATUS_AVAILABLE


def test_index_json_counts_as_weight_marker(tmp_path):
    cache = _cache(
        tmp_path,
        REV,
        ["config.json", "tokenizer.json", "model.safetensors.index.json"],
    )
    result = check_model_snapshot(MODEL, REV, cache_dir=cache)
    assert result.status == STATUS_AVAILABLE


def test_snapshot_check_never_imports_torch():
    code = (
        "import sys; "
        "from amali.model_gateway.availability import check_model_snapshot; "
        "check_model_snapshot('x/y', 'a'*40); "
        "raise SystemExit(0 if 'torch' not in sys.modules else 1)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
