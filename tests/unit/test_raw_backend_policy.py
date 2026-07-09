"""Raw model paths enforce owner allowlist before local model work."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from amali.model_gateway.adapter_loader import load_raw_backend

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
EVIL_MODEL = "evil/backdoored-model"


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_policy_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _latest_json(root: Path, name: str) -> dict:
    reports = sorted(root.glob(f"*/{name}"))
    assert reports
    return json.loads(reports[-1].read_text(encoding="utf-8"))


def test_load_raw_backend_refuses_non_allowlisted_before_local_work(
    monkeypatch, tmp_path
):
    def fail_deps():
        raise AssertionError("deps check must not run after policy refusal")

    def fail_snapshot(*args, **kwargs):
        raise AssertionError("snapshot check must not run after policy refusal")

    def fail_load(*args, **kwargs):
        raise AssertionError("model load must not run after policy refusal")

    monkeypatch.setattr("amali.model_gateway.adapter_loader._deps_missing", fail_deps)
    monkeypatch.setattr(
        "amali.model_gateway.adapter_loader.check_model_snapshot", fail_snapshot
    )
    monkeypatch.setattr(
        "amali.model_gateway.adapter_loader.LocalModelBackend.from_local", fail_load
    )

    result, backend = load_raw_backend(EVIL_MODEL, REVISION, cache_dir=tmp_path)

    assert backend is None
    assert result.status == "POLICY_BLOCKED"
    assert result.base_model_id == EVIL_MODEL
    assert result.base_model_revision == REVISION
    assert "MODEL_NOT_ALLOWLISTED" in result.reasons[0]


def test_load_raw_backend_allowlisted_missing_deps_behavior_unchanged(monkeypatch):
    monkeypatch.setattr(
        "amali.model_gateway.adapter_loader._deps_missing", lambda: ["torch"]
    )

    result, backend = load_raw_backend(MODEL, REVISION)

    assert backend is None
    assert result.status == "DEPS_MISSING"
    assert "torch" in result.reasons[0]


def test_evaluate_model_raw_base_non_allowlisted_exits_typed_refusal(
    monkeypatch, tmp_path, capsys
):
    mod = _load_script("evaluate_model")
    monkeypatch.setattr(mod, "ARTIFACT_BASE", tmp_path / "base_model_gate")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_model.py",
            "--model",
            "raw_base",
            "--model-id",
            EVIL_MODEL,
        ],
    )

    assert mod.main() == 1
    captured = capsys.readouterr()
    assert "Eval raw_base: POLICY_BLOCKED" in captured.out
    assert "MODEL_NOT_ALLOWLISTED" in captured.err
    report = _latest_json(mod.ARTIFACT_BASE, "eval_report_raw_base.json")
    assert report["status"] == "POLICY_BLOCKED"
    assert report["decisions"][0]["model_id"] == EVIL_MODEL


def test_chat_smoke_raw_base_non_allowlisted_exits_typed_refusal(
    monkeypatch, tmp_path, capsys
):
    mod = _load_script("run_local_amali_chat_smoke")
    monkeypatch.setattr(mod, "READINESS_BASE", tmp_path / "base_model_readiness")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_local_amali_chat_smoke.py",
            "--mode",
            "raw_base",
            "--model-id",
            EVIL_MODEL,
        ],
    )

    assert mod.main() == 1
    captured = capsys.readouterr()
    assert "Chat smoke [raw_base]: POLICY_BLOCKED" in captured.out
    assert "MODEL_NOT_ALLOWLISTED" in captured.err
    report = _latest_json(mod.READINESS_BASE, "local_chat_smoke_report.json")
    assert report["status"] == "POLICY_BLOCKED"
    assert report["metrics"]["model_id"] == EVIL_MODEL
