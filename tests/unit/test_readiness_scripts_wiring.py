"""Script wiring: base readiness uses the base comparison, never the FT one.

Loads the real scripts (not copies of their logic) against temp artifact
trees and proves:

* ``compare_models.py`` defaults to the base-readiness comparison and
  PASSes with raw_base + amali_wrapped_raw_base only;
* ``compare_models.py --mode ft_promotion`` still refuses a missing
  amali_wrapped_ft_v0;
* ``check_base_model_readiness.py`` reaches BASE_MODEL_READY from base
  prerequisites alone while the FT verdict stays AMALI_FT_NOT_READY;
* the readiness checker requires the base comparison artifact — a
  three-way report cannot stand in for it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

from amali.eval.model_eval import ModelEvalReport
from amali.model_gateway.availability import SnapshotAvailability
from amali.model_gateway.revision_pin import RevisionPinReport

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
SUITE_HASH = "suite_x"

METRICS = {
    "overall_score": 0.5,
    "honest_unknown_score": 0.5,
    "safety_score": 0.5,
    "unsupported_success_rate": 0.0,
}


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_wiring_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_eval(gate_dir: Path, stamp: str, name: str, mode: str) -> None:
    report = ModelEvalReport(
        mode=mode,
        status="PASS",
        model_id=MODEL,
        revision=REVISION,
        suite_hash=SUITE_HASH,
        metrics=dict(METRICS),
    )
    _write_json(
        gate_dir / stamp / f"{name}.json",
        {"status": "PASS", "decisions": [report.model_dump(mode="json")]},
    )


def _compare_models_fixture(tmp_path: Path, monkeypatch):
    mod = _load_script("compare_models")
    gate_dir = tmp_path / "base_model_gate"
    _write_eval(gate_dir, "20260101T000000Z", "eval_report_raw_base", "raw_base")
    _write_eval(
        gate_dir,
        "20260101T000000Z",
        "eval_report_wrapped_raw_base",
        "amali_wrapped_raw_base",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"suite_hash": SUITE_HASH}), encoding="utf-8"
    )
    monkeypatch.setattr(mod, "ARTIFACT_BASE", gate_dir)
    monkeypatch.setattr(mod, "MANIFEST_PATH", manifest)
    return mod, gate_dir


def test_compare_models_default_base_mode_passes_without_any_ft(
    tmp_path, monkeypatch
):
    mod, gate_dir = _compare_models_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["compare_models.py"])

    assert mod.main() == 0
    reports = sorted(gate_dir.glob("*/comparison_base_readiness.json"))
    assert reports
    data = json.loads(reports[-1].read_text(encoding="utf-8"))
    assert data["status"] == "PASS"
    assert set(data["metrics"]) == {"raw_base", "amali_wrapped_raw_base"}
    # base mode never emits the FT artifact
    assert not list(gate_dir.glob("*/comparison_three_way.json"))


def test_compare_models_ft_promotion_mode_refuses_missing_wrapped_ft(
    tmp_path, monkeypatch
):
    mod, gate_dir = _compare_models_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        sys, "argv", ["compare_models.py", "--mode", "ft_promotion"]
    )

    assert mod.main() == 1
    reports = sorted(gate_dir.glob("*/comparison_three_way.json"))
    assert reports
    data = json.loads(reports[-1].read_text(encoding="utf-8"))
    assert data["status"] == "NOT_RUN"
    assert any("amali_wrapped_ft_v0" in risk for risk in data["risks"])


def _readiness_fixture(tmp_path: Path, monkeypatch):
    mod = _load_script("check_base_model_readiness")
    readiness_dir = tmp_path / "base_model_readiness"
    gate_dir = tmp_path / "base_model_gate"
    preflight_dir = tmp_path / "base_model_gate_preflight"
    ft_dir = tmp_path / "amali_ft_v0"
    stamp = "20260101T000000Z"

    _write_json(preflight_dir / stamp / "preflight_report.json", {"status": "PASS"})
    _write_json(
        readiness_dir / stamp / "model_gateway_smoke_report.json",
        {"status": "SUCCESS"},
    )
    for name in (
        "eval_report_raw_base.json",
        "eval_report_wrapped_raw_base.json",
        "security_validation_report.json",
    ):
        _write_json(gate_dir / stamp / name, {"status": "PASS"})
    # a stale FT three-way report must not influence the base verdict
    _write_json(
        gate_dir / stamp / "comparison_three_way.json", {"status": "NOT_RUN"}
    )
    # deliberately no promotion_report.json: AMALI-FT-v0 does not exist

    pin = RevisionPinReport(
        status="REVISION_PINNED", model_id=MODEL, revision=REVISION
    )
    monkeypatch.setattr(mod, "READINESS_BASE", readiness_dir)
    monkeypatch.setattr(mod, "GATE_BASE", gate_dir)
    monkeypatch.setattr(mod, "PREFLIGHT_BASE", preflight_dir)
    monkeypatch.setattr(mod, "FT_BASE", ft_dir)
    monkeypatch.setattr(mod, "load_latest_pin", lambda root: pin)
    monkeypatch.setattr(
        mod,
        "check_model_snapshot",
        lambda model_id, revision: SnapshotAvailability(
            model_id=model_id,
            revision_requested=revision,
            status="AVAILABLE",
        ),
    )
    monkeypatch.setattr(mod, "_core_import_torch_free", lambda: True)
    return mod, readiness_dir, gate_dir


def _latest_readiness_report(readiness_dir: Path) -> dict:
    reports = sorted(readiness_dir.glob("*/base_model_readiness_report.json"))
    assert reports
    return json.loads(reports[-1].read_text(encoding="utf-8"))


def test_readiness_script_reaches_ready_from_base_prerequisites_only(
    tmp_path, monkeypatch
):
    mod, readiness_dir, gate_dir = _readiness_fixture(tmp_path, monkeypatch)
    _write_json(
        gate_dir / "20260101T000000Z" / "comparison_base_readiness.json",
        {"status": "PASS"},
    )

    assert mod.main() == 0
    data = _latest_readiness_report(readiness_dir)
    assert data["status"] == "BASE_MODEL_READY"
    assert data["metrics"]["amali_ft_status"] == "AMALI_FT_NOT_READY"


def test_readiness_script_requires_base_comparison_not_three_way(
    tmp_path, monkeypatch
):
    mod, readiness_dir, gate_dir = _readiness_fixture(tmp_path, monkeypatch)
    # only a (passing!) three-way report exists — still not the base artifact
    _write_json(
        gate_dir / "20260102T000000Z" / "comparison_three_way.json",
        {"status": "PASS"},
    )

    assert mod.main() == 1
    data = _latest_readiness_report(readiness_dir)
    assert data["status"] == "BASE_MODEL_NOT_READY"
    assert data["metrics"]["comparison"] == "NOT_RUN"
    assert data["metrics"]["amali_ft_status"] == "AMALI_FT_NOT_READY"


def test_core_import_subprocess_gets_source_checkout_pythonpath(monkeypatch):
    mod = _load_script("check_base_model_readiness")
    captured: dict = {}
    monkeypatch.setenv("PYTHONPATH", "existing_path")

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    assert mod._core_import_torch_free() is True
    pythonpath = captured["env"]["PYTHONPATH"].split(os.pathsep)
    assert pythonpath[0] == str(REPO_ROOT / "src")
    assert "existing_path" in pythonpath
