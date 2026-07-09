"""Phase 12: Gate B demo integration — honest optional model paths."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from amali.demo.base_ai_gate import run_demo

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_default_demo_reports_gate_b_not_run(tmp_path):
    out = run_demo(repo_root=REPO_ROOT, output_dir=tmp_path / "a")
    report = json.loads(
        (out / "gate_b_model_report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "NOT_RUN"
    assert report["metrics"]["requested"] is False


def test_missing_ft_adapter_reports_honest_status(tmp_path):
    out = run_demo(
        repo_root=REPO_ROOT,
        output_dir=tmp_path / "b",
        with_local_model=True,
        model_mode="amali_ft_v0",
    )
    report = json.loads(
        (out / "gate_b_model_report.json").read_text(encoding="utf-8")
    )
    # No trained adapter exists on a fresh machine: the status must be a
    # typed honest skip, never PASS and never an exception.
    assert report["status"] in (
        "MODEL_NOT_AVAILABLE",
        "CHECKPOINT_TAMPERED",
        "DEPS_MISSING",
        "SKIPPED",
    )
    demo = json.loads((out / "demo_report.json").read_text(encoding="utf-8"))
    assert demo["status"] == "PASS"  # Gate A unaffected


def test_missing_raw_base_reports_honest_status(tmp_path):
    out = run_demo(
        repo_root=REPO_ROOT,
        output_dir=tmp_path / "c",
        with_local_model=True,
        model_mode="raw_base",
    )
    report = json.loads(
        (out / "gate_b_model_report.json").read_text(encoding="utf-8")
    )
    assert report["status"] in (
        "PASS",
        "SKIPPED",
        "SKIPPED_LOCAL_MODEL_NOT_AVAILABLE",
        "DEPS_MISSING",
    )


def test_gate_a_demo_never_imports_torch():
    code = (
        "import sys, tempfile; from pathlib import Path; "
        "from amali.demo.base_ai_gate import run_demo; "
        f"run_demo(repo_root=r'{REPO_ROOT}', "
        "output_dir=Path(tempfile.mkdtemp()) / 'x'); "
        "print('torch_loaded=', 'torch' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert out.returncode == 0, out.stderr
    assert "torch_loaded= False" in out.stdout
