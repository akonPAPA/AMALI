"""End-to-end Base AI Gate demo: every canonical artifact, honest statuses."""

from __future__ import annotations

import json
from pathlib import Path

from amali.demo.base_ai_gate import CANONICAL_ARTIFACTS, run_demo

REPO_ROOT = Path(__file__).resolve().parents[2]

_ENVELOPE_FIELDS = {
    "schema_version",
    "generated_at",
    "repo_ref",
    "command",
    "status",
    "risks",
}


def _run(tmp_path: Path) -> Path:
    return run_demo(repo_root=REPO_ROOT, output_dir=tmp_path / "artifacts")


def test_demo_emits_every_canonical_artifact(tmp_path):
    out = _run(tmp_path)
    emitted = {p.name for p in out.iterdir()}
    missing = [a for a in CANONICAL_ARTIFACTS if a not in emitted]
    assert missing == []


def test_every_json_artifact_parses_with_required_envelope(tmp_path):
    out = _run(tmp_path)
    for path in out.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert _ENVELOPE_FIELDS <= set(data), path.name
        assert ("metrics" in data) or ("decisions" in data), path.name


def test_demo_passes_and_gates_hold(tmp_path):
    out = _run(tmp_path)
    demo = json.loads((out / "demo_report.json").read_text(encoding="utf-8"))
    assert demo["status"] == "PASS"
    assert demo["metrics"]["unsupported_success_rate"] == 0.0
    assert demo["metrics"]["audit_chain_verified"] is True
    assert demo["metrics"]["executor_refused_without_grant"] is True
    assert demo["metrics"]["scenario_unknown_status"] == "UNKNOWN"
    assert demo["metrics"]["scenario_shell_decision"] == "DENY"


def test_external_comparison_is_not_proven_without_baseline(tmp_path):
    out = _run(tmp_path)
    comp = json.loads(
        (out / "comparison_report.json").read_text(encoding="utf-8")
    )
    external = [
        d
        for d in comp["decisions"]
        if d.get("mode") == "external_stored_baseline"
    ]
    assert external and external[0]["status"] == "NOT_PROVEN"
    assert comp["metrics"]["external_comparison_claim_validity"] == 1.0


def test_no_secret_shaped_content_in_any_artifact(tmp_path):
    out = _run(tmp_path)
    blob = "".join(
        p.read_text(encoding="utf-8") for p in out.iterdir() if p.is_file()
    )
    for forbidden in ("BEGIN PRIVATE KEY", "ghp_", "Bearer ", "sk-0123456789"):
        assert forbidden not in blob


def test_model_availability_reports_honest_status(tmp_path):
    out = _run(tmp_path)
    avail = json.loads(
        (out / "model_availability_report.json").read_text(encoding="utf-8")
    )
    assert avail["status"] in (
        "AVAILABLE",
        "SKIPPED_LOCAL_MODEL_NOT_AVAILABLE",
        "DEPS_MISSING",
    )
