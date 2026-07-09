"""Phase 5/6: training config validation, dry-run gates, trainer refusals."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from amali.model_gateway.availability import ModelAvailability
from amali.training.config import TrainingConfig, estimate_vram_gb, path_is_inside_repo
from amali.training import trainer as trainer_mod
from amali.training.trainer import (
    build_training_plan,
    run_dry_run,
    run_training,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    REPO_ROOT / "fixtures" / "eval" / "base_model_gate_manifest.json"
)


def _config(tmp_path: Path, **overrides) -> TrainingConfig:
    defaults = dict(
        run_id="run_test",
        base_model_id="Qwen/Qwen2.5-1.5B-Instruct",
        base_model_revision="0123abc0123abc0123abc0123abc0123abc01234",
        output_dir=str(tmp_path / "outside_repo" / "adapter"),
        dataset_manifest_path="",
        eval_suite_manifest_path=str(MANIFEST_PATH),
        method="lora",
        quantization="none",
        optimizer="adamw_torch",
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)


def _write_dataset(tmp_path: Path, count: int, *, contaminated: bool = False) -> Path:
    """A synthetic-but-honestly-shaped dataset file for gate tests."""
    path = tmp_path / "train.jsonl"
    rows = []
    for i in range(count):
        rows.append(
            {
                "example_id": f"t_{i:04d}",
                "source_id": f"docs/doc_{i}.md",
                "source_hash": "a" * 64,
                "source_path": f"docs/doc_{i}.md",
                "data_class": "documentation",
                "synthetic": False,
                "instruction": "Restate the passage faithfully and cite the source.",
                "input_context": (
                    f"Local passage number {i}: the audit ledger chains events "
                    f"with SHA-256 and records actor, action, and target for "
                    f"entry {i} of the documentation corpus."
                ),
                "expected_response": f"Passage {i} restated with citation.",
                "expected_status": "SUCCESS",
                "required_citations": [f"docs/doc_{i}.md"],
                "forbidden_outputs": [],
                "behavior_tags": ["evidence_grounding"],
                "data_wall_decision_id": "decision_x",
                "content_hash": f"{i:064d}",
            }
        )
    if contaminated:
        seed = json.loads(
            (REPO_ROOT / "fixtures" / "eval" / "base_model_gate_seed.json")
            .read_text(encoding="utf-8")
        )
        rows[0]["input_context"] = seed["items"][0]["prompt"]
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    return path


@pytest.fixture
def all_gates_pass(monkeypatch):
    """Make deps + weights gates pass so the pure gating logic is testable
    on a machine without owner-installed training dependencies."""
    monkeypatch.setattr(trainer_mod, "_missing_train_deps", lambda method: [])
    monkeypatch.setattr(
        trainer_mod,
        "check_local_model_availability",
        lambda model_id, cache_dir=None: ModelAvailability(
            model_id=model_id,
            status="AVAILABLE",
            torch_installed=True,
            transformers_installed=True,
            weights_cached_locally=True,
            cache_dir_checked="fake",
            notes=[],
        ),
    )


# --- config validation ---------------------------------------------------------


def test_config_rejects_unknown_method(tmp_path):
    with pytest.raises(ValueError):
        _config(tmp_path, method="full_finetune")


def test_config_rejects_bad_seq_len_and_rank(tmp_path):
    with pytest.raises(ValueError):
        _config(tmp_path, max_seq_len=16)
    with pytest.raises(ValueError):
        _config(tmp_path, lora_rank=128)


def test_config_hash_stable_and_sensitive(tmp_path):
    a = _config(tmp_path)
    b = _config(tmp_path)
    c = _config(tmp_path, lora_rank=8)
    assert a.config_hash() == b.config_hash()
    assert a.config_hash() != c.config_hash()


def test_output_inside_repo_detected(tmp_path):
    inside = _config(tmp_path, output_dir=str(REPO_ROOT / "models" / "x"))
    outside = _config(tmp_path)
    assert inside.output_inside(REPO_ROOT) is True
    assert outside.output_inside(REPO_ROOT) is False


def test_windows_absolute_default_output_not_inside_repo_on_posix_ci():
    assert path_is_inside_repo("D:/AMALI/models/amali_ft_v0", REPO_ROOT) is False
    assert path_is_inside_repo("//server/share/amali_ft_v0", REPO_ROOT) is False


# --- VRAM heuristic -------------------------------------------------------------


def test_vram_estimate_1_5b_qlora_fits_8gb():
    estimate = estimate_vram_gb(
        parameter_count="1.5B",
        quantization="4bit_nf4",
        precision="bf16",
        max_seq_len=1024,
        batch_size=1,
        lora_rank=16,
    )
    assert estimate.total_gb <= 8.0


def test_vram_estimate_3b_fp16_exceeds_8gb():
    estimate = estimate_vram_gb(
        parameter_count="3B",
        quantization="none",
        precision="fp16",
        max_seq_len=1024,
        batch_size=1,
        lora_rank=16,
    )
    assert estimate.total_gb > 8.0


# --- dry-run gates --------------------------------------------------------------


def test_feasible_config_passes_dry_run(tmp_path, all_gates_pass):
    dataset = _write_dataset(tmp_path, 220)
    report = run_dry_run(
        _config(tmp_path), repo_root=REPO_ROOT, dataset_path=dataset
    )
    assert report.status == "DRY_RUN_PASS", (report.status, report.reasons)
    # sourcepack_valid is NOT_RUN when no raw-dir examples exist.
    assert all(v in ("PASS", "NOT_RUN") for v in report.checks.values())
    assert report.checks["registry_outside_repo"] == "PASS"


def test_small_dataset_refused(tmp_path, all_gates_pass):
    dataset = _write_dataset(tmp_path, 50)
    report = run_dry_run(
        _config(tmp_path), repo_root=REPO_ROOT, dataset_path=dataset
    )
    assert report.status == "DATASET_NOT_READY"


def test_contaminated_dataset_refused(tmp_path, all_gates_pass):
    dataset = _write_dataset(tmp_path, 220, contaminated=True)
    report = run_dry_run(
        _config(tmp_path), repo_root=REPO_ROOT, dataset_path=dataset
    )
    assert report.status == "CONTAMINATION_FAIL"


def test_unpinned_revision_refused(tmp_path, all_gates_pass):
    dataset = _write_dataset(tmp_path, 220)
    report = run_dry_run(
        _config(tmp_path, base_model_revision="main"),
        repo_root=REPO_ROOT,
        dataset_path=dataset,
    )
    assert report.status == "REVISION_NOT_PINNED"


def test_output_inside_repo_refused(tmp_path, all_gates_pass):
    dataset = _write_dataset(tmp_path, 220)
    report = run_dry_run(
        _config(tmp_path, output_dir=str(REPO_ROOT / "models" / "bad")),
        repo_root=REPO_ROOT,
        dataset_path=dataset,
    )
    assert report.checks["output_outside_repo"] == "FAIL"
    assert report.status != "DRY_RUN_PASS"


def test_missing_train_deps_reported_honestly(tmp_path, monkeypatch):
    # Only the deps gate should decide here: make earlier gates pass.
    monkeypatch.setattr(
        trainer_mod,
        "check_local_model_availability",
        lambda model_id, cache_dir=None: ModelAvailability(
            model_id=model_id,
            status="AVAILABLE",
            torch_installed=True,
            transformers_installed=True,
            weights_cached_locally=True,
            cache_dir_checked="fake",
            notes=[],
        ),
    )
    monkeypatch.setattr(
        trainer_mod,
        "_missing_train_deps",
        lambda method: ["transformers", "peft"],
    )
    dataset = _write_dataset(tmp_path, 220)
    report = run_dry_run(
        _config(tmp_path), repo_root=REPO_ROOT, dataset_path=dataset
    )
    assert report.status == "TRAIN_DEPS_MISSING"
    assert report.missing_deps == ["transformers", "peft"]
    assert any("pip install" in a for a in report.owner_actions)


def test_vram_infeasible_refused(tmp_path, all_gates_pass):
    dataset = _write_dataset(tmp_path, 220)
    report = run_dry_run(
        _config(tmp_path, vram_budget_gb=1.0),
        repo_root=REPO_ROOT,
        dataset_path=dataset,
    )
    assert report.status == "VRAM_INFEASIBLE"


def test_registry_inside_repo_refused(tmp_path, all_gates_pass):
    dataset = _write_dataset(tmp_path, 220)
    report = run_dry_run(
        _config(tmp_path),
        repo_root=REPO_ROOT,
        dataset_path=dataset,
        registry_path=REPO_ROOT / "models" / "registry.json",
    )
    assert report.checks["registry_outside_repo"] == "FAIL"
    assert report.status != "DRY_RUN_PASS"


def test_raw_sourced_examples_require_valid_sourcepack(tmp_path, all_gates_pass):
    # Dataset examples claim to come from the raw owner dir, but no valid
    # sourcepack exists there -> OWNER_DATA_REQUIRED.
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    dataset = tmp_path / "train.jsonl"
    rows = []
    for i in range(220):
        rows.append(
            {
                "example_id": f"t_{i:04d}",
                "source_id": f"raw/doc_{i}.md",
                "source_hash": "a" * 64,
                "source_path": str(raw_root / f"doc_{i}.md"),
                "data_class": "internal",
                "synthetic": False,
                "instruction": "Restate the passage faithfully.",
                "input_context": f"Owner passage number {i} about the ledger.",
                "expected_response": f"Passage {i} restated.",
                "expected_status": "SUCCESS",
                "required_citations": [],
                "forbidden_outputs": [],
                "behavior_tags": [],
                "data_wall_decision_id": "decision_x",
                "content_hash": f"{i:064d}",
            }
        )
    dataset.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    report = run_dry_run(
        _config(tmp_path),
        repo_root=REPO_ROOT,
        dataset_path=dataset,
        sourcepack_root=raw_root,
    )
    assert report.checks["sourcepack_valid"] == "FAIL"
    assert report.status == "OWNER_DATA_REQUIRED"


# --- trainer refusals -------------------------------------------------------------


def test_trainer_refuses_when_gates_fail_and_produces_no_artifacts(tmp_path):
    dataset = _write_dataset(tmp_path, 10)
    report = run_training(
        _config(tmp_path),
        repo_root=REPO_ROOT,
        execute=True,
        dataset_path=dataset,
    )
    assert report.status == "DATASET_NOT_READY"
    assert report.adapter_dir == ""
    assert report.loss_curve == []
    assert report.step_count == 0
    assert not (tmp_path / "outside_repo" / "adapter").exists()


def test_dry_run_flag_never_trains(tmp_path, all_gates_pass):
    dataset = _write_dataset(tmp_path, 220)
    report = run_training(
        _config(tmp_path),
        repo_root=REPO_ROOT,
        execute=False,
        dataset_path=dataset,
    )
    assert report.status == "DRY_RUN_PASS"
    assert report.adapter_dir == ""
    assert report.started_at == ""


def test_training_report_schema_stable(tmp_path):
    report = run_training(
        _config(tmp_path),
        repo_root=REPO_ROOT,
        execute=False,
        dataset_path=tmp_path / "missing.jsonl",
    )
    dumped = report.model_dump(mode="json")
    for field in (
        "status",
        "run_id",
        "started_at",
        "ended_at",
        "wall_clock_seconds",
        "device",
        "gpu_name",
        "peak_vram_gb",
        "step_count",
        "loss_curve",
        "config_hash",
        "dataset_hash",
        "eval_suite_hash",
        "base_model_id",
        "base_model_revision",
        "adapter_dir",
        "adapter_file_hashes",
        "degradations_applied",
        "dry_run",
        "risks",
        "owner_actions",
    ):
        assert field in dumped


# --- training plan (resume / max-steps) ----------------------------------------


def test_training_plan_resume_and_max_steps(tmp_path):
    config = _config(tmp_path, max_steps=0)
    plan = build_training_plan(config, max_steps_override=50, resume=True)
    assert plan["max_steps"] == 50
    assert plan["resume_from_checkpoint"] is True
    assert plan["seed"] == config.seed
    assert plan["gradient_checkpointing"] is True

    default_plan = build_training_plan(config)
    assert default_plan["max_steps"] == -1
    assert default_plan["resume_from_checkpoint"] is False


def test_training_plan_precision_flags(tmp_path):
    bf16 = build_training_plan(_config(tmp_path, precision="bf16"))
    fp16 = build_training_plan(_config(tmp_path, precision="fp16"))
    assert bf16["bf16"] is True and bf16["fp16"] is False
    assert fp16["fp16"] is True and fp16["bf16"] is False


# --- torch isolation --------------------------------------------------------------


def test_trainer_import_is_torch_free():
    code = (
        "import sys; import amali.training.trainer; "
        "import amali.training.config; import amali.training.reports; "
        "print('torch_loaded=', 'torch' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    assert "torch_loaded= False" in out.stdout
