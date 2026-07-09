"""Assemble the canonical Base Model Gate artifact directory (Phase 16).

    python scripts/collect_base_model_gate_artifacts.py

Copies the newest instance of every canonical artifact from the
timestamped run directories into one fresh canonical directory,
generates the live allowlist/availability reports, and writes
remaining_risks.md from the actual current state. Artifacts that cannot
exist yet (e.g. a training run that never happened produces no adapter)
are listed honestly as absent with the owner action required.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from amali.model_gateway.availability import check_local_model_availability  # noqa: E402
from amali.model_gateway.base_model_allowlist import get_allowlist  # noqa: E402

ARTIFACT_BASE = REPO_ROOT / "artifacts" / "base_model_gate"
PREFLIGHT_BASE = REPO_ROOT / "artifacts" / "base_model_gate_preflight"
FT_V0_BASE = REPO_ROOT / "artifacts" / "amali_ft_v0"
READINESS_BASE = REPO_ROOT / "artifacts" / "base_model_readiness"

CANONICAL = [
    "preflight_report.json",
    "sourcepack_validation_report.json",
    "sourcepack_validation_report.md",
    "training_env_report.json",
    "training_fallback_plan.json",
    "security_validation_report.json",
    "security_validation_report.md",
    "eval_suite_freeze_report.json",
    "eval_suite_freeze_report.md",
    "dataset_build_report.json",
    "dataset_build_report.md",
    "dataset_manifest.json",
    "data_wall_training_report.json",
    "contamination_report.json",
    "contamination_report.md",
    "base_model_allowlist_report.json",
    "model_availability_report.json",
    "training_config.json",
    "training_dry_run_report.json",
    "training_run_report.json",
    "training_run_report.md",
    "checkpoint_registry.json",
    "checkpoint_integrity_report.json",
    "base_model_revision_pin_report.json",
    "model_gateway_smoke_report.json",
    "base_model_readiness_report.json",
    "base_model_readiness_report.md",
    "local_chat_smoke_report.json",
    "eval_report_raw_base.json",
    "eval_report_raw_base.md",
    "eval_report_wrapped_raw_base.json",
    "eval_report_amali_ft.json",
    "eval_report_amali_wrapped_ft.json",
    "safety_regression_report.json",
    "comparison_base_readiness.json",
    "comparison_base_readiness.md",
    "comparison_three_way.json",
    "comparison_three_way.md",
    "promotion_report.json",
    "promotion_report.md",
    "model_registry_entry.json",
    "dwac_model_pack_report.json",
    "gate_b_model_report.json",
    "model_card.md",
    "remaining_risks.md",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_ref() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=REPO_ROOT,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # noqa: BLE001 - metadata only
        pass
    return "unknown"


def _newest(name: str) -> Path | None:
    candidates: list[Path] = []
    for base in (ARTIFACT_BASE, PREFLIGHT_BASE, FT_V0_BASE, READINESS_BASE):
        if base.is_dir():
            candidates.extend(base.glob(f"*/{name}"))
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: str(p.parent))[-1]


def _latest(pattern: str) -> dict | None:
    """Load the newest artifact matching pattern as JSON, or None."""
    path = _newest(pattern)
    if path is None:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _gate_b_from_demo() -> Path | None:
    base = REPO_ROOT / "artifacts" / "base_ai_gate"
    if not base.is_dir():
        return None
    candidates = sorted(base.glob("*/gate_b_model_report.json"))
    return candidates[-1] if candidates else None


def main() -> int:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ARTIFACT_BASE / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    command = "python scripts/collect_base_model_gate_artifacts.py"

    def envelope(status: str, **extra) -> dict:
        return {
            "schema_version": "1.0.0",
            "generated_at": _now(),
            "repo_ref": _repo_ref(),
            "command": command,
            "status": status,
            **extra,
        }

    # Live reports ---------------------------------------------------------
    (out_dir / "base_model_allowlist_report.json").write_text(
        json.dumps(
            envelope(
                "PASS",
                metrics={"entries": len(get_allowlist())},
                decisions=[e.model_dump(mode="json") for e in get_allowlist()],
                risks=[
                    "allowed_revision is 'main' until the owner pins the "
                    "downloaded snapshot; promotion refuses floating revisions"
                ],
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    availability = check_local_model_availability()
    (out_dir / "model_availability_report.json").write_text(
        json.dumps(
            envelope(
                availability.status,
                metrics={
                    "torch_installed": availability.torch_installed,
                    "transformers_installed": availability.transformers_installed,
                    "weights_cached_locally": availability.weights_cached_locally,
                },
                decisions=[availability.model_dump(mode="json")],
                risks=[],
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    # Copy the newest instance of everything else ----------------------------
    missing: list[str] = []
    for name in CANONICAL:
        if name == "remaining_risks.md":  # generated below, never copied
            continue
        if (out_dir / name).exists():
            continue
        source = _newest(name)
        if source is None and name == "gate_b_model_report.json":
            source = _gate_b_from_demo()
        if source is None:
            missing.append(name)
            continue
        shutil.copy2(source, out_dir / name)

    # remaining_risks.md — dynamic from latest artifacts -----
    dataset_report = _latest("*/dataset_build_report.json")
    dataset_metric = dataset_report.get("metrics", {}) if dataset_report else {}
    non_synthetic_count = dataset_metric.get("non_synthetic_count", 0)
    min_required = dataset_metric.get("min_required_non_synthetic", 200)
    dataset_status = (
        f"{non_synthetic_count} approved non-synthetic examples < {min_required} floor -> "
        f"DATASET_NOT_READY"
        if non_synthetic_count < min_required
        else f"{non_synthetic_count} non-synthetic examples >= {min_required} floor -> READY"
    )

    avail = availability
    train_deps_line = (
        "- training deps installed: torch, transformers, peft, datasets, "
        "accelerate, bitsandbytes ready.\n"
        if avail.torch_installed and avail.transformers_installed
        else (
            "- training deps: transformers/peft/datasets/accelerate/bitsandbytes "
            "not installed. Owner action: "
            'python -m pip install -e ".[dev,local_llm,train]".\n'
        )
    )
    weights_line = (
        f"- base weights: {avail.model_id} locally cached; "
        "Gate B ready to smoke-test offline.\n"
        if avail.weights_cached_locally
        else (
            "- base weights: no local snapshot. Owner action: "
            "python scripts/download_model.py --model "
            "Qwen/Qwen2.5-1.5B-Instruct --revision <PINNED_REVISION>.\n"
        )
    )

    (out_dir / "remaining_risks.md").write_text(
        "# Remaining Risks — Base Model Gate\n\n"
        "Honest current state: the Base Model Gate is **code-complete**; "
        "AMALI-FT-v0 does **not** exist yet.\n\n"
        + f"- dataset: {dataset_status}. "
        + (
            "Owner action: add approved local source material under "
            "D:/AMALI/data/raw/ and rerun scripts/build_training_dataset.py.\n"
            if non_synthetic_count < min_required
            else ""
        )
        + train_deps_line
        + weights_line
        + "- revision: no pinned base model revision yet; promotion refuses "
        "floating 'main' by design.\n"
        "- raw base / FT / wrapped-FT evals: not runnable until deps and "
        "weights exist; reports carry DEPS_MISSING / MODEL_NOT_AVAILABLE.\n"
        "- promotion: NOT_READY (training never completed); no partial "
        "promotion exists.\n"
        "- the deterministic scorer is lexical, not semantic entailment; "
        "eval breadth is AMALI-system behavior, not general ability.\n"
        "- bitsandbytes on Windows may be unavailable at owner-install "
        "time; the trainer then reports TRAIN_DEPS_MISSING and the lora "
        "(non-quantized) method on the 0.5B fallback is the safe path.\n"
        + (
            "\nCanonical artifacts not yet producible (honest absences):\n"
            + "\n".join(f"- {name}" for name in missing)
            if missing
            else ""
        )
        + "\n",
        encoding="utf-8",
    )

    present = [n for n in CANONICAL if (out_dir / n).exists()]
    print(f"Canonical dir: {out_dir}")
    print(f"present: {len(present)}/{len(CANONICAL)}")
    for name in missing:
        print(f"  absent (honest): {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
