"""Train (or dry-run) the AMALI-FT-v0 adapter.

    python scripts/train_amali_adapter.py --dry-run
    python scripts/train_amali_adapter.py                 # owner-invoked real run
    python scripts/train_amali_adapter.py --resume
    python scripts/train_amali_adapter.py --max-steps 50

The dry-run re-verifies every gate (frozen eval suite, dataset floor,
contamination, allowlist, revision pin, local weights, train deps, VRAM
heuristic, output dir outside git) and never imports torch. The real run
refuses unless the dry-run passes. Nothing is faked: a blocked run emits a
typed report and exits nonzero.

Artifacts under ``artifacts/base_model_gate/<timestamp>/``:
    training_config.json
    training_dry_run_report.json
    training_run_report.json (+ .md) when a real run was attempted
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from amali.training.config import DEFAULT_OUTPUT_DIR, TrainingConfig  # noqa: E402
from amali.training.trainer import run_training  # noqa: E402

MANIFEST_PATH = REPO_ROOT / "fixtures" / "eval" / "base_model_gate_manifest.json"


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


def _latest_dataset_manifest() -> str:
    """Most recent dataset_manifest.json emitted by the dataset build."""
    base = REPO_ROOT / "artifacts" / "base_model_gate"
    if not base.is_dir():
        return ""
    candidates = sorted(base.glob("*/dataset_manifest.json"))
    return str(candidates[-1]) if candidates else ""


def _default_config(args: argparse.Namespace) -> TrainingConfig:
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    return TrainingConfig(
        run_id=run_id,
        base_model_id=args.model,
        base_model_revision=args.revision,
        output_dir=args.output_dir,
        dataset_manifest_path=_latest_dataset_manifest(),
        eval_suite_manifest_path=str(MANIFEST_PATH),
        max_steps=args.max_steps,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="AMALI adapter trainer")
    parser.add_argument("--dry-run", action="store_true", help="gates only; no training")
    parser.add_argument("--resume", action="store_true", help="resume from last checkpoint")
    parser.add_argument("--max-steps", type=int, default=0, help="cap training steps")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--revision",
        default="main",
        help="pinned base model revision (required for a promotable run)",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    config = _default_config(args)
    report = run_training(
        config,
        repo_root=REPO_ROOT,
        execute=not args.dry_run,
        max_steps_override=args.max_steps,
        resume=args.resume,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO_ROOT / "artifacts" / "base_model_gate" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    command = "python scripts/train_amali_adapter.py" + (
        " --dry-run" if args.dry_run else ""
    )

    def envelope(status: str, **extra) -> dict:
        return {
            "schema_version": "1.0.0",
            "generated_at": _now(),
            "repo_ref": _repo_ref(),
            "command": command,
            "status": status,
            **extra,
        }

    (out_dir / "training_config.json").write_text(
        json.dumps(
            envelope(
                report.status,
                metrics={"config_hash": config.config_hash()},
                decisions=[config.model_dump(mode="json")],
                risks=[],
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    dry = report.dry_run
    (out_dir / "training_dry_run_report.json").write_text(
        json.dumps(
            envelope(
                dry.status if dry else "NOT_RUN",
                metrics={
                    "estimated_vram_gb": dry.estimated_vram_gb if dry else None,
                    "vram_budget_gb": dry.vram_budget_gb if dry else None,
                    "checks": dry.checks if dry else {},
                },
                decisions=[dry.model_dump(mode="json")] if dry else [],
                risks=dry.reasons if dry else [],
                owner_actions=dry.owner_actions if dry else [],
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    if not args.dry_run:
        (out_dir / "training_run_report.json").write_text(
            json.dumps(
                envelope(
                    report.status,
                    metrics={
                        "step_count": report.step_count,
                        "wall_clock_seconds": report.wall_clock_seconds,
                        "peak_vram_gb": report.peak_vram_gb,
                        "device": report.device,
                        "gpu_name": report.gpu_name,
                        "loss_first": report.loss_curve[0] if report.loss_curve else None,
                        "loss_last": report.loss_curve[-1] if report.loss_curve else None,
                    },
                    decisions=[report.model_dump(mode="json")],
                    risks=report.risks,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        (out_dir / "training_run_report.md").write_text(
            f"# Training Run Report\n\nStatus: **{report.status}**\n\n"
            f"- run_id: {report.run_id}\n"
            f"- base model: {report.base_model_id} @ {report.base_model_revision}\n"
            f"- device: {report.device or 'n/a'} {report.gpu_name}\n"
            f"- steps: {report.step_count}\n"
            f"- wall clock: {report.wall_clock_seconds or 'n/a'} s\n"
            f"- peak VRAM: {report.peak_vram_gb or 'n/a'} GB\n"
            f"- adapter: {report.adapter_dir or 'not produced'}\n\n"
            "Owner actions:\n"
            + "\n".join(f"- {a}" for a in report.owner_actions or ["none"])
            + "\n\nRisks:\n"
            + "\n".join(f"- {r}" for r in report.risks or ["none recorded"])
            + "\n",
            encoding="utf-8",
        )

    print(f"Training: {report.status}")
    if dry:
        for name, outcome in dry.checks.items():
            print(f"  gate {name}: {outcome}")
    for action in report.owner_actions:
        print(f"  owner action: {action}", file=sys.stderr)
    print(f"Reports: {out_dir}")
    return 0 if report.status in ("DRY_RUN_PASS", "TRAINING_COMPLETED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
