"""Verify registered checkpoints against disk (AMALI-FT-v0 Stage 7).

    python scripts/verify_checkpoint.py [--registry PATH] [--checkpoint-id ID]

Recomputes SHA-256 of every adapter file listed in the registry. A
missing file is CHECKPOINT_INVALID; a hash mismatch is
CHECKPOINT_TAMPERED. A floating base revision or a non-completed
train_status marks the checkpoint NOT_PROMOTABLE. Verification never
mutates the registry.

Exit 0 only when at least one checkpoint exists and every checkpoint
verifies clean.
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

from amali.model_gateway.base_model_allowlist import (  # noqa: E402
    FLOATING_REVISIONS,
)
from amali.training.registry import (  # noqa: E402
    DEFAULT_REGISTRY_PATH,
    load_registry,
    verify_checkpoint,
)


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


def main() -> int:
    parser = argparse.ArgumentParser(description="AMALI checkpoint verifier")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument(
        "--checkpoint-id", default=None, help="verify only this checkpoint"
    )
    args = parser.parse_args()

    manifests = load_registry(Path(args.registry))
    if args.checkpoint_id:
        manifests = [
            m for m in manifests if m.checkpoint_id == args.checkpoint_id
        ]

    findings: list[dict] = []
    for manifest in manifests:
        integrity = verify_checkpoint(manifest)
        if integrity.missing_files:
            verdict = "CHECKPOINT_INVALID"
        elif integrity.hash_mismatches:
            verdict = "CHECKPOINT_TAMPERED"
        else:
            verdict = "PASS"
        promotable = (
            verdict == "PASS"
            and manifest.base_model_revision not in FLOATING_REVISIONS
            and manifest.train_status == "TRAINING_COMPLETED"
        )
        not_promotable_reasons = []
        if manifest.base_model_revision in FLOATING_REVISIONS:
            not_promotable_reasons.append("floating base revision")
        if manifest.train_status != "TRAINING_COMPLETED":
            not_promotable_reasons.append(
                f"train_status={manifest.train_status}"
            )
        findings.append(
            {
                "checkpoint_id": manifest.checkpoint_id,
                "verdict": verdict,
                "promotable": promotable,
                "not_promotable_reasons": not_promotable_reasons,
                "missing_files": integrity.missing_files,
                "hash_mismatches": integrity.hash_mismatches,
            }
        )

    if not manifests:
        status = "NOT_RUN"
    elif all(f["verdict"] == "PASS" for f in findings):
        status = "PASS"
    else:
        status = "FAIL"

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO_ROOT / "artifacts" / "amali_ft_v0" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoint_integrity_report.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "generated_at": _now(),
                "repo_ref": _repo_ref(),
                "command": "python scripts/verify_checkpoint.py",
                "status": status,
                "metrics": {
                    "checkpoints": len(manifests),
                    "clean": sum(1 for f in findings if f["verdict"] == "PASS"),
                    "invalid": sum(
                        1
                        for f in findings
                        if f["verdict"] == "CHECKPOINT_INVALID"
                    ),
                    "tampered": sum(
                        1
                        for f in findings
                        if f["verdict"] == "CHECKPOINT_TAMPERED"
                    ),
                },
                "decisions": findings,
                "risks": (
                    ["no checkpoints registered; train the adapter first"]
                    if not manifests
                    else [
                        f"{f['checkpoint_id']}: {f['verdict']}"
                        for f in findings
                        if f["verdict"] != "PASS"
                    ]
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Checkpoint verification: {status} ({len(manifests)} checkpoint(s))")
    for finding in findings:
        promotable = "promotable" if finding["promotable"] else "NOT_PROMOTABLE"
        print(f"  {finding['checkpoint_id']}: {finding['verdict']} [{promotable}]")
    print(f"Report: {out_dir / 'checkpoint_integrity_report.json'}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
