"""Build the AMALI Base Model Gate training dataset (local sources only).

    python scripts/build_training_dataset.py

Every candidate example passes DataWall(flow=TRAINING). Blocked content is
never written anywhere. The dataset lands outside git at
``D:/AMALI/data/processed/amali_bmg_train.jsonl``; reports and the manifest
go to ``artifacts/base_model_gate/<timestamp>/``.

Exit code 0 only when the dataset is READY for training (>=200 approved
non-synthetic examples). DATASET_NOT_READY exits 1 with an honest report —
never padding, never fabrication.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from amali.audit.ledger import AuditLedger  # noqa: E402
from amali.training.dataset import (  # noqa: E402
    MIN_NON_SYNTHETIC,
    build_training_dataset,
)
from amali.training.sourcepack import (  # noqa: E402
    SOURCEPACK_ROOT,
    validate_sourcepack,
)

OUTPUT_PATH = Path("D:/AMALI/data/processed/amali_bmg_train.jsonl")


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
    sourcepack_report = validate_sourcepack(SOURCEPACK_ROOT)
    ledger = AuditLedger()
    report, manifest, _examples = build_training_dataset(
        repo_root=REPO_ROOT,
        output_path=OUTPUT_PATH,
        ledger=ledger,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO_ROOT / "artifacts" / "base_model_gate" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    command = "python scripts/build_training_dataset.py"

    def envelope(status: str, **extra) -> dict:
        return {
            "schema_version": "1.0.0",
            "generated_at": _now(),
            "repo_ref": _repo_ref(),
            "command": command,
            "status": status,
            **extra,
        }

    (out_dir / "dataset_build_report.json").write_text(
        json.dumps(
            envelope(
                report.status,
                metrics={
                    "raw_sources": report.raw_sources,
                    "candidate_examples": report.candidate_examples,
                    "blocked": report.blocked,
                    "redacted": report.redacted,
                    "deduped": report.deduped,
                    "final_examples": report.final_examples,
                    "non_synthetic_count": report.non_synthetic_count,
                    "synthetic_count": report.synthetic_count,
                    "min_required_non_synthetic": MIN_NON_SYNTHETIC,
                    "sourcepack_validation_status": sourcepack_report.status,
                    "ready_for_training": report.status == "READY",
                },
                decisions=[],
                risks=report.risks,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "dataset_build_report.md").write_text(
        f"# Dataset Build Report\n\nStatus: **{report.status}**\n\n"
        f"| metric | value |\n|--------|-------|\n"
        f"| raw sources | {report.raw_sources} |\n"
        f"| candidates | {report.candidate_examples} |\n"
        f"| blocked | {report.blocked} |\n"
        f"| redacted | {report.redacted} |\n"
        f"| deduped away | {report.deduped} |\n"
        f"| final examples | {report.final_examples} |\n"
        f"| non-synthetic | {report.non_synthetic_count} |\n"
        f"| synthetic (labeled) | {report.synthetic_count} |\n"
        f"| training floor | {MIN_NON_SYNTHETIC} |\n"
        f"| sourcepack validation | {sourcepack_report.status} |\n"
        f"| ready for training | {report.status == 'READY'} |\n\n"
        "Risks:\n"
        + "\n".join(f"- {r}" for r in report.risks or ["none recorded"])
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "data_wall_training_report.json").write_text(
        json.dumps(
            envelope(
                "PASS",
                metrics={
                    "wall_checks": len(report.wall_decisions),
                    "blocks": sum(
                        1
                        for d in report.wall_decisions
                        if d["outcome"] == "BLOCK"
                    ),
                    "redactions": sum(
                        1
                        for d in report.wall_decisions
                        if d["outcome"] == "REDACT"
                    ),
                    "audit_chain_verified": ledger.verify_chain(),
                },
                decisions=report.wall_decisions,
                risks=[],
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    if manifest is not None:
        stamped = manifest.model_copy(update={"created_at": _now()})
        (out_dir / "dataset_manifest.json").write_text(
            json.dumps(stamped.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )

    print(f"Dataset build: {report.status}")
    print(
        f"sources={report.raw_sources} candidates={report.candidate_examples} "
        f"blocked={report.blocked} redacted={report.redacted} "
        f"deduped={report.deduped} final={report.final_examples} "
        f"(non-synthetic={report.non_synthetic_count}, "
        f"synthetic={report.synthetic_count})"
    )
    if manifest is not None:
        print(f"dataset: {OUTPUT_PATH}")
        print(f"dataset_hash: {manifest.dataset_hash}")
    for risk in report.risks:
        print(f"  risk: {risk}", file=sys.stderr)
    print(f"Reports: {out_dir}")
    return 0 if report.status == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
