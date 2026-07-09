"""Validate the owner sourcepack at D:/AMALI/data/raw (AMALI-FT-v0 Stage 1).

    python scripts/validate_owner_sourcepack.py [--root PATH]

Exit 0 only when the sourcepack is VALID for training. Missing root,
missing manifest, secret content, holdout markers, unlisted or missing
files all exit nonzero with a typed report. Reports land under
``artifacts/amali_ft_v0/<timestamp>/``.
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

from amali.training.sourcepack import (  # noqa: E402
    SOURCEPACK_ROOT,
    validate_sourcepack,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(SOURCEPACK_ROOT),
        help="sourcepack root (default: D:/AMALI/data/raw)",
    )
    args = parser.parse_args()

    report = validate_sourcepack(args.root)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO_ROOT / "artifacts" / "amali_ft_v0" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "sourcepack_validation_report.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "generated_at": _now(),
                "repo_ref": _repo_ref(),
                "command": "python scripts/validate_owner_sourcepack.py",
                "status": report.status,
                "metrics": {
                    "files_listed": report.files_listed,
                    "files_on_disk": report.files_on_disk,
                    "approved_files": report.approved_files,
                    "issues": len(report.issues),
                    "duplicate_hashes": len(report.duplicate_hashes),
                },
                "report": report.model_dump(mode="json"),
                "risks": report.risks,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "sourcepack_validation_report.md").write_text(
        f"# Owner Sourcepack Validation\n\nStatus: **{report.status}**\n\n"
        f"- root: `{report.sourcepack_root}`\n"
        f"- sourcepack_id: {report.sourcepack_id or 'n/a'}\n"
        f"- files listed / on disk / approved: "
        f"{report.files_listed} / {report.files_on_disk} / "
        f"{report.approved_files}\n\n"
        + (
            "## Issues\n"
            + "\n".join(
                f"- **{i.code}** ({i.severity}) `{i.path}` — {i.detail}"
                for i in report.issues
            )
            if report.issues
            else "No issues."
        )
        + "\n\nRisks:\n"
        + "\n".join(f"- {r}" for r in report.risks or ["none recorded"])
        + "\n",
        encoding="utf-8",
    )

    print(f"Sourcepack validation: {report.status}")
    print(
        f"listed={report.files_listed} on_disk={report.files_on_disk} "
        f"approved={report.approved_files} issues={len(report.issues)}"
    )
    for issue in report.issues:
        print(
            f"  {issue.severity}: {issue.code} {issue.path} {issue.detail}",
            file=sys.stderr,
        )
    print(f"Report: {out_dir / 'sourcepack_validation_report.json'}")
    return 0 if report.status == "VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
