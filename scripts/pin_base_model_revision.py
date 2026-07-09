"""Pin the exact base model snapshot revision (owner command).

    python scripts/pin_base_model_revision.py --model Qwen/Qwen2.5-1.5B-Instruct --revision <HF_SNAPSHOT_COMMIT_SHA>

Only an allowlisted model plus a full 40-hex snapshot commit SHA yields
REVISION_PINNED. Floating revisions (main/latest/HEAD/tags) are refused —
they can never anchor a promotable artifact. Missing --revision reports
OWNER_MODEL_REVISION_REQUIRED with the exact next command; nothing is
invented on the owner's behalf.

Writes base_model_revision_pin_report.{json,md} under
``artifacts/base_model_readiness/<timestamp>/``. Exit 0 only on
REVISION_PINNED.
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

from amali.model_gateway.revision_pin import (  # noqa: E402
    STATUS_REVISION_PINNED,
    decide_revision_pin,
)

READINESS_BASE = REPO_ROOT / "artifacts" / "base_model_readiness"


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
    parser = argparse.ArgumentParser(
        description="Pin the exact base model snapshot revision"
    )
    parser.add_argument("--model", required=True, help="allowlisted model id")
    parser.add_argument(
        "--revision",
        default=None,
        help="exact Hugging Face snapshot commit SHA (40 hex chars)",
    )
    args = parser.parse_args()

    report = decide_revision_pin(args.model, args.revision)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = READINESS_BASE / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    command = f"python scripts/pin_base_model_revision.py --model {args.model}" + (
        f" --revision {args.revision}" if args.revision else ""
    )

    (out_dir / "base_model_revision_pin_report.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "generated_at": _now(),
                "repo_ref": _repo_ref(),
                "command": command,
                "status": report.status,
                "metrics": {
                    "model_id": report.model_id,
                    "revision": report.revision,
                    "promotion_eligible": report.promotion_eligible,
                },
                "decisions": [report.model_dump(mode="json")],
                "risks": report.reasons,
                "owner_actions": report.owner_actions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "base_model_revision_pin_report.md").write_text(
        f"# Base Model Revision Pin Report\n\nStatus: **{report.status}**\n\n"
        f"- model: `{report.model_id}`\n"
        f"- revision: `{report.revision or '(none provided)'}`\n"
        f"- role: {report.role or 'n/a'}\n"
        f"- license: {report.license_label or 'n/a'}\n"
        f"- promotion eligible: {report.promotion_eligible}\n\n"
        + (
            "Reasons:\n" + "\n".join(f"- {r}" for r in report.reasons) + "\n\n"
            if report.reasons
            else ""
        )
        + (
            "Owner actions:\n"
            + "\n".join(f"- {a}" for a in report.owner_actions)
            + "\n"
            if report.owner_actions
            else "No owner action required; the pin is recorded.\n"
        ),
        encoding="utf-8",
    )

    print(f"Revision pin: {report.status}")
    if report.revision:
        print(f"  {report.model_id} @ {report.revision}")
    for reason in report.reasons:
        print(f"  reason: {reason}", file=sys.stderr)
    for action in report.owner_actions:
        print(f"  owner action: {action}", file=sys.stderr)
    print(f"Report: {out_dir / 'base_model_revision_pin_report.json'}")
    return 0 if report.status == STATUS_REVISION_PINNED else 1


if __name__ == "__main__":
    raise SystemExit(main())
