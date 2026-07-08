"""Freeze (or verify) the Base Model Gate eval suite.

    python scripts/freeze_eval_suite.py

First run: validates the seed fixture, computes item/suite/scorer hashes,
and writes the frozen manifest to
``fixtures/eval/base_model_gate_manifest.json`` (tracked in git — it *is*
the freeze). Later runs verify the seed against the manifest; any mutation
is a tamper failure with a nonzero exit code.

Artifacts (every run):
    artifacts/base_model_gate/<timestamp>/eval_suite_freeze_report.json
    artifacts/base_model_gate/<timestamp>/eval_suite_freeze_report.md
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from amali.eval.suite import (  # noqa: E402
    EvalSuiteManifest,
    freeze_suite,
    load_seed_items,
    verify_frozen_suite,
)

SEED_PATH = REPO_ROOT / "fixtures" / "eval" / "base_model_gate_seed.json"
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


def main() -> int:
    items = load_seed_items(SEED_PATH)
    action = "verify" if MANIFEST_PATH.exists() else "freeze"

    if action == "freeze":
        suite, report = freeze_suite(
            items, suite_id="base_model_gate_v0", frozen_at=_now()
        )
        if suite is not None:
            MANIFEST_PATH.write_text(
                json.dumps(
                    suite.manifest.model_dump(mode="json"), indent=2
                ),
                encoding="utf-8",
            )
    else:
        manifest = EvalSuiteManifest(
            **json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        )
        report = verify_frozen_suite(items, manifest)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO_ROOT / "artifacts" / "base_model_gate" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    envelope = {
        "schema_version": "1.0.0",
        "generated_at": _now(),
        "repo_ref": _repo_ref(),
        "command": "python scripts/freeze_eval_suite.py",
        "status": report.status,
        "metrics": {
            "action": action,
            "item_count": report.item_count,
            "section_counts": report.section_counts,
            "suite_hash": report.suite_hash,
            "scorer_hash": report.scorer_hash,
        },
        "decisions": [],
        "risks": report.problems,
    }
    (out_dir / "eval_suite_freeze_report.json").write_text(
        json.dumps(envelope, indent=2), encoding="utf-8"
    )
    (out_dir / "eval_suite_freeze_report.md").write_text(
        f"# Eval Suite Freeze Report\n\n"
        f"Action: **{action}**  |  Status: **{report.status}**\n\n"
        f"- items: {report.item_count}\n"
        f"- sections: {report.section_counts}\n"
        f"- suite_hash: `{report.suite_hash}`\n"
        f"- scorer_hash: `{report.scorer_hash}`\n"
        + (
            "\n## Problems\n" + "\n".join(f"- {p}" for p in report.problems)
            if report.problems
            else "\nNo problems. The suite is frozen; mutations now fail "
            "verification."
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Eval suite {action}: {report.status}")
    print(f"suite_hash:  {report.suite_hash}")
    print(f"scorer_hash: {report.scorer_hash}")
    for problem in report.problems:
        print(f"  problem: {problem}", file=sys.stderr)
    print(f"Report: {out_dir / 'eval_suite_freeze_report.json'}")
    return 0 if report.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
