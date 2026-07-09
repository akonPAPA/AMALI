"""Run the contamination gate: training dataset vs frozen eval suite.

    python scripts/check_contamination.py

Reads the built dataset from ``D:/AMALI/data/processed/amali_bmg_train.jsonl``
and the frozen eval seed, and writes contamination_report.{json,md} under
``artifacts/base_model_gate/<timestamp>/``. Exit 0 only on PASS (zero
leakage); contaminated ids are listed, never silently removed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from amali.eval.suite import load_seed_items  # noqa: E402
from amali.training.contamination import check_contamination  # noqa: E402
from amali.training.dataset import TrainingExample  # noqa: E402

DATASET_PATH = Path("D:/AMALI/data/processed/amali_bmg_train.jsonl")
SEED_PATH = REPO_ROOT / "fixtures" / "eval" / "base_model_gate_seed.json"


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
    eval_items = load_seed_items(SEED_PATH)

    if not DATASET_PATH.exists():
        training = []
    else:
        training = [
            TrainingExample(**json.loads(line))
            for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    report = check_contamination(training, eval_items)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO_ROOT / "artifacts" / "base_model_gate" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "contamination_report.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "generated_at": _now(),
                "repo_ref": _repo_ref(),
                "command": "python scripts/check_contamination.py",
                "status": report.status,
                "metrics": {
                    "training_examples": report.training_examples,
                    "eval_items": report.eval_items,
                    "findings": len(report.findings),
                    "contaminated_ids": report.contaminated_ids,
                },
                "decisions": [f.model_dump(mode="json") for f in report.findings],
                "risks": (
                    []
                    if report.status == "PASS"
                    else ["training must not start until contamination is resolved"]
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "contamination_report.md").write_text(
        f"# Contamination Report\n\nStatus: **{report.status}**\n\n"
        f"- training examples: {report.training_examples}\n"
        f"- eval items: {report.eval_items}\n"
        f"- findings: {len(report.findings)}\n"
        + (
            "\n## Contaminated training example ids\n"
            + "\n".join(f"- {i}" for i in report.contaminated_ids)
            + "\n\n## Findings\n"
            + "\n".join(
                f"- {f.kind}: {f.training_example_id} vs {f.eval_item_id or f.detail}"
                f" (sim={f.similarity})"
                for f in report.findings
            )
            if report.findings
            else "\nZero exact and zero near-duplicate leakage. Training may proceed."
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Contamination gate: {report.status}")
    print(
        f"training={report.training_examples} eval={report.eval_items} "
        f"findings={len(report.findings)}"
    )
    for cid in report.contaminated_ids:
        print(f"  contaminated: {cid}", file=sys.stderr)
    print(f"Report: {out_dir / 'contamination_report.json'}")
    if report.status == "NOT_RUN":
        print(
            "dataset or eval suite missing; build the dataset first",
            file=sys.stderr,
        )
        return 1
    return 0 if report.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
