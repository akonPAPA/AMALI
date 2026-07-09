"""Verify and report the local checkpoint registry.

    python scripts/register_checkpoint.py                # verify all entries
    python scripts/register_checkpoint.py --adapter-dir D:/AMALI/models/amali_ft_v0 \
        --checkpoint-id ckpt_manual --base-model Qwen/Qwen2.5-1.5B-Instruct \
        --revision <PINNED_REVISION>                     # register manually

The registry lives outside git (D:/AMALI/models/registry/). Registration
hashes every adapter file; verification recomputes hashes — a missing
file or mismatch marks the checkpoint invalid. Artifacts:
checkpoint_registry.json (copy) + checkpoint_integrity_report.json.
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

from amali.training.registry import (  # noqa: E402
    DEFAULT_REGISTRY_PATH,
    CheckpointManifest,
    hash_directory_files,
    load_registry,
    register_checkpoint,
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
    parser = argparse.ArgumentParser(description="AMALI checkpoint registry")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--checkpoint-id", default=None)
    parser.add_argument("--adapter-name", default="amali_ft_v0")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    registry_path = Path(args.registry)

    if args.adapter_dir:
        adapter_dir = Path(args.adapter_dir)
        if not adapter_dir.is_dir():
            print(f"adapter dir not found: {adapter_dir}", file=sys.stderr)
            return 1
        manifest = CheckpointManifest(
            checkpoint_id=args.checkpoint_id
            or f"ckpt_manual_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
            adapter_name=args.adapter_name,
            base_model_id=args.base_model,
            base_model_revision=args.revision,
            training_config_hash="",
            dataset_manifest_hash="",
            eval_suite_hash="",
            adapter_dir=str(adapter_dir),
            file_hashes=hash_directory_files(adapter_dir),
            created_at=_now(),
            train_status="TRAINING_COMPLETED",
            risks=["manually registered; training provenance hashes absent"],
        )
        register_checkpoint(manifest, registry_path=registry_path)
        print(f"registered: {manifest.checkpoint_id}")

    manifests = load_registry(registry_path)
    integrity = [verify_checkpoint(m) for m in manifests]
    all_pass = all(i.status == "PASS" for i in integrity)
    status = (
        "PASS" if manifests and all_pass
        else ("NOT_RUN" if not manifests else "FAIL")
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO_ROOT / "artifacts" / "base_model_gate" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    command = "python scripts/register_checkpoint.py"

    (out_dir / "checkpoint_registry.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "generated_at": _now(),
                "repo_ref": _repo_ref(),
                "command": command,
                "status": status,
                "metrics": {"checkpoints": len(manifests)},
                "decisions": [m.model_dump(mode="json") for m in manifests],
                "risks": (
                    []
                    if manifests
                    else ["no checkpoints registered; train the adapter first"]
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "checkpoint_integrity_report.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "generated_at": _now(),
                "repo_ref": _repo_ref(),
                "command": command,
                "status": status,
                "metrics": {
                    "verified": sum(1 for i in integrity if i.status == "PASS"),
                    "invalid": sum(1 for i in integrity if i.status != "PASS"),
                },
                "decisions": [i.model_dump(mode="json") for i in integrity],
                "risks": [
                    f"{i.checkpoint_id}: missing={i.missing_files} "
                    f"mismatched={i.hash_mismatches}"
                    for i in integrity
                    if i.status != "PASS"
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Registry: {status} ({len(manifests)} checkpoint(s))")
    for check in integrity:
        print(f"  {check.checkpoint_id}: {check.status}")
    print(f"Reports: {out_dir}")
    return 0 if status in ("PASS", "NOT_RUN") else 1


if __name__ == "__main__":
    raise SystemExit(main())
