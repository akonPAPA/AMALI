"""One-time, owner-initiated download of an allowlisted local base model.

This script is the *only* sanctioned network moment for model weights.
Everything else in AMALI runs offline: backends load with
``local_files_only=True`` and fail loudly rather than reach the network.

Usage (explicit owner command; nothing here runs automatically):

    python scripts/download_model.py --model Qwen/Qwen2.5-1.5B-Instruct --revision <PINNED_REVISION>

Rules enforced here:
* the model must be on the owner allowlist
  (``amali.model_gateway.base_model_allowlist``);
* ``--model`` is required — no implicit default download;
* ``--revision`` pins the exact snapshot; downloading a floating ``main``
  is allowed for experimentation but the manifest records it as
  NOT promotable;
* a ``model_download_manifest.json`` artifact records what was fetched.

Weights land in the standard Hugging Face cache (~/.cache/huggingface or
HF_HOME), never in the git repository.
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
    check_model_allowed,
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
    parser = argparse.ArgumentParser(
        description="Owner-initiated download of an allowlisted base model"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="allowlisted model id, e.g. Qwen/Qwen2.5-1.5B-Instruct",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="pinned revision (commit sha or tag); required for promotion",
    )
    args = parser.parse_args()

    decision = check_model_allowed(args.model)
    if not decision.allowed:
        print(
            f"REFUSED: {args.model} is not on the owner allowlist "
            f"({decision.reasons}). Edit "
            "src/amali/model_gateway/base_model_allowlist.py deliberately "
            "if this model should be approved.",
            file=sys.stderr,
        )
        return 1

    revision = args.revision
    promotable = revision is not None and revision not in FLOATING_REVISIONS
    if not promotable:
        print(
            "WARNING: no pinned --revision given. The download will use the "
            "repo default branch and CANNOT anchor a promoted model. "
            "Re-run with --revision <commit_sha> for a reproducible pin.",
            file=sys.stderr,
        )

    print(f"Downloading weights for: {args.model} (revision: {revision or 'main'})")
    print("This is a one-time, owner-approved network action.")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "huggingface_hub is missing. Install the local runtime first:\n"
            "    pip install -e .[local_llm]",
            file=sys.stderr,
        )
        return 1

    path = snapshot_download(repo_id=args.model, revision=revision)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO_ROOT / "artifacts" / "base_model_gate" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1.0.0",
        "generated_at": _now(),
        "repo_ref": _repo_ref(),
        "command": (
            f"python scripts/download_model.py --model {args.model}"
            + (f" --revision {revision}" if revision else "")
        ),
        "status": "PASS",
        "metrics": {
            "model_id": args.model,
            "revision": revision or "main",
            "revision_pinned": promotable,
            "local_path": str(path),
        },
        "decisions": [decision.model_dump(mode="json")],
        "risks": (
            []
            if promotable
            else ["floating revision: this snapshot cannot anchor promotion"]
        ),
    }
    (out_dir / "model_download_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"Done. Cached at: {path}")
    print(f"Manifest: {out_dir / 'model_download_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
