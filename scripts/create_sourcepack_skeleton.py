"""Create the owner sourcepack skeleton outside the repository.

    python scripts/create_sourcepack_skeleton.py --root D:\\AMALI\\data\\raw

Writes a manifest *template* and a next-steps README into the raw data
root. It never creates example content, never marks anything ready, and
never touches Git — the owner curates real, approved, non-synthetic
material and renames the template deliberately.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

TEMPLATE_NAME = "sourcepack_manifest.template.json"
README_NAME = "README_SOURCEPACK_NEXT_STEPS.md"

MANIFEST_TEMPLATE = {
    "schema_version": "1.0.0",
    "pack_id": "owner_sourcepack_v0",
    "created_at": "<fill in ISO timestamp>",
    "owner_approval": False,
    "sources": [
        {
            "path": "<relative path to an approved file in this directory>",
            "kind": "<doc | notes | qa | code>",
            "license_ok": False,
            "contains_secrets": False,
            "synthetic": False,
            "description": "<what this source is and why it is approved>",
        }
    ],
    "notes": (
        "Rename to sourcepack_manifest.json only after every listed source "
        "is real, approved, license-clean, secret-free, and non-synthetic."
    ),
}

README = """# Owner Sourcepack — Next Steps

This directory holds RAW OWNER DATA. It must stay OUTSIDE the Git repo.

1. Add approved, non-synthetic source files here (docs, notes, Q&A, code
   you own or may use). Synthetic examples never count toward the
   200-example floor.
2. Copy `sourcepack_manifest.template.json` to `sourcepack_manifest.json`
   and fill in one entry per file. One bad file invalidates the pack.
3. Never include secrets, holdout items, or eval fixtures.
4. Validate and build:

       python scripts/validate_owner_sourcepack.py
       python scripts/build_training_dataset.py
       python scripts/check_contamination.py

Nothing in this skeleton is training data. Creating it does not make the
dataset ready; only real approved content does.
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create the owner sourcepack skeleton (no fake data)"
    )
    parser.add_argument(
        "--root",
        default="D:/AMALI/data/raw",
        help="raw owner data root, outside the repository",
    )
    args = parser.parse_args()

    root = Path(args.root)
    repo_root = Path(__file__).resolve().parents[1]
    try:
        root.resolve().relative_to(repo_root.resolve())
        print(
            f"REFUSED: {root} is inside the repository; raw owner data "
            "must live outside Git.",
            file=sys.stderr,
        )
        return 1
    except ValueError:
        pass  # outside the repo — correct

    root.mkdir(parents=True, exist_ok=True)

    template_path = root / TEMPLATE_NAME
    if template_path.exists():
        print(f"kept existing: {template_path}")
    else:
        template = dict(MANIFEST_TEMPLATE)
        template["created_at"] = datetime.now(timezone.utc).isoformat()
        template_path.write_text(
            json.dumps(template, indent=2), encoding="utf-8"
        )
        print(f"created: {template_path}")

    readme_path = root / README_NAME
    if readme_path.exists():
        print(f"kept existing: {readme_path}")
    else:
        readme_path.write_text(README, encoding="utf-8")
        print(f"created: {readme_path}")

    print(
        "Skeleton only: the dataset is NOT ready until the owner adds real "
        "approved sources and the validators pass."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
