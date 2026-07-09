"""Run the AMALI Base AI Gate demo (Gate A; Gate B optional).

Usage, from the repo root:

    python scripts/run_base_ai_gate_demo.py
    python scripts/run_base_ai_gate_demo.py --with-local-model   # optional

Gate A is deterministic and fully local: no network, no GPU, no torch, no
model weights. It emits the complete canonical artifact set under
``artifacts/base_ai_gate/<timestamp>/``.

``--with-local-model`` additionally probes for owner-downloaded local
weights and runs a tiny offline smoke if (and only if) they exist. A missing
model is reported as SKIPPED_LOCAL_MODEL_NOT_AVAILABLE — never a failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the repo root without an installed package.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from amali.demo.base_ai_gate import CANONICAL_ARTIFACTS, run_demo  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AMALI Base AI Gate demo (local, deterministic)"
    )
    parser.add_argument(
        "--with-local-model",
        action="store_true",
        help="also probe/smoke the optional local model (Gate B)",
    )
    parser.add_argument(
        "--model",
        default="raw_base",
        choices=["raw_base", "amali_ft_v0"],
        help="Gate B target: raw base weights or the promoted adapter",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="override the artifact output directory",
    )
    args = parser.parse_args()

    out = run_demo(
        repo_root=REPO_ROOT,
        output_dir=args.output_dir,
        with_local_model=args.with_local_model,
        model_mode=args.model,
    )

    emitted = sorted(p.name for p in out.iterdir())
    missing = [a for a in CANONICAL_ARTIFACTS if a not in emitted]

    print(f"Artifact directory: {out}")
    print(f"Artifacts emitted:  {len(emitted)}")
    for name in emitted:
        print(f"  - {name}")
    if missing:
        print(f"MISSING artifacts: {missing}", file=sys.stderr)
        return 1
    print("All canonical artifacts emitted. Gate A complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
