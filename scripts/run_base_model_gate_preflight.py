"""Alias for preflight_base_model_gate.py (convenience wrapper).

    python scripts/run_base_model_gate_preflight.py

Runs Gate A preflight checks: pytest, demo, torch-free import, git tracking.
Exits nonzero on failure. Artifact: preflight_report.json.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    preflight_script = REPO_ROOT / "scripts" / "preflight_base_model_gate.py"
    result = subprocess.run(
        [sys.executable, str(preflight_script)],
        cwd=REPO_ROOT,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
