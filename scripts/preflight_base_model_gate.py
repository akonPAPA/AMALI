"""Base Model Gate preflight: verify Gate A is stable before model work.

Runs the four required checks and writes an honest, typed report:

    python scripts/preflight_base_model_gate.py

Checks:
1. full pytest suite passes;
2. Gate A demo emits every canonical artifact;
3. core import stays torch-free;
4. no generated egg-info / artifacts tracked by git.

Exit code is nonzero when any check fails. The report is written to
``artifacts/base_model_gate_preflight/<timestamp>/preflight_report.json``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

TORCH_FREE_SNIPPET = (
    "import sys; import amali; import amali.model_gateway; "
    "import amali.router; print('torch_loaded=', 'torch' in sys.modules)"
)


def _run(cmd: list[str], timeout: int = 900) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _repo_ref() -> str:
    out = _run(["git", "rev-parse", "HEAD"], timeout=30)
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def main() -> int:
    risks: list[str] = []

    tests = _run([sys.executable, "-m", "pytest", "-q"])
    tests_pass = tests.returncode == 0

    demo = _run([sys.executable, "scripts/run_base_ai_gate_demo.py"])
    demo_pass = demo.returncode == 0

    torch_check = _run([sys.executable, "-c", TORCH_FREE_SNIPPET], timeout=120)
    torch_free = (
        torch_check.returncode == 0
        and "torch_loaded= False" in torch_check.stdout
    )

    tracked = _run(["git", "ls-files"], timeout=60).stdout.splitlines()
    bad_tracked = [
        p
        for p in tracked
        if ".egg-info" in p
        or p.startswith(("artifacts/", "data/", "models/", "cache/", "logs/"))
        or p.endswith(".env")
    ]
    clean_tree = not bad_tracked

    if not tests_pass:
        risks.append("pytest failed; fix Base AI Gate before model work")
    if not demo_pass:
        risks.append("Gate A demo failed; canonical artifacts incomplete")
    if not torch_free:
        risks.append("core import pulled in torch; Gate A must stay torch-free")
    if bad_tracked:
        risks.append(f"generated/local files tracked by git: {bad_tracked}")

    status = "PASS" if (tests_pass and demo_pass and torch_free and clean_tree) else "FAIL"

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO_ROOT / "artifacts" / "base_model_gate_preflight" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_ref": _repo_ref(),
        "command": "python scripts/preflight_base_model_gate.py",
        "status": status,
        "base_ai_gate_tests": "PASS" if tests_pass else "FAIL",
        "base_ai_gate_demo": "PASS" if demo_pass else "FAIL",
        "torch_free_core_import": "PASS" if torch_free else "FAIL",
        "git_tree_clean_of_generated_files": "PASS" if clean_tree else "FAIL",
        "metrics": {
            "pytest_returncode": tests.returncode,
            "demo_returncode": demo.returncode,
        },
        "decisions": [],
        "risks": risks,
    }
    (out_dir / "preflight_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Preflight status: {status}")
    print(f"Report: {out_dir / 'preflight_report.json'}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
