"""Decide Base Model Readiness from the newest typed artifacts.

    python scripts/check_base_model_readiness.py

Gathers the latest evidence (Gate A preflight, live torch-free import
check, revision pin, live snapshot availability, gateway smoke, raw and
wrapped evals, comparison, security validation) and emits one typed
verdict. The first missing prerequisite names the verdict. Nothing is
inferred optimistically: a missing artifact is NOT_RUN, never PASS.

BASE_MODEL_READY does not mean AMALI-FT-v0 exists; the FT verdict is
reported separately and requires a real PROMOTED promotion report.

Writes base_model_readiness_report.{json,md} under
``artifacts/base_model_readiness/<timestamp>/``. Exit 0 only on
BASE_MODEL_READY.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from amali.model_gateway.availability import check_model_snapshot  # noqa: E402
from amali.model_gateway.base_readiness import (  # noqa: E402
    BaseModelReadinessInputs,
    decide_base_model_readiness,
)
from amali.model_gateway.revision_pin import load_latest_pin  # noqa: E402

READINESS_BASE = REPO_ROOT / "artifacts" / "base_model_readiness"
GATE_BASE = REPO_ROOT / "artifacts" / "base_model_gate"
PREFLIGHT_BASE = REPO_ROOT / "artifacts" / "base_model_gate_preflight"
FT_BASE = REPO_ROOT / "artifacts" / "amali_ft_v0"


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


def _latest_status(name: str, *bases: Path, key: str = "status") -> str:
    """Status field of the newest artifact named ``name``, else NOT_RUN."""
    candidates: list[Path] = []
    for base in bases:
        if base.is_dir():
            candidates.extend(base.glob(f"*/{name}"))
    if not candidates:
        return "NOT_RUN"
    newest = sorted(candidates, key=lambda p: str(p.parent))[-1]
    try:
        return str(
            json.loads(newest.read_text(encoding="utf-8")).get(key, "NOT_RUN")
        )
    except (OSError, json.JSONDecodeError):
        return "NOT_RUN"


def _core_import_torch_free() -> bool:
    """Live proof, not a stale artifact: import the core in a fresh process."""
    code = (
        "import sys; import amali; import amali.model_gateway; "
        "import amali.router; "
        "raise SystemExit(0 if 'torch' not in sys.modules else 1)"
    )
    try:
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=REPO_ROOT,
        )
        return out.returncode == 0
    except Exception:  # noqa: BLE001 - a broken interpreter is a failure
        return False


def main() -> int:
    pin = load_latest_pin(READINESS_BASE)
    if pin is not None:
        availability = check_model_snapshot(pin.model_id, pin.revision)
        availability_status = availability.status
    else:
        availability_status = "NOT_RUN"

    inputs = BaseModelReadinessInputs(
        gate_a_status=_latest_status("preflight_report.json", PREFLIGHT_BASE),
        core_import_torch_free=_core_import_torch_free(),
        revision_pin_status=pin.status if pin else "NOT_RUN",
        model_availability_status=availability_status,
        gateway_smoke_status=_latest_status(
            "model_gateway_smoke_report.json", READINESS_BASE
        ),
        raw_eval_status=_latest_status("eval_report_raw_base.json", GATE_BASE),
        wrapped_eval_status=_latest_status(
            "eval_report_wrapped_raw_base.json", GATE_BASE
        ),
        comparison_status=_latest_status("comparison_three_way.json", GATE_BASE),
        security_validation_status=_latest_status(
            "security_validation_report.json", FT_BASE, GATE_BASE
        ),
        promotion_decision=_latest_status(
            "promotion_report.json", GATE_BASE, FT_BASE
        ),
    )
    report = decide_base_model_readiness(inputs)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = READINESS_BASE / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    command = "python scripts/check_base_model_readiness.py"

    (out_dir / "base_model_readiness_report.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "generated_at": _now(),
                "repo_ref": _repo_ref(),
                "command": command,
                "status": report.status,
                "metrics": {
                    "base_model_ready": report.base_model_ready,
                    "amali_ft_status": report.amali_ft_status,
                    **report.checks,
                },
                "decisions": [
                    report.model_dump(mode="json"),
                    inputs.model_dump(mode="json"),
                ],
                "risks": report.reasons,
                "owner_actions": report.owner_actions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "base_model_readiness_report.md").write_text(
        f"# Base Model Readiness Report\n\nVerdict: **{report.status}**\n\n"
        f"AMALI-FT verdict: **{report.amali_ft_status}** "
        "(separate from base model readiness by design)\n\n"
        "| check | status |\n|-------|--------|\n"
        + "\n".join(f"| {k} | {v} |" for k, v in report.checks.items())
        + "\n\n"
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
            else ""
        )
        + "\nNo claim of AGI, ASI, or frontier-model superiority is made "
        "or implied by this report.\n",
        encoding="utf-8",
    )

    print(f"Base model readiness: {report.status}")
    print(f"AMALI-FT: {report.amali_ft_status}")
    for name, value in report.checks.items():
        print(f"  {name}: {value}")
    for action in report.owner_actions:
        print(f"  owner action: {action}", file=sys.stderr)
    print(f"Report: {out_dir / 'base_model_readiness_report.json'}")
    return 0 if report.base_model_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
