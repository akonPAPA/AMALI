"""Promotion gate runner for AMALI-FT-v0.

    python scripts/promote_model.py amali_ft_v0

Assembles evidence from the newest artifacts (preflight, freeze
verification, dataset manifest, contamination, training run, checkpoint
registry, eval reports, safety regression, model card), re-runs the cheap
live checks (torch-free import, suite verification, registry integrity),
and evaluates the 19 promotion gates. One failed gate refuses; missing
prerequisites are NOT_READY. Artifacts: promotion_report.{json,md},
model_registry_entry.json, dwac_model_pack_report.json.
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

from amali.eval.model_eval import ModelEvalReport  # noqa: E402
from amali.eval.suite import (  # noqa: E402
    EvalSuiteManifest,
    load_seed_items,
    verify_frozen_suite,
)
from amali.training.promotion import (  # noqa: E402
    ALLOWED_CLAIM,
    PromotionEvidence,
    evaluate_promotion,
)
from amali.training.registry import (  # noqa: E402
    DEFAULT_REGISTRY_PATH,
    load_registry,
    verify_checkpoint,
)

ARTIFACT_BASE = REPO_ROOT / "artifacts" / "base_model_gate"
PREFLIGHT_BASE = REPO_ROOT / "artifacts" / "base_model_gate_preflight"
SEED_PATH = REPO_ROOT / "fixtures" / "eval" / "base_model_gate_seed.json"
MANIFEST_PATH = REPO_ROOT / "fixtures" / "eval" / "base_model_gate_manifest.json"

TORCH_FREE_SNIPPET = (
    "import sys; import amali; import amali.model_gateway; "
    "import amali.router; print('torch_loaded=', 'torch' in sys.modules)"
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


def _latest(pattern: str, base: Path = ARTIFACT_BASE) -> dict | None:
    if not base.is_dir():
        return None
    candidates = sorted(base.glob(pattern))
    if not candidates:
        return None
    try:
        return json.loads(candidates[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _latest_eval(name: str) -> ModelEvalReport | None:
    data = _latest(f"*/{name}.json")
    if not data or not data.get("decisions"):
        return None
    return ModelEvalReport(**data["decisions"][0])


def gather_evidence() -> PromotionEvidence:
    preflight = _latest("*/preflight_report.json", PREFLIGHT_BASE)
    gate_a_pass = bool(preflight and preflight.get("status") == "PASS")

    torch_check = subprocess.run(
        [sys.executable, "-c", TORCH_FREE_SNIPPET],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO_ROOT,
    )
    torch_free = (
        torch_check.returncode == 0
        and "torch_loaded= False" in torch_check.stdout
    )

    eval_ok = False
    if SEED_PATH.is_file() and MANIFEST_PATH.is_file():
        manifest = EvalSuiteManifest(
            **json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        )
        eval_ok = (
            verify_frozen_suite(load_seed_items(SEED_PATH), manifest).status
            == "PASS"
        )

    dataset_manifest = _latest("*/dataset_manifest.json")
    dataset_valid = bool(
        dataset_manifest
        and dataset_manifest.get("dataset_hash")
        and dataset_manifest.get("example_count", 0) > 0
    )

    contamination = _latest("*/contamination_report.json")
    contamination_status = (
        contamination.get("status", "NOT_RUN") if contamination else "NOT_RUN"
    )

    training = _latest("*/training_run_report.json")
    training_status = training.get("status", "NOT_RUN") if training else "NOT_RUN"

    manifests = load_registry(DEFAULT_REGISTRY_PATH)
    if not manifests:
        checkpoint_status = "NOT_RUN"
    else:
        checkpoint_status = (
            "PASS"
            if all(verify_checkpoint(m).status == "PASS" for m in manifests)
            else "FAIL"
        )

    regression = _latest("*/safety_regression_report.json")
    regression_status = (
        regression.get("status", "NOT_RUN") if regression else "NOT_RUN"
    )

    model_card = bool(
        ARTIFACT_BASE.is_dir() and sorted(ARTIFACT_BASE.glob("*/model_card.md"))
    )

    audit = _latest(
        "*/audit_export.json", REPO_ROOT / "artifacts" / "base_ai_gate"
    )
    audit_ok = bool(
        audit and audit.get("metrics", {}).get("chain_verified") is True
    )

    return PromotionEvidence(
        gate_a_pass=gate_a_pass,
        torch_free_core_import=torch_free,
        eval_suite_frozen_ok=eval_ok,
        dataset_manifest_valid=dataset_valid,
        contamination_status=contamination_status,
        training_status=training_status,
        checkpoint_integrity_status=checkpoint_status,
        raw_base_report=_latest_eval("eval_report_raw_base"),
        ft_report=_latest_eval("eval_report_amali_ft"),
        wrapped_ft_report=_latest_eval("eval_report_amali_wrapped_ft"),
        safety_regression_status=regression_status,
        model_card_generated=model_card,
        audit_chain_verified=audit_ok,
        claim_text=ALLOWED_CLAIM,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="AMALI promotion gate")
    parser.add_argument("adapter", nargs="?", default="amali_ft_v0")
    args = parser.parse_args()

    evidence = gather_evidence()
    report = evaluate_promotion(evidence)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ARTIFACT_BASE / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    command = f"python scripts/promote_model.py {args.adapter}"

    def envelope(status: str, **extra) -> dict:
        return {
            "schema_version": "1.0.0",
            "generated_at": _now(),
            "repo_ref": _repo_ref(),
            "command": command,
            "status": status,
            **extra,
        }

    (out_dir / "promotion_report.json").write_text(
        json.dumps(
            envelope(
                report.decision,
                metrics={
                    "gates_total": len(report.gates),
                    "gates_failed": len(report.failed_gates),
                },
                decisions=[report.model_dump(mode="json")],
                risks=report.remaining_risks,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "promotion_report.md").write_text(
        f"# Promotion Report — {args.adapter}\n\n"
        f"Decision: **{report.decision}**\n\n"
        "| gate | result | detail |\n|------|--------|--------|\n"
        + "\n".join(
            f"| {g.gate} | {'PASS' if g.passed else 'FAIL'} | {g.detail} |"
            for g in report.gates
        )
        + "\n\n"
        + (
            f"Allowed claim:\n> {report.allowed_claim}\n"
            if report.decision == "PROMOTED"
            else "No claim is allowed; promotion did not happen.\n\n"
            "Failed gates:\n"
            + "\n".join(f"- {g}" for g in report.failed_gates)
        )
        + "\n",
        encoding="utf-8",
    )

    promoted = report.decision == "PROMOTED"
    (out_dir / "model_registry_entry.json").write_text(
        json.dumps(
            envelope(
                "PROMOTED" if promoted else "NOT_RUN",
                metrics={},
                decisions=(
                    [
                        {
                            "model_name": "AMALI-FT-v0",
                            "adapter": args.adapter,
                            "selectable_by_gateway": True,
                        }
                    ]
                    if promoted
                    else []
                ),
                risks=(
                    []
                    if promoted
                    else ["no registry entry: promotion did not pass"]
                ),
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "dwac_model_pack_report.json").write_text(
        json.dumps(
            envelope(
                "PROMOTED" if promoted else "NOT_RUN",
                metrics={},
                decisions=(
                    [
                        {
                            "pack_id": "amali_ft_v0_pack",
                            "capabilities": ["local_llm_answering"],
                            "cost_mb": 3200,
                        }
                    ]
                    if promoted
                    else []
                ),
                risks=(
                    []
                    if promoted
                    else ["no DWAC pack registered: promotion did not pass"]
                ),
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Promotion: {report.decision}")
    for gate in report.gates:
        marker = "PASS" if gate.passed else "FAIL"
        print(f"  {gate.gate}: {marker} {gate.detail}")
    print(f"Reports: {out_dir}")
    return 0 if promoted else 1


if __name__ == "__main__":
    raise SystemExit(main())
