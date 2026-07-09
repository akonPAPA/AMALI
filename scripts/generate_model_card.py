"""Generate the truthful AMALI-FT-v0 model card from existing artifacts.

    python scripts/generate_model_card.py

Reads the newest artifacts (dataset manifest, freeze manifest,
contamination report, training run report, eval reports, safety
regression, promotion report) and renders model_card.md. Absent evidence
renders as an explicit absence — never as an invented value.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from amali.training.model_card import ModelCardInputs, generate_model_card  # noqa: E402

ARTIFACT_BASE = REPO_ROOT / "artifacts" / "base_model_gate"
MANIFEST_PATH = REPO_ROOT / "fixtures" / "eval" / "base_model_gate_manifest.json"


def _latest(pattern: str) -> dict | None:
    if not ARTIFACT_BASE.is_dir():
        return None
    candidates = sorted(ARTIFACT_BASE.glob(pattern))
    if not candidates:
        return None
    try:
        return json.loads(candidates[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _metrics(report_name: str) -> dict:
    data = _latest(f"*/{report_name}.json")
    if not data or data.get("status") != "PASS":
        return {}
    return data.get("metrics", {})


def main() -> int:
    suite_hash = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))[
        "suite_hash"
    ]
    dataset_manifest = _latest("*/dataset_manifest.json") or {}
    contamination = _latest("*/contamination_report.json")
    contamination_hash = (
        hashlib.sha256(
            json.dumps(contamination, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if contamination
        else ""
    )
    training = _latest("*/training_run_report.json") or {}
    training_decisions = (training.get("decisions") or [{}])[0]
    promotion = _latest("*/promotion_report.json") or {}
    regression = _latest("*/safety_regression_report.json") or {}
    config = _latest("*/training_config.json") or {}
    config_decisions = (config.get("decisions") or [{}])[0]

    inputs = ModelCardInputs(
        base_model_id=config_decisions.get(
            "base_model_id", "Qwen/Qwen2.5-1.5B-Instruct"
        ),
        base_model_revision=(
            config_decisions.get("base_model_revision", "")
            if config_decisions.get("base_model_revision") != "main"
            else ""
        ),
        dataset_manifest_hash=dataset_manifest.get("dataset_hash", ""),
        eval_suite_hash=suite_hash,
        contamination_report_hash=contamination_hash,
        training_status=training.get("status", "NOT_RUN"),
        training_config={
            k: config_decisions[k]
            for k in (
                "method",
                "quantization",
                "lora_rank",
                "lora_alpha",
                "max_seq_len",
                "batch_size",
                "learning_rate",
                "epochs",
                "seed",
            )
            if k in config_decisions
        },
        training_run_facts={
            k: v
            for k, v in {
                "steps": training_decisions.get("step_count"),
                "wall_clock_seconds": training_decisions.get(
                    "wall_clock_seconds"
                ),
                "peak_vram_gb": training_decisions.get("peak_vram_gb"),
                "device": training_decisions.get("device"),
            }.items()
            if v
        },
        raw_base_metrics=_metrics("eval_report_raw_base"),
        ft_metrics=_metrics("eval_report_amali_ft"),
        wrapped_ft_metrics=_metrics("eval_report_amali_wrapped_ft"),
        safety_regression_status=(
            regression.get("status", "NOT_RUN") if regression else "NOT_RUN"
        ),
        promotion_decision=promotion.get("status", "NOT_RUN"),
    )

    card = generate_model_card(inputs)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ARTIFACT_BASE / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_card.md").write_text(card, encoding="utf-8")
    print(f"Model card: {out_dir / 'model_card.md'}")
    print(f"  training status: {inputs.training_status}")
    print(f"  promotion: {inputs.promotion_decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
