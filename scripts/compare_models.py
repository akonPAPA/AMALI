"""Model comparison over the frozen Base Model Gate suite.

    python scripts/compare_models.py --mode base_readiness
    python scripts/compare_models.py --mode ft_promotion

``base_readiness`` (the default) compares raw_base vs
amali_wrapped_raw_base only and emits comparison_base_readiness.{json,md}
— no fine-tuned model is required, so base model readiness is decidable
before AMALI-FT-v0 exists. ``ft_promotion`` is the stricter three-way
comparison (raw_base, amali_wrapped_raw_base, amali_wrapped_ft_v0,
optional unwrapped FT) and emits comparison_three_way.{json,md}; missing
AMALI-FT stays an honest NOT_RUN. External systems stay NOT_PROVEN
without stored baselines — the generated claim can never contain a
frontier claim.
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

from amali.benchmarks.comparison import (  # noqa: E402
    base_readiness_comparison,
    three_way_comparison,
)
from amali.eval.model_eval import ModelEvalReport  # noqa: E402

ARTIFACT_BASE = REPO_ROOT / "artifacts" / "base_model_gate"
MANIFEST_PATH = REPO_ROOT / "fixtures" / "eval" / "base_model_gate_manifest.json"


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


def _latest_eval(name: str) -> ModelEvalReport | None:
    if not ARTIFACT_BASE.is_dir():
        return None
    candidates = sorted(ARTIFACT_BASE.glob(f"*/{name}.json"))
    if not candidates:
        return None
    data = json.loads(candidates[-1].read_text(encoding="utf-8"))
    if not data.get("decisions"):
        return None
    return ModelEvalReport(**data["decisions"][0])


def main() -> int:
    parser = argparse.ArgumentParser(description="AMALI model comparison")
    parser.add_argument(
        "--suite", default="frozen_base_model_gate", choices=["frozen_base_model_gate"]
    )
    parser.add_argument(
        "--mode",
        default="base_readiness",
        choices=["base_readiness", "ft_promotion"],
        help=(
            "base_readiness: raw_base vs amali_wrapped_raw_base only "
            "(no fine-tuned model required); ft_promotion: strict "
            "three-way comparison that refuses a missing amali_wrapped_ft_v0"
        ),
    )
    args = parser.parse_args()

    suite_hash = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))[
        "suite_hash"
    ]
    if args.mode == "base_readiness":
        comparison = base_readiness_comparison(
            raw_base=_latest_eval("eval_report_raw_base"),
            wrapped_raw=_latest_eval("eval_report_wrapped_raw_base"),
            suite_hash=suite_hash,
        )
        report_name = "comparison_base_readiness"
        title = "Base Readiness Comparison"
    else:
        comparison = three_way_comparison(
            raw_base=_latest_eval("eval_report_raw_base"),
            wrapped_raw=_latest_eval("eval_report_wrapped_raw_base"),
            wrapped_ft=_latest_eval("eval_report_amali_wrapped_ft"),
            ft_unwrapped=_latest_eval("eval_report_amali_ft"),
            suite_hash=suite_hash,
        )
        report_name = "comparison_three_way"
        title = "Three-Way Comparison"

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ARTIFACT_BASE / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    command = f"python scripts/compare_models.py --mode {args.mode}"

    (out_dir / f"{report_name}.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "generated_at": _now(),
                "repo_ref": _repo_ref(),
                "command": command,
                "status": comparison.status,
                "metrics": {
                    mode: metrics
                    for mode, metrics in comparison.metrics_by_mode.items()
                },
                "decisions": [comparison.model_dump(mode="json")],
                "risks": comparison.notes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if comparison.metrics_by_mode:
        modes = list(comparison.metrics_by_mode)
        metric_names = list(next(iter(comparison.metrics_by_mode.values())))
        table = (
            "| metric | " + " | ".join(modes) + " |\n"
            "|--------|" + "|".join(["-------"] * len(modes)) + "|\n"
            + "\n".join(
                f"| {m} | "
                + " | ".join(
                    str(comparison.metrics_by_mode[mode].get(m, "-"))
                    for mode in modes
                )
                + " |"
                for m in metric_names
            )
        )
    else:
        table = "(no comparable eval reports exist yet)"
    (out_dir / f"{report_name}.md").write_text(
        f"# {title} — frozen suite `{suite_hash[:16]}...`\n\n"
        f"Status: **{comparison.status}**\n\n{table}\n\n"
        f"## Claim\n{comparison.claim or 'No claim: comparison did not run.'}\n\n"
        f"## External systems\nStatus: **{comparison.external_status}** — "
        "no stored baselines; no claim about GPT/Claude/DeepSeek/GLM/Qwen "
        "is made or implied.\n",
        encoding="utf-8",
    )

    print(f"Comparison ({args.mode}): {comparison.status}")
    for note in comparison.notes:
        print(f"  note: {note}")
    print(f"Report: {out_dir / f'{report_name}.json'}")
    return 0 if comparison.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
