"""Evaluate a model mode over the frozen Base Model Gate eval suite.

    python scripts/evaluate_model.py --model deterministic_gate_a
    python scripts/evaluate_model.py --model raw_base
    python scripts/evaluate_model.py --model amali_wrapped_raw_base
    python scripts/evaluate_model.py --model amali_ft_v0
    python scripts/evaluate_model.py --model amali_wrapped_ft_v0

Same frozen items, same deterministic scorer, greedy decoding for local
models. Missing weights/deps/adapters produce honest MODEL_NOT_AVAILABLE /
DEPS_MISSING artifacts and a nonzero exit — never invented scores.

When evaluating an FT mode and a raw-base report exists in the artifact
tree, a safety_regression_report.json is emitted alongside.
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

from amali.eval.model_eval import (  # noqa: E402
    MODEL_MODES,
    ControlPlaneWrapper,
    DeterministicGateABackend,
    ModelEvalReport,
    evaluate_model,
    safety_regression,
)
from amali.eval.suite import EvalSuiteManifest, load_seed_items  # noqa: E402

SEED_PATH = REPO_ROOT / "fixtures" / "eval" / "base_model_gate_seed.json"
MANIFEST_PATH = REPO_ROOT / "fixtures" / "eval" / "base_model_gate_manifest.json"
ARTIFACT_BASE = REPO_ROOT / "artifacts" / "base_model_gate"

REPORT_NAMES = {
    "deterministic_gate_a": "eval_report_gate_a",
    "raw_base": "eval_report_raw_base",
    "amali_wrapped_raw_base": "eval_report_wrapped_raw_base",
    "amali_ft_v0": "eval_report_amali_ft",
    "amali_wrapped_ft_v0": "eval_report_amali_wrapped_ft",
}


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


def _build_backend(mode: str) -> tuple[str, object | None, list[str]]:
    """Return (status, backend, reasons) for the requested mode."""
    if mode == "deterministic_gate_a":
        return "PASS", DeterministicGateABackend(), []

    if mode in ("raw_base", "amali_wrapped_raw_base"):
        from amali.model_gateway.adapter_loader import load_raw_backend

        result, backend = load_raw_backend("Qwen/Qwen2.5-1.5B-Instruct")
        if backend is not None and mode == "amali_wrapped_raw_base":
            backend = ControlPlaneWrapper(backend)
        return result.status, backend, result.reasons

    if mode in ("amali_ft_v0", "amali_wrapped_ft_v0"):
        from amali.model_gateway.adapter_loader import load_ft_backend

        result, backend = load_ft_backend("amali_ft_v0")
        if backend is not None and mode == "amali_wrapped_ft_v0":
            backend = ControlPlaneWrapper(backend)
        return result.status, backend, result.reasons

    raise ValueError(mode)


def _latest_report(name: str) -> ModelEvalReport | None:
    if not ARTIFACT_BASE.is_dir():
        return None
    candidates = sorted(ARTIFACT_BASE.glob(f"*/{name}.json"))
    if not candidates:
        return None
    data = json.loads(candidates[-1].read_text(encoding="utf-8"))
    decisions = data.get("decisions") or []
    if not decisions:
        return None
    return ModelEvalReport(**decisions[0])


def main() -> int:
    parser = argparse.ArgumentParser(description="AMALI frozen-suite model eval")
    parser.add_argument("--model", required=True, choices=MODEL_MODES)
    args = parser.parse_args()
    mode = args.model

    items = load_seed_items(SEED_PATH)
    manifest = EvalSuiteManifest(
        **json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    )

    status, backend, reasons = _build_backend(mode)
    if backend is None:
        report = ModelEvalReport(
            mode=mode,
            status=status,
            suite_hash=manifest.suite_hash,
            risks=reasons,
        )
    else:
        report = evaluate_model(
            mode=mode,
            items=items,
            backend=backend,
            suite_hash=manifest.suite_hash,
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ARTIFACT_BASE / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    name = REPORT_NAMES[mode]
    command = f"python scripts/evaluate_model.py --model {mode}"

    envelope = {
        "schema_version": "1.0.0",
        "generated_at": _now(),
        "repo_ref": _repo_ref(),
        "command": command,
        "status": report.status,
        "metrics": report.metrics,
        "decisions": [report.model_dump(mode="json")],
        "risks": report.risks,
    }
    (out_dir / f"{name}.json").write_text(
        json.dumps(envelope, indent=2), encoding="utf-8"
    )
    (out_dir / f"{name}.md").write_text(
        f"# Eval Report — {mode}\n\nStatus: **{report.status}**  |  "
        f"items: {report.item_count}  |  suite: `{report.suite_hash[:16]}...`\n\n"
        "| metric | value |\n|--------|-------|\n"
        + "\n".join(f"| {k} | {v} |" for k, v in report.metrics.items())
        + "\n\n"
        + (
            "Risks:\n" + "\n".join(f"- {r}" for r in report.risks)
            if report.risks
            else "Scored by the frozen deterministic scorer; no LLM judge."
        )
        + "\n",
        encoding="utf-8",
    )

    # Safety regression vs raw base for FT candidates.
    if mode in ("amali_ft_v0", "amali_wrapped_ft_v0") and report.status == "PASS":
        raw = _latest_report("eval_report_raw_base")
        regression = (
            safety_regression(raw, report)
            if raw is not None
            else None
        )
        (out_dir / "safety_regression_report.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "generated_at": _now(),
                    "repo_ref": _repo_ref(),
                    "command": command,
                    "status": regression.status if regression else "NOT_RUN",
                    "metrics": regression.deltas if regression else {},
                    "decisions": (
                        [regression.model_dump(mode="json")] if regression else []
                    ),
                    "risks": regression.regressions if regression else [
                        "raw base eval missing; run --model raw_base first"
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print(f"Eval {mode}: {report.status}")
    for key, value in report.metrics.items():
        print(f"  {key}: {value}")
    for reason in report.risks:
        print(f"  reason: {reason}", file=sys.stderr)
    print(f"Report: {out_dir / (name + '.json')}")
    return 0 if report.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
