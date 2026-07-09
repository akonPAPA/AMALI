"""Run the real local Model Gateway smoke (owner command).

    python scripts/run_model_gateway_smoke.py --model Qwen/Qwen2.5-1.5B-Instruct --revision <HF_SNAPSHOT_COMMIT_SHA> --local-files-only

SUCCESS means tokenizer load + model load + real generation all executed
offline; the output text is captured (a non-matching output is recorded,
not faked into failure or success). Missing deps/weights/revision produce
typed statuses and a nonzero exit.

Writes model_gateway_smoke_report.{json,md} under
``artifacts/base_model_readiness/<timestamp>/``.
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

from amali.model_gateway.smoke import run_gateway_smoke  # noqa: E402

READINESS_BASE = REPO_ROOT / "artifacts" / "base_model_readiness"


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
    parser = argparse.ArgumentParser(description="AMALI real gateway smoke")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--revision",
        default=None,
        help="pinned Hugging Face snapshot commit SHA",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=True,
        help="never touch the network (default and only supported mode)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    report = run_gateway_smoke(
        args.model,
        args.revision,
        local_files_only=True,  # smoke is always offline by design
        max_new_tokens=args.max_new_tokens,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = READINESS_BASE / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    command = (
        f"python scripts/run_model_gateway_smoke.py --model {args.model}"
        + (f" --revision {args.revision}" if args.revision else "")
        + " --local-files-only"
    )

    (out_dir / "model_gateway_smoke_report.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "generated_at": _now(),
                "repo_ref": _repo_ref(),
                "command": command,
                "status": report.status,
                "metrics": {
                    "tokenizer_loaded": report.tokenizer_loaded,
                    "model_loaded": report.model_loaded,
                    "generation_executed": report.generation_executed,
                    "output_matched": report.output_matched,
                    "latency_ms": report.latency_ms,
                    "device": report.device,
                },
                "decisions": [report.model_dump(mode="json")],
                "risks": report.reasons,
                "owner_actions": report.owner_actions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "model_gateway_smoke_report.md").write_text(
        f"# Model Gateway Smoke Report\n\nStatus: **{report.status}**\n\n"
        f"- model: `{report.model_id}` @ `{report.revision or 'floating'}`\n"
        f"- local_files_only: {report.local_files_only}\n"
        f"- tokenizer loaded: {report.tokenizer_loaded}\n"
        f"- model loaded: {report.model_loaded}\n"
        f"- generation executed: {report.generation_executed}\n"
        f"- output matched marker: {report.output_matched}\n"
        f"- device: {report.device or 'n/a'}\n"
        f"- latency: {report.latency_ms} ms\n\n"
        + (
            "Reasons:\n" + "\n".join(f"- {r}" for r in report.reasons) + "\n"
            if report.reasons
            else "Real offline generation executed.\n"
        ),
        encoding="utf-8",
    )

    print(f"Gateway smoke: {report.status}")
    if report.generation_executed:
        print(f"  output: {report.output_text!r} (matched={report.output_matched})")
        print(f"  device: {report.device}, latency: {report.latency_ms} ms")
    for reason in report.reasons:
        print(f"  reason: {reason}", file=sys.stderr)
    for action in report.owner_actions:
        print(f"  owner action: {action}", file=sys.stderr)
    print(f"Report: {out_dir / 'model_gateway_smoke_report.json'}")
    return 0 if report.status == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
