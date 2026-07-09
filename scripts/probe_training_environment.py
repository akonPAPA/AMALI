"""Probe the local training environment (AMALI-FT-v0 Stage 4).

    python scripts/probe_training_environment.py

Gathers Python/torch/CUDA/GPU/dependency facts, runs the fallback
planner, and writes ``training_env_report.json`` plus
``training_fallback_plan.json`` under ``artifacts/amali_ft_v0/<ts>/``.

Exit 0 only on TRAIN_ENV_PASS (full preferred path: deps + CUDA +
bitsandbytes + VRAM). Any degradation or gap exits nonzero with a typed
status — degraded plans require explicit owner approval before training.

torch is imported here (explicit owner diagnostics command), never on
the core import path.
"""

from __future__ import annotations

import importlib.util
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from amali.training.environment import (  # noqa: E402
    EnvProbe,
    plan_training_environment,
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


def _installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _bitsandbytes_usable() -> bool:
    """True only when bitsandbytes can actually run its native kernels.

    On Windows the wheel can install and even import cleanly while the
    CUDA binary for the local toolkit version is absent; bitsandbytes then
    swallows the load error and exposes a mock/None library object. QLoRA
    with that state fails at first quantized op — so it counts as
    unavailable here. No fake QLoRA.
    """
    if not _installed("bitsandbytes"):
        return False
    try:
        import bitsandbytes  # noqa: F401
        from bitsandbytes.cextension import lib
    except Exception:  # noqa: BLE001 - any import failure means unusable
        return False
    if lib is None:
        return False
    # newer versions swap in an error-handler mock when the binary is missing
    if "mock" in type(lib).__name__.lower() or "error" in type(lib).__name__.lower():
        return False
    return True


def gather_probe() -> EnvProbe:
    torch_installed = _installed("torch")
    torch_version = ""
    cuda_available = False
    gpu_name = ""
    vram_total_gb = 0.0
    bf16_supported = False

    if torch_installed:
        import torch

        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            vram_total_gb = round(props.total_memory / (1024**3), 2)
            try:
                bf16_supported = torch.cuda.is_bf16_supported()
            except Exception:  # noqa: BLE001 - optional signal
                bf16_supported = False

    return EnvProbe(
        python_version=platform.python_version(),
        torch_installed=torch_installed,
        torch_version=torch_version,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        vram_total_gb=vram_total_gb,
        bf16_supported=bf16_supported,
        transformers_installed=_installed("transformers"),
        peft_installed=_installed("peft"),
        datasets_installed=_installed("datasets"),
        accelerate_installed=_installed("accelerate"),
        bitsandbytes_installed=_bitsandbytes_usable(),
        huggingface_hub_installed=_installed("huggingface_hub"),
    )


def main() -> int:
    probe = gather_probe()
    report, plan = plan_training_environment(probe)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO_ROOT / "artifacts" / "amali_ft_v0" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    envelope = {
        "schema_version": "1.0.0",
        "generated_at": _now(),
        "repo_ref": _repo_ref(),
        "command": "python scripts/probe_training_environment.py",
    }
    (out_dir / "training_env_report.json").write_text(
        json.dumps(
            {**envelope, "status": report.status,
             "report": report.model_dump(mode="json")},
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "training_fallback_plan.json").write_text(
        json.dumps(
            {**envelope, "status": plan.status,
             "plan": plan.model_dump(mode="json")},
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Training environment: {report.status}")
    print(
        f"python={probe.python_version} torch={probe.torch_version or 'missing'} "
        f"cuda={probe.cuda_available} gpu={probe.gpu_name or 'n/a'} "
        f"vram={probe.vram_total_gb}GB bf16={probe.bf16_supported} "
        f"bitsandbytes={probe.bitsandbytes_installed}"
    )
    if report.missing_deps:
        print(f"missing deps: {', '.join(report.missing_deps)}", file=sys.stderr)
    print(
        f"Fallback plan: {plan.status}"
        + (f" -> {plan.method} on {plan.model_id}" if plan.model_id else "")
        + (" (degraded; owner approval required)" if plan.degraded else "")
    )
    for decision in plan.decisions:
        print(f"  decision: {decision.code} — {decision.detail}", file=sys.stderr)
    for action in report.risks:
        print(f"  risk: {action}", file=sys.stderr)
    for action in plan.owner_actions:
        print(f"  owner action: {action}", file=sys.stderr)
    print(f"Reports: {out_dir}")
    return 0 if report.status == "TRAIN_ENV_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
