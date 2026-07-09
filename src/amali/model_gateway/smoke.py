"""Real local Model Gateway smoke (Base Model Readiness).

One question, answered honestly: does the pinned local base model actually
tokenize, load, and generate on this machine, fully offline?

SUCCESS means all three of tokenizer load, model load, and generation
really executed and text was captured — never that the output matched the
expected marker (a mismatched output is recorded but is still a real
generation). Every failure mode is a typed status; availability alone can
never produce SUCCESS.

torch/transformers are imported only inside the real load path. Tests
inject a fake ``load_fn`` and never touch torch or the network.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

from amali.model_gateway.availability import (
    STATUS_AVAILABLE,
    check_model_snapshot,
)
from amali.model_gateway.base_model_allowlist import check_model_allowed

__all__ = [
    "SMOKE_PROMPT",
    "SMOKE_EXPECTED",
    "GatewaySmokeReport",
    "run_gateway_smoke",
]

SMOKE_EXPECTED = "AMALI_MODEL_GATEWAY_OK"
SMOKE_PROMPT = f"Answer with exactly: {SMOKE_EXPECTED}"

STATUS_SUCCESS = "SUCCESS"
STATUS_DEPS_MISSING = "DEPS_MISSING"
STATUS_MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
STATUS_TOKENIZER_LOAD_FAILED = "TOKENIZER_LOAD_FAILED"
STATUS_GENERATION_FAILED = "GENERATION_FAILED"
STATUS_POLICY_BLOCKED = "POLICY_BLOCKED"


class GatewaySmokeReport(BaseModel):
    """Typed outcome of one gateway smoke attempt."""

    model_config = {"extra": "forbid", "protected_namespaces": ()}

    status: str
    model_id: str
    revision: str = ""
    local_files_only: bool = True
    tokenizer_loaded: bool = False
    model_loaded: bool = False
    generation_executed: bool = False
    output_text: str = ""
    output_matched: bool = False
    prompt: str = SMOKE_PROMPT
    max_new_tokens: int = 16
    device: str = ""
    latency_ms: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    owner_actions: list[str] = Field(default_factory=list)


def _deps_missing() -> list[str]:
    return [
        m
        for m in ("torch", "transformers")
        if importlib.util.find_spec(m) is None
    ]


def _real_generate(
    model_id: str,
    revision: str | None,
    local_files_only: bool,
    max_new_tokens: int,
    report: GatewaySmokeReport,
) -> GatewaySmokeReport:
    """The only code path that touches torch. Split per stage so every
    failure carries the stage it died in."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, local_files_only=local_files_only
        )
    except Exception as exc:  # noqa: BLE001 - typed report, not a crash
        return report.model_copy(
            update={
                "status": STATUS_TOKENIZER_LOAD_FAILED,
                "reasons": [f"tokenizer load failed: {exc}"],
            }
        )
    report = report.model_copy(update={"tokenizer_loaded": True})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
        model.to(device)
        model.eval()
    except Exception as exc:  # noqa: BLE001 - typed report, not a crash
        return report.model_copy(
            update={
                "status": STATUS_MODEL_LOAD_FAILED,
                "device": device,
                "reasons": [f"model load failed: {exc}"],
            }
        )
    report = report.model_copy(update={"model_loaded": True, "device": device})

    try:
        start = time.perf_counter()
        inputs = tokenizer(SMOKE_PROMPT, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        text = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        latency = (time.perf_counter() - start) * 1000.0
    except Exception as exc:  # noqa: BLE001 - typed report, not a crash
        return report.model_copy(
            update={
                "status": STATUS_GENERATION_FAILED,
                "reasons": [f"generation failed: {exc}"],
            }
        )

    return report.model_copy(
        update={
            "status": STATUS_SUCCESS,
            "generation_executed": True,
            "output_text": text,
            "output_matched": SMOKE_EXPECTED in text,
            "latency_ms": round(latency, 3),
            "reasons": (
                []
                if SMOKE_EXPECTED in text
                else [
                    "output did not match the expected marker; generation "
                    "still really executed (recorded honestly)"
                ]
            ),
        }
    )


def run_gateway_smoke(
    model_id: str,
    revision: str | None = None,
    *,
    local_files_only: bool = True,
    max_new_tokens: int = 16,
    cache_dir: Path | None = None,
    load_fn: Callable[..., GatewaySmokeReport] | None = None,
) -> GatewaySmokeReport:
    """Gate order: allowlist -> deps -> real snapshot -> real generation.

    ``load_fn`` exists for tests only; the availability gates before it
    still run, so a fake loader can never turn a missing model into
    SUCCESS.
    """
    report = GatewaySmokeReport(
        status=STATUS_POLICY_BLOCKED,
        model_id=model_id,
        revision=revision or "",
        local_files_only=local_files_only,
        max_new_tokens=max_new_tokens,
    )

    decision = check_model_allowed(model_id)
    if not decision.allowed:
        return report.model_copy(
            update={
                "status": STATUS_POLICY_BLOCKED,
                "reasons": [f"allowlist refused: {decision.reasons}"],
            }
        )

    missing = _deps_missing()
    if missing:
        return report.model_copy(
            update={
                "status": STATUS_DEPS_MISSING,
                "reasons": [f"missing deps: {missing}"],
                "owner_actions": [
                    'python -m pip install -e ".[dev,local_llm]"'
                ],
            }
        )

    snapshot = check_model_snapshot(model_id, revision, cache_dir=cache_dir)
    if snapshot.status != STATUS_AVAILABLE:
        return report.model_copy(
            update={
                "status": snapshot.status,
                "reasons": snapshot.notes,
                "owner_actions": [
                    f"python scripts/download_model.py --model {model_id} "
                    "--revision <HF_SNAPSHOT_COMMIT_SHA>"
                ],
            }
        )

    loader = load_fn or _real_generate
    return loader(model_id, revision, local_files_only, max_new_tokens, report)
