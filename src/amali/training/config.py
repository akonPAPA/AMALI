"""Typed training configuration + deterministic feasibility checks (Phase 5).

The config is validated data, the dry-run is a gate: nothing trains until
every precondition holds, and every failure is a typed status the owner
can act on — never a stack trace, never a fake pass.

No torch is imported here. VRAM feasibility uses a documented heuristic
over the allowlist's parameter counts, not a live CUDA probe, so the
dry-run works on machines without training deps installed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from amali.eval.suite import canonical_json

__all__ = [
    "TrainingConfig",
    "VramEstimate",
    "estimate_vram_gb",
    "DEFAULT_OUTPUT_DIR",
]

DEFAULT_OUTPUT_DIR = "D:/AMALI/models/amali_ft_v0"

_METHODS = ("lora", "qlora")
_QUANTIZATIONS = ("none", "4bit_nf4")
_PRECISIONS = ("bf16", "fp16", "fp32")
_OPTIMIZERS = ("adamw_torch", "adamw_8bit", "paged_adamw_8bit")


class TrainingConfig(BaseModel):
    """Everything a reproducible adapter training run needs."""

    model_config = {"extra": "forbid", "protected_namespaces": ()}

    run_id: str
    base_model_id: str
    base_model_revision: str
    adapter_name: str = "amali_ft_v0"
    output_dir: str = DEFAULT_OUTPUT_DIR
    dataset_manifest_path: str
    eval_suite_manifest_path: str
    contamination_report_path: str = ""
    method: str = "qlora"
    quantization: str = "4bit_nf4"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]
    )
    max_seq_len: int = 1024
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    epochs: int = 2
    max_steps: int = 0  # 0 = derive from epochs
    warmup_ratio: float = 0.03
    seed: int = 20260709
    precision: str = "bf16"
    optimizer: str = "paged_adamw_8bit"
    vram_budget_gb: float = 8.0
    degradation_policy: list[str] = Field(
        default_factory=lambda: [
            "max_seq_len:1024->768->512",
            "lora_rank:16->8",
        ]
    )

    @field_validator("method")
    @classmethod
    def _method_known(cls, value: str) -> str:
        if value not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}")
        return value

    @field_validator("quantization")
    @classmethod
    def _quant_known(cls, value: str) -> str:
        if value not in _QUANTIZATIONS:
            raise ValueError(f"quantization must be one of {_QUANTIZATIONS}")
        return value

    @field_validator("precision")
    @classmethod
    def _precision_known(cls, value: str) -> str:
        if value not in _PRECISIONS:
            raise ValueError(f"precision must be one of {_PRECISIONS}")
        return value

    @field_validator("optimizer")
    @classmethod
    def _optimizer_known(cls, value: str) -> str:
        if value not in _OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {_OPTIMIZERS}")
        return value

    @field_validator("max_seq_len")
    @classmethod
    def _seq_len_sane(cls, value: int) -> int:
        if not 64 <= value <= 4096:
            raise ValueError("max_seq_len must be within [64, 4096]")
        return value

    @field_validator("lora_rank")
    @classmethod
    def _rank_sane(cls, value: int) -> int:
        if not 1 <= value <= 64:
            raise ValueError("lora_rank must be within [1, 64]")
        return value

    def config_hash(self) -> str:
        return hashlib.sha256(
            canonical_json(self.model_dump(mode="json")).encode("utf-8")
        ).hexdigest()

    def output_inside(self, repo_root: str | Path) -> bool:
        """True when output_dir is (wrongly) inside the git repository."""
        try:
            out = Path(self.output_dir).resolve()
            root = Path(repo_root).resolve()
            return out == root or root in out.parents
        except OSError:
            return False


class VramEstimate(BaseModel):
    """Documented heuristic VRAM estimate; honest about being a heuristic."""

    model_config = {"extra": "forbid"}

    params_billion: float
    weights_gb: float
    adapter_and_optimizer_gb: float
    activations_gb: float
    runtime_overhead_gb: float
    total_gb: float
    assumptions: list[str]


def _parse_params_billion(parameter_count: str) -> float:
    text = parameter_count.strip().upper().rstrip("B")
    try:
        return float(text)
    except ValueError:
        return 0.0


def estimate_vram_gb(
    *,
    parameter_count: str,
    quantization: str,
    precision: str,
    max_seq_len: int,
    batch_size: int,
    lora_rank: int,
) -> VramEstimate:
    """Deterministic VRAM heuristic for a LoRA/QLoRA fine-tune.

    Weights: 0.55 GB/B params at 4-bit NF4 (incl. quantization overhead),
    2.2 GB/B at 16-bit. Adapter+optimizer: proportional to rank, small.
    Activations: calibrated to ~1.2 GB for 1.5B at seq 1024 batch 1 with
    gradient checkpointing. Plus a fixed CUDA/runtime overhead. This is a
    planning heuristic, not a measurement — the real run records actual
    peak VRAM.
    """
    params_b = _parse_params_billion(parameter_count)
    bytes_per_b = 0.55 if quantization == "4bit_nf4" else (
        2.2 if precision in ("bf16", "fp16") else 4.4
    )
    weights = params_b * bytes_per_b
    adapter = 0.3 + 0.02 * lora_rank
    activations = 1.2 * batch_size * (max_seq_len / 1024.0) * (params_b / 1.5)
    overhead = 0.8
    total = round(weights + adapter + activations + overhead, 2)
    return VramEstimate(
        params_billion=params_b,
        weights_gb=round(weights, 2),
        adapter_and_optimizer_gb=round(adapter, 2),
        activations_gb=round(activations, 2),
        runtime_overhead_gb=overhead,
        total_gb=total,
        assumptions=[
            "gradient checkpointing enabled",
            "0.55 GB per B params at 4bit_nf4, 2.2 GB per B at 16-bit",
            "activations calibrated at 1.2 GB for 1.5B/seq1024/batch1",
            "heuristic for planning; the real run records measured peak VRAM",
        ],
    )
