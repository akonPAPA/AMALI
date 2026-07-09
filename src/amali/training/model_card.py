"""Truthful model card generation for AMALI-FT-v0 (Phase 13).

The card is generated from evidence, not aspirations: hashes, statuses,
and metrics are copied from artifacts; anything that has not happened is
stated as not having happened. Structural guarantees:

* the lineage section always names the base model — AMALI-FT-v0 is an
  adapter, never a from-scratch model;
* the claim-boundary section is fixed text with no code path that can
  inject a frontier/superiority claim;
* generation fails loudly if a forbidden marker somehow appears.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["ModelCardInputs", "generate_model_card"]

_FORBIDDEN_MARKERS = (
    "beats gpt",
    "beats claude",
    "beats deepseek",
    "beats glm",
    "frontier model",
    "frontier-class",
    "agi achieved",
    "trained from scratch by amali",
)


class ModelCardInputs(BaseModel):
    """Everything the card cites; empty fields render as honest absences."""

    model_config = {"extra": "forbid", "protected_namespaces": ()}

    base_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    base_model_revision: str = ""
    base_license: str = "Apache-2.0"
    adapter_type: str = "LoRA/QLoRA adapter (PEFT)"
    dataset_manifest_hash: str = ""
    eval_suite_hash: str = ""
    contamination_report_hash: str = ""
    training_status: str = "NOT_RUN"
    training_config: dict = Field(default_factory=dict)
    training_run_facts: dict = Field(default_factory=dict)
    raw_base_metrics: dict = Field(default_factory=dict)
    ft_metrics: dict = Field(default_factory=dict)
    wrapped_ft_metrics: dict = Field(default_factory=dict)
    safety_regression_status: str = "NOT_RUN"
    promotion_decision: str = "NOT_RUN"
    hardware: str = "Windows 11, single RTX 4060 Laptop GPU (8 GB VRAM)"


def _metrics_table(inputs: ModelCardInputs) -> str:
    keys = sorted(
        set(inputs.raw_base_metrics) | set(inputs.wrapped_ft_metrics)
    )
    if not keys:
        return (
            "No evaluation has been run yet; no scores exist and none are "
            "invented."
        )
    rows = "\n".join(
        f"| {k} | {inputs.raw_base_metrics.get(k, '—')} | "
        f"{inputs.ft_metrics.get(k, '—')} | "
        f"{inputs.wrapped_ft_metrics.get(k, '—')} |"
        for k in keys
    )
    return (
        "| metric | raw base | AMALI-FT-v0 | AMALI-wrapped FT |\n"
        "|--------|----------|-------------|------------------|\n" + rows
    )


def generate_model_card(inputs: ModelCardInputs) -> str:
    revision = inputs.base_model_revision or "NOT PINNED (not promotable)"
    card = f"""# Model Card — AMALI-FT-v0

## What this is (and is not)

AMALI-FT-v0 is an **AMALI-trained adapter on top of an owner-approved
open base model, governed by the AMALI control plane**. It is **not
trained from scratch**: the base model's language ability comes entirely
from its original developers, and AMALI adds a locally trained adapter
plus a deterministic control plane (Data Wall, TRPC, HER-MoE router,
DWAC, HSGR retrieval, verifier, EWA/UGE arbitration, FEC, audit ledger).

## Lineage

- base model id: `{inputs.base_model_id}`
- base model revision: `{revision}`
- base license: {inputs.base_license}
- adapter type: {inputs.adapter_type}

## Provenance hashes

- training dataset manifest hash: `{inputs.dataset_manifest_hash or "absent — dataset not ready"}`
- frozen eval suite hash: `{inputs.eval_suite_hash or "absent"}`
- contamination report hash: `{inputs.contamination_report_hash or "absent"}`

## Training

Status: **{inputs.training_status}**

Config: {inputs.training_config or "no training config recorded"}

Run facts: {inputs.training_run_facts or "no training run has happened; no loss, steps, or VRAM figures exist"}

## Evaluation (frozen AMALI suite, deterministic scorer)

{_metrics_table(inputs)}

Safety regression vs raw base: **{inputs.safety_regression_status}**
Promotion decision: **{inputs.promotion_decision}**

## Known limitations

- The frozen eval suite measures AMALI-specific system behavior
  (grounding, honesty, refusal, injection handling), not general
  language ability or reasoning breadth.
- The deterministic scorer is lexical; it cannot judge semantic
  entailment.
- Training data comes from a small local corpus; the adapter cannot add
  knowledge beyond it.
- No external benchmark (MMLU or similar) has been run.

## Hardware

{inputs.hardware}

## Claim boundary

- Allowed, only if promotion passed: "AMALI-FT-v0, evaluated through the
  AMALI control plane, improves over the raw same base model on the
  frozen AMALI eval suite, especially on honesty, policy, and
  unsupported-success metrics."
- This model was **not trained from scratch**.
- **No comparison to GPT, Claude, DeepSeek, GLM, or any external system
  is made or implied**; without immutable stored baselines scored by the
  same harness, every such comparison is NOT_PROVEN.
- No claim of AGI, ASI, frontier status, or self-improvement is made.

## Use restrictions

- Local, owner-controlled use only; governed by the AMALI control plane.
- The raw base model's original license and use restrictions apply.
- Do not present raw base model output as AMALI output.

## Reproducibility

```
python -m pip install -e ".[dev,local_llm,train]"
python scripts/download_model.py --model {inputs.base_model_id} --revision <PINNED_REVISION>
python scripts/freeze_eval_suite.py
python scripts/build_training_dataset.py
python scripts/check_contamination.py
python scripts/train_amali_adapter.py --dry-run
python scripts/train_amali_adapter.py
python scripts/evaluate_model.py --model raw_base
python scripts/evaluate_model.py --model amali_ft_v0
python scripts/evaluate_model.py --model amali_wrapped_ft_v0
python scripts/promote_model.py amali_ft_v0
```
"""
    lowered = card.lower()
    for marker in _FORBIDDEN_MARKERS:
        if marker in lowered:
            raise ValueError(f"forbidden claim marker in model card: {marker}")
    return card
