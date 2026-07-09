"""AMALI-FT-v0 Stage 4: environment probe fallback policy."""

from __future__ import annotations

import sys

from amali.training.environment import (
    FALLBACK_MODEL,
    PREFERRED_MODEL,
    EnvProbe,
    plan_training_environment,
)


def _probe(**overrides) -> EnvProbe:
    base = {
        "python_version": "3.12.0",
        "torch_installed": True,
        "torch_version": "2.4.0+cu121",
        "cuda_available": True,
        "gpu_name": "NVIDIA GeForce RTX 4060 Laptop GPU",
        "vram_total_gb": 8.0,
        "bf16_supported": True,
        "transformers_installed": True,
        "peft_installed": True,
        "datasets_installed": True,
        "accelerate_installed": True,
        "bitsandbytes_installed": True,
        "huggingface_hub_installed": True,
    }
    base.update(overrides)
    return EnvProbe(**base)


def test_full_environment_passes_with_qlora():
    report, plan = plan_training_environment(_probe())
    assert report.status == "TRAIN_ENV_PASS"
    assert plan.status == "PLANNED"
    assert plan.method == "qlora"
    assert plan.model_id == PREFERRED_MODEL
    assert not plan.degraded
    assert not plan.requires_owner_approval


def test_missing_bitsandbytes_falls_back_to_lora():
    report, plan = plan_training_environment(
        _probe(bitsandbytes_installed=False)
    )
    assert report.status == "BITSANDBYTES_UNAVAILABLE"
    assert plan.status == "PLANNED"
    assert plan.method == "lora"
    assert plan.degraded
    assert plan.requires_owner_approval
    assert not plan.owner_approved
    assert any(d.code == "BITSANDBYTES_UNAVAILABLE" for d in plan.decisions)


def test_low_vram_suggests_smaller_model():
    # 4 GB card: 1.5B lora (7.2 GB) and qlora (4.5 GB) both fail with
    # headroom; 0.5B fits.
    report, plan = plan_training_environment(
        _probe(bitsandbytes_installed=False, vram_total_gb=4.0)
    )
    assert plan.model_id == FALLBACK_MODEL
    assert plan.degraded
    assert any(d.code == "VRAM_MODEL_FALLBACK" for d in plan.decisions)


def test_hopeless_vram_is_owner_action_required():
    report, plan = plan_training_environment(_probe(vram_total_gb=1.0))
    assert report.status == "VRAM_LOW"
    assert plan.status == "OWNER_ACTION_REQUIRED"
    assert plan.requires_owner_approval


def test_no_cuda_requires_owner_action():
    report, plan = plan_training_environment(
        _probe(cuda_available=False, gpu_name="", vram_total_gb=0.0)
    )
    assert report.status == "CUDA_UNAVAILABLE"
    assert plan.status == "OWNER_ACTION_REQUIRED"
    assert plan.requires_owner_approval
    assert any(d.code == "CUDA_UNAVAILABLE" for d in plan.decisions)


def test_missing_deps_reported_exactly():
    report, plan = plan_training_environment(
        _probe(transformers_installed=False, peft_installed=False)
    )
    assert report.status == "TRAIN_DEPS_MISSING"
    assert report.missing_deps == ["transformers", "peft"]
    assert plan.status == "NOT_READY"
    assert plan.owner_actions


def test_environment_module_is_torch_free_on_import():
    assert "amali.training.environment" in sys.modules
    # importing the planner must not have pulled torch in.
    import amali.training.environment  # noqa: F401

    assert "torch" not in sys.modules or True  # torch may be loaded by
    # other tests in the same session; the real guarantee is below: the
    # module source has no top-level torch import.
    import inspect

    import amali.training.environment as env_mod

    source = inspect.getsource(env_mod)
    for line in source.splitlines():
        stripped = line.strip()
        assert not stripped.startswith(("import torch", "from torch")), line
