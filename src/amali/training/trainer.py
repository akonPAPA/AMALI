"""AMALI adapter trainer (Phase 5/6): dry-run gate + real QLoRA/LoRA path.

Two strictly separated layers:

* **Gate layer (always importable, torch-free):** :func:`run_dry_run`
  re-verifies every precondition — frozen eval suite, dataset floor,
  contamination, allowlist, revision pin, local weights, train deps,
  VRAM heuristic, output dir outside git. Any failure is a typed status.
* **Execution layer (owner-invoked only):** :func:`run_training` runs the
  dry-run first and refuses on anything but ``DRY_RUN_PASS``. torch /
  transformers / peft / datasets are imported lazily inside the real
  train path only.

Nothing here fakes progress: loss, steps, VRAM, and checkpoints exist
only when a real run produced them.
"""

from __future__ import annotations

import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from amali.eval.suite import (
    EvalSuiteManifest,
    load_seed_items,
    verify_frozen_suite,
)
from amali.model_gateway.availability import check_local_model_availability
from amali.model_gateway.base_model_allowlist import (
    FLOATING_REVISIONS,
    check_model_allowed,
    find_entry,
)
from amali.training.config import TrainingConfig, estimate_vram_gb, path_is_inside_repo
from amali.training.contamination import check_contamination
from amali.training.dataset import MIN_NON_SYNTHETIC, TrainingExample
from amali.training.registry import (
    DEFAULT_REGISTRY_PATH,
    CheckpointManifest,
    hash_directory_files,
    register_checkpoint,
)
from amali.training.reports import DryRunReport, TrainingRunReport
from amali.training.sourcepack import SOURCEPACK_ROOT, validate_sourcepack

__all__ = [
    "DATASET_JSONL",
    "run_dry_run",
    "run_training",
    "build_training_plan",
    "load_dataset_examples",
]

DATASET_JSONL = Path("D:/AMALI/data/processed/amali_bmg_train.jsonl")

TRAIN_DEP_MODULES = ("torch", "transformers", "peft", "datasets", "accelerate")
QLORA_EXTRA_DEPS = ("bitsandbytes",)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_dataset_examples(path: Path = DATASET_JSONL) -> list[TrainingExample]:
    if not path.exists():
        return []
    return [
        TrainingExample(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _missing_train_deps(method: str) -> list[str]:
    modules = list(TRAIN_DEP_MODULES)
    if method == "qlora":
        modules.extend(QLORA_EXTRA_DEPS)
    return [m for m in modules if importlib.util.find_spec(m) is None]


def _path_is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def run_dry_run(
    config: TrainingConfig,
    *,
    repo_root: str | Path,
    dataset_path: Path = DATASET_JSONL,
    cache_dir: Path | None = None,
    min_non_synthetic: int = MIN_NON_SYNTHETIC,
    sourcepack_root: Path | None = None,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> DryRunReport:
    """Verify every training precondition without touching a GPU."""
    root = Path(repo_root)
    checks: dict[str, str] = {}
    reasons: list[str] = []
    owner_actions: list[str] = []
    status: str | None = None

    def gate(name: str, ok: bool, fail_status: str, reason: str, action: str = "") -> None:
        nonlocal status
        checks[name] = "PASS" if ok else "FAIL"
        if not ok:
            reasons.append(reason)
            if action:
                owner_actions.append(action)
            if status is None:
                status = fail_status

    # 1. frozen eval suite ---------------------------------------------------
    seed_path = root / "fixtures" / "eval" / "base_model_gate_seed.json"
    manifest_path = (
        Path(config.eval_suite_manifest_path)
        if config.eval_suite_manifest_path
        else None
    )
    eval_suite_hash = ""
    eval_ok = False
    eval_items = []
    if seed_path.is_file() and manifest_path is not None and manifest_path.is_file():
        eval_items = load_seed_items(seed_path)
        manifest = EvalSuiteManifest(
            **json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        eval_ok = verify_frozen_suite(eval_items, manifest).status == "PASS"
        eval_suite_hash = manifest.suite_hash
    gate(
        "eval_suite_frozen",
        eval_ok,
        "NOT_READY",
        "eval suite is not frozen or failed tamper verification",
        "run: python scripts/freeze_eval_suite.py",
    )

    # 2. dataset floor ---------------------------------------------------------
    examples = load_dataset_examples(dataset_path)
    non_synthetic = sum(1 for e in examples if not e.synthetic)
    dataset_hash = ""
    manifest_file = (
        Path(config.dataset_manifest_path)
        if config.dataset_manifest_path
        else None
    )
    if manifest_file is not None and manifest_file.is_file():
        dataset_hash = json.loads(
            manifest_file.read_text(encoding="utf-8")
        ).get("dataset_hash", "")
    gate(
        "dataset_ready",
        non_synthetic >= min_non_synthetic,
        "DATASET_NOT_READY",
        f"dataset has {non_synthetic} non-synthetic examples; floor is "
        f"{min_non_synthetic}",
        "add approved local sources under D:/AMALI/data/raw/ and rerun "
        "scripts/build_training_dataset.py",
    )

    # 3. contamination ---------------------------------------------------------
    if examples and eval_items:
        contamination = check_contamination(examples, eval_items)
        gate(
            "contamination",
            contamination.status == "PASS",
            "CONTAMINATION_FAIL",
            f"contaminated ids: {contamination.contaminated_ids}",
            "remove contaminated items from sources and rebuild the dataset",
        )
    else:
        checks["contamination"] = "NOT_RUN"

    # 4. allowlist + owner approval ---------------------------------------------
    allow = check_model_allowed(config.base_model_id, for_training=True)
    gate(
        "base_model_allowlisted",
        allow.allowed,
        "NOT_OWNER_APPROVED",
        f"model {config.base_model_id} not approved for training: "
        f"{allow.reasons}",
        "approve the model in base_model_allowlist.py deliberately",
    )

    # 5. revision pin --------------------------------------------------------------
    gate(
        "revision_pinned",
        config.base_model_revision not in FLOATING_REVISIONS,
        "REVISION_NOT_PINNED",
        "base_model_revision is floating; promotion-grade training needs "
        "a pinned snapshot",
        "download with --revision <commit_sha> and set base_model_revision",
    )

    # 6. local weights ----------------------------------------------------------------
    availability = check_local_model_availability(
        config.base_model_id, cache_dir=cache_dir
    )
    gate(
        "weights_available_locally",
        availability.status == "AVAILABLE",
        "MODEL_NOT_AVAILABLE",
        f"local weights: {availability.status}",
        "owner: python scripts/download_model.py --model "
        f"{config.base_model_id} --revision <PINNED_REVISION>",
    )

    # 7. train deps ---------------------------------------------------------------------
    missing = _missing_train_deps(config.method)
    gate(
        "train_deps_installed",
        not missing,
        "TRAIN_DEPS_MISSING",
        f"missing training dependencies: {missing}",
        'owner: python -m pip install -e ".[dev,local_llm,train]"',
    )

    # 8. VRAM heuristic ---------------------------------------------------------------------
    entry = find_entry(config.base_model_id)
    estimate = None
    if entry is not None:
        estimate = estimate_vram_gb(
            parameter_count=entry.parameter_count,
            quantization=config.quantization,
            precision=config.precision,
            max_seq_len=config.max_seq_len,
            batch_size=config.batch_size,
            lora_rank=config.lora_rank,
        )
        gate(
            "vram_feasible",
            estimate.total_gb <= config.vram_budget_gb,
            "VRAM_INFEASIBLE",
            f"estimated {estimate.total_gb} GB exceeds budget "
            f"{config.vram_budget_gb} GB",
            "apply degradation policy: reduce max_seq_len (1024->768->512) "
            "or lora_rank (16->8), or pick the smaller base model",
        )
    else:
        checks["vram_feasible"] = "NOT_RUN"

    # 9. output dir outside git ------------------------------------------------------------------
    gate(
        "output_outside_repo",
        not config.output_inside(root),
        "NOT_READY",
        f"output_dir {config.output_dir} is inside the git repository",
        "point output_dir outside the repo, e.g. D:/AMALI/models/amali_ft_v0",
    )

    # 10. registry outside git ---------------------------------------------------------------------
    gate(
        "registry_outside_repo",
        not path_is_inside_repo(registry_path, root),
        "NOT_READY",
        f"checkpoint registry {registry_path} is inside the git repository",
        "point the registry outside the repo, e.g. D:/AMALI/models/registry/",
    )

    # 11. sourcepack re-verification --------------------------------------------------------------
    # Any dataset example built from the raw owner dir must be backed by a
    # currently-VALID sourcepack; catches post-build tampering. When no
    # raw-sourced examples exist there is nothing to re-verify.
    sp_root = Path(sourcepack_root) if sourcepack_root else SOURCEPACK_ROOT
    raw_sourced = [
        e for e in examples if _path_is_under(Path(e.source_path), sp_root)
    ]
    if raw_sourced:
        sp_report = validate_sourcepack(sp_root)
        gate(
            "sourcepack_valid",
            sp_report.status == "VALID",
            "OWNER_DATA_REQUIRED",
            f"sourcepack validation: {sp_report.status}",
            "fix the sourcepack and rerun "
            "scripts/validate_owner_sourcepack.py",
        )
    else:
        checks["sourcepack_valid"] = "NOT_RUN"

    return DryRunReport(
        status=status or "DRY_RUN_PASS",
        checks=checks,
        reasons=reasons,
        missing_deps=missing,
        estimated_vram_gb=estimate.total_gb if estimate else None,
        vram_budget_gb=config.vram_budget_gb,
        config_hash=config.config_hash(),
        dataset_hash=dataset_hash,
        eval_suite_hash=eval_suite_hash,
        owner_actions=owner_actions,
    )


def build_training_plan(
    config: TrainingConfig,
    *,
    max_steps_override: int = 0,
    resume: bool = False,
) -> dict:
    """Pure, torch-free construction of the HF training arguments.

    Kept separate so the exact run parameters are testable and auditable
    without any training dependency installed.
    """
    max_steps = max_steps_override or config.max_steps
    return {
        "output_dir": config.output_dir,
        "per_device_train_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.epochs,
        "max_steps": max_steps if max_steps > 0 else -1,
        "warmup_ratio": config.warmup_ratio,
        "seed": config.seed,
        "bf16": config.precision == "bf16",
        "fp16": config.precision == "fp16",
        "optim": config.optimizer,
        "gradient_checkpointing": True,
        "logging_steps": 1,
        "save_strategy": "epoch",
        "report_to": [],
        "resume_from_checkpoint": resume,
    }


def run_training(
    config: TrainingConfig,
    *,
    repo_root: str | Path,
    execute: bool = True,
    max_steps_override: int = 0,
    resume: bool = False,
    dataset_path: Path = DATASET_JSONL,
    cache_dir: Path | None = None,
) -> TrainingRunReport:
    """Run the full gate, then (optionally) the real training.

    ``execute=False`` is the ``--dry-run`` path: the report carries the
    dry-run outcome and no training side effects exist.
    """
    dry = run_dry_run(
        config,
        repo_root=repo_root,
        dataset_path=dataset_path,
        cache_dir=cache_dir,
    )
    base_report = TrainingRunReport(
        status=dry.status,
        run_id=config.run_id,
        config_hash=dry.config_hash,
        dataset_hash=dry.dataset_hash,
        eval_suite_hash=dry.eval_suite_hash,
        base_model_id=config.base_model_id,
        base_model_revision=config.base_model_revision,
        dry_run=dry,
        risks=list(dry.reasons),
        owner_actions=list(dry.owner_actions),
    )
    if dry.status != "DRY_RUN_PASS" or not execute:
        return base_report

    return _execute_training(
        config,
        base_report,
        max_steps_override=max_steps_override,
        resume=resume,
        dataset_path=dataset_path,
    )


def _format_example(example: TrainingExample) -> str:
    parts = [f"Instruction: {example.instruction}"]
    if example.input_context:
        parts.append(f"Context: {example.input_context}")
    parts.append(f"Response: {example.expected_response}")
    return "\n\n".join(parts)


def _execute_training(
    config: TrainingConfig,
    report: TrainingRunReport,
    *,
    max_steps_override: int,
    resume: bool,
    dataset_path: Path,
) -> TrainingRunReport:
    """The real, owner-invoked training path. Lazy imports only here."""
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        report.status = "TRAIN_DEPS_MISSING"
        report.risks.append(f"import failed inside train path: {exc}")
        report.owner_actions.append(
            'owner: python -m pip install -e ".[dev,local_llm,train]"'
        )
        return report

    started = time.monotonic()
    report.started_at = _now()
    report.status = "TRAINING_STARTED"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report.device = device
    if device == "cuda":
        report.gpu_name = torch.cuda.get_device_name(0)
        torch.cuda.reset_peak_memory_stats()

    try:
        quant_kwargs = {}
        if config.method == "qlora" and config.quantization == "4bit_nf4":
            from transformers import BitsAndBytesConfig

            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=(
                    torch.bfloat16 if config.precision == "bf16" else torch.float16
                ),
                bnb_4bit_use_double_quant=True,
            )

        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_id,
            revision=config.base_model_revision,
            local_files_only=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_id,
            revision=config.base_model_revision,
            local_files_only=True,
            torch_dtype=(
                torch.bfloat16 if config.precision == "bf16" else torch.float16
            ),
            **quant_kwargs,
        )
        if config.method == "qlora":
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(model)
        model = get_peft_model(
            model,
            LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.target_modules,
                task_type="CAUSAL_LM",
            ),
        )

        examples = load_dataset_examples(dataset_path)
        encodings = []
        for example in examples:
            tokens = tokenizer(
                _format_example(example),
                truncation=True,
                max_length=config.max_seq_len,
                padding="max_length",
                return_tensors="pt",
            )
            encodings.append(
                {
                    "input_ids": tokens["input_ids"][0],
                    "attention_mask": tokens["attention_mask"][0],
                    "labels": tokens["input_ids"][0].clone(),
                }
            )

        losses: list[float] = []

        class _LossRecorder(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and "loss" in logs:
                    losses.append(float(logs["loss"]))

        plan = build_training_plan(
            config, max_steps_override=max_steps_override, resume=resume
        )
        resume_flag = plan.pop("resume_from_checkpoint")
        trainer = Trainer(
            model=model,
            args=TrainingArguments(**plan),
            train_dataset=encodings,
            callbacks=[_LossRecorder()],
        )
        result = trainer.train(
            resume_from_checkpoint=resume_flag if resume_flag else None
        )

        adapter_dir = Path(config.output_dir)
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

        report.step_count = int(result.global_step)
        report.loss_curve = losses
        report.adapter_dir = str(adapter_dir)
        report.adapter_file_hashes = hash_directory_files(adapter_dir)
        if device == "cuda":
            report.peak_vram_gb = round(
                torch.cuda.max_memory_allocated() / (1024**3), 2
            )
        report.status = "TRAINING_COMPLETED"

        manifest = CheckpointManifest(
            checkpoint_id=f"ckpt_{config.run_id}",
            adapter_name=config.adapter_name,
            base_model_id=config.base_model_id,
            base_model_revision=config.base_model_revision,
            training_config_hash=report.config_hash,
            dataset_manifest_hash=report.dataset_hash,
            eval_suite_hash=report.eval_suite_hash,
            adapter_dir=str(adapter_dir),
            file_hashes=report.adapter_file_hashes,
            created_at=_now(),
            train_status="TRAINING_COMPLETED",
        )
        register_checkpoint(manifest)
    except Exception as exc:  # noqa: BLE001 - real failures must surface, typed
        is_oom = isinstance(
            exc, torch.cuda.OutOfMemoryError
        ) or "out of memory" in str(exc).lower()
        if is_oom:
            report.status = "TRAINING_FAILED_OOM"
            report.risks.append(f"CUDA out of memory: {exc}")
            report.owner_actions.extend(
                [
                    "degradation options (owner picks; nothing is retried "
                    "silently):",
                    "  1. reduce max_seq_len 1024 -> 768 -> 512",
                    "  2. reduce lora_rank 16 -> 8",
                    "  3. fall back to Qwen/Qwen2.5-0.5B-Instruct",
                ]
            )
        else:
            report.status = "TRAINING_FAILED"
            report.risks.append(
                f"training failed: {type(exc).__name__}: {exc}"
            )
    finally:
        report.ended_at = _now()
        report.wall_clock_seconds = round(time.monotonic() - started, 2)

    return report
