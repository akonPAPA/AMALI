# AMALI Core — Local Kernel

AMALI (Adaptive Multi-Agent LLM Infrastructure) is an **owner-governed,
evaluation-first, secure model-team AI infrastructure**. This repository
holds the **local source-of-truth kernel** for *Investor Gate Alpha* — a
technical proof-of-architecture, not a product.

> This is **not** a B2B SaaS MVP, not a chatbot wrapper, and not a
> frontend demo. It is a small, typed, auditable kernel that proves a few
> architectural invariants are real and enforceable in code.

## What this kernel proves

- **Generated text is not trusted state.** Model output becomes a
  `ClaimRecord` that starts `UNKNOWN` / `unsupported`.
- **Claims are explicitly classified** via `EvidenceLevel`
  (`KNOWN`, `INFERRED`, `UNKNOWN`, `HYPOTHESIS`, `RISK`).
- **Evidence is explicit** and carries trust, sensitivity, and
  adversarial-risk metadata (poison / injection).
- **A claim may only be `KNOWN` with supporting evidence** — enforced by
  model validation.
- **Trusted-state mutations are auditable.** Every create/add operation
  appends to a hash-chained audit ledger.
- **Audit tampering is detectable.** Mutating any prior event or breaking
  a hash link makes `verify_chain()` return `False`.

- **Tools run only under least-privilege grants.** The TRPC compiles every
  tool request into a typed, audited allow/deny/review decision
  (default-deny; forbidden shell stays forbidden).
- **Answers come from evidence or are honestly UNKNOWN.** The Base AI Gate
  pipeline retrieves, quotes, verifies, and arbitrates — an unsupported
  claim can never produce SUCCESS.

## Base AI Gate (Gate A)

The deterministic, fully local execution path — no network, no GPU, no
torch, no model weights:

```text
TaskRequest -> Data Wall -> TRPC policy -> HER-MoE router -> DWAC plan
  -> HSGR retrieval -> deterministic answer -> claim extraction
  -> verifier -> EWA arbitration -> audit -> failure-to-eval
```

Run it (emits the full canonical artifact set under
`artifacts/base_ai_gate/<timestamp>/`):

```powershell
python scripts/run_base_ai_gate_demo.py
```

Gate B (`--with-local-model`) optionally smokes an owner-downloaded local
model; a missing model reports `SKIPPED_LOCAL_MODEL_NOT_AVAILABLE` and
never fails Gate A. No frontier-superiority claims are made anywhere:
external comparisons are `NOT_PROVEN` without a stored baseline
(`fixtures/baselines/README.md`).

## Base Model Gate (AMALI-FT-v0 stage)

The Base Model Gate turns the deterministic kernel into an
owner-controlled path toward **AMALI-FT-v0** — an AMALI-trained adapter
on top of an owner-approved open base model, governed by the AMALI
control plane. It is **not** a from-scratch model, and no claim about
GPT/Claude/DeepSeek/GLM/Qwen is made anywhere: external comparisons stay
`NOT_PROVEN` without stored baselines.

Honest current state: **code-complete**. The local corpus yields fewer
than 200 approved non-synthetic training examples, training dependencies
and base weights are owner-install/download actions that have not
happened, so the pipeline reports `DATASET_NOT_READY` /
`TRAIN_DEPS_MISSING` / `MODEL_NOT_AVAILABLE` and promotion is
`NOT_READY`. Nothing is faked.

Preflight (verify Gate A stability before model work):

```powershell
python scripts/preflight_base_model_gate.py
python scripts/run_base_model_gate_preflight.py  # alias
```

Full stage (run these in order when ready):

```powershell
# owner data intake (see docs/data/OWNER_SOURCEPACK_GUIDE.md)
python scripts/validate_owner_sourcepack.py

# eval foundation (frozen before any data/training work)
python scripts/freeze_eval_suite.py
python scripts/build_training_dataset.py
python scripts/check_contamination.py

# environment + feasibility gates (torch only inside the probe)
python scripts/probe_training_environment.py
python scripts/train_amali_adapter.py --dry-run

# owner-invoked real path (installs + download are explicit owner actions)
python -m pip install -e ".[dev,local_llm,train]"
python scripts/download_model.py --model Qwen/Qwen2.5-1.5B-Instruct --revision <PINNED_REVISION>
python scripts/train_amali_adapter.py --model Qwen/Qwen2.5-1.5B-Instruct --revision <PINNED_REVISION>

# checkpoint integrity
python scripts/register_checkpoint.py
python scripts/verify_checkpoint.py

# evaluation / promotion / comparison
python scripts/evaluate_model.py --model raw_base
python scripts/evaluate_model.py --model amali_wrapped_raw_base
python scripts/evaluate_model.py --model amali_ft_v0
python scripts/evaluate_model.py --model amali_wrapped_ft_v0
python scripts/promote_model.py amali_ft_v0
python scripts/compare_models.py --mode base_readiness   # raw_base vs amali_wrapped_raw_base (no FT needed)
python scripts/compare_models.py --mode ft_promotion     # strict three-way; refuses missing amali_wrapped_ft_v0
python scripts/run_base_ai_gate_demo.py --with-local-model --model amali_ft_v0

# card + security validation
python scripts/generate_model_card.py
python scripts/security_validate_amali_ft_v0.py
```

Every command exits nonzero on real failure, emits a typed artifact
under `artifacts/base_model_gate/<timestamp>/`, never downloads
silently, and never requires torch for Gate A.

## Layout

```text
src/amali/
  core/          ids + domain errors
  state/         typed Pydantic state models + in-memory store
  audit/         hash-chained audit ledger + permission events
  evidence/      claim construction helpers
  contracts/     TaskRequest/TaskHandle boundary schemas
  policy/        TRPC permission compiler + manifests registry
  tools/         tool requests, grant checks, executor boundary
  security/      redaction + secret handles
  data_wall/     flow-aware data admission matrix
  model_gateway/ model invocation boundary + availability gate
  router/        HER-MoE system router
  retrieval/     RICARDO-HSGR deterministic retrieval
  verifier/      claim extraction + verification
  arbiter/       RICARDO-EWA + AMALI-UGE
  activation/    AMALI-DWAC activation planner
  debate/        rule-based structured debate
  eval/          RICARDO-FEC failure-to-eval compiler
  benchmarks/    deterministic benchmark + honest comparison
  training/      training-readiness manifests + contamination check
  runtime/       Gate A orchestrator + deterministic answerer
  demo/          Base AI Gate demo emitter
docs/
  architecture/  plans + component docs
  adr/           ADR_001..ADR_004
  algorithms/    algorithm proof documentation
tests/unit/  tests/integration/
```

## Activate the virtualenv (Windows / PowerShell)

```powershell
cd D:\amali\amali-core
.\.venv\Scripts\Activate.ps1
```

## Run the tests

Install the package in editable mode for local development, then run pytest:

```powershell
python -m pip install -e ".[dev]"
python -m pytest
```

For a quick local run without installing, set `PYTHONPATH` so imports resolve
(the same approach CI uses):

```powershell
$env:PYTHONPATH = "src"
python -m pytest
```

`pyproject.toml` also sets `pythonpath = ["src"]` for pytest when you invoke it
from the repo root after install.

`uv.lock` is an optional local dependency lock; CI still uses direct `pip install`
for Gate A (see `.github/workflows/ci.yml`).

## Scope

Local-only. No network calls, no external services, no database server,
no Docker, no cloud, no frontend. Later Investor Gate Alpha stages
(policy, routing, RAG, verification, eval) are described in
[`docs/architecture/investor_gate_alpha_plan.md`](docs/architecture/investor_gate_alpha_plan.md).
