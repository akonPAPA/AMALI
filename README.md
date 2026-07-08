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

```powershell
python -m pytest
```

`pytest` is configured (in `pyproject.toml`) with `pythonpath = ["src"]`,
so no install step is required to run the suite.

## Scope

Local-only. No network calls, no external services, no database server,
no Docker, no cloud, no frontend. Later Investor Gate Alpha stages
(policy, routing, RAG, verification, eval) are described in
[`docs/architecture/investor_gate_alpha_plan.md`](docs/architecture/investor_gate_alpha_plan.md).
