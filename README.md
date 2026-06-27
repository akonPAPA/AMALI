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

## Layout

```text
src/amali/
  core/      ids + domain errors
  state/     typed Pydantic state models + in-memory store
  audit/     hash-chained audit ledger
  evidence/  claim construction helpers
docs/
  architecture/investor_gate_alpha_plan.md
  adr/ADR_001..ADR_003
tests/unit/  unit tests
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
