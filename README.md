# AMALI

**Evidence-governed runtime and evaluation framework for local LLM and agent systems.**

AMALI explores one core question:

> **Can AI-generated output be treated as an untrusted proposal that must pass through evidence, policy, verification, and audit boundaries before it can influence trusted state?**

AMALI is not presented as a new foundation model, AGI system, or frontier-model replacement. The current project focuses on runtime control, evidence handling, secure tool execution, auditability, and reproducible evaluation.

---

## Why AMALI?

A typical agent pipeline often looks like:

```text
Prompt
  ↓
LLM
  ↓
Answer / Tool Call
```

AMALI uses a stricter execution model:

```text
Input
  ↓
Model / Retrieval
  ↓
Untrusted Claims
  ↓
Evidence
  ↓
Verification
  ↓
Policy
  ↓
Audited Result
```

The goal is not to make a model "more intelligent" by declaration. The goal is to make AI behavior more constrained, inspectable, and measurable.

---

## Core Properties

### Evidence before trust

Generated statements are represented as typed claims and classified as:

- `KNOWN`
- `INFERRED`
- `UNKNOWN`
- `HYPOTHESIS`
- `RISK`

A claim cannot become `KNOWN` without supporting evidence.

### Explicit uncertainty

The runtime is allowed to return `UNKNOWN` instead of forcing an unsupported answer.

### Least-privilege tool execution

Tool requests pass through a typed policy boundary:

```text
Agent Request
    ↓
Typed Tool Request
    ↓
Policy Evaluation
    ↓
ALLOW / DENY / REVIEW
    ↓
Executor
```

The policy model is default-deny.

### Auditable operations

Trusted-state operations append events to a hash-linked audit ledger. Historical tampering can be detected by chain verification.

### Local-first execution

The deterministic AMALI core can run without:

- external APIs
- cloud services
- GPU
- model weights
- network access

This allows the control plane to be tested independently from model quality.

---

## Current Architecture

```text
                    Task Request
                         │
                         ▼
                ┌─────────────────┐
                │ Data Admission  │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Policy Boundary │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Model/Retrieval │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Claim Extraction│
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  Verification   │
                └────────┬────────┘
                         │
               ┌─────────┴─────────┐
               ▼                   ▼
        Audited Result         Evaluation
```

---

## What Is Implemented

| Area | Status |
|---|---|
| Typed claim model | Implemented |
| Evidence records | Implemented |
| Explicit uncertainty states | Implemented |
| Hash-linked audit ledger | Implemented |
| Typed tool permission boundary | Implemented |
| Deterministic local runtime | Implemented |
| Local-model gateway | Implemented |
| Evaluation infrastructure | Implemented |
| Training pipeline scaffolding | Implemented |
| Fine-tuned AMALI model | Experimental / not validated |
| Production readiness | Not claimed |

---

## Base AI Gate

AMALI includes a deterministic execution path that exercises the control plane without requiring an LLM.

Run:

```bash
python scripts/run_base_ai_gate_demo.py
```

Artifacts are emitted under:

```text
artifacts/base_ai_gate/<timestamp>/
```

The Base AI Gate is used to validate:

- claim lifecycle
- evidence handling
- policy enforcement
- verification
- audit integrity
- evaluation output

independently from model quality.

---

## Local Model Experiments

AMALI also supports an explicit owner-controlled local-model path.

The currently tested readiness target is:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

Model weights are not bundled with this repository and are never downloaded implicitly.

The planned comparison is:

```text
A. Raw base model

B. Raw base model + AMALI runtime

C. Fine-tuned adapter + AMALI runtime
```

---

## AMALI-FT-v0

`AMALI-FT-v0` is the planned first AMALI fine-tuned adapter.

**Current status: NOT READY**

The repository contains infrastructure for:

- base-model readiness checks
- dataset construction
- contamination checks
- training environment validation
- adapter training
- checkpoint registration and verification
- evaluation
- model comparison
- promotion
- model-card generation
- security validation

AMALI-FT-v0 should not be considered complete until a real training and evaluation cycle has been reproduced successfully.

---

## Research Question

The next major milestone is empirical evaluation:

> **Does evidence-governed execution reduce unsupported claims and successful prompt-injection attacks compared with an unmodified base-model runtime?**

Planned metrics include:

- supported-claim precision
- unsupported-claim rate
- abstention accuracy
- prompt-injection attack success rate
- secret-leakage rate
- task-completion rate
- latency
- token overhead

No superiority claim is made until these measurements are reproducible.

---

## Repository Structure

```text
src/amali/
├── audit/
├── contracts/
├── core/
├── data_wall/
├── evidence/
├── eval/
├── model_gateway/
├── policy/
├── retrieval/
├── runtime/
├── security/
├── state/
├── tools/
├── training/
└── verifier/

docs/
├── architecture/
├── algorithms/
└── adr/

tests/
├── unit/
└── integration/
```

Detailed internal component names, training procedures, architecture decisions, and algorithm notes live under `docs/`.

---

## Quick Start

```bash
git clone https://github.com/akonPAPA/AMALI.git
cd AMALI

python -m pip install -e ".[dev]"
python -m pytest
python scripts/run_base_ai_gate_demo.py
```

---

## Current Scope

AMALI is experimental engineering and evaluation software.

It is not presented as:

- a foundation model
- an AGI system
- an unrestricted autonomous agent
- a production security platform
- a replacement for existing frontier models

The current priority is to move from architecture claims to reproducible empirical evaluation.

---

## Limitations

Current limitations include:

- no production-readiness claim
- no frontier-model superiority claim
- no completed AMALI-FT-v0 benchmark yet
- no guarantee that evidence metadata eliminates adversarial manipulation
- no guarantee that policy enforcement prevents every possible tool-abuse strategy
- local-model performance depends on user hardware and model availability

Security properties must be measured empirically rather than inferred from architecture alone.

---

## Documentation

See `docs/` for:

- architecture decisions
- component designs
- algorithm notes
- model readiness
- data preparation
- evaluation
- training
- security constraints

---

## License

See `LICENSE` for the project's current license and usage terms.
