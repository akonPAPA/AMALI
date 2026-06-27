# Investor Gate Alpha — Architecture Plan

Investor Gate Alpha (IGA) is a sequence of small, local-only kernels. Each
kernel proves one architectural property in code and tests before the next
is built. The goal is to demonstrate that AMALI's core invariants are real
and enforceable — not to ship a product.

Principles that hold across every stage:

- Generated text is never trusted state.
- Claims are explicitly classified; evidence is explicit.
- Trusted-state mutations are auditable and tamper-evident.
- No agent gets direct, unmediated tool or external-write authority.
- Everything is local-only until a later stage explicitly justifies more.

## Stages

### AMALI-IGA-00 — Local Source-of-Truth Kernel
Project structure, packaging, stable IDs, domain errors, and an in-memory
task store. Establishes a single local source of truth for tasks and the
discipline that state lives in typed records, not in free text.

**Status: implemented.**

### AMALI-IGA-01 — Epistemic State + Audit Kernel
Typed epistemic state models (`ClaimRecord`, `EvidenceRecord`, `TaskState`,
`AMALIResponse`) and a hash-chained audit ledger. Proves: generated text is
not KNOWN without evidence, claims are classified, and state mutations are
auditable and tamper-evident.

**Status: implemented.**

### AMALI-IGA-02 — Policy + Risk + Tool Permission Kernel
A local policy layer that decides what an actor/task is permitted to do,
classifies risk, and gates tool permissions. No tool is executed directly;
the kernel produces permission decisions that a future execution layer must
honor. **Status: planned.**

### AMALI-IGA-03 — HER-MoE Local Routing Kernel
A local, deterministic routing kernel that selects among model/expert
profiles based on task features and recorded uncertainty. Routing decisions
are recorded as `route_decision_refs` and audited. No neural MoE training;
routing is configuration- and rule-driven. **Status: planned.**

### AMALI-IGA-04 — Evidence / RAG Kernel
A local evidence-retrieval kernel that turns sources into `EvidenceRecord`s
with trust, sensitivity, and poison/injection risk. Retrieval is local
(files/fixtures), with explicit provenance and hashing. **Status: planned.**

### AMALI-IGA-05 — Verifier + EWA Arbiter Kernel
A verification kernel that checks claims against attached evidence and an
Evidence-Weighted Arbiter (EWA) that decides final epistemic levels and
flags contradictions. Promotion of a claim to KNOWN is verifier-gated.
**Status: planned.**

### AMALI-IGA-06 — Failure-to-Eval Kernel
A kernel that converts failures (blocked, partial, contradicted, low
confidence) into structured evaluation records for offline review. Closes
the loop from runtime outcomes to evaluation. **Status: planned.**

### AMALI-IGA-07 — Investor Technical Demo Pack
A local, reproducible demonstration that walks through the prior kernels on
fixture data and renders the epistemic + audit story end to end. No hosting,
no external dependencies. **Status: planned.**

## Out of scope for Investor Gate Alpha

GitHub workflows, Docker, FastAPI/HTTP servers, frontend, database servers,
cloud services, external APIs, model training/fine-tuning, neural MoE,
JEPA/PIM implementations, and any SaaS/billing framing.
