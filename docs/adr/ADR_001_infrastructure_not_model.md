# ADR 001 — AMALI is Infrastructure, Not a Model

**Status:** Accepted

## Context

It is tempting to equate "AI system" with "the model." AMALI's value,
however, is in the infrastructure around models: governance, epistemic
state, evidence handling, auditability, routing, and evaluation. If the
first artifacts we build are model wrappers, the architecture's real
guarantees (what is trusted, what is auditable) never get proven.

## Decision

The kernel is built as **infrastructure first**. The earliest code defines
typed state, explicit evidence classification, and a tamper-evident audit
ledger — with **no model training, no neural MoE, and no inference engine**.
Models are treated as untrusted text generators whose output must enter the
system as classified, evidence-gated claims.

## Consequences

- Architectural invariants (trusted state, auditability) are testable now,
  independent of any specific model.
- Model choice/training is deferred and pluggable behind the state and
  evidence boundary.
- More upfront structure (records, ledger, validation) before any
  "intelligent" behavior is visible — accepted as the cost of provable
  correctness.
