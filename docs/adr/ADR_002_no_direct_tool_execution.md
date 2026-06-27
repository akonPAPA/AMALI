# ADR 002 — No Direct Tool Execution

**Status:** Accepted

## Context

A common failure mode in agent systems is letting a model directly call
tools or write to external systems. This removes the owner's control,
erases the boundary between "intent" and "authorized action," and makes
audit and rollback impossible.

## Decision

The kernel contains **no direct tool/shell/external-write execution layer**.
Trusted-state mutations go through explicit store methods that validate
inputs and append audit events. Tool permission and execution are deferred
to later, owner-governed stages (IGA-02+), where permission decisions are
produced separately from any execution and every action is audited.

## Consequences

- Every trusted-state change in this kernel is mediated and auditable.
- Models and (future) agents cannot mutate trusted business state directly.
- Real tool execution requires an explicit, owner-approved permission and
  execution layer added later — it is intentionally absent here.
