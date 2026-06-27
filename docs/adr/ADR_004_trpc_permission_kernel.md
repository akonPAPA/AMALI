# ADR 004 — TRPC Permission Kernel and No Direct Tool Execution

**Status:** Accepted
**Stage:** AMALI-IGA-02
**Relates to:** ADR 002 (No Direct Tool Execution)

## Context

ADR 002 established that the kernel contains no direct tool/shell/external
execution layer and deferred permission and execution to a later,
owner-governed stage. AMALI-IGA-02 implements the *permission-decision* half
of that promise — without implementing execution.

The threat model is concrete. A model or agent (or a document it retrieved)
must never be able to:

- run a shell command or spawn a process;
- reach the network by default;
- read or write the filesystem outside an explicit, bounded scope;
- escalate its own permissions by setting authority fields in the request;
- have a raw secret routed into model/agent context, a log, or an audit
  event;
- forge an approval with a plain string;
- escape the workspace via path traversal.

## Decision

Introduce the **Tool Risk Permission Compiler (TRPC)**: a deterministic,
default-deny function `compile_permission` that turns an untrusted
`ToolInvocationRequest` into exactly one typed, audited decision —
`PermissionGrant`, `PermissionDeny`, `PermissionReviewRequired`, or
`InvalidPermissionRequest`.

Key design commitments:

1. **Default deny.** A grant is produced only when every hard check passes.
   The decision order runs all hard-deny checks before any permissive branch.
2. **Closed vocabularies.** Risk levels, tool types, scope policies, and
   reason codes are enums, so an unknown value cannot be smuggled through.
3. **Manifests and policy are versioned and strictly validated.** A malformed
   manifest or policy resolves to deny/invalid, never allow. The compiler
   re-validates defensively (`assert_valid`) so a bypassed (`model_construct`)
   or mutated manifest is still caught.
4. **Authority is never taken from the request.** Effective risk is the *max*
   of task risk and tool risk; request `metadata` and `request_source` confer
   no authority and cannot lower risk.
5. **Secrets are structurally absent.** No raw secret is retrieved this stage;
   `SecretHandle` never reveals a value; all audit metadata is redacted before
   it is written.
6. **Every decision is audited** on the existing hash-chained ledger.
7. **Execution stays absent.** The executor boundary consumes a grant but
   performs no real action and refuses any call lacking a valid, matching,
   unexpired grant.

## Consequences

- AMALI now has an auditable, testable permission core that can gate a future
  sandbox executor, while remaining local-only with no network, no shell, and
  no secret retrieval.
- The grant object and `check_grant` define the exact contract the future
  executor must honor before any real execution.
- Reason codes give tests and audits a stable, machine-readable explanation
  for every decision.

## Limitations (explicitly not solved here)

- Path checks are lexical only; a real executor must re-resolve paths against
  a real workspace root (including symlink handling) before disk access.
- Approvals and the audit ledger are in-memory; durable, owner-governed
  storage is a later stage.
- No sandbox exists yet; `sandbox_required` is asserted and recorded only.
