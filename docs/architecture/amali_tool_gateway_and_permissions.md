# AMALI Tool Gateway and Permissions (AMALI-IGA-02)

## Why this exists

AMALI must never allow any of these paths:

```text
LLM output                      -> direct shell execution
Agent                           -> unrestricted filesystem / network / database
Tool response                   -> trusted policy decision without validation
Retrieved document              -> system / developer instructions
Web page / email / PDF / code   -> tool permission expansion
```

The Tool Risk Permission Compiler (**TRPC**) is the control point that makes
the *allowed* path the only path:

```text
Agent requests a tool
  -> task / risk / data / tool policy is checked
  -> TRPC compiles a least-privilege PermissionGrant, or a Deny / Review / Invalid
  -> an audit event is emitted (always)
  -> only a later sandbox / tool executor may consume the grant
  -> no raw secret is ever exposed to the model / agent context
```

## What AMALI-IGA-02 implements

- **Versioned manifests** (`manifests/tools/*.yaml`) and a single
  **permission policy** (`manifests/policies/*.yaml`), loaded locally with no
  network by `ManifestRegistry`.
- **Strict validation** of manifests, policy, and requests
  (`amali.policy.manifests`, `amali.tools.requests`). A malformed manifest or
  policy can never resolve to "allow".
- **A deterministic decision kernel** `compile_permission`
  (`amali.policy.trpc`) returning exactly one of four typed results:
  `PermissionGrant`, `PermissionDeny`, `PermissionReviewRequired`,
  `InvalidPermissionRequest`.
- **Audit-backed decisions**: every decision (allow / deny / review /
  invalid) appends one redacted event to the existing hash-chained
  `AuditLedger` (`amali.audit.events`).
- **A refusal-only executor boundary** (`amali.tools.executor_boundary`):
  it consumes a grant but performs **no** real shell, network, or filesystem
  action. Without a valid, matching, unexpired grant it raises
  `PermissionRequiredError`.
- **Conservative redaction** and **non-revealing secret handles**
  (`amali.security`).

## What AMALI-IGA-02 does NOT implement (by design)

- No real tool execution, no shell command running, no process spawning.
- No network calls of any kind. Network is denied by default this stage.
- No real secret retrieval. `SecretHandle.reveal()` raises
  `NotImplementedError`.
- No owner-approval UI. Approvals exist only as a typed, registry-verified
  object used to prove the scope-checking property.
- No sandbox. `sandbox_required` is recorded and reasoned about, but the
  sandbox itself is a later stage.

## Decision algorithm (deterministic, default-deny)

Hard-deny checks run first, in a fixed order; a single failing invariant can
never be overridden by a later permissive branch. See the module docstring in
`src/amali/policy/trpc.py` for the canonical 14-step ordering. Summary:

1. request schema -> `INVALID`
2. policy integrity -> `DENY`
3. tool resolution -> `DENY` (unknown tool)
4. manifest integrity -> `DENY`
5. dangerous patterns -> `DENY`
6. forbidden operation -> `DENY` (forbidden overrides allowed)
7. operation not allowed -> `DENY` (default-deny)
8. secret / raw-secret -> `DENY`
9. network -> `DENY` (network denied this stage)
10. filesystem scope (traversal / out-of-scope / write) -> `DENY`
11. risk L5 -> `DENY`
12. ttl exceeds policy -> `DENY`
13. high risk or sensitive data, no valid scoped approval -> `REVIEW`
14. otherwise -> `GRANT` (least privilege)

## Adding a new tool manifest safely

1. Copy an existing manifest under `manifests/tools/` and give it a unique
   `tool_id` and a `version`.
2. Keep `default_permission: deny` unless the tool is genuinely read-only and
   `risk_level <= L1`.
3. Keep `audit_required: true` (the kernel rejects anything else).
4. List `allowed_operations` explicitly. Anything not listed is denied.
5. List `forbidden_operations` for anything dangerous; forbidden always wins.
6. Set `network_policy: none` unless a later networked stage is in scope.
7. For filesystem tools, prefer `explicit_paths_only` with a minimal
   `allowed_paths`. Never include `..` or absolute/host paths.
8. For shell / test_runner / static_analysis / db tools, set
   `sandbox_required: true` (the kernel requires it).
9. For any `secret_access` other than `none`, set `approval_required: true`
   (or use `handle_only` on a `secret_broker` tool).

## Testing a new tool manifest

- Add a load + validation test mirroring
  `tests/unit/test_policy_manifest_validation.py` (valid loads; each broken
  variant is rejected).
- Add allow/deny/review cases mirroring
  `tests/unit/test_trpc_permission_decisions.py`.
- Add negative cases for the tool's specific dangers in
  `tests/unit/test_trpc_negative_security.py`.
- Assert reason codes, not prose. Assert an audit event was emitted.

## Known limitations / remaining risks

- Path checks are **lexical only** (no filesystem access, no symlink
  resolution). They reject `..`, absolute, and drive-letter paths, but a real
  executor stage must still resolve and re-check paths against a real
  workspace root before touching the disk.
- Approvals are local and in-memory; there is no durable owner-approval store
  or UI yet.
- The audit ledger is in-memory (as in IGA-00/01); durable persistence is a
  later concern.
- "Sandbox required" is asserted and granted, but no sandbox executes
  anything yet.

## Relationship to the future sandbox executor

The grant produced here is the *only* token a future executor may consume.
`amali.tools.grants.check_grant` already defines the exact match (tool,
operation, expiry, scope, no escalation). The future executor must call this
check (or an equivalent) before any real execution, and must run inside the
sandbox the manifest demands.
