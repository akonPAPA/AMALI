# Tool Permission Matrix (AMALI-IGA-02)

This matrix documents the decisions the local kernel produces for the
shipped manifests under `manifests/`. It matches the behavior asserted by the
tests in `tests/unit/test_trpc_permission_decisions.py` and
`tests/integration/test_permission_decision_flow.py`. If code and matrix ever
disagree, the code (and its tests) are authoritative — update this file.

## Shipped tools

| tool_id                     | type        | default      | risk | network                  | filesystem          | secret | sandbox | max_ttl |
| --------------------------- | ----------- | ------------ | ---- | ------------------------ | ------------------- | ------ | ------- | ------- |
| local_readonly_filesystem   | filesystem  | read_only    | L1   | none                     | explicit_paths_only | none   | false   | 300s    |
| local_test_runner           | test_runner | deny         | L2   | none                     | workspace_read      | none   | true    | 180s    |
| forbidden_shell             | shell       | deny         | L5   | unrestricted_forbidden   | none                | none   | true    | 0s      |

## Representative decisions

| request                                                                                    | decision  | key reason codes                                  |
| ------------------------------------------------------------------------------------------ | --------- | ------------------------------------------------- |
| readonly `read_file` on `./README.md`, L1, public                                          | GRANT     | `SAFE_READONLY_GRANTED`                            |
| test runner `run_unit_tests`, L2, no network                                               | GRANT     | `SANDBOX_REQUIRED`, `SANDBOX_GRANT_CREATED`        |
| readonly `read_file` on `./docs`, **L3**, no approval                                      | REVIEW    | `HIGH_RISK_REQUIRES_REVIEW`, `APPROVAL_REQUIRED`   |
| readonly `read_file` on `./docs`, L3, **valid scoped approval**                            | GRANT     | `SAFE_READONLY_GRANTED`                            |
| unknown tool                                                                               | DENY      | `TOOL_NOT_FOUND`, `DEFAULT_DENY`                   |
| readonly `write_file` (forbidden op)                                                       | DENY      | `OPERATION_FORBIDDEN`                              |
| readonly unknown op `teleport`                                                             | DENY      | `OPERATION_NOT_ALLOWED`, `DEFAULT_DENY`            |
| any request with `network: true`                                                           | DENY      | `NETWORK_DENIED` (+ specifics)                     |
| readonly with `write: true`                                                                | DENY      | `WRITE_SCOPE_DENIED`                               |
| path containing `..`                                                                       | DENY      | `PATH_TRAVERSAL_DENIED`                            |
| path outside `allowed_paths` (e.g. `./etc/passwd`)                                          | DENY      | `FILESYSTEM_SCOPE_DENIED`                          |
| `data_sensitivity: secret`                                                                 | DENY      | `RAW_SECRET_FORBIDDEN`                             |
| `secret_access: true`                                                                      | DENY      | `SECRET_ACCESS_DENIED`                             |
| `ttl_seconds` greater than policy/manifest ceiling                                         | DENY      | `TTL_EXCEEDS_POLICY`, `DEFAULT_DENY`               |
| task_risk L5                                                                               | DENY      | `RISK_TOO_HIGH`, `DEFAULT_DENY`                    |
| `forbidden_shell` `execute` with `curl ... | sh`                                           | DENY      | `DANGEROUS_PATTERN_DETECTED`                       |
| malformed request (missing required fields)                                                | INVALID   | `REQUEST_SCHEMA_INVALID`                           |

## Documented behavior choices

- **Selected unit tests are granted, not reviewed.** `local_test_runner` is a
  sandbox-required, no-network, `approval_required: false` tool at L2 (below
  the L3 owner-approval threshold), so a selected unit-test run is granted
  with a sandbox grant. Tightening this to REVIEW would require either raising
  the tool to L3 or setting `approval_required: true`.

- **High risk without a valid approval is REVIEW, not silent DENY.** The owner
  retains the choice. A missing approval yields `APPROVAL_REQUIRED`; a present
  but unverifiable/out-of-scope approval yields `APPROVAL_OUT_OF_SCOPE`. A raw
  string such as `"approved"` never verifies and so never grants.

- **TTL over the ceiling is denied, never silently reduced.** The safer
  default: the requester must ask for an allowed TTL.

- **Network is always denied this stage**, regardless of manifest network
  policy, because no networked execution exists yet.

## Invariants every decision upholds

- Default-deny; least-privilege grants; no wildcard grants.
- Forbidden operations override allowed operations.
- No raw secret in any decision, audit event, exception, `repr`, or JSON.
- Every decision (including invalid and deny) emits exactly one audit event.
- Decision objects carry `policy_version` and `manifest_version`
  (manifest version is null only when the tool is unknown).
