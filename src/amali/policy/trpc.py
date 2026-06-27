"""TRPC — the Tool Risk Permission Compiler.

``compile_permission`` is the heart of AMALI-IGA-02. It takes an untrusted
:class:`~amali.tools.requests.ToolInvocationRequest` (or a raw mapping) and
returns exactly one typed decision: grant, deny, review-required, or
invalid-request. Every path emits an audit event before returning.

The algorithm is deterministic and default-deny. Hard-deny checks run
*before* any permissive reasoning, in a fixed order, so that a single
failing invariant can never be overridden by a later permissive branch:

    1. request schema           -> INVALID
    2. policy integrity         -> DENY (never allow on broken policy)
    3. tool resolution          -> DENY (unknown tool)
    4. manifest integrity       -> DENY (malformed manifest)
    5. dangerous patterns       -> DENY
    6. forbidden operation      -> DENY (forbidden overrides allowed)
    7. operation not allowed    -> DENY (default-deny)
    8. secret / raw-secret      -> DENY
    9. network                  -> DENY (network denied this stage)
   10. filesystem scope         -> DENY (traversal / out-of-scope / write)
   11. risk L5                  -> DENY
   12. ttl exceeds policy       -> DENY
   13. high risk / sensitive    -> REVIEW (unless valid scoped approval)
   14. otherwise                -> GRANT (least privilege)

There is no ML scoring here and no fall-open branch.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import ValidationError

from amali.audit.events import emit_permission_audit
from amali.audit.ledger import AuditLedger
from amali.core.ids import new_decision_id
from amali.policy.approvals import ApprovalRegistry
from amali.policy.decisions import (
    InvalidPermissionRequest,
    PermissionDecision,
    PermissionDeny,
    PermissionGrant,
    PermissionReviewRequired,
)
from amali.policy.enums import (
    DataSensitivity,
    DecisionType,
    DefaultPermission,
    FilesystemPolicy,
    NetworkPolicy,
    RiskLevel,
    ToolType,
    max_risk,
    risk_at_least,
    risk_rank,
)
from amali.policy.errors import (
    ManifestValidationError,
    PolicyValidationError,
)
from amali.policy.manifests import ToolManifest, ToolPermissionPolicy
from amali.policy.reason_codes import ReasonCode
from amali.policy.registry import ManifestRegistry
from amali.policy.scopes import GrantedScope, RequestedScope
from amali.tools.requests import ToolInvocationRequest

__all__ = ["compile_permission"]

_SENSITIVE_DATA = frozenset(
    {DataSensitivity.PERSONAL, DataSensitivity.CONFIDENTIAL}
)


# --------------------------------------------------------------------------
# Normalization / detection helpers (lexical only — no filesystem access).
# --------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace so casing/spacing can't hide intent."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def _build_haystack(request: ToolInvocationRequest) -> str:
    parts: list[str] = [request.operation, request.reason]
    parts.extend(request.requested_scope.paths)
    parts.extend(request.requested_scope.domains)
    parts.extend(request.requested_scope.env)
    for value in request.metadata.values():
        if isinstance(value, str):
            parts.append(value)
    return _normalize_text(" \n ".join(parts))


def _detect_dangerous(haystack: str, patterns: list[str]) -> list[str]:
    """Return the dangerous patterns present in ``haystack``.

    Multi-token / punctuated patterns are matched as substrings; short bare
    word patterns (``nc``, ``ssh`` ...) are matched on word boundaries to
    avoid false positives inside ordinary words.
    """
    matched: list[str] = []
    for raw in patterns:
        pattern = _normalize_text(raw)
        if not pattern:
            continue
        if pattern.isalnum() and " " not in pattern:
            if re.search(rf"(?<![a-z0-9]){re.escape(pattern)}(?![a-z0-9])", haystack):
                matched.append(pattern)
        elif pattern in haystack:
            matched.append(pattern)
    return matched


def _normalize_path(path: str) -> str:
    norm = path.replace("\\", "/").strip()
    while norm.startswith("./"):
        norm = norm[2:]
    return norm.rstrip("/")


def _is_traversal(path: str) -> bool:
    norm = path.replace("\\", "/").strip()
    segments = norm.split("/")
    if ".." in segments:
        return True
    # Absolute POSIX path or Windows drive path: escapes the workspace.
    if norm.startswith("/"):
        return True
    if re.match(r"^[a-zA-Z]:", norm):
        return True
    return False


def _within(target: str, base: str) -> bool:
    t = _normalize_path(target)
    b = _normalize_path(base)
    return t == b or t.startswith(b + "/")


# --------------------------------------------------------------------------
# Public entry point.
# --------------------------------------------------------------------------


def compile_permission(
    request: ToolInvocationRequest | dict[str, Any],
    *,
    registry: ManifestRegistry,
    ledger: AuditLedger,
    approval_registry: ApprovalRegistry | None = None,
    now: datetime | None = None,
) -> PermissionDecision:
    """Compile a tool-invocation request into a typed permission decision."""
    now = now or datetime.now(timezone.utc)
    approval_registry = approval_registry or ApprovalRegistry()
    policy = registry.policy

    # 1. Request schema validation. -----------------------------------
    if isinstance(request, dict):
        try:
            request = ToolInvocationRequest(**request)
        except ValidationError as exc:
            errors = [
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
                for e in exc.errors()
            ]
            return _invalid(
                ledger,
                now=now,
                validation_errors=errors,
                task_id=_safe_get(request, "task_id"),
                actor_ref=_safe_get(request, "actor_ref"),
                tool_id=_safe_get(request, "tool_id"),
                policy_version=policy.version,
            )

    # 2. Policy integrity. Never allow on a broken policy. -------------
    try:
        policy.assert_valid()
    except PolicyValidationError:
        return _deny(
            ledger,
            request=request,
            now=now,
            reason_codes=[ReasonCode.POLICY_INVALID, ReasonCode.DEFAULT_DENY],
            policy_version=policy.version,
            manifest_version=None,
        )

    # 3. Tool resolution. ---------------------------------------------
    manifest = registry.get_tool(request.tool_id)
    if manifest is None:
        return _deny(
            ledger,
            request=request,
            now=now,
            reason_codes=[ReasonCode.TOOL_NOT_FOUND, ReasonCode.DEFAULT_DENY],
            policy_version=policy.version,
            manifest_version=None,
        )

    # 4. Manifest integrity (defensive: catch construct/mutation). -----
    try:
        manifest.assert_valid()
    except ManifestValidationError:
        return _deny(
            ledger,
            request=request,
            now=now,
            reason_codes=[ReasonCode.MANIFEST_INVALID, ReasonCode.DEFAULT_DENY],
            policy_version=policy.version,
            manifest_version=manifest.version,
        )

    effective_risk = max_risk(request.task_risk, manifest.risk_level)
    scope = request.requested_scope

    # 5. Dangerous-pattern detection. ---------------------------------
    haystack = _build_haystack(request)
    if _detect_dangerous(haystack, policy.dangerous_operation_patterns):
        return _deny(
            ledger,
            request=request,
            now=now,
            reason_codes=[
                ReasonCode.DANGEROUS_PATTERN_DETECTED,
                ReasonCode.DEFAULT_DENY,
            ],
            policy_version=policy.version,
            manifest_version=manifest.version,
        )

    # 6. Forbidden operation (manifest forbidden + policy denied). -----
    if (
        request.operation in manifest.forbidden_operations
        or request.operation in policy.denied_operations
    ):
        return _deny(
            ledger,
            request=request,
            now=now,
            reason_codes=[ReasonCode.OPERATION_FORBIDDEN],
            policy_version=policy.version,
            manifest_version=manifest.version,
        )

    # 7. Operation must be explicitly allowed (default-deny). ----------
    if request.operation not in manifest.allowed_operations:
        return _deny(
            ledger,
            request=request,
            now=now,
            reason_codes=[
                ReasonCode.OPERATION_NOT_ALLOWED,
                ReasonCode.DEFAULT_DENY,
            ],
            policy_version=policy.version,
            manifest_version=manifest.version,
        )

    # 8. Secret / raw-secret exposure. --------------------------------
    if scope.secret_access or request.data_sensitivity is DataSensitivity.SECRET:
        codes: list[ReasonCode] = []
        if request.data_sensitivity is DataSensitivity.SECRET:
            codes.append(ReasonCode.RAW_SECRET_FORBIDDEN)
        if scope.secret_access:
            codes.append(ReasonCode.SECRET_ACCESS_DENIED)
        return _deny(
            ledger,
            request=request,
            now=now,
            reason_codes=codes,
            policy_version=policy.version,
            manifest_version=manifest.version,
        )

    # 9. Network scope (denied by default this stage). ----------------
    if scope.network:
        codes = [ReasonCode.NETWORK_DENIED]
        if manifest.network_policy is NetworkPolicy.UNRESTRICTED_FORBIDDEN:
            codes.insert(0, ReasonCode.UNRESTRICTED_NETWORK_FORBIDDEN)
        elif manifest.network_policy is NetworkPolicy.ALLOWLIST:
            allowed = set(manifest.allowed_domains or [])
            if not set(scope.domains).issubset(allowed):
                codes.insert(0, ReasonCode.DOMAIN_NOT_ALLOWLISTED)
        return _deny(
            ledger,
            request=request,
            now=now,
            reason_codes=codes,
            policy_version=policy.version,
            manifest_version=manifest.version,
        )

    # 10. Filesystem scope. -------------------------------------------
    fs_codes = _check_filesystem(scope, manifest)
    if fs_codes:
        return _deny(
            ledger,
            request=request,
            now=now,
            reason_codes=fs_codes,
            policy_version=policy.version,
            manifest_version=manifest.version,
        )

    # 11. Risk L5 is forbidden outright. ------------------------------
    if effective_risk is RiskLevel.L5:
        return _deny(
            ledger,
            request=request,
            now=now,
            reason_codes=[ReasonCode.RISK_TOO_HIGH, ReasonCode.DEFAULT_DENY],
            policy_version=policy.version,
            manifest_version=manifest.version,
        )

    # 12. TTL ceiling. Requesting more than allowed is a hard deny. ----
    max_ttl = policy.max_grant_ttl_seconds
    if manifest.max_ttl_seconds is not None:
        max_ttl = min(max_ttl, manifest.max_ttl_seconds)
    if request.ttl_seconds is not None and request.ttl_seconds > max_ttl:
        return _deny(
            ledger,
            request=request,
            now=now,
            reason_codes=[ReasonCode.TTL_EXCEEDS_POLICY, ReasonCode.DEFAULT_DENY],
            policy_version=policy.version,
            manifest_version=manifest.version,
        )

    # 13. Review gate: high risk or sensitive data needs owner approval.
    high_risk = risk_at_least(
        effective_risk, policy.owner_approval_required_from_risk
    )
    sensitive = request.data_sensitivity in _SENSITIVE_DATA
    if high_risk or sensitive:
        approval = approval_registry.get(request.approval_ref)
        valid_approval = approval is not None and approval.covers(
            tool_id=request.tool_id,
            operation=request.operation,
            paths=scope.paths,
            domains=scope.domains,
            effective_risk=effective_risk,
            now=now,
        )
        if not valid_approval:
            review_codes: list[ReasonCode] = []
            if high_risk:
                review_codes.append(ReasonCode.HIGH_RISK_REQUIRES_REVIEW)
            if sensitive:
                review_codes.append(ReasonCode.DATA_MINIMIZATION_REQUIRED)
            if request.approval_ref:
                review_codes.append(ReasonCode.APPROVAL_OUT_OF_SCOPE)
            else:
                review_codes.append(ReasonCode.APPROVAL_REQUIRED)
            return _review(
                ledger,
                request=request,
                now=now,
                reason_codes=review_codes,
                policy_version=policy.version,
                manifest_version=manifest.version,
                effective_risk=effective_risk,
            )

    # 14. Grant (least privilege). ------------------------------------
    if manifest.max_ttl_seconds == 0:
        # A zero-ttl tool can never be granted; treat as deny defensively.
        return _deny(
            ledger,
            request=request,
            now=now,
            reason_codes=[ReasonCode.TTL_EXCEEDS_POLICY, ReasonCode.DEFAULT_DENY],
            policy_version=policy.version,
            manifest_version=manifest.version,
        )
    ttl = request.ttl_seconds if request.ttl_seconds is not None else max_ttl
    grant_codes = _grant_reason_codes(manifest)
    return _grant(
        ledger,
        request=request,
        now=now,
        reason_codes=grant_codes,
        policy_version=policy.version,
        manifest_version=manifest.version,
        effective_risk=effective_risk,
        ttl=ttl,
        granted_scope=GrantedScope(
            paths=[_normalize_path(p) for p in scope.paths],
            domains=[],
            env=[],
            write=False,
            network=False,
            secret_access=False,
        ),
    )


# --------------------------------------------------------------------------
# Sub-checks.
# --------------------------------------------------------------------------


def _check_filesystem(
    scope: RequestedScope, manifest: ToolManifest
) -> list[ReasonCode]:
    """Return deny reason codes for a bad filesystem scope, else []."""
    # Path traversal / absolute escape always loses, regardless of policy.
    for path in scope.paths:
        if _is_traversal(path):
            return [ReasonCode.PATH_TRAVERSAL_DENIED]

    # Write requested against a non-write tool.
    if scope.write and manifest.filesystem_policy is not (
        FilesystemPolicy.WORKSPACE_WRITE
    ):
        return [ReasonCode.WRITE_SCOPE_DENIED]

    if not scope.paths:
        return []

    if manifest.filesystem_policy is FilesystemPolicy.NONE:
        return [ReasonCode.FILESYSTEM_SCOPE_DENIED]

    if manifest.filesystem_policy is FilesystemPolicy.EXPLICIT_PATHS_ONLY:
        allowed = manifest.allowed_paths or []
        for path in scope.paths:
            if not any(_within(path, base) for base in allowed):
                return [ReasonCode.FILESYSTEM_SCOPE_DENIED]
    # workspace_read / workspace_write: any non-traversal relative path is
    # in-workspace by construction (traversal already rejected above).
    return []


def _grant_reason_codes(manifest: ToolManifest) -> list[ReasonCode]:
    codes: list[ReasonCode] = []
    if (
        manifest.tool_type is ToolType.FILESYSTEM
        and manifest.default_permission is DefaultPermission.READ_ONLY
    ):
        codes.append(ReasonCode.SAFE_READONLY_GRANTED)
    if manifest.sandbox_required:
        codes.append(ReasonCode.SANDBOX_REQUIRED)
        codes.append(ReasonCode.SANDBOX_GRANT_CREATED)
    if not codes:
        codes.append(ReasonCode.POLICY_ALLOW_LOW_RISK)
    return codes


# --------------------------------------------------------------------------
# Decision constructors (each emits exactly one audit event).
# --------------------------------------------------------------------------


def _audit_payload(request: ToolInvocationRequest) -> dict[str, Any]:
    """A minimal, non-secret summary of the request for the audit trail."""
    return {
        "request_source": request.request_source.value,
        "requested_paths": len(request.requested_scope.paths),
        "requested_domains": len(request.requested_scope.domains),
        "write": request.requested_scope.write,
        "network": request.requested_scope.network,
        "secret_access": request.requested_scope.secret_access,
        "ttl_requested": request.ttl_seconds,
    }


def _deny(
    ledger: AuditLedger,
    *,
    request: ToolInvocationRequest,
    now: datetime,
    reason_codes: list[ReasonCode],
    policy_version: str,
    manifest_version: str | None,
) -> PermissionDeny:
    decision_id = new_decision_id()
    audit_id = emit_permission_audit(
        ledger,
        decision_type=DecisionType.DENY,
        decision_id=decision_id,
        task_id=request.task_id,
        actor_ref=request.actor_ref,
        tool_id=request.tool_id,
        operation=request.operation,
        reason_codes=reason_codes,
        risk_level=request.task_risk,
        data_sensitivity=request.data_sensitivity,
        policy_version=policy_version,
        manifest_version=manifest_version,
        redacted_payload=_audit_payload(request),
    )
    return PermissionDeny(
        decision_id=decision_id,
        audit_event_id=audit_id,
        task_id=request.task_id,
        actor_ref=request.actor_ref,
        tool_id=request.tool_id,
        operation=request.operation,
        denied_scope=request.requested_scope,
        risk_level=request.task_risk,
        data_sensitivity=request.data_sensitivity,
        reason_codes=reason_codes,
        policy_version=policy_version,
        manifest_version=manifest_version,
        created_at=now,
    )


def _review(
    ledger: AuditLedger,
    *,
    request: ToolInvocationRequest,
    now: datetime,
    reason_codes: list[ReasonCode],
    policy_version: str,
    manifest_version: str,
    effective_risk: RiskLevel,
) -> PermissionReviewRequired:
    decision_id = new_decision_id()
    audit_id = emit_permission_audit(
        ledger,
        decision_type=DecisionType.REVIEW_REQUIRED,
        decision_id=decision_id,
        task_id=request.task_id,
        actor_ref=request.actor_ref,
        tool_id=request.tool_id,
        operation=request.operation,
        reason_codes=reason_codes,
        risk_level=effective_risk,
        data_sensitivity=request.data_sensitivity,
        policy_version=policy_version,
        manifest_version=manifest_version,
        redacted_payload=_audit_payload(request),
    )
    return PermissionReviewRequired(
        decision_id=decision_id,
        audit_event_id=audit_id,
        task_id=request.task_id,
        actor_ref=request.actor_ref,
        tool_id=request.tool_id,
        operation=request.operation,
        requested_scope=request.requested_scope,
        risk_level=effective_risk,
        data_sensitivity=request.data_sensitivity,
        reason_codes=reason_codes,
        required_approval_type="owner",
        policy_version=policy_version,
        manifest_version=manifest_version,
        created_at=now,
    )


def _grant(
    ledger: AuditLedger,
    *,
    request: ToolInvocationRequest,
    now: datetime,
    reason_codes: list[ReasonCode],
    policy_version: str,
    manifest_version: str,
    effective_risk: RiskLevel,
    ttl: int,
    granted_scope: GrantedScope,
) -> PermissionGrant:
    decision_id = new_decision_id()
    audit_id = emit_permission_audit(
        ledger,
        decision_type=DecisionType.GRANT,
        decision_id=decision_id,
        task_id=request.task_id,
        actor_ref=request.actor_ref,
        tool_id=request.tool_id,
        operation=request.operation,
        reason_codes=reason_codes,
        risk_level=effective_risk,
        data_sensitivity=request.data_sensitivity,
        policy_version=policy_version,
        manifest_version=manifest_version,
        redacted_payload=_audit_payload(request),
    )
    return PermissionGrant(
        decision_id=decision_id,
        audit_event_id=audit_id,
        task_id=request.task_id,
        actor_ref=request.actor_ref,
        tool_id=request.tool_id,
        operation=request.operation,
        granted_scope=granted_scope,
        risk_level=effective_risk,
        data_sensitivity=request.data_sensitivity,
        ttl_seconds=ttl,
        expires_at=now + timedelta(seconds=ttl),
        reason_codes=reason_codes,
        policy_version=policy_version,
        manifest_version=manifest_version,
        created_at=now,
    )


def _invalid(
    ledger: AuditLedger,
    *,
    now: datetime,
    validation_errors: list[str],
    task_id: str | None,
    actor_ref: str | None,
    tool_id: str | None,
    policy_version: str | None,
) -> InvalidPermissionRequest:
    decision_id = new_decision_id()
    reason_codes = [ReasonCode.REQUEST_SCHEMA_INVALID]
    audit_id = emit_permission_audit(
        ledger,
        decision_type=DecisionType.INVALID_REQUEST,
        decision_id=decision_id,
        task_id=task_id,
        actor_ref=actor_ref,
        tool_id=tool_id,
        operation=None,
        reason_codes=reason_codes,
        risk_level=None,
        data_sensitivity=None,
        policy_version=policy_version,
        manifest_version=None,
        redacted_payload={"validation_error_count": len(validation_errors)},
    )
    return InvalidPermissionRequest(
        decision_id=decision_id,
        audit_event_id=audit_id,
        task_id=task_id,
        actor_ref=actor_ref,
        tool_id=tool_id,
        validation_errors=validation_errors,
        reason_codes=reason_codes,
        policy_version=policy_version,
        created_at=now,
    )


def _safe_get(request: ToolInvocationRequest | dict[str, Any], key: str) -> Any:
    if isinstance(request, dict):
        value = request.get(key)
        return value if isinstance(value, str) else None
    return getattr(request, key, None)
