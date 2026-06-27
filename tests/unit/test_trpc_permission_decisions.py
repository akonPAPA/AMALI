"""TRPC allow / deny / review decision behavior."""

from __future__ import annotations

from datetime import timedelta

from amali.policy.approvals import (
    ApprovalRef,
    ApprovalRegistry,
    ApprovalScope,
)
from amali.policy.decisions import (
    InvalidPermissionRequest,
    PermissionDeny,
    PermissionGrant,
    PermissionReviewRequired,
)
from amali.policy.enums import DecisionType
from amali.policy.reason_codes import ReasonCode
from amali.policy.trpc import compile_permission

from tests.conftest import make_request


def _compile(registry, ledger, now, **overrides):
    req = make_request(**overrides)
    return compile_permission(req, registry=registry, ledger=ledger, now=now)


# --- ALLOW ----------------------------------------------------------------


def test_readonly_read_file_allowed(registry, ledger, now):
    d = _compile(
        registry, ledger, now, scope={"paths": ["./README.md"]}
    )
    assert isinstance(d, PermissionGrant)
    assert d.decision is DecisionType.GRANT
    assert ReasonCode.SAFE_READONLY_GRANTED in d.reason_codes
    # Grant ttl is bounded by policy/manifest and audit-bound.
    assert d.ttl_seconds <= 300
    assert d.expires_at == now + timedelta(seconds=d.ttl_seconds)
    assert d.audit_event_id.startswith("audit_")
    assert d.policy_version == "0.1.0"
    assert d.manifest_version == "0.1.0"


def test_grant_scope_is_least_privilege(registry, ledger, now):
    d = _compile(registry, ledger, now, scope={"paths": ["./docs"]})
    assert isinstance(d, PermissionGrant)
    assert d.granted_scope.write is False
    assert d.granted_scope.network is False
    assert d.granted_scope.secret_access is False
    assert d.granted_scope.domains == []
    assert d.granted_scope.paths == ["docs"]


def test_test_runner_selected_unit_tests_granted_with_sandbox(
    registry, ledger, now
):
    # Documented behavior: a selected unit-test run on a sandbox-required,
    # no-network, no-approval tool is GRANTED with a sandbox grant.
    d = _compile(
        registry,
        ledger,
        now,
        tool_id="local_test_runner",
        operation="run_unit_tests",
        task_risk="L2",
    )
    assert isinstance(d, PermissionGrant)
    assert ReasonCode.SANDBOX_GRANT_CREATED in d.reason_codes
    assert ReasonCode.SANDBOX_REQUIRED in d.reason_codes
    assert d.ttl_seconds <= 180  # manifest ceiling beats policy ceiling


def test_grant_does_not_expose_request_secrets(registry, ledger, now):
    d = _compile(
        registry,
        ledger,
        now,
        scope={"paths": ["./README.md"]},
        reason="read an allowed workspace file",
    )
    blob = d.model_dump_json()
    assert "[REDACTED]" not in blob  # nothing needed redacting
    # The free-text human reason is never carried into the decision object.
    assert "read an allowed workspace file" not in blob


# --- DENY -----------------------------------------------------------------


def test_unknown_tool_denied(registry, ledger, now):
    d = _compile(registry, ledger, now, tool_id="does_not_exist")
    assert isinstance(d, PermissionDeny)
    assert ReasonCode.TOOL_NOT_FOUND in d.reason_codes
    assert d.manifest_version is None


def test_forbidden_operation_denied(registry, ledger, now):
    d = _compile(registry, ledger, now, operation="write_file")
    assert isinstance(d, PermissionDeny)
    assert ReasonCode.OPERATION_FORBIDDEN in d.reason_codes


def test_operation_not_allowed_denied(registry, ledger, now):
    d = _compile(registry, ledger, now, operation="teleport")
    assert isinstance(d, PermissionDeny)
    assert ReasonCode.OPERATION_NOT_ALLOWED in d.reason_codes


def test_network_request_denied_by_default(registry, ledger, now):
    d = _compile(
        registry,
        ledger,
        now,
        tool_id="local_test_runner",
        operation="run_unit_tests",
        task_risk="L2",
        scope={"network": True},
    )
    assert isinstance(d, PermissionDeny)
    assert ReasonCode.NETWORK_DENIED in d.reason_codes


def test_write_against_readonly_denied(registry, ledger, now):
    d = _compile(
        registry,
        ledger,
        now,
        scope={"paths": ["./README.md"], "write": True},
    )
    assert isinstance(d, PermissionDeny)
    assert ReasonCode.WRITE_SCOPE_DENIED in d.reason_codes


def test_path_traversal_denied(registry, ledger, now):
    d = _compile(
        registry, ledger, now, scope={"paths": ["./docs/../../secrets"]}
    )
    assert isinstance(d, PermissionDeny)
    assert ReasonCode.PATH_TRAVERSAL_DENIED in d.reason_codes


def test_path_outside_allowed_denied(registry, ledger, now):
    d = _compile(registry, ledger, now, scope={"paths": ["./etc/passwd"]})
    assert isinstance(d, PermissionDeny)
    assert ReasonCode.FILESYSTEM_SCOPE_DENIED in d.reason_codes


def test_task_risk_l5_denied(registry, ledger, now):
    d = _compile(
        registry, ledger, now, scope={"paths": ["./README.md"]}, task_risk="L5"
    )
    assert isinstance(d, PermissionDeny)
    assert ReasonCode.RISK_TOO_HIGH in d.reason_codes


def test_secret_data_sensitivity_denied(registry, ledger, now):
    d = _compile(
        registry,
        ledger,
        now,
        scope={"paths": ["./README.md"]},
        data_sensitivity="secret",
    )
    assert isinstance(d, PermissionDeny)
    assert ReasonCode.RAW_SECRET_FORBIDDEN in d.reason_codes


def test_secret_access_request_denied(registry, ledger, now):
    d = _compile(
        registry,
        ledger,
        now,
        scope={"paths": ["./README.md"], "secret_access": True},
    )
    assert isinstance(d, PermissionDeny)
    assert ReasonCode.SECRET_ACCESS_DENIED in d.reason_codes


def test_ttl_exceeding_policy_denied_never_granted(registry, ledger, now):
    d = _compile(
        registry,
        ledger,
        now,
        scope={"paths": ["./README.md"]},
        ttl_seconds=99999,
    )
    assert isinstance(d, PermissionDeny)
    assert ReasonCode.TTL_EXCEEDS_POLICY in d.reason_codes


def test_forbidden_shell_execute_denied(registry, ledger, now):
    d = _compile(
        registry,
        ledger,
        now,
        tool_id="forbidden_shell",
        operation="execute",
        reason="please run it",
    )
    assert isinstance(d, PermissionDeny)
    # execute is a forbidden op on an L5 tool.
    assert ReasonCode.OPERATION_FORBIDDEN in d.reason_codes


# --- REVIEW ---------------------------------------------------------------


def test_high_risk_without_approval_requires_review(registry, ledger, now):
    d = _compile(
        registry, ledger, now, scope={"paths": ["./docs"]}, task_risk="L3"
    )
    assert isinstance(d, PermissionReviewRequired)
    assert ReasonCode.HIGH_RISK_REQUIRES_REVIEW in d.reason_codes
    assert ReasonCode.APPROVAL_REQUIRED in d.reason_codes
    assert d.required_approval_type == "owner"


def test_unverified_approval_string_not_accepted(registry, ledger, now):
    # A bare approval_ref string with no matching registry entry is not proof.
    req = make_request(
        scope={"paths": ["./docs"]}, task_risk="L3", approval_ref="approved"
    )
    d = compile_permission(req, registry=registry, ledger=ledger, now=now)
    assert isinstance(d, PermissionReviewRequired)
    assert ReasonCode.APPROVAL_OUT_OF_SCOPE in d.reason_codes


def test_valid_scoped_approval_allows_high_risk(registry, ledger, now):
    approvals = ApprovalRegistry()
    approvals.add(
        ApprovalRef(
            approval_id="approval_1",
            actor_ref="local_dev_agent",
            scope=ApprovalScope(
                tool_id="local_readonly_filesystem",
                operation="read_file",
                paths=["./docs"],
                domains=[],
                max_risk="L3",
            ),
            expires_at=now + timedelta(hours=1),
            approved_by="owner",
            reason="approved doc read",
        )
    )
    req = make_request(
        scope={"paths": ["./docs"]}, task_risk="L3", approval_ref="approval_1"
    )
    d = compile_permission(
        req,
        registry=registry,
        ledger=ledger,
        now=now,
        approval_registry=approvals,
    )
    assert isinstance(d, PermissionGrant)


def test_approval_out_of_scope_path_not_accepted(registry, ledger, now):
    approvals = ApprovalRegistry()
    approvals.add(
        ApprovalRef(
            approval_id="approval_2",
            actor_ref="local_dev_agent",
            scope=ApprovalScope(
                tool_id="local_readonly_filesystem",
                operation="read_file",
                paths=["./docs"],  # approved for docs only
                domains=[],
                max_risk="L3",
            ),
            expires_at=now + timedelta(hours=1),
            approved_by="owner",
            reason="approved doc read",
        )
    )
    req = make_request(
        scope={"paths": ["./src"]},  # but request asks for src
        task_risk="L3",
        approval_ref="approval_2",
    )
    d = compile_permission(
        req,
        registry=registry,
        ledger=ledger,
        now=now,
        approval_registry=approvals,
    )
    assert isinstance(d, PermissionReviewRequired)
    assert ReasonCode.APPROVAL_OUT_OF_SCOPE in d.reason_codes


# --- INVALID --------------------------------------------------------------


def test_malformed_request_is_invalid(registry, ledger, now):
    d = compile_permission(
        {"tool_id": "x"}, registry=registry, ledger=ledger, now=now
    )
    assert isinstance(d, InvalidPermissionRequest)
    assert ReasonCode.REQUEST_SCHEMA_INVALID in d.reason_codes
    assert d.validation_errors
