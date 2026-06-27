"""Strict validation of tool manifests and the permission policy."""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from amali.policy.manifests import ToolManifest, ToolPermissionPolicy

# A known-valid read-only filesystem manifest, mutated per test.
_VALID_FS = dict(
    tool_id="local_readonly_filesystem",
    version="0.1.0",
    tool_type="filesystem",
    default_permission="read_only",
    risk_level="L1",
    allowed_operations=["read_file", "list_dir"],
    forbidden_operations=["write_file", "delete_file", "execute", "network"],
    sandbox_required=False,
    network_policy="none",
    filesystem_policy="explicit_paths_only",
    secret_access="none",
    approval_required=False,
    audit_required=True,
    allowed_paths=["./README.md", "./docs"],
    max_ttl_seconds=300,
)

_VALID_POLICY = dict(
    policy_id="tool_permission_policy_v0",
    version="0.1.0",
    default_decision="deny",
    max_grant_ttl_seconds=300,
    risk_thresholds=dict(
        allow_max="L1",
        readonly_max="L1",
        sandbox_max="L2",
        review_max="L4",
        deny_min="L5",
    ),
    owner_approval_required_from_risk="L3",
    denied_operations=["execute_shell", "raw_secret_read"],
    dangerous_operation_patterns=[
        "rm -rf",
        "format",
        "shutdown",
        "reboot",
        "curl",
        "wget",
        "nc",
        "netcat",
        "ssh",
        "scp",
        "powershell -encodedcommand",
        "invoke-webrequest",
        "invoke-expression",
        "chmod 777",
        "sudo",
        "private_key",
        "access_token",
        "refresh_token",
        "bearer",
    ],
)


def _fs(**overrides) -> dict:
    data = copy.deepcopy(_VALID_FS)
    data.update(overrides)
    return data


def _policy(**overrides) -> dict:
    data = copy.deepcopy(_VALID_POLICY)
    data.update(overrides)
    return data


# --- manifest acceptance --------------------------------------------------


def test_valid_readonly_manifest_loads():
    manifest = ToolManifest(**_VALID_FS)
    assert manifest.tool_id == "local_readonly_filesystem"
    assert manifest.audit_required is True


# --- manifest rejection ---------------------------------------------------


def test_operation_in_both_allowed_and_forbidden_rejected():
    with pytest.raises(ValidationError):
        ToolManifest(**_fs(allowed_operations=["read_file", "execute"]))


def test_shell_default_not_deny_rejected():
    with pytest.raises(ValidationError):
        ToolManifest(
            **_fs(
                tool_id="bad_shell",
                tool_type="shell",
                default_permission="allowed_low_risk",
                risk_level="L1",
                filesystem_policy="none",
                allowed_paths=None,
                sandbox_required=True,
                allowed_operations=[],
                forbidden_operations=["execute"],
            )
        )


def test_audit_required_false_rejected():
    with pytest.raises(ValidationError):
        ToolManifest(**_fs(audit_required=False))


def test_allowlist_network_without_domains_rejected():
    with pytest.raises(ValidationError):
        ToolManifest(**_fs(network_policy="allowlist", allowed_domains=None))


def test_explicit_paths_only_without_paths_rejected():
    with pytest.raises(ValidationError):
        ToolManifest(**_fs(allowed_paths=None))


def test_secret_access_without_constraints_rejected():
    # secret_access != none on a non-broker tool without approval is invalid.
    with pytest.raises(ValidationError):
        ToolManifest(
            **_fs(secret_access="handle_only", approval_required=False)
        )


def test_secret_handle_only_on_secret_broker_allowed():
    manifest = ToolManifest(
        **_fs(
            tool_id="broker",
            tool_type="secret_broker",
            secret_access="handle_only",
            approval_required=False,
            filesystem_policy="none",
            allowed_paths=None,
            allowed_operations=["get_handle"],
            forbidden_operations=["reveal"],
        )
    )
    assert manifest.secret_access.value == "handle_only"


def test_non_deny_default_above_l1_rejected():
    with pytest.raises(ValidationError):
        ToolManifest(**_fs(default_permission="read_only", risk_level="L3"))


def test_test_runner_without_sandbox_rejected():
    with pytest.raises(ValidationError):
        ToolManifest(
            **_fs(
                tool_id="tr",
                tool_type="test_runner",
                default_permission="deny",
                risk_level="L2",
                sandbox_required=False,
                filesystem_policy="workspace_read",
                allowed_paths=None,
                allowed_operations=["run_unit_tests"],
                forbidden_operations=["shell"],
            )
        )


# --- policy acceptance / rejection ---------------------------------------


def test_valid_policy_loads():
    policy = ToolPermissionPolicy(**_VALID_POLICY)
    assert policy.default_decision == "deny"


def test_policy_default_decision_not_deny_rejected():
    with pytest.raises(ValidationError):
        ToolPermissionPolicy(**_policy(default_decision="allow"))


def test_policy_audit_off_rejected():
    with pytest.raises(ValidationError):
        ToolPermissionPolicy(**_policy(require_audit_event=False))


def test_policy_network_default_deny_off_rejected():
    with pytest.raises(ValidationError):
        ToolPermissionPolicy(**_policy(network_default_deny=False))


def test_policy_non_monotonic_thresholds_rejected():
    bad = _policy(
        risk_thresholds=dict(
            allow_max="L3",
            readonly_max="L1",
            sandbox_max="L2",
            review_max="L4",
            deny_min="L5",
        )
    )
    with pytest.raises(ValidationError):
        ToolPermissionPolicy(**bad)


def test_policy_empty_denied_operations_rejected():
    with pytest.raises(ValidationError):
        ToolPermissionPolicy(**_policy(denied_operations=[]))


def test_policy_missing_dangerous_patterns_rejected():
    with pytest.raises(ValidationError):
        ToolPermissionPolicy(
            **_policy(dangerous_operation_patterns=["rm -rf", "curl"])
        )
