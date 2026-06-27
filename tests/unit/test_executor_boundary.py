"""The executor boundary refuses any call without a valid, matching grant."""

from __future__ import annotations

from datetime import timedelta

import pytest

from amali.policy.decisions import PermissionGrant
from amali.policy.errors import PermissionRequiredError
from amali.policy.trpc import compile_permission
from amali.tools.executor_boundary import BOUNDARY_RESULT, execute_tool

from tests.conftest import make_request


def _grant(registry, ledger, now, **overrides):
    req = make_request(scope={"paths": ["./README.md"]}, **overrides)
    decision = compile_permission(
        req, registry=registry, ledger=ledger, now=now
    )
    assert isinstance(decision, PermissionGrant)
    return req, decision


def test_execute_without_grant_blocked(registry, ledger, now):
    req, _ = _grant(registry, ledger, now)
    with pytest.raises(PermissionRequiredError):
        execute_tool(req, None, now=now)


def test_execute_with_valid_grant_is_boundary_only(registry, ledger, now):
    req, grant = _grant(registry, ledger, now)
    result = execute_tool(req, grant, now=now)
    assert result["executed"] is False
    assert result["status"] == "boundary_only"
    assert result["detail"] == BOUNDARY_RESULT


def test_expired_grant_blocked(registry, ledger, now):
    req, grant = _grant(registry, ledger, now)
    later = grant.expires_at + timedelta(seconds=1)
    with pytest.raises(PermissionRequiredError):
        execute_tool(req, grant, now=later)


def test_wrong_tool_id_blocked(registry, ledger, now):
    req, grant = _grant(registry, ledger, now)
    mismatched = make_request(
        tool_id="local_test_runner",
        operation="read_file",
        scope={"paths": ["./README.md"]},
    )
    with pytest.raises(PermissionRequiredError):
        execute_tool(mismatched, grant, now=now)


def test_wrong_operation_blocked(registry, ledger, now):
    req, grant = _grant(registry, ledger, now)
    mismatched = make_request(
        operation="list_dir", scope={"paths": ["./README.md"]}
    )
    with pytest.raises(PermissionRequiredError):
        execute_tool(mismatched, grant, now=now)


def test_scope_mismatch_path_blocked(registry, ledger, now):
    req, grant = _grant(registry, ledger, now)
    mismatched = make_request(
        operation="read_file", scope={"paths": ["./src"]}
    )
    with pytest.raises(PermissionRequiredError):
        execute_tool(mismatched, grant, now=now)


def test_scope_escalation_write_blocked(registry, ledger, now):
    req, grant = _grant(registry, ledger, now)
    escalated = make_request(
        operation="read_file",
        scope={"paths": ["./README.md"], "write": True},
    )
    with pytest.raises(PermissionRequiredError):
        execute_tool(escalated, grant, now=now)
