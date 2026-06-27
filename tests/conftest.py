"""Shared fixtures for AMALI-IGA-02 permission-kernel tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from amali.audit.ledger import AuditLedger
from amali.policy.registry import ManifestRegistry
from amali.tools.requests import RequestedScope, ToolInvocationRequest

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFESTS_DIR = REPO_ROOT / "manifests"

# A fixed instant so decisions (expiry, audit ordering) are deterministic.
FIXED_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def now() -> datetime:
    return FIXED_NOW


@pytest.fixture
def ledger() -> AuditLedger:
    return AuditLedger()


@pytest.fixture
def registry() -> ManifestRegistry:
    return ManifestRegistry.load_from_dir(MANIFESTS_DIR)


def make_request(**overrides) -> ToolInvocationRequest:
    """Build a ToolInvocationRequest with safe, valid defaults."""
    scope_kwargs = overrides.pop("scope", None)
    defaults = dict(
        task_id="task_fixed_1",
        actor_ref="local_dev_agent",
        tool_id="local_readonly_filesystem",
        operation="read_file",
        task_risk="L1",
        data_sensitivity="public",
        reason="read an allowed workspace file",
        request_source="agent",
    )
    defaults.update(overrides)
    if scope_kwargs is not None:
        defaults["requested_scope"] = RequestedScope(**scope_kwargs)
    return ToolInvocationRequest(**defaults)
