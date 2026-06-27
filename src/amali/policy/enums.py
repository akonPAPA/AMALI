"""Typed enums for the Tool Risk Permission Compiler (TRPC).

These enums define the closed vocabularies the permission kernel reasons
over. Using closed enums (rather than free strings) is itself a security
control: an agent or a manifest cannot smuggle in an unknown risk level,
tool type, or scope policy that the decision algorithm has not accounted
for.

``RiskLevel`` is ordered (L0 < L1 < ... < L5). Because the values are
strings (to stay readable in YAML and audit logs) the ordering is provided
explicitly via :func:`risk_rank` and the comparison helpers rather than by
relying on enum definition order.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "RiskLevel",
    "DataSensitivity",
    "ToolType",
    "NetworkPolicy",
    "FilesystemPolicy",
    "SecretAccess",
    "DefaultPermission",
    "DecisionType",
    "RequestSource",
    "risk_rank",
    "risk_at_least",
    "max_risk",
]


class RiskLevel(str, Enum):
    """Ordered risk classification for a task or tool.

    L0: harmless / no tool effect.
    L1: local read-only / low risk.
    L2: local write-limited or test-only.
    L3: sensitive, destructive-adjacent, external, or privilege-relevant.
    L4: destructive, networked, secret-adjacent, production-impacting.
    L5: forbidden or owner-only.
    """

    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"
    L5 = "L5"


# Explicit ordering. Defined separately from the enum so that comparisons
# never depend on accidental declaration order.
_RISK_ORDER: dict[RiskLevel, int] = {
    RiskLevel.L0: 0,
    RiskLevel.L1: 1,
    RiskLevel.L2: 2,
    RiskLevel.L3: 3,
    RiskLevel.L4: 4,
    RiskLevel.L5: 5,
}


def risk_rank(level: RiskLevel) -> int:
    """Return the integer rank of a :class:`RiskLevel` (L0=0 .. L5=5)."""
    return _RISK_ORDER[level]


def risk_at_least(level: RiskLevel, threshold: RiskLevel) -> bool:
    """Return True iff ``level`` is at or above ``threshold``."""
    return risk_rank(level) >= risk_rank(threshold)


def max_risk(a: RiskLevel, b: RiskLevel) -> RiskLevel:
    """Return the higher of two risk levels.

    This is how the effective risk of a request is computed: it is never
    lower than either the task's declared risk or the tool's own risk, so
    a low task-risk can never pull a high-risk tool down.
    """
    return a if risk_rank(a) >= risk_rank(b) else b


class DataSensitivity(str, Enum):
    """Sensitivity classification of the data a tool operation touches."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    PERSONAL = "personal"


class ToolType(str, Enum):
    """Category of tool, used to apply type-specific safety constraints."""

    FILESYSTEM = "filesystem"
    SHELL = "shell"
    GIT = "git"
    TEST_RUNNER = "test_runner"
    STATIC_ANALYSIS = "static_analysis"
    WEB_DOCS = "web_docs"
    DB = "db"
    SECRET_BROKER = "secret_broker"
    MODEL = "model"
    CUSTOM = "custom"


class NetworkPolicy(str, Enum):
    """Network capability a tool manifest declares.

    ``none``: no network at all.
    ``allowlist``: only explicitly listed domains (not exercised this stage).
    ``unrestricted_forbidden``: a marker that unrestricted network is never
    granted; the compiler must never produce a network grant for it.
    """

    NONE = "none"
    ALLOWLIST = "allowlist"
    UNRESTRICTED_FORBIDDEN = "unrestricted_forbidden"


class FilesystemPolicy(str, Enum):
    """Filesystem capability a tool manifest declares."""

    NONE = "none"
    WORKSPACE_READ = "workspace_read"
    WORKSPACE_WRITE = "workspace_write"
    EXPLICIT_PATHS_ONLY = "explicit_paths_only"


class SecretAccess(str, Enum):
    """Secret-access capability a tool manifest declares.

    ``none``: no secret access.
    ``handle_only``: may receive an opaque handle, never a raw value.
    ``raw_to_trusted_tool_only``: a raw value may flow only to a trusted
    tool boundary, *never* to a model/agent. Not exercised this stage.
    """

    NONE = "none"
    HANDLE_ONLY = "handle_only"
    RAW_TO_TRUSTED_TOOL_ONLY = "raw_to_trusted_tool_only"


class DefaultPermission(str, Enum):
    """Default permission posture declared by a tool manifest."""

    DENY = "deny"
    READ_ONLY = "read_only"
    ALLOWED_LOW_RISK = "allowed_low_risk"


class DecisionType(str, Enum):
    """The four possible TRPC outcomes."""

    GRANT = "GRANT"
    DENY = "DENY"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    INVALID_REQUEST = "INVALID_REQUEST"


class RequestSource(str, Enum):
    """Origin of a tool-invocation request.

    The source is recorded for audit only. It never grants authority: an
    ``agent`` or ``model`` source can never obtain more than a ``user`` or
    ``system`` source for the same request.
    """

    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"
    TEST = "test"
