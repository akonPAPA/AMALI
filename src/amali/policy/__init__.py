"""Policy layer: tool manifests, the permission policy, and the TRPC.

The Tool Risk Permission Compiler (TRPC) turns an untrusted tool-invocation
request into one of four typed, audited decisions under a strict,
default-deny policy. Nothing in this package executes a tool.
"""

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
    RequestSource,
    RiskLevel,
    SecretAccess,
    ToolType,
)
from amali.policy.manifests import ToolManifest, ToolPermissionPolicy
from amali.policy.reason_codes import ReasonCode
from amali.policy.registry import ManifestRegistry
from amali.policy.trpc import compile_permission

__all__ = [
    "compile_permission",
    "ManifestRegistry",
    "ToolManifest",
    "ToolPermissionPolicy",
    "ReasonCode",
    "PermissionDecision",
    "PermissionGrant",
    "PermissionDeny",
    "PermissionReviewRequired",
    "InvalidPermissionRequest",
    "DecisionType",
    "RiskLevel",
    "DataSensitivity",
    "ToolType",
    "NetworkPolicy",
    "FilesystemPolicy",
    "SecretAccess",
    "DefaultPermission",
    "RequestSource",
]
