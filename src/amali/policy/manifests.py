"""Versioned tool manifests and the tool-permission policy.

A :class:`ToolManifest` declares what a single tool *may* do; a
:class:`ToolPermissionPolicy` declares the global, default-deny rules the
compiler enforces across all tools. Both are strictly validated: a manifest
or policy that violates a hard invariant is rejected at construction time
(pydantic ``ValidationError``) and, defensively, re-checked by the compiler
via :meth:`ToolManifest.assert_valid` / :meth:`ToolPermissionPolicy.assert_valid`.

The validation rules are intentionally conservative. They encode the
security posture: shell tools default to deny, audit is mandatory, secret
access is tightly constrained, and the dangerous-pattern set must cover both
POSIX and Windows/PowerShell forms (the owner develops on Windows).
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from amali.policy.enums import (
    DataSensitivity,
    DefaultPermission,
    FilesystemPolicy,
    NetworkPolicy,
    RiskLevel,
    SecretAccess,
    ToolType,
    risk_at_least,
    risk_rank,
)
from amali.policy.errors import ManifestValidationError, PolicyValidationError

__all__ = [
    "ToolManifest",
    "ToolPermissionPolicy",
    "RiskThresholds",
    "REQUIRED_DANGEROUS_PATTERNS",
]

# Tool types that are inherently capable enough to demand sandbox isolation
# before any execution could occur.
_SANDBOX_REQUIRED_TYPES = frozenset(
    {
        ToolType.SHELL,
        ToolType.TEST_RUNNER,
        ToolType.STATIC_ANALYSIS,
        ToolType.DB,
    }
)

# The compiler-side dangerous-pattern set must include at least these tokens.
# They are normalized (lowercased) substrings, spanning POSIX and Windows.
REQUIRED_DANGEROUS_PATTERNS: tuple[str, ...] = (
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
    "powershell",
    "encodedcommand",
    "invoke-webrequest",
    "invoke-expression",
    "chmod 777",
    "sudo",
    "private_key",
    "access_token",
    "refresh_token",
    "bearer",
)


class ToolManifest(BaseModel):
    """Versioned declaration of a single tool's permitted behavior."""

    model_config = {"extra": "forbid"}

    tool_id: str
    version: str
    tool_type: ToolType
    default_permission: DefaultPermission
    risk_level: RiskLevel
    allowed_operations: list[str] = Field(default_factory=list)
    forbidden_operations: list[str] = Field(default_factory=list)
    sandbox_required: bool
    network_policy: NetworkPolicy
    filesystem_policy: FilesystemPolicy
    secret_access: SecretAccess
    approval_required: bool
    audit_required: bool = True
    allowed_paths: list[str] | None = None
    allowed_domains: list[str] | None = None
    max_ttl_seconds: int | None = None
    description: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> "ToolManifest":
        # Surface domain violations as pydantic ValidationError at
        # construction, while still letting callers invoke assert_valid()
        # directly to get the typed ManifestValidationError.
        try:
            self.assert_valid()
        except ManifestValidationError as exc:
            raise ValueError(str(exc)) from exc
        return self

    def assert_valid(self) -> None:
        """Re-assert every hard manifest invariant.

        Raises :class:`ManifestValidationError` on the first violation. The
        compiler calls this defensively so that a manifest constructed by a
        bypass (``model_construct``) or mutated after construction is still
        caught before it can influence a decision.
        """
        if not self.tool_id or not self.tool_id.strip():
            raise ManifestValidationError("tool_id must not be empty")
        if not self.version or not self.version.strip():
            raise ManifestValidationError("version must not be empty")

        # Audit is mandatory for every tool, always.
        if self.audit_required is not True:
            raise ManifestValidationError(
                "audit_required must be true for all tool manifests"
            )

        # Forbidden overrides allowed: an operation cannot be both.
        overlap = set(self.allowed_operations) & set(self.forbidden_operations)
        if overlap:
            raise ManifestValidationError(
                f"operations listed as both allowed and forbidden: "
                f"{sorted(overlap)}"
            )

        # Default permission must be deny unless the tool is genuinely
        # low-risk (<= L1). read_only / allowed_low_risk are only valid for
        # low-risk tools.
        if self.default_permission is not DefaultPermission.DENY:
            if risk_rank(self.risk_level) > risk_rank(RiskLevel.L1):
                raise ManifestValidationError(
                    "default_permission may be non-deny only when "
                    "risk_level <= L1"
                )

        # Shell tools must default to deny, period.
        if (
            self.tool_type is ToolType.SHELL
            and self.default_permission is not DefaultPermission.DENY
        ):
            raise ManifestValidationError("shell tool default must be deny")

        # Network allowlist requires explicit domains.
        if self.network_policy is NetworkPolicy.ALLOWLIST and not (
            self.allowed_domains
        ):
            raise ManifestValidationError(
                "network_policy=allowlist requires non-empty allowed_domains"
            )

        # explicit_paths_only requires explicit paths.
        if self.filesystem_policy is FilesystemPolicy.EXPLICIT_PATHS_ONLY and (
            not self.allowed_paths
        ):
            raise ManifestValidationError(
                "filesystem_policy=explicit_paths_only requires "
                "non-empty allowed_paths"
            )

        # Secret access must be tightly constrained.
        if self.secret_access is not SecretAccess.NONE:
            handle_broker = (
                self.secret_access is SecretAccess.HANDLE_ONLY
                and self.tool_type is ToolType.SECRET_BROKER
            )
            if not (self.approval_required or handle_broker):
                raise ManifestValidationError(
                    "secret_access != none requires approval_required=true, "
                    "or handle_only access on a secret_broker tool"
                )

        # Capable tool types must require a sandbox.
        if self.tool_type in _SANDBOX_REQUIRED_TYPES and not (
            self.sandbox_required
        ):
            raise ManifestValidationError(
                f"sandbox_required must be true for tool_type "
                f"{self.tool_type.value}"
            )
        # Write-capable custom tools must require a sandbox.
        if (
            self.tool_type is ToolType.CUSTOM
            and self.filesystem_policy is FilesystemPolicy.WORKSPACE_WRITE
            and not self.sandbox_required
        ):
            raise ManifestValidationError(
                "write-capable custom tool must set sandbox_required=true"
            )


class RiskThresholds(BaseModel):
    """Monotonic risk thresholds for the permission policy."""

    model_config = {"extra": "forbid"}

    allow_max: RiskLevel
    readonly_max: RiskLevel
    sandbox_max: RiskLevel
    review_max: RiskLevel
    deny_min: RiskLevel


class ToolPermissionPolicy(BaseModel):
    """Global, default-deny tool-permission policy."""

    model_config = {"extra": "forbid"}

    policy_id: str
    version: str
    default_decision: str
    max_grant_ttl_seconds: int
    risk_thresholds: RiskThresholds
    high_risk_requires_review: bool = True
    secret_requires_handle_only: bool = True
    network_default_deny: bool = True
    forbid_unrestricted_network: bool = True
    forbid_raw_secret_to_model: bool = True
    require_audit_event: bool = True
    owner_approval_required_from_risk: RiskLevel
    denied_operations: list[str] = Field(default_factory=list)
    dangerous_operation_patterns: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> "ToolPermissionPolicy":
        try:
            self.assert_valid()
        except PolicyValidationError as exc:
            raise ValueError(str(exc)) from exc
        return self

    def assert_valid(self) -> None:
        """Re-assert every hard policy invariant.

        Raises :class:`PolicyValidationError` on the first violation.
        """
        if self.default_decision != "deny":
            raise PolicyValidationError("default_decision must be deny")
        if self.max_grant_ttl_seconds <= 0:
            raise PolicyValidationError("max_grant_ttl_seconds must be > 0")

        # Security switches that must never be off.
        if not self.require_audit_event:
            raise PolicyValidationError("require_audit_event must be true")
        if not self.forbid_raw_secret_to_model:
            raise PolicyValidationError(
                "forbid_raw_secret_to_model must be true"
            )
        if not self.forbid_unrestricted_network:
            raise PolicyValidationError(
                "forbid_unrestricted_network must be true"
            )
        if not self.network_default_deny:
            raise PolicyValidationError("network_default_deny must be true")

        # Thresholds must be monotonic and non-overlapping with deny.
        t = self.risk_thresholds
        ordered = [
            t.allow_max,
            t.readonly_max,
            t.sandbox_max,
            t.review_max,
        ]
        ranks = [risk_rank(level) for level in ordered]
        if ranks != sorted(ranks):
            raise PolicyValidationError(
                "risk thresholds must be monotonic non-decreasing: "
                "allow_max <= readonly_max <= sandbox_max <= review_max"
            )
        if risk_at_least(t.review_max, t.deny_min):
            raise PolicyValidationError(
                "deny_min must be strictly above review_max"
            )

        if not self.denied_operations:
            raise PolicyValidationError("denied_operations must be non-empty")

        # The dangerous-pattern set must cover the required tokens.
        normalized = {p.strip().lower() for p in self.dangerous_operation_patterns}
        joined = " | ".join(normalized)
        missing = [
            token
            for token in REQUIRED_DANGEROUS_PATTERNS
            if token not in joined
        ]
        if missing:
            raise PolicyValidationError(
                f"dangerous_operation_patterns missing required tokens: "
                f"{missing}"
            )

    def sensitivity_is_secret(self, sensitivity: DataSensitivity) -> bool:
        """Convenience: is this sensitivity a hard raw-secret class?"""
        return sensitivity is DataSensitivity.SECRET
