"""Typed reason codes for permission decisions.

Every decision carries one or more reason codes. They are a closed,
machine-readable vocabulary so that audits and tests can assert on *why* a
decision was made without parsing human prose. Prose can be injected or
mistranslated; a reason code cannot.
"""

from __future__ import annotations

from enum import Enum

__all__ = ["ReasonCode"]


class ReasonCode(str, Enum):
    """Closed vocabulary explaining a permission decision."""

    # Resolution / validation failures.
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    MANIFEST_INVALID = "MANIFEST_INVALID"
    POLICY_INVALID = "POLICY_INVALID"
    REQUEST_SCHEMA_INVALID = "REQUEST_SCHEMA_INVALID"

    # Operation gating.
    OPERATION_FORBIDDEN = "OPERATION_FORBIDDEN"
    OPERATION_NOT_ALLOWED = "OPERATION_NOT_ALLOWED"
    DEFAULT_DENY = "DEFAULT_DENY"
    DANGEROUS_PATTERN_DETECTED = "DANGEROUS_PATTERN_DETECTED"

    # Risk.
    RISK_TOO_HIGH = "RISK_TOO_HIGH"
    HIGH_RISK_REQUIRES_REVIEW = "HIGH_RISK_REQUIRES_REVIEW"

    # Secrets.
    SECRET_ACCESS_DENIED = "SECRET_ACCESS_DENIED"
    RAW_SECRET_FORBIDDEN = "RAW_SECRET_FORBIDDEN"

    # Network.
    NETWORK_DENIED = "NETWORK_DENIED"
    UNRESTRICTED_NETWORK_FORBIDDEN = "UNRESTRICTED_NETWORK_FORBIDDEN"
    DOMAIN_NOT_ALLOWLISTED = "DOMAIN_NOT_ALLOWLISTED"

    # Filesystem.
    FILESYSTEM_SCOPE_DENIED = "FILESYSTEM_SCOPE_DENIED"
    PATH_TRAVERSAL_DENIED = "PATH_TRAVERSAL_DENIED"
    WRITE_SCOPE_DENIED = "WRITE_SCOPE_DENIED"

    # Data minimization.
    DATA_MINIMIZATION_REQUIRED = "DATA_MINIMIZATION_REQUIRED"

    # TTL.
    TTL_EXCEEDS_POLICY = "TTL_EXCEEDS_POLICY"

    # Approval.
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    APPROVAL_MISSING = "APPROVAL_MISSING"
    APPROVAL_OUT_OF_SCOPE = "APPROVAL_OUT_OF_SCOPE"

    # Audit.
    AUDIT_REQUIRED = "AUDIT_REQUIRED"

    # Positive (grant) outcomes.
    SAFE_READONLY_GRANTED = "SAFE_READONLY_GRANTED"
    SANDBOX_REQUIRED = "SANDBOX_REQUIRED"
    SANDBOX_GRANT_CREATED = "SANDBOX_GRANT_CREATED"
    POLICY_ALLOW_LOW_RISK = "POLICY_ALLOW_LOW_RISK"
