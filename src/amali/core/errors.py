"""Domain exceptions for the AMALI kernel.

These are intentionally simple. They exist so that callers can
distinguish a validation-boundary rejection from an audit-integrity
failure without inspecting error strings.
"""

from __future__ import annotations

__all__ = [
    "AMALIError",
    "ValidationBoundaryError",
    "TrustedStateMutationError",
    "AuditIntegrityError",
]


class AMALIError(Exception):
    """Base class for all AMALI domain errors."""


class ValidationBoundaryError(AMALIError):
    """Raised when input crossing a trust boundary fails validation."""


class TrustedStateMutationError(AMALIError):
    """Raised when a trusted-state mutation is attempted unsafely."""


class AuditIntegrityError(AMALIError):
    """Raised when the audit ledger fails an integrity check."""
