"""Domain errors for the policy / permission layer.

These extend the kernel's :class:`~amali.core.errors.AMALIError` so callers
can catch the whole family, while still distinguishing a malformed manifest
from a malformed policy from a refused execution.
"""

from __future__ import annotations

from amali.core.errors import AMALIError

__all__ = [
    "PolicyError",
    "ManifestValidationError",
    "PolicyValidationError",
    "RequestValidationError",
    "PermissionRequiredError",
]


class PolicyError(AMALIError):
    """Base class for policy/permission errors."""


class ManifestValidationError(PolicyError):
    """Raised when a tool manifest violates a hard manifest invariant."""


class PolicyValidationError(PolicyError):
    """Raised when a tool-permission policy violates a hard invariant."""


class RequestValidationError(PolicyError):
    """Raised when a tool-invocation request fails schema validation."""


class PermissionRequiredError(PolicyError):
    """Raised by the executor boundary when no valid grant authorizes a call.

    The message intentionally carries only a reason code and structural
    facts (tool/operation), never request payload or secret material.
    """
