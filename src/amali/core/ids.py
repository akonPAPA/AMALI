"""Stable ID helpers.

All IDs are prefixed so that the kind of an entity is obvious from the
string alone. Values come from ``uuid.uuid4().hex`` (128 bits of entropy,
no hyphens), which is local-only and requires no network.
"""

from __future__ import annotations

import uuid

__all__ = [
    "new_task_id",
    "new_claim_id",
    "new_evidence_id",
    "new_audit_id",
    "new_route_id",
    "new_decision_id",
    "new_grant_id",
    "new_approval_id",
]


def _hex() -> str:
    return uuid.uuid4().hex


def new_task_id() -> str:
    """Return a new task ID, e.g. ``task_<uuid>``."""
    return f"task_{_hex()}"


def new_claim_id() -> str:
    """Return a new claim ID, e.g. ``claim_<uuid>``."""
    return f"claim_{_hex()}"


def new_evidence_id() -> str:
    """Return a new evidence ID, e.g. ``ev_<uuid>``."""
    return f"ev_{_hex()}"


def new_audit_id() -> str:
    """Return a new audit event ID, e.g. ``audit_<uuid>``."""
    return f"audit_{_hex()}"


def new_route_id() -> str:
    """Return a new route decision ID, e.g. ``route_<uuid>``."""
    return f"route_{_hex()}"


def new_decision_id() -> str:
    """Return a new permission-decision ID, e.g. ``decision_<uuid>``."""
    return f"decision_{_hex()}"


def new_grant_id() -> str:
    """Return a new permission-grant ID, e.g. ``grant_<uuid>``."""
    return f"grant_{_hex()}"


def new_approval_id() -> str:
    """Return a new owner-approval ID, e.g. ``approval_<uuid>``."""
    return f"approval_{_hex()}"
