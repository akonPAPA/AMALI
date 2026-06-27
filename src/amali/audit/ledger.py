"""Append-only audit ledger with SHA-256 hash chaining.

Every trusted-state mutation is recorded as an :class:`AuditEvent`. Each
event's hash is computed over the previous event's hash plus the canonical
JSON of the event body. This makes the ledger tamper-evident: mutating any
field of any prior event, or breaking a ``previous_hash`` link, causes
:meth:`AuditLedger.verify_chain` to fail.

This is local-only and in-memory. No network, no external service.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, Field

from amali.core.ids import new_audit_id

__all__ = ["AuditEvent", "AuditLedger", "GENESIS_HASH"]

GENESIS_HASH = "GENESIS"


def _canonical_json(data: dict[str, Any]) -> str:
    """Serialize a mapping deterministically for hashing."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


class AuditEvent(BaseModel):
    """A single immutable audit record in the chain."""

    audit_id: str
    event_type: str
    actor: str
    action: str
    target: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    previous_hash: str
    event_hash: str

    def body_for_hash(self) -> dict[str, Any]:
        """Return the hashed body: everything except ``event_hash``."""
        return self.model_dump(exclude={"event_hash"})


def compute_event_hash(previous_hash: str, body: dict[str, Any]) -> str:
    """Compute ``sha256(previous_hash + canonical_json(body))``.

    ``body`` must not contain ``event_hash``.
    """
    payload = previous_hash + _canonical_json(body)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AuditLedger:
    """An in-memory, append-only, hash-chained ledger.

    The internal list is private; callers read it via :meth:`events`,
    which returns a defensive copy so the chain cannot be mutated in
    place through the returned reference.
    """

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    @property
    def last_hash(self) -> str:
        """Hash of the most recent event, or ``GENESIS`` if empty."""
        if not self._events:
            return GENESIS_HASH
        return self._events[-1].event_hash

    def append(
        self,
        event_type: str,
        actor: str,
        action: str,
        target: str,
        metadata: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Append a new event, chaining it to the current tail."""
        previous_hash = self.last_hash
        body = {
            "audit_id": new_audit_id(),
            "event_type": event_type,
            "actor": actor,
            "action": action,
            "target": target,
            "metadata": dict(metadata or {}),
            "previous_hash": previous_hash,
        }
        event_hash = compute_event_hash(previous_hash, body)
        event = AuditEvent(**body, event_hash=event_hash)
        self._events.append(event)
        return event

    def events(self) -> list[AuditEvent]:
        """Return a defensive copy of the event list."""
        return list(self._events)

    def verify_chain(self) -> bool:
        """Return True iff every link and every event hash is intact.

        Detects:
        * mutation of any field of any prior event (recomputed hash differs);
        * a broken ``previous_hash`` link between consecutive events;
        * a wrong genesis link on the first event.
        """
        expected_prev = GENESIS_HASH
        for event in self._events:
            if event.previous_hash != expected_prev:
                return False
            recomputed = compute_event_hash(
                event.previous_hash, event.body_for_hash()
            )
            if recomputed != event.event_hash:
                return False
            expected_prev = event.event_hash
        return True
