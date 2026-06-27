"""In-memory, audited task store.

The store is the local source of truth for tasks, claims, and evidence.
Every mutation that changes trusted state also appends an audit event, so
state and audit cannot drift apart. There is no hidden global state: a
store instance owns its own dictionaries and its own audit ledger
reference.
"""

from __future__ import annotations

from amali.audit.ledger import AuditLedger
from amali.core.errors import ValidationBoundaryError
from amali.core.ids import new_task_id
from amali.state.models import ClaimRecord, EvidenceRecord, TaskState

__all__ = ["InMemoryTaskStore"]

# Keys the store will read from a create_task payload. Everything else is
# ignored so that callers cannot inject authority fields (status, etc.)
# through the payload.
_ALLOWED_PAYLOAD_KEYS = frozenset({"uncertainty_score", "uncertainty_reasons"})


class InMemoryTaskStore:
    """Local source of truth for tasks, claims, and evidence."""

    def __init__(self, audit_ledger: AuditLedger) -> None:
        self._audit = audit_ledger
        self._tasks: dict[str, TaskState] = {}
        self._claims: dict[str, ClaimRecord] = {}
        self._evidence: dict[str, EvidenceRecord] = {}

    def create_task(
        self, payload: dict, actor_ref: str, project_ref: str
    ) -> TaskState:
        """Create a new task, record it, and append an audit event.

        Only a whitelist of fields is read from ``payload``; authority
        fields such as ``status`` can never be set by a caller here.
        """
        task_id = new_task_id()
        init_kwargs: dict = {
            "task_id": task_id,
            "actor_context_ref": actor_ref,
            "project_context_ref": project_ref,
        }
        for key in _ALLOWED_PAYLOAD_KEYS:
            if key in payload:
                init_kwargs[key] = payload[key]

        task = TaskState(**init_kwargs)

        event = self._audit.append(
            event_type="task.created",
            actor=actor_ref,
            action="create_task",
            target=task_id,
            # Record only payload key names, never raw values, to avoid
            # leaking potentially sensitive content into the ledger.
            # Maybe need review from higher authorized personnel forexmple-owner
            # sensitive payload keys.
            metadata={"payload_keys": sorted(payload.keys())},
        )
        task.audit_event_refs.append(event.audit_id)
        self._tasks[task_id] = task
        return task

    def add_claim(self, task_id: str, claim: ClaimRecord) -> TaskState:
        """Attach a claim to a task and append an audit event."""
        task = self._require_task(task_id)
        if claim.task_id != task_id:
            raise ValidationBoundaryError(
                "claim.task_id does not match target task_id"
            )

        self._claims[claim.claim_id] = claim
        task.claim_refs.append(claim.claim_id)
        event = self._audit.append(
            event_type="claim.added",
            actor="system",
            action="add_claim",
            target=task_id,
            metadata={
                "claim_id": claim.claim_id,
                "evidence_level": claim.evidence_level.value,
                "claim_status": claim.status,
            },
        )
        task.audit_event_refs.append(event.audit_id)
        return task

    def add_evidence(self, task_id: str, evidence: EvidenceRecord) -> TaskState:
        """Attach an evidence record to a task and append an audit event."""
        task = self._require_task(task_id)
        if evidence.task_id != task_id:
            raise ValidationBoundaryError(
                "evidence.task_id does not match target task_id"
            )

        self._evidence[evidence.evidence_id] = evidence
        task.evidence_refs.append(evidence.evidence_id)
        event = self._audit.append(
            event_type="evidence.added",
            actor="system",
            action="add_evidence",
            target=task_id,
            metadata={
                "evidence_id": evidence.evidence_id,
                "trust_level": evidence.trust_level.value,
                "sensitivity": evidence.sensitivity.value,
                "poison_risk": evidence.poison_risk,
                "injection_risk": evidence.injection_risk,
            },
        )
        task.audit_event_refs.append(event.audit_id)
        return task

    def get_task(self, task_id: str) -> TaskState:
        """Return a task or raise if it does not exist."""
        return self._require_task(task_id)

    def _require_task(self, task_id: str) -> TaskState:
        task = self._tasks.get(task_id)
        if task is None:
            raise ValidationBoundaryError(f"unknown task_id: {task_id}")
        return task
