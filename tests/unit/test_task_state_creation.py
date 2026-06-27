"""Creating a task yields ACCEPTED status and an audit event."""

from amali.audit.ledger import AuditLedger
from amali.state.models import TaskStatus
from amali.state.store import InMemoryTaskStore


def test_create_task_defaults_and_audit():
    ledger = AuditLedger()
    store = InMemoryTaskStore(ledger)

    task = store.create_task(
        payload={}, actor_ref="actor_ctx_1", project_ref="project_ctx_1"
    )

    assert task.task_id.startswith("task_")
    assert task.status is TaskStatus.ACCEPTED
    assert task.actor_context_ref == "actor_ctx_1"
    assert task.project_context_ref == "project_ctx_1"

    # An audit event was appended and linked onto the task.
    assert len(task.audit_event_refs) == 1
    events = ledger.events()
    assert len(events) == 1
    assert events[0].event_type == "task.created"
    assert events[0].target == task.task_id
    assert events[0].audit_id == task.audit_event_refs[0]
    assert ledger.verify_chain() is True


def test_get_task_returns_same_state():
    ledger = AuditLedger()
    store = InMemoryTaskStore(ledger)
    task = store.create_task(payload={}, actor_ref="a", project_ref="p")
    assert store.get_task(task.task_id) is task
