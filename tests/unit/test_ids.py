"""IDs have correct prefixes and are unique."""

from amali.core.ids import (
    new_audit_id,
    new_claim_id,
    new_evidence_id,
    new_route_id,
    new_task_id,
)


def test_id_prefixes():
    assert new_task_id().startswith("task_")
    assert new_claim_id().startswith("claim_")
    assert new_evidence_id().startswith("ev_")
    assert new_audit_id().startswith("audit_")
    assert new_route_id().startswith("route_")


def test_ids_are_unique():
    generators = [
        new_task_id,
        new_claim_id,
        new_evidence_id,
        new_audit_id,
        new_route_id,
    ]
    for generator in generators:
        values = {generator() for _ in range(1000)}
        assert len(values) == 1000
