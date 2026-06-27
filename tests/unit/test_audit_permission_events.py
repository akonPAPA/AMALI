"""Every permission decision emits a redacted, chained audit event."""

from __future__ import annotations

import json

from amali.audit.events import PERMISSION_EVENT_TYPE
from amali.policy.reason_codes import ReasonCode
from amali.policy.trpc import compile_permission

from tests.conftest import make_request


def _compile(registry, ledger, now, **overrides):
    req = make_request(**overrides)
    return compile_permission(req, registry=registry, ledger=ledger, now=now)


def _last_event(ledger):
    return ledger.events()[-1]


def test_grant_emits_audit_event(registry, ledger, now):
    d = _compile(registry, ledger, now, scope={"paths": ["./README.md"]})
    event = _last_event(ledger)
    assert event.event_type == PERMISSION_EVENT_TYPE
    assert event.audit_id == d.audit_event_id
    assert event.metadata["outcome"] == "allow"
    assert event.metadata["decision_id"] == d.decision_id


def test_deny_emits_audit_event(registry, ledger, now):
    d = _compile(registry, ledger, now, operation="write_file")
    event = _last_event(ledger)
    assert event.audit_id == d.audit_event_id
    assert event.metadata["outcome"] == "deny"
    assert ReasonCode.OPERATION_FORBIDDEN.value in event.metadata["reason_codes"]


def test_review_emits_audit_event(registry, ledger, now):
    d = _compile(
        registry, ledger, now, scope={"paths": ["./docs"]}, task_risk="L3"
    )
    event = _last_event(ledger)
    assert event.audit_id == d.audit_event_id
    assert event.metadata["outcome"] == "review"


def test_invalid_request_emits_audit_event(registry, ledger, now):
    d = compile_permission(
        {"tool_id": "x"}, registry=registry, ledger=ledger, now=now
    )
    event = _last_event(ledger)
    assert event.audit_id == d.audit_event_id
    assert event.metadata["outcome"] == "invalid"


def test_audit_chain_stays_verifiable_across_decisions(registry, ledger, now):
    _compile(registry, ledger, now, scope={"paths": ["./README.md"]})
    _compile(registry, ledger, now, operation="write_file")
    _compile(registry, ledger, now, scope={"paths": ["./docs"]}, task_risk="L3")
    assert ledger.verify_chain() is True
    assert len(ledger.events()) == 3


def test_audit_event_redacts_secret_metadata(registry, ledger, now):
    # A request whose metadata *name* looks secret-bearing is rejected at the
    # schema, so here we prove the audit layer itself redacts: feed a value
    # that looks like a token via an allowed key, through the reason path.
    d = _compile(
        registry,
        ledger,
        now,
        operation="write_file",  # denied, but still audited
        reason="token=ghp_0123456789abcdefghijABCDEFGHIJ extra",
    )
    event = _last_event(ledger)
    blob = json.dumps(event.metadata, default=str)
    # The audit trail must not carry the raw reason text or token value.
    assert "ghp_0123456789abcdefghijABCDEFGHIJ" not in blob
    assert "token=ghp_" not in blob
    assert d.audit_event_id == event.audit_id


def test_audit_event_never_contains_raw_secret_value(registry, ledger, now):
    # Even on the grant path, the serialized event holds only structural
    # facts: codes, counts, ids — never raw payload.
    _compile(registry, ledger, now, scope={"paths": ["./README.md"]})
    event = _last_event(ledger)
    blob = json.dumps(event.metadata, default=str)
    for forbidden in ("BEGIN PRIVATE KEY", "Bearer ", "sk-", "ghp_"):
        assert forbidden not in blob
