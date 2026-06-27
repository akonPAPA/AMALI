"""Mutating a prior event or breaking a link is detectable."""

from amali.audit.ledger import AuditLedger


def _build_ledger() -> AuditLedger:
    ledger = AuditLedger()
    ledger.append("a", "actor", "act", "t1")
    ledger.append("b", "actor", "act", "t2")
    ledger.append("c", "actor", "act", "t3")
    return ledger


def test_mutating_older_event_field_breaks_chain():
    ledger = _build_ledger()
    assert ledger.verify_chain() is True

    # Tamper with an older event's content (its stored hash no longer
    # matches the recomputed hash).
    ledger._events[0].action = "tampered"
    assert ledger.verify_chain() is False


def test_mutating_metadata_breaks_chain():
    ledger = _build_ledger()
    ledger._events[1].metadata["injected"] = "payload"
    assert ledger.verify_chain() is False


def test_breaking_previous_hash_link_breaks_chain():
    ledger = _build_ledger()
    ledger._events[2].previous_hash = "GENESIS"
    assert ledger.verify_chain() is False
