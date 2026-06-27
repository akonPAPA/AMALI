"""Hash-chaining links events correctly and verifies."""

from amali.audit.ledger import GENESIS_HASH, AuditLedger


def test_chain_links_and_verifies():
    ledger = AuditLedger()
    e1 = ledger.append("a", "actor", "act", "t1")
    e2 = ledger.append("b", "actor", "act", "t2")
    e3 = ledger.append("c", "actor", "act", "t3", metadata={"k": "v"})

    # First event links to GENESIS; each subsequent event links to the
    # previous event's hash.
    assert e1.previous_hash == GENESIS_HASH
    assert e2.previous_hash == e1.event_hash
    assert e3.previous_hash == e2.event_hash

    assert len({e1.event_hash, e2.event_hash, e3.event_hash}) == 3
    assert ledger.verify_chain() is True


def test_empty_ledger_verifies():
    assert AuditLedger().verify_chain() is True


def test_events_returns_defensive_copy():
    ledger = AuditLedger()
    ledger.append("a", "actor", "act", "t1")
    snapshot = ledger.events()
    snapshot.clear()
    # Clearing the returned list must not affect the ledger.
    assert len(ledger.events()) == 1
