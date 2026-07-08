"""Data Wall: operation matrix, redaction, block, holdout isolation."""

from amali.audit.ledger import AuditLedger
from amali.data_wall import DataFlow, DataWall, WallOutcome
from amali.policy.enums import DataSensitivity


def _wall():
    ledger = AuditLedger()
    return DataWall(ledger), ledger


# --- operation matrix ------------------------------------------------------


def test_public_runtime_allowed():
    wall, _ = _wall()
    d = wall.check(
        flow=DataFlow.RUNTIME,
        sensitivity=DataSensitivity.PUBLIC,
        payload={"text": "hello"},
    )
    assert d.outcome is WallOutcome.ALLOW
    assert d.payload == {"text": "hello"}


def test_confidential_runtime_redacted():
    wall, _ = _wall()
    d = wall.check(
        flow=DataFlow.RUNTIME,
        sensitivity=DataSensitivity.CONFIDENTIAL,
        payload={"password": "hunter2", "note": "fine"},
    )
    assert d.outcome is WallOutcome.REDACT
    assert d.payload["password"] == "[REDACTED]"
    assert d.payload["note"] == "fine"


def test_personal_training_blocked_by_default_matrix():
    wall, _ = _wall()
    d = wall.check(
        flow=DataFlow.TRAINING,
        sensitivity=DataSensitivity.PERSONAL,
        payload={"name": "alice"},
    )
    assert d.outcome is WallOutcome.BLOCK
    assert d.payload is None
    assert "DEFAULT_BLOCK_NOT_IN_MATRIX" in d.reasons


def test_secret_blocked_in_every_flow():
    wall, _ = _wall()
    for flow in DataFlow:
        d = wall.check(
            flow=flow,
            sensitivity=DataSensitivity.SECRET,
            payload={"k": "v"},
        )
        assert d.outcome is WallOutcome.BLOCK, flow
        assert "SECRET_ALWAYS_BLOCKED" in d.reasons
        assert d.payload is None


# --- content veto ----------------------------------------------------------


def test_secret_shaped_content_blocks_allow_cell():
    wall, _ = _wall()
    d = wall.check(
        flow=DataFlow.RUNTIME,
        sensitivity=DataSensitivity.PUBLIC,  # claims public...
        payload={"note": "Bearer abcdefgh12345678"},  # ...but carries a token
    )
    assert d.outcome is WallOutcome.BLOCK
    assert "SECRET_SHAPED_CONTENT_DETECTED" in d.reasons


def test_redaction_does_not_mutate_original():
    wall, _ = _wall()
    payload = {"password": "hunter2"}
    wall.check(
        flow=DataFlow.RUNTIME,
        sensitivity=DataSensitivity.CONFIDENTIAL,
        payload=payload,
    )
    assert payload["password"] == "hunter2"  # original untouched


# --- holdout isolation -----------------------------------------------------


def test_holdout_blocked_from_training_even_if_public():
    wall, _ = _wall()
    d = wall.check(
        flow=DataFlow.TRAINING,
        sensitivity=DataSensitivity.PUBLIC,
        payload={"q": "eval item"},
        is_holdout=True,
    )
    assert d.outcome is WallOutcome.BLOCK
    assert "HOLDOUT_ISOLATION" in d.reasons


def test_holdout_blocked_from_benchmark():
    wall, _ = _wall()
    d = wall.check(
        flow=DataFlow.BENCHMARK,
        sensitivity=DataSensitivity.PUBLIC,
        payload={"q": "eval item"},
        is_holdout=True,
    )
    assert d.outcome is WallOutcome.BLOCK


def test_holdout_still_allowed_in_eval():
    wall, _ = _wall()
    d = wall.check(
        flow=DataFlow.EVAL,
        sensitivity=DataSensitivity.PUBLIC,
        payload={"q": "eval item"},
        is_holdout=True,
    )
    assert d.outcome is WallOutcome.ALLOW


# --- audit -----------------------------------------------------------------


def test_every_decision_is_audited_and_payload_free():
    wall, ledger = _wall()
    wall.check(
        flow=DataFlow.RUNTIME,
        sensitivity=DataSensitivity.PUBLIC,
        payload={"api_key": "sk-0123456789abcdef0123"},
    )
    events = ledger.events()
    assert len(events) == 1
    assert events[0].event_type == "data_wall.decision"
    import json

    blob = json.dumps(events[0].metadata, default=str)
    assert "sk-0123456789abcdef0123" not in blob
    assert ledger.verify_chain() is True
