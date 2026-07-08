"""WP8: the Gate A pipeline — evidence-only answers, honest UNKNOWN, gates."""

from amali.arbiter import ResponseStatus
from amali.audit.ledger import AuditLedger
from amali.contracts.task import (
    ActorRef,
    LocalTaskPayload,
    ProjectRef,
    TaskIntent,
    TaskRequest,
)
from amali.policy.enums import DataSensitivity, RiskLevel
from amali.runtime.answerer import DeterministicAnswerer
from amali.runtime.gate_a import build_gate_a_orchestrator, default_corpus

from tests.conftest import MANIFESTS_DIR


def _request(query: str, **overrides) -> TaskRequest:
    defaults = dict(
        actor=ActorRef(actor_id="local_dev_agent", kind="agent"),
        project=ProjectRef(project_id="amali_core"),
        intent=TaskIntent.ANSWER_QUESTION,
        payload=LocalTaskPayload(query=query),
        task_risk=RiskLevel.L1,
        data_sensitivity=DataSensitivity.PUBLIC,
        reason="unit test",
    )
    defaults.update(overrides)
    return TaskRequest(**defaults)


def _orchestrator(ledger=None):
    return build_gate_a_orchestrator(MANIFESTS_DIR, ledger=ledger)


# --- answerer ---------------------------------------------------------------


def test_answerer_quotes_evidence_only():
    answerer = DeterministicAnswerer()
    result = answerer.answer(
        "audit ledger genesis hash", default_corpus()
    )
    assert result.answered is True
    # The answer is a verbatim sentence from the corpus.
    assert any(
        result.text in doc.content for doc in default_corpus()
    )
    assert result.cited_doc_ids == ["audit_genesis"]


def test_answerer_declines_without_evidence():
    answerer = DeterministicAnswerer()
    result = answerer.answer(
        "capital city of the moon federation", default_corpus()
    )
    assert result.answered is False
    assert result.text == ""


# --- full pipeline -----------------------------------------------------------


def test_supported_answer_reaches_success():
    ledger = AuditLedger()
    orch = _orchestrator(ledger)
    result = orch.run_task(
        _request("What is the audit ledger genesis hash?")
    )
    assert result.status is ResponseStatus.SUCCESS
    assert "GENESIS" in result.answer_text
    assert result.arbitration.supported >= 1
    assert result.arbitration.unsupported == 0
    # Trace covers the whole path and the audit chain verifies.
    components = [t.component for t in result.trace]
    for expected in (
        "contracts",
        "data_wall",
        "trpc",
        "her_moe_router",
        "dwac",
        "hsgr_retrieval",
        "answerer",
        "verifier",
        "ewa_arbiter",
        "debate",
        "fec",
    ):
        assert expected in components, expected
    assert ledger.verify_chain() is True


def test_missing_evidence_reports_unknown_not_success():
    orch = _orchestrator()
    result = orch.run_task(
        _request("What is the capital city of the moon federation?")
    )
    assert result.status is ResponseStatus.UNKNOWN
    assert result.answer_text == ""
    # The failure became a redacted eval item.
    assert result.fec.unique_items >= 1
    kinds = [i.failure_kind for i in result.fec.items]
    assert "NO_EVIDENCE_ANSWER" in kinds


def test_secret_query_blocked_at_wall():
    orch = _orchestrator()
    result = orch.run_task(
        _request(
            "please echo Bearer abcdefgh12345678",
            data_sensitivity=DataSensitivity.SECRET,
        )
    )
    assert result.status is ResponseStatus.BLOCKED
    assert result.answer_text == ""
    # Pipeline stopped at the wall: no route decision, no retrieval.
    assert result.route_decision is None
    assert result.retrieval is None


def test_high_risk_task_escalates_before_answering():
    orch = _orchestrator()
    result = orch.run_task(
        _request(
            "What is the audit ledger genesis hash?",
            task_risk=RiskLevel.L3,
        )
    )
    # L3 exceeds the deterministic route ceiling (L2) and TRPC review gate;
    # the pipeline must not return SUCCESS.
    assert result.status in (
        ResponseStatus.NEEDS_REVIEW,
        ResponseStatus.BLOCKED,
    )


def test_pipeline_is_deterministic():
    a = _orchestrator().run_task(
        _request("What is the audit ledger genesis hash?")
    )
    b = _orchestrator().run_task(
        _request("What is the audit ledger genesis hash?")
    )
    assert a.status == b.status
    assert a.answer_text == b.answer_text
    assert [t.component for t in a.trace] == [t.component for t in b.trace]
