"""WP8: the Gate A pipeline — evidence-only answers, honest UNKNOWN, gates."""

from amali.activation.planner import ActivationPlan, ActivationRequest, DWACPlanner
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
from amali.policy.registry import ManifestRegistry
from amali.retrieval.hsgr import HSGRRetriever, RetrievalResult
from amali.router.router import HERMoERouter
from amali.runtime.answerer import DeterministicAnswerer
from amali.runtime.gate_a import (
    DETERMINISTIC_MODEL_ID,
    build_gate_a_orchestrator,
    default_corpus,
    default_packs,
    deterministic_model_manifest,
    deterministic_route,
)
from amali.runtime.orchestrator import RuntimeOrchestrator

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


class _ExcludeDocRetriever(HSGRRetriever):
    """Test helper: retrieval omits one corpus doc the answerer would need."""

    def __init__(self, corpus, exclude_doc_id: str) -> None:
        super().__init__(corpus)
        self._exclude_doc_id = exclude_doc_id

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        result = super().retrieve(query, k=k)
        ranked = [s for s in result.ranked if s.doc_id != self._exclude_doc_id]
        return RetrievalResult(query=query, ranked=ranked)


def _orchestrator_with_planner_and_retriever(planner, retriever):
    ledger = AuditLedger()
    registry = ManifestRegistry.load_from_dir(MANIFESTS_DIR)
    router = HERMoERouter(
        routes=[deterministic_route()],
        models={DETERMINISTIC_MODEL_ID: deterministic_model_manifest()},
        ledger=ledger,
    )
    return RuntimeOrchestrator(
        ledger=ledger,
        registry=registry,
        router=router,
        planner=planner,
        corpus=default_corpus(),
        retriever=retriever,
    )


def test_answer_requires_retrieved_evidence_not_full_corpus():
    """Corpus holds the answer, but HSGR did not retrieve it -> no SUCCESS."""
    retriever = _ExcludeDocRetriever(default_corpus(), "audit_genesis")
    orch = _orchestrator_with_planner_and_retriever(
        DWACPlanner(default_packs(), fallback_pack_id="deterministic_answerer_pack"),
        retriever,
    )
    result = orch.run_task(_request("What is the audit ledger genesis hash?"))
    assert result.status is not ResponseStatus.SUCCESS
    assert result.answer_text == ""


class _InfeasiblePlanner:
    def plan(self, request: ActivationRequest) -> ActivationPlan:
        return ActivationPlan(
            feasible=False,
            budget_mb=request.budget_mb,
            uncovered_capabilities=list(request.required_capabilities),
            reasons=["TEST_INFEASIBLE"],
        )


def test_infeasible_activation_plan_never_success():
    orch = _orchestrator_with_planner_and_retriever(
        _InfeasiblePlanner(),
        HSGRRetriever(default_corpus()),
    )
    result = orch.run_task(_request("What is the audit ledger genesis hash?"))
    assert result.status is ResponseStatus.NEEDS_REVIEW
    assert "ACTIVATION_INFEASIBLE" in [
        i.failure_kind for i in result.fec.items
    ]


class _FallbackUncoveredPlanner:
    def plan(self, request: ActivationRequest) -> ActivationPlan:
        return ActivationPlan(
            feasible=True,
            fallback_used=True,
            selected_pack_ids=["deterministic_answerer_pack"],
            total_cost_mb=64,
            budget_mb=request.budget_mb,
            covered_capabilities=["answer_from_evidence"],
            uncovered_capabilities=["retrieve"],
            reasons=["TEST_FALLBACK"],
        )


def test_fallback_with_uncovered_capabilities_not_success():
    orch = _orchestrator_with_planner_and_retriever(
        _FallbackUncoveredPlanner(),
        HSGRRetriever(default_corpus()),
    )
    result = orch.run_task(_request("What is the audit ledger genesis hash?"))
    assert result.status is ResponseStatus.PARTIAL
