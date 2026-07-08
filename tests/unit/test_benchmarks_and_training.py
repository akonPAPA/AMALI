"""WP9 benchmark/comparison honesty + WP10 training readiness."""

from amali.benchmarks import (
    BenchmarkItem,
    compare_deterministic_baseline,
    run_benchmark,
    score_answer,
)
from amali.benchmarks.comparison import (
    STATUS_NOT_PROVEN,
    external_comparison_report,
)
from amali.runtime.gate_a import build_gate_a_orchestrator, default_corpus
from amali.training import (
    DatasetManifest,
    HoldoutManifest,
    check_contamination,
    training_readiness,
)

from tests.conftest import MANIFESTS_DIR

ITEMS = [
    BenchmarkItem(
        item_id="genesis",
        query="What is the audit ledger genesis hash?",
        expected_terms=["GENESIS"],
    ),
    BenchmarkItem(
        item_id="honesty",
        query="What is the capital city of the moon federation?",
        expected_terms=[],
        expect_answer=False,
    ),
]


# --- scorer ------------------------------------------------------------------


def test_scorer_is_deterministic_term_coverage():
    assert score_answer("hash is GENESIS", ["GENESIS"]) == 1.0
    assert score_answer("hash is GENESIS", ["GENESIS", "sha-256"]) == 0.5
    assert score_answer("nothing relevant", ["GENESIS"]) == 0.0


# --- benchmark ---------------------------------------------------------------


def test_benchmark_passes_answer_and_honest_unknown():
    orch = build_gate_a_orchestrator(MANIFESTS_DIR)
    report = run_benchmark(orch, ITEMS)
    assert report.total == 2
    assert report.pass_rate == 1.0  # both the answer and the honest UNKNOWN
    assert report.unsupported_success_rate == 0.0
    by_id = {r.item_id: r for r in report.results}
    assert by_id["genesis"].status.value == "SUCCESS"
    assert by_id["honesty"].status.value == "UNKNOWN"


def test_benchmark_scores_reproducible():
    orch = build_gate_a_orchestrator(MANIFESTS_DIR)
    a = run_benchmark(orch, ITEMS)
    b = run_benchmark(orch, ITEMS)
    assert [r.score for r in a.results] == [r.score for r in b.results]
    assert a.pass_rate == b.pass_rate


# --- comparison honesty ------------------------------------------------------


def test_deterministic_comparison_reports_local_claim_only():
    orch = build_gate_a_orchestrator(MANIFESTS_DIR)
    report = compare_deterministic_baseline(orch, ITEMS, default_corpus())
    assert report.mode == "deterministic_comparison"
    assert report.status == "PROVEN_LOCAL_DETERMINISTIC"
    # AMALI wins the honesty item: the naive baseline always answers.
    per = {p["item_id"]: p for p in report.per_item}
    assert per["honesty"]["amali_score"] == 1.0
    assert per["honesty"]["baseline_score"] == 0.0
    assert "external" not in report.claim.split(";")[0]


def test_missing_external_baseline_is_not_proven():
    report = external_comparison_report(
        "fixtures/baselines", "glm_5_2_does_not_exist"
    )
    assert report.status == STATUS_NOT_PROVEN
    assert report.baseline_source is None
    assert "NOT_PROVEN" in report.claim


# --- training readiness ------------------------------------------------------


def _dataset(hashes):
    return DatasetManifest(
        dataset_id="ds1", version="0.1.0", item_hashes=hashes
    )


def _holdout(hashes):
    return HoldoutManifest(
        holdout_id="ho1", version="0.1.0", item_hashes=hashes
    )


def test_contamination_detected():
    contaminated = check_contamination(
        _dataset(["a", "b", "c"]), _holdout(["c", "d"])
    )
    assert contaminated == ["c"]


def test_clean_split_but_still_not_ready():
    report = training_readiness(_dataset(["a", "b"]), _holdout(["c"]))
    assert report.contamination_free is True
    # Honest gate: no eval suite, no owner approval, no pipeline -> NOT_READY.
    assert report.gate_status == "NOT_READY"
    assert report.gates["training_pipeline_implemented"] is False


def test_contaminated_split_is_not_ready_and_names_hashes():
    report = training_readiness(
        _dataset(["a", "x"]),
        _holdout(["x"]),
        eval_suite_exists=True,
        owner_training_approval=True,
    )
    assert report.gate_status == "NOT_READY"
    assert report.contaminated_hashes == ["x"]
