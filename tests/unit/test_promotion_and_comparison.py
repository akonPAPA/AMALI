"""Phases 10/11: promotion gate decisions + three-way comparison claims."""

from __future__ import annotations

import pytest

from amali.benchmarks.comparison import (
    FORBIDDEN_CLAIM_MARKERS,
    base_readiness_comparison,
    three_way_comparison,
)
from amali.eval.model_eval import ModelEvalReport
from amali.training.promotion import (
    ALLOWED_CLAIM,
    PromotionEvidence,
    evaluate_promotion,
)

RAW_METRICS = {
    "overall_score": 0.5,
    "evidence_grounded_score": 0.6,
    "honest_unknown_score": 0.4,
    "safety_score": 0.7,
    "tool_permission_score": 0.6,
    "injection_as_data_score": 0.5,
    "unsupported_success_rate": 0.15,
    "refusal_correctness": 0.6,
    "citation_support_score": 0.5,
}

BETTER_METRICS = {
    "overall_score": 0.8,
    "evidence_grounded_score": 0.85,
    "honest_unknown_score": 0.9,
    "safety_score": 0.95,
    "tool_permission_score": 0.9,
    "injection_as_data_score": 0.8,
    "unsupported_success_rate": 0.0,
    "refusal_correctness": 0.95,
    "citation_support_score": 0.9,
}


def _report(mode: str, metrics: dict, status: str = "PASS") -> ModelEvalReport:
    return ModelEvalReport(mode=mode, status=status, metrics=dict(metrics))


def _passing_evidence(**overrides) -> PromotionEvidence:
    defaults = dict(
        gate_a_pass=True,
        torch_free_core_import=True,
        eval_suite_frozen_ok=True,
        dataset_manifest_valid=True,
        contamination_status="PASS",
        training_status="TRAINING_COMPLETED",
        checkpoint_integrity_status="PASS",
        raw_base_report=_report("raw_base", RAW_METRICS),
        ft_report=_report("amali_ft_v0", BETTER_METRICS),
        wrapped_ft_report=_report("amali_wrapped_ft_v0", BETTER_METRICS),
        safety_regression_status="PASS",
        model_card_generated=True,
        audit_chain_verified=True,
    )
    defaults.update(overrides)
    return PromotionEvidence(**defaults)


# --- promotion -----------------------------------------------------------------


def test_fully_passing_fake_candidate_promoted():
    report = evaluate_promotion(_passing_evidence())
    assert report.decision == "PROMOTED", report.failed_gates
    assert report.failed_gates == []
    assert report.allowed_claim == ALLOWED_CLAIM


def test_better_but_unsafe_candidate_refused():
    unsafe = dict(BETTER_METRICS, safety_score=0.5)  # better overall, worse safety
    report = evaluate_promotion(
        _passing_evidence(
            wrapped_ft_report=_report("amali_wrapped_ft_v0", unsafe),
            safety_regression_status="FAIL",
        )
    )
    assert report.decision == "REFUSED"
    assert "10_safety_regression_pass" in report.failed_gates
    assert "12_safety_score_ge_raw" in report.failed_gates
    assert report.allowed_claim == ""


def test_safe_but_not_better_candidate_refused():
    same = dict(RAW_METRICS, unsupported_success_rate=0.0)  # safe, no improvement
    report = evaluate_promotion(
        _passing_evidence(
            wrapped_ft_report=_report("amali_wrapped_ft_v0", same),
        )
    )
    assert report.decision == "REFUSED"
    assert "13_honest_unknown_gt_raw" in report.failed_gates


def test_nonzero_unsupported_success_refused():
    leaky = dict(BETTER_METRICS, unsupported_success_rate=0.05)
    report = evaluate_promotion(
        _passing_evidence(
            wrapped_ft_report=_report("amali_wrapped_ft_v0", leaky),
        )
    )
    assert report.decision == "REFUSED"
    assert "11_wrapped_unsupported_success_zero" in report.failed_gates


def test_missing_eval_is_not_ready():
    report = evaluate_promotion(_passing_evidence(raw_base_report=None))
    assert report.decision == "NOT_READY"
    assert "08_raw_base_eval_exists" in report.failed_gates


def test_training_not_completed_is_not_ready():
    report = evaluate_promotion(
        _passing_evidence(training_status="TRAIN_DEPS_MISSING")
    )
    assert report.decision == "NOT_READY"
    assert "06_training_completed" in report.failed_gates


def test_tampered_checkpoint_refused():
    report = evaluate_promotion(
        _passing_evidence(checkpoint_integrity_status="FAIL")
    )
    assert report.decision == "REFUSED"
    assert "07_checkpoint_registry_valid" in report.failed_gates


def test_forbidden_claim_text_refused():
    report = evaluate_promotion(
        _passing_evidence(claim_text="AMALI beats GPT and is a frontier model")
    )
    assert report.decision == "REFUSED"
    assert "17_no_frontier_superiority_claim" in report.failed_gates


def test_no_partial_promotion_single_gate_failure():
    report = evaluate_promotion(_passing_evidence(model_card_generated=False))
    assert report.decision == "REFUSED"
    assert report.failed_gates == ["18_model_card_generated"]


# --- three-way comparison ---------------------------------------------------------


def test_missing_ft_is_not_run():
    comparison = three_way_comparison(
        raw_base=_report("raw_base", RAW_METRICS),
        wrapped_raw=None,
        wrapped_ft=None,
    )
    assert comparison.status == "NOT_RUN"
    assert comparison.claim == ""


def test_missing_raw_base_is_not_run():
    comparison = three_way_comparison(
        raw_base=None,
        wrapped_raw=None,
        wrapped_ft=_report("amali_wrapped_ft_v0", BETTER_METRICS),
    )
    assert comparison.status == "NOT_RUN"


def test_improved_candidate_generates_allowed_claim_only():
    comparison = three_way_comparison(
        raw_base=_report("raw_base", RAW_METRICS),
        wrapped_raw=_report("amali_wrapped_raw_base", RAW_METRICS),
        wrapped_ft=_report("amali_wrapped_ft_v0", BETTER_METRICS),
        suite_hash="abc",
    )
    assert comparison.status == "PASS"
    assert "improves over the raw same base model" in comparison.claim
    lowered = comparison.claim.lower()
    assert not any(marker in lowered for marker in FORBIDDEN_CLAIM_MARKERS)
    assert comparison.external_status == "NOT_PROVEN"


def test_not_improved_candidate_makes_no_improvement_claim():
    comparison = three_way_comparison(
        raw_base=_report("raw_base", RAW_METRICS),
        wrapped_raw=None,
        wrapped_ft=_report("amali_wrapped_ft_v0", RAW_METRICS),
    )
    assert comparison.status == "PASS"
    assert "did not improve" in comparison.claim


def test_ft_promotion_comparison_still_refuses_missing_wrapped_ft():
    # both base sides PASS; the FT comparison must still be NOT_RUN
    comparison = three_way_comparison(
        raw_base=_report("raw_base", RAW_METRICS),
        wrapped_raw=_report("amali_wrapped_raw_base", BETTER_METRICS),
        wrapped_ft=None,
    )
    assert comparison.status == "NOT_RUN"
    assert any("amali_wrapped_ft_v0" in note for note in comparison.notes)
    assert comparison.claim == ""


def test_external_comparison_stays_not_proven():
    comparison = three_way_comparison(
        raw_base=_report("raw_base", RAW_METRICS),
        wrapped_raw=None,
        wrapped_ft=_report("amali_wrapped_ft_v0", BETTER_METRICS),
    )
    assert comparison.external_status == "NOT_PROVEN"
    assert any("NOT_PROVEN" in note for note in comparison.notes)


# --- base readiness comparison ------------------------------------------------


def test_base_comparison_passes_with_raw_and_wrapped_raw_only():
    comparison = base_readiness_comparison(
        raw_base=_report("raw_base", RAW_METRICS),
        wrapped_raw=_report("amali_wrapped_raw_base", BETTER_METRICS),
        suite_hash="abc",
    )
    assert comparison.status == "PASS"
    assert set(comparison.metrics_by_mode) == {
        "raw_base",
        "amali_wrapped_raw_base",
    }
    lowered = comparison.claim.lower()
    assert not any(marker in lowered for marker in FORBIDDEN_CLAIM_MARKERS)
    assert comparison.external_status == "NOT_PROVEN"


def test_base_comparison_never_requires_wrapped_ft():
    # no FT report exists anywhere, and nothing asks for one
    comparison = base_readiness_comparison(
        raw_base=_report("raw_base", RAW_METRICS),
        wrapped_raw=_report("amali_wrapped_raw_base", BETTER_METRICS),
    )
    assert comparison.status == "PASS"
    assert "amali_wrapped_ft_v0" not in str(comparison.metrics_by_mode)
    assert not any("missing" in note for note in comparison.notes)


def test_base_comparison_missing_wrapped_raw_is_not_run():
    comparison = base_readiness_comparison(
        raw_base=_report("raw_base", RAW_METRICS),
        wrapped_raw=None,
    )
    assert comparison.status == "NOT_RUN"
    assert comparison.claim == ""
    assert any("amali_wrapped_raw_base" in note for note in comparison.notes)


def test_base_comparison_missing_raw_base_is_not_run():
    comparison = base_readiness_comparison(
        raw_base=None,
        wrapped_raw=_report("amali_wrapped_raw_base", BETTER_METRICS),
    )
    assert comparison.status == "NOT_RUN"
    assert any("raw_base" in note for note in comparison.notes)


def test_base_comparison_not_improved_makes_no_improvement_claim():
    comparison = base_readiness_comparison(
        raw_base=_report("raw_base", RAW_METRICS),
        wrapped_raw=_report("amali_wrapped_raw_base", RAW_METRICS),
    )
    assert comparison.status == "PASS"
    assert "did not improve" in comparison.claim
