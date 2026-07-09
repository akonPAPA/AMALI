"""Eval reports carry model identity; comparisons refuse mixed identities."""

from __future__ import annotations

from amali.benchmarks.comparison import (
    base_readiness_comparison,
    three_way_comparison,
)
from amali.eval.model_eval import (
    FakeDeterministicBackend,
    ModelEvalReport,
    evaluate_model,
)
from amali.eval.suite import EvalItem, EvalSection

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REV_A = "a" * 40
REV_B = "b" * 40

METRICS = {
    "overall_score": 0.5,
    "honest_unknown_score": 0.5,
    "safety_score": 0.5,
    "unsupported_success_rate": 0.0,
}


def _items() -> list[EvalItem]:
    return [
        EvalItem(
            item_id="it_001",
            section=EvalSection.HONEST_UNKNOWN,
            prompt="What is unknowable?",
            expected_behavior="honest unknown",
            expected_status="UNKNOWN",
        )
    ]


def _report(mode: str, **overrides) -> ModelEvalReport:
    values = dict(
        mode=mode,
        status="PASS",
        model_id=MODEL,
        revision=REV_A,
        suite_hash="suite_x",
        metrics=dict(METRICS),
    )
    values.update(overrides)
    return ModelEvalReport(**values)


def test_evaluate_model_records_identity_and_decoding_config():
    report = evaluate_model(
        mode="raw_base",
        items=_items(),
        backend=FakeDeterministicBackend(),
        suite_hash="suite_x",
        model_id=MODEL,
        revision=REV_A,
        local_files_only=True,
    )
    assert report.model_id == MODEL
    assert report.revision == REV_A
    assert report.local_files_only
    assert report.decoding_config["do_sample"] is False


def test_comparison_fails_on_revision_mismatch():
    comparison = three_way_comparison(
        raw_base=_report("raw_base"),
        wrapped_raw=None,
        wrapped_ft=_report("amali_wrapped_ft_v0", revision=REV_B),
        suite_hash="suite_x",
    )
    assert comparison.status == "FAIL"
    assert any("revision" in note for note in comparison.notes)
    assert comparison.claim == ""


def test_comparison_fails_on_model_id_mismatch():
    comparison = three_way_comparison(
        raw_base=_report("raw_base"),
        wrapped_raw=None,
        wrapped_ft=_report(
            "amali_wrapped_ft_v0", model_id="Qwen/Qwen2.5-0.5B-Instruct"
        ),
    )
    assert comparison.status == "FAIL"
    assert any("model_id" in note for note in comparison.notes)


def test_comparison_fails_on_suite_hash_mismatch():
    comparison = three_way_comparison(
        raw_base=_report("raw_base"),
        wrapped_raw=None,
        wrapped_ft=_report("amali_wrapped_ft_v0", suite_hash="suite_y"),
    )
    assert comparison.status == "FAIL"
    assert any("suite_hash" in note for note in comparison.notes)


def test_comparison_passes_on_consistent_identity():
    comparison = three_way_comparison(
        raw_base=_report("raw_base"),
        wrapped_raw=_report("amali_wrapped_raw_base"),
        wrapped_ft=_report("amali_wrapped_ft_v0"),
        suite_hash="suite_x",
    )
    assert comparison.status == "PASS"


def test_legacy_reports_without_identity_still_compare():
    # older reports carry no model_id/revision; absence is not a mismatch
    comparison = three_way_comparison(
        raw_base=_report("raw_base", model_id="", revision=""),
        wrapped_raw=None,
        wrapped_ft=_report("amali_wrapped_ft_v0", model_id="", revision=""),
    )
    assert comparison.status == "PASS"


def test_base_comparison_fails_on_revision_mismatch():
    comparison = base_readiness_comparison(
        raw_base=_report("raw_base"),
        wrapped_raw=_report("amali_wrapped_raw_base", revision=REV_B),
        suite_hash="suite_x",
    )
    assert comparison.status == "FAIL"
    assert any("revision" in note for note in comparison.notes)
    assert comparison.claim == ""


def test_base_comparison_passes_on_consistent_identity():
    comparison = base_readiness_comparison(
        raw_base=_report("raw_base"),
        wrapped_raw=_report("amali_wrapped_raw_base"),
        suite_hash="suite_x",
    )
    assert comparison.status == "PASS"


def test_non_run_report_carries_no_scores():
    report = ModelEvalReport(
        mode="raw_base",
        status="REVISION_NOT_PINNED",
        model_id=MODEL,
        risks=["no pinned revision"],
    )
    assert report.metrics == {}
    assert report.item_scores == []
