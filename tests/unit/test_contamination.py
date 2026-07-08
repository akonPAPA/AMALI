"""Phase 3: contamination gate — planted leaks must be caught."""

from __future__ import annotations

from amali.eval.suite import EvalItem, EvalSection
from amali.training.contamination import (
    check_contamination,
    jaccard_shingles,
    normalized_hash,
)
from amali.training.dataset import TrainingExample


def _example(example_id: str, **overrides) -> TrainingExample:
    defaults = dict(
        example_id=example_id,
        source_id="docs/sample.md",
        source_hash="a" * 64,
        source_path="docs/sample.md",
        data_class="documentation",
        synthetic=False,
        instruction="Restate the passage faithfully and cite the source.",
        input_context=(
            "The audit ledger chains every event with SHA-256 hashing so "
            "that mutation of history is detectable by verification."
        ),
        expected_response=(
            "The audit ledger chains every event with SHA-256 hashing so "
            "that mutation of history is detectable by verification."
        ),
        expected_status="SUCCESS",
        required_citations=["docs/sample.md"],
        forbidden_outputs=[],
        behavior_tags=["evidence_grounding"],
        data_wall_decision_id="decision_x",
        content_hash="b" * 64,
    )
    defaults.update(overrides)
    return TrainingExample(**defaults)


def _eval_item(item_id: str, prompt: str, **overrides) -> EvalItem:
    defaults = dict(
        item_id=item_id,
        section=EvalSection.EVIDENCE_GROUNDED_QA,
        prompt=prompt,
        expected_behavior="answers",
        expected_status="SUCCESS",
        scoring_rubric="term coverage",
    )
    defaults.update(overrides)
    return EvalItem(**defaults)


EVAL_PROMPT = (
    "Evidence [doc_1]: 'The Data Wall blocks secret data in every flow "
    "without exception.' Question: what happens to secret data at the "
    "wall? Cite the evidence id."
)


def test_clean_split_passes():
    report = check_contamination(
        [_example("t1")], [_eval_item("e1", EVAL_PROMPT)]
    )
    assert report.status == "PASS"
    assert report.findings == []


def test_planted_exact_duplicate_detected():
    dirty = _example(
        "t_exact",
        input_context=EVAL_PROMPT,
    )
    report = check_contamination(
        [dirty], [_eval_item("e1", EVAL_PROMPT)]
    )
    assert report.status == "FAIL"
    assert any(f.kind == "exact_hash_overlap" for f in report.findings)
    assert "t_exact" in report.contaminated_ids


def test_planted_normalized_duplicate_detected():
    dirty = _example(
        "t_norm",
        input_context=EVAL_PROMPT.upper().replace(".", " . "),
    )
    report = check_contamination(
        [dirty], [_eval_item("e1", EVAL_PROMPT)]
    )
    assert report.status == "FAIL"
    assert any(
        f.kind in ("normalized_hash_overlap", "near_duplicate")
        for f in report.findings
    )


def test_planted_paraphrase_detected_as_near_duplicate():
    paraphrase = (
        "Evidence [doc_1]: 'The Data Wall blocks secret data in every "
        "flow without exception.' Question: what happens to the secret "
        "data at the wall? Please cite the evidence id."
    )
    dirty = _example("t_para", input_context=paraphrase)
    report = check_contamination(
        [dirty], [_eval_item("e1", EVAL_PROMPT)]
    )
    assert report.status == "FAIL"
    assert any(f.kind == "near_duplicate" for f in report.findings)


def test_forbidden_source_path_detected():
    dirty = _example(
        "t_path",
        source_id="fixtures/eval/base_model_gate_seed.json",
        source_path="fixtures/eval/base_model_gate_seed.json",
    )
    report = check_contamination(
        [dirty], [_eval_item("e1", EVAL_PROMPT)]
    )
    assert report.status == "FAIL"
    assert any(f.kind == "forbidden_source_path" for f in report.findings)


def test_holdout_overlap_detected():
    holdout_text = (
        "Reserved holdout question about the arbitration hard gates and "
        "their exact ordering inside the EWA pipeline components."
    )
    dirty = _example("t_hold", input_context=holdout_text)
    report = check_contamination(
        [dirty],
        [_eval_item("e1", EVAL_PROMPT)],
        holdout_texts=[holdout_text],
    )
    assert report.status == "FAIL"
    assert any(f.kind == "holdout_overlap" for f in report.findings)


def test_contaminated_items_reported_not_silently_removed():
    dirty = _example("t_exact", input_context=EVAL_PROMPT)
    clean = _example("t_clean")
    report = check_contamination(
        [dirty, clean], [_eval_item("e1", EVAL_PROMPT)]
    )
    assert report.status == "FAIL"
    # the clean item is still counted; nothing was dropped to pass
    assert report.training_examples == 2
    assert report.contaminated_ids == ["t_exact"]


def test_empty_inputs_are_not_run():
    report = check_contamination([], [])
    assert report.status == "NOT_RUN"


def test_report_schema_stable():
    report = check_contamination(
        [_example("t1")], [_eval_item("e1", EVAL_PROMPT)]
    )
    dumped = report.model_dump(mode="json")
    assert set(dumped) == {
        "status",
        "training_examples",
        "eval_items",
        "findings",
    }


def test_similarity_helpers_deterministic():
    a = "the data wall blocks secret data in every flow"
    b = "the data wall blocks secret data in every flow without fail"
    assert normalized_hash(a) == normalized_hash("The  DATA wall blocks secret data in every flow!")
    s1 = jaccard_shingles(a, b)
    s2 = jaccard_shingles(a, b)
    assert s1 == s2
    assert 0.0 < s1 < 1.0
