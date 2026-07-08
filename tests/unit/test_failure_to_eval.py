"""RICARDO-FEC: redacted, deduplicated eval item creation."""

from amali.eval import FailureTrace, compile_failures


def _trace(**overrides) -> FailureTrace:
    defaults = dict(
        task_id="task_1",
        component="verifier",
        failure_kind="UNSUPPORTED_CLAIM",
        query="What is the genesis hash?",
        observed="claim unsupported by evidence",
    )
    defaults.update(overrides)
    return FailureTrace(**defaults)


def test_single_failure_becomes_eval_item():
    result = compile_failures([_trace()])
    assert result.unique_items == 1
    assert result.conversion_rate == 1.0
    item = result.items[0]
    assert item.eval_id.startswith("eval_")
    assert item.regression_guard is True
    assert item.source_task_ids == ["task_1"]


def test_duplicates_are_merged_not_repeated():
    result = compile_failures(
        [_trace(), _trace(task_id="task_2"), _trace(task_id="task_1")]
    )
    assert result.total_traces == 3
    assert result.unique_items == 1
    item = result.items[0]
    assert item.duplicate_count == 3
    assert set(item.source_task_ids) == {"task_1", "task_2"}
    assert result.conversion_rate == round(1 / 3, 6)


def test_different_failures_stay_distinct():
    result = compile_failures(
        [
            _trace(),
            _trace(failure_kind="WALL_BLOCK"),
            _trace(component="policy"),
        ]
    )
    assert result.unique_items == 3


def test_eval_ids_are_stable_across_runs():
    a = compile_failures([_trace()]).items[0].eval_id
    b = compile_failures([_trace()]).items[0].eval_id
    assert a == b


def test_secret_material_is_redacted_from_items():
    result = compile_failures(
        [
            _trace(
                query="call failed with Bearer abcdefgh12345678 token",
                observed="header Bearer abcdefgh12345678 rejected",
                expected="no Bearer abcdefgh12345678 anywhere",
            )
        ]
    )
    item = result.items[0]
    for text in (item.prompt, item.observed, item.expected):
        assert "abcdefgh12345678" not in text
        assert "[REDACTED]" in text


def test_empty_input_is_honest_zero():
    result = compile_failures([])
    assert result.unique_items == 0
    assert result.conversion_rate == 0.0
