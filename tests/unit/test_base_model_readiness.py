"""Base Model Readiness decision matrix: first missing gate names the verdict."""

from __future__ import annotations

from amali.model_gateway.base_readiness import (
    BaseModelReadinessInputs,
    decide_base_model_readiness,
)

FORBIDDEN_WORDS = ("agi", "asi", "frontier", "beats gpt", "beats claude")


def _all_green(**overrides) -> BaseModelReadinessInputs:
    values = dict(
        gate_a_status="PASS",
        core_import_torch_free=True,
        revision_pin_status="REVISION_PINNED",
        model_availability_status="AVAILABLE",
        gateway_smoke_status="SUCCESS",
        raw_eval_status="PASS",
        wrapped_eval_status="PASS",
        comparison_status="PASS",
        security_validation_status="PASS",
        promotion_decision="NOT_RUN",
    )
    values.update(overrides)
    return BaseModelReadinessInputs(**values)


def test_all_prerequisites_ready():
    report = decide_base_model_readiness(_all_green())
    assert report.status == "BASE_MODEL_READY"
    assert report.base_model_ready


def test_ready_base_model_never_implies_ft_exists():
    report = decide_base_model_readiness(_all_green())
    assert report.amali_ft_status == "AMALI_FT_NOT_READY"


def test_ft_ready_only_after_real_promotion():
    report = decide_base_model_readiness(
        _all_green(promotion_decision="PROMOTED")
    )
    assert report.amali_ft_status == "AMALI_FT_READY"
    # anything but PROMOTED stays NOT_READY
    for decision in ("REFUSED", "NOT_READY", "NOT_RUN", "SECURITY_BLOCKED"):
        assert (
            decide_base_model_readiness(
                _all_green(promotion_decision=decision)
            ).amali_ft_status
            == "AMALI_FT_NOT_READY"
        )


def test_gate_a_failure_is_core_regression():
    report = decide_base_model_readiness(_all_green(gate_a_status="FAIL"))
    assert report.status == "CORE_REGRESSION_FAILED"
    assert not report.base_model_ready


def test_torch_in_core_import_is_core_regression():
    report = decide_base_model_readiness(
        _all_green(core_import_torch_free=False)
    )
    assert report.status == "CORE_REGRESSION_FAILED"


def test_missing_pin_requires_owner_revision():
    for pin in ("NOT_RUN", "OWNER_MODEL_REVISION_REQUIRED",
                "REVISION_FLOATING_REFUSED"):
        report = decide_base_model_readiness(
            _all_green(revision_pin_status=pin)
        )
        assert report.status == "OWNER_MODEL_REVISION_REQUIRED"
        assert report.owner_actions


def test_missing_weights_not_ready():
    for availability in ("MODEL_NOT_AVAILABLE", "MODEL_INCOMPLETE",
                         "REVISION_MISMATCH", "NOT_RUN"):
        report = decide_base_model_readiness(
            _all_green(model_availability_status=availability)
        )
        assert report.status == "MODEL_NOT_AVAILABLE"
        assert not report.base_model_ready


def test_failed_smoke_is_gateway_failed():
    report = decide_base_model_readiness(
        _all_green(gateway_smoke_status="MODEL_LOAD_FAILED")
    )
    assert report.status == "MODEL_GATEWAY_FAILED"


def test_missing_raw_eval_not_ready():
    report = decide_base_model_readiness(_all_green(raw_eval_status="NOT_RUN"))
    assert report.status == "RAW_EVAL_NOT_RUN"


def test_missing_wrapped_eval_not_ready():
    report = decide_base_model_readiness(
        _all_green(wrapped_eval_status="NOT_RUN")
    )
    assert report.status == "WRAPPED_EVAL_NOT_RUN"


def test_missing_comparison_not_ready():
    report = decide_base_model_readiness(
        _all_green(comparison_status="NOT_RUN")
    )
    assert report.status == "BASE_MODEL_NOT_READY"


def test_comparison_mismatch_not_ready():
    report = decide_base_model_readiness(_all_green(comparison_status="FAIL"))
    assert report.status == "BASE_MODEL_NOT_READY"


def test_failed_security_validation_blocks():
    for status in ("FAIL", "NOT_RUN"):
        report = decide_base_model_readiness(
            _all_green(security_validation_status=status)
        )
        assert report.status == "SECURITY_VALIDATION_FAILED"


def test_report_wording_has_no_forbidden_claims():
    for inputs in (_all_green(), _all_green(raw_eval_status="NOT_RUN")):
        report = decide_base_model_readiness(inputs)
        text = " ".join(
            report.reasons + report.owner_actions + [report.status]
        ).lower()
        for word in FORBIDDEN_WORDS:
            assert word not in text


def test_checks_table_lists_every_gate():
    report = decide_base_model_readiness(_all_green())
    for gate in (
        "gate_a",
        "core_import_torch_free",
        "revision_pin",
        "model_availability",
        "gateway_smoke",
        "raw_base_eval",
        "wrapped_raw_base_eval",
        "comparison",
        "security_validation",
    ):
        assert gate in report.checks
