"""Revision pinning: only an allowlisted model + exact SHA pins."""

from __future__ import annotations

import json

from amali.model_gateway.revision_pin import (
    STATUS_MODEL_NOT_ALLOWLISTED,
    STATUS_OWNER_MODEL_REVISION_REQUIRED,
    STATUS_REVISION_FLOATING_REFUSED,
    STATUS_REVISION_PINNED,
    decide_revision_pin,
    load_latest_pin,
)

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SHA = "0123abc0123abc0123abc0123abc0123abc01234".replace("g", "0")
VALID_SHA = "a" * 40


def test_valid_sha_is_pinned_and_promotable():
    report = decide_revision_pin(MODEL, VALID_SHA)
    assert report.status == STATUS_REVISION_PINNED
    assert report.promotion_eligible
    assert report.revision == VALID_SHA
    assert report.pinned_at


def test_main_refused():
    report = decide_revision_pin(MODEL, "main")
    assert report.status == STATUS_REVISION_FLOATING_REFUSED
    assert not report.promotion_eligible


def test_latest_refused():
    assert (
        decide_revision_pin(MODEL, "latest").status
        == STATUS_REVISION_FLOATING_REFUSED
    )


def test_head_refused():
    assert (
        decide_revision_pin(MODEL, "HEAD").status
        == STATUS_REVISION_FLOATING_REFUSED
    )


def test_short_or_non_hex_revision_refused():
    assert (
        decide_revision_pin(MODEL, "abc123").status
        == STATUS_REVISION_FLOATING_REFUSED
    )
    assert (
        decide_revision_pin(MODEL, "v1.0.0").status
        == STATUS_REVISION_FLOATING_REFUSED
    )
    # 40 chars but not hex
    assert (
        decide_revision_pin(MODEL, "z" * 40).status
        == STATUS_REVISION_FLOATING_REFUSED
    )


def test_empty_revision_requires_owner_action():
    for empty in (None, "", "   "):
        report = decide_revision_pin(MODEL, empty)
        assert report.status == STATUS_OWNER_MODEL_REVISION_REQUIRED
        assert report.owner_actions


def test_unknown_model_refused():
    report = decide_revision_pin("evil/backdoored-model", VALID_SHA)
    assert report.status == STATUS_MODEL_NOT_ALLOWLISTED
    assert not report.promotion_eligible


def test_eval_only_model_pins_but_is_not_trainable():
    report = decide_revision_pin("Qwen/Qwen2.5-3B-Instruct", VALID_SHA)
    assert report.status == STATUS_REVISION_PINNED
    assert report.role == "eval_only"
    assert not report.permitted_for_training


def _write_pin_artifact(root, timestamp: str, decision: dict) -> None:
    out = root / timestamp
    out.mkdir(parents=True)
    (out / "base_model_revision_pin_report.json").write_text(
        json.dumps({"status": decision["status"], "decisions": [decision]}),
        encoding="utf-8",
    )


def test_load_latest_pin_returns_newest_successful_pin(tmp_path):
    pinned = decide_revision_pin(MODEL, VALID_SHA).model_dump(mode="json")
    refused = decide_revision_pin(MODEL, "main").model_dump(mode="json")
    _write_pin_artifact(tmp_path, "20260101T000000Z", pinned)
    _write_pin_artifact(tmp_path, "20260102T000000Z", refused)

    # the newer refusal never overrides the older real pin
    latest = load_latest_pin(tmp_path)
    assert latest is not None
    assert latest.status == STATUS_REVISION_PINNED
    assert latest.revision == VALID_SHA


def test_load_latest_pin_none_when_no_pin_exists(tmp_path):
    refused = decide_revision_pin(MODEL, "main").model_dump(mode="json")
    _write_pin_artifact(tmp_path, "20260101T000000Z", refused)
    assert load_latest_pin(tmp_path) is None
    assert load_latest_pin(tmp_path / "does-not-exist") is None


def test_report_schema_stable():
    report = decide_revision_pin(MODEL, VALID_SHA)
    dumped = report.model_dump(mode="json")
    for field in (
        "status",
        "model_id",
        "revision",
        "role",
        "license_label",
        "promotion_eligible",
        "permitted_for_training",
        "pinned_at",
        "reasons",
        "owner_actions",
    ):
        assert field in dumped
