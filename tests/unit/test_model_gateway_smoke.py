"""Gateway smoke: availability can never fake a SUCCESS generation."""

from __future__ import annotations

from pathlib import Path

from amali.model_gateway.smoke import (
    SMOKE_EXPECTED,
    GatewaySmokeReport,
    run_gateway_smoke,
)

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REV = "a" * 40

FULL_SNAPSHOT = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors",
]


def _cache(tmp_path: Path, files: list[str]) -> Path:
    snap = (
        tmp_path
        / ("models--" + MODEL.replace("/", "--"))
        / "snapshots"
        / REV
    )
    snap.mkdir(parents=True)
    for name in files:
        (snap / name).write_text("x", encoding="utf-8")
    return tmp_path


def _fake_success_loader(model_id, revision, local_files_only, max_new_tokens, report):
    return report.model_copy(
        update={
            "status": "SUCCESS",
            "tokenizer_loaded": True,
            "model_loaded": True,
            "generation_executed": True,
            "output_text": SMOKE_EXPECTED,
            "output_matched": True,
        }
    )


def test_non_allowlisted_model_is_policy_blocked(tmp_path):
    report = run_gateway_smoke(
        "evil/backdoored-model",
        REV,
        cache_dir=tmp_path,
        load_fn=_fake_success_loader,
    )
    assert report.status == "POLICY_BLOCKED"
    assert not report.generation_executed


def test_missing_weights_cannot_reach_the_loader(tmp_path):
    # even a loader that would report SUCCESS is never invoked without
    # a real, complete local snapshot — fake availability cannot pass.
    report = run_gateway_smoke(
        MODEL, REV, cache_dir=tmp_path, load_fn=_fake_success_loader
    )
    assert report.status == "MODEL_NOT_AVAILABLE"
    assert not report.generation_executed


def test_incomplete_snapshot_is_typed_not_success(tmp_path):
    cache = _cache(tmp_path, ["config.json", "model.safetensors"])
    report = run_gateway_smoke(
        MODEL, REV, cache_dir=cache, load_fn=_fake_success_loader
    )
    assert report.status == "MODEL_INCOMPLETE"


def test_revision_mismatch_is_typed(tmp_path):
    cache = _cache(tmp_path, FULL_SNAPSHOT)
    report = run_gateway_smoke(
        MODEL, "b" * 40, cache_dir=cache, load_fn=_fake_success_loader
    )
    assert report.status == "REVISION_MISMATCH"


def test_missing_deps_reported(monkeypatch, tmp_path):
    cache = _cache(tmp_path, FULL_SNAPSHOT)
    monkeypatch.setattr(
        "amali.model_gateway.smoke._deps_missing", lambda: ["torch"]
    )
    report = run_gateway_smoke(
        MODEL, REV, cache_dir=cache, load_fn=_fake_success_loader
    )
    assert report.status == "DEPS_MISSING"
    assert report.owner_actions


def test_real_snapshot_reaches_loader_and_records_generation(tmp_path):
    cache = _cache(tmp_path, FULL_SNAPSHOT)
    report = run_gateway_smoke(
        MODEL, REV, cache_dir=cache, load_fn=_fake_success_loader
    )
    assert report.status == "SUCCESS"
    assert report.generation_executed
    assert report.output_matched


def test_mismatched_output_is_recorded_but_still_a_real_generation(tmp_path):
    def loader(model_id, revision, local_files_only, max_new_tokens, report):
        return report.model_copy(
            update={
                "status": "SUCCESS",
                "tokenizer_loaded": True,
                "model_loaded": True,
                "generation_executed": True,
                "output_text": "something else entirely",
                "output_matched": False,
            }
        )

    cache = _cache(tmp_path, FULL_SNAPSHOT)
    report = run_gateway_smoke(MODEL, REV, cache_dir=cache, load_fn=loader)
    assert report.status == "SUCCESS"
    assert not report.output_matched
    assert report.output_text == "something else entirely"


def test_load_failure_is_typed_not_success(tmp_path):
    def loader(model_id, revision, local_files_only, max_new_tokens, report):
        return report.model_copy(
            update={
                "status": "MODEL_LOAD_FAILED",
                "tokenizer_loaded": True,
                "reasons": ["model load failed: simulated OOM"],
            }
        )

    cache = _cache(tmp_path, FULL_SNAPSHOT)
    report = run_gateway_smoke(MODEL, REV, cache_dir=cache, load_fn=loader)
    assert report.status == "MODEL_LOAD_FAILED"
    assert not report.generation_executed


def test_report_schema_stable():
    report = GatewaySmokeReport(status="DEPS_MISSING", model_id=MODEL)
    dumped = report.model_dump(mode="json")
    for field in (
        "status",
        "model_id",
        "revision",
        "local_files_only",
        "tokenizer_loaded",
        "model_loaded",
        "generation_executed",
        "output_text",
        "output_matched",
        "device",
        "latency_ms",
    ):
        assert field in dumped
