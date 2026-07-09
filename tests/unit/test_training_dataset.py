"""Phase 2: dataset pipeline through the Data Wall."""

from __future__ import annotations

import json
from pathlib import Path

from amali.audit.ledger import AuditLedger
from amali.training.dataset import (
    build_training_dataset,
    collect_source_files,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

SECRET_TEXT = (
    "Deployment notes for the internal gateway service. "
    "The gateway rotates credentials monthly and the current value is "
    "sk-abcdefghijklmnopqrstuvwx which must be kept out of every dataset. "
    "This paragraph exists to verify that secret-bearing chunks are "
    "blocked at the wall and never written to disk in any output file."
)

CLEAN_TEXT = (
    "AMALI is an owner-governed local AI infrastructure. The audit ledger "
    "chains every event with SHA-256 so that any mutation of history is "
    "detectable. The Data Wall applies a default-block operation matrix "
    "before data enters any flow, and the TRPC permission compiler is "
    "default deny for every tool invocation without a matching manifest."
)


def _build(tmp_path: Path, sources: list[Path]):
    out = tmp_path / "out" / "train.jsonl"
    return build_training_dataset(
        repo_root=REPO_ROOT,
        output_path=out,
        ledger=AuditLedger(),
        source_files=sources,
    ), out


def test_secret_bearing_chunk_blocked_and_absent_from_output(tmp_path):
    src = tmp_path / "secret_doc.md"
    src.write_text(SECRET_TEXT + "\n\n" + CLEAN_TEXT, encoding="utf-8")
    (report, manifest, examples), out = _build(tmp_path, [src])

    assert report.blocked >= 1
    assert all("sk-abcdefghijklmnopqrstuvwx" not in e.input_context for e in examples)
    assert all("sk-abcdefghijklmnopqrstuvwx" not in e.expected_response for e in examples)
    if out.exists():
        assert "sk-abcdefghijklmnopqrstuvwx" not in out.read_text(encoding="utf-8")


def test_holdout_source_blocked(tmp_path):
    fixtures_dir = tmp_path / "fixtures" / "holdout"
    fixtures_dir.mkdir(parents=True)
    src = fixtures_dir / "holdout_doc.md"
    src.write_text(CLEAN_TEXT, encoding="utf-8")
    (report, manifest, examples), _ = build_training_dataset(
        repo_root=tmp_path,
        output_path=tmp_path / "out" / "train.jsonl",
        ledger=AuditLedger(),
        source_files=[src],
    ), None

    non_synthetic = [e for e in examples if not e.synthetic]
    assert non_synthetic == []
    holdout_blocks = [
        d
        for d in report.wall_decisions
        if "HOLDOUT_ISOLATION" in d["reasons"]
    ]
    assert holdout_blocks


def test_duplicate_chunks_deduped(tmp_path):
    a = tmp_path / "doc_a.md"
    b = tmp_path / "doc_b.md"
    a.write_text(CLEAN_TEXT, encoding="utf-8")
    b.write_text(CLEAN_TEXT, encoding="utf-8")
    (report, manifest, examples), _ = _build(tmp_path, [a, b])

    assert report.deduped >= 1
    hashes = [e.content_hash for e in examples]
    assert len(set(hashes)) == len(hashes)


def test_empty_sources_not_ready_without_fabrication(tmp_path):
    # No sources at all: even synthetic behaviors exist, but the report
    # must be honest about the non-synthetic floor.
    (report, manifest, examples), _ = _build(tmp_path, [])
    assert report.status in ("DATASET_NOT_READY", "NOT_READY")
    assert report.non_synthetic_count == 0


def test_manifest_hash_stable_across_rebuilds(tmp_path):
    src = tmp_path / "doc.md"
    src.write_text(CLEAN_TEXT, encoding="utf-8")
    (r1, m1, _), _ = _build(tmp_path, [src])
    (r2, m2, _), _ = _build(tmp_path, [src])
    assert m1 is not None and m2 is not None
    assert m1.dataset_hash == m2.dataset_hash
    assert m1.example_hashes == m2.example_hashes


def test_real_repo_build_is_honest_about_floor(tmp_path):
    report, manifest, examples = build_training_dataset(
        repo_root=REPO_ROOT,
        output_path=tmp_path / "train.jsonl",
        ledger=AuditLedger(),
    )
    # The local corpus is small; the status must reflect reality either way.
    if report.non_synthetic_count >= 200:
        assert report.status == "READY"
    else:
        assert report.status in ("DATASET_NOT_READY", "NOT_READY")
    # Synthetic examples are always labeled.
    for example in examples:
        if example.source_id.startswith("synthetic:"):
            assert example.synthetic is True


def test_collect_source_files_excludes_fixtures():
    files = collect_source_files(REPO_ROOT)
    assert files, "approved sources should exist in this repo"
    assert all("fixtures" not in str(f).lower() for f in files)


def test_synthetic_examples_never_count_toward_floor(tmp_path):
    # One real source yields a couple of non-synthetic chunks plus the
    # labeled synthetic behaviors. With a floor above the non-synthetic
    # count the build must not be READY, no matter how many synthetic
    # examples exist.
    src = tmp_path / "doc.md"
    src.write_text(CLEAN_TEXT, encoding="utf-8")
    report, _, examples = build_training_dataset(
        repo_root=REPO_ROOT,
        output_path=tmp_path / "train.jsonl",
        ledger=AuditLedger(),
        source_files=[src],
        min_non_synthetic=10_000,
    )
    assert report.synthetic_count > 0
    assert report.status == "DATASET_NOT_READY"


def test_dataset_at_floor_is_ready(tmp_path):
    src = tmp_path / "doc.md"
    src.write_text(CLEAN_TEXT, encoding="utf-8")
    report, manifest, examples = build_training_dataset(
        repo_root=REPO_ROOT,
        output_path=tmp_path / "train.jsonl",
        ledger=AuditLedger(),
        source_files=[src],
        min_non_synthetic=1,
    )
    assert report.non_synthetic_count >= 1
    assert report.status == "READY"
    assert manifest is not None


def test_owner_jsonl_rows_become_non_synthetic_examples(tmp_path):
    rows = [
        {
            "instruction": "Explain the AMALI audit ledger.",
            "expected_response": CLEAN_TEXT,
            "expected_status": "SUCCESS",
        },
        {
            "instruction": "What score did the unrun benchmark give?",
            "expected_response": "UNKNOWN. No such evaluation has been run.",
            "expected_status": "UNKNOWN",
        },
    ]
    src = tmp_path / "owner_examples.jsonl"
    src.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    report, _, examples = build_training_dataset(
        repo_root=REPO_ROOT,
        output_path=tmp_path / "train.jsonl",
        ledger=AuditLedger(),
        source_files=[src],
    )
    owner_examples = [e for e in examples if e.data_class == "owner_example"]
    assert len(owner_examples) == 2
    assert all(not e.synthetic for e in owner_examples)
    assert owner_examples[1].expected_status == "UNKNOWN"


def test_owner_jsonl_secret_row_blocked(tmp_path):
    rows = [
        {
            "instruction": "Store this key.",
            "expected_response": "sk-abcdefghijklmnopqrstuvwx is the key.",
        }
    ]
    src = tmp_path / "owner_examples.jsonl"
    src.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    report, _, examples = build_training_dataset(
        repo_root=REPO_ROOT,
        output_path=tmp_path / "train.jsonl",
        ledger=AuditLedger(),
        source_files=[src],
    )
    assert report.blocked >= 1
    assert all(
        "sk-abcdefghijklmnopqrstuvwx" not in e.expected_response
        for e in examples
    )


def test_jsonl_output_examples_carry_wall_decisions(tmp_path):
    src = tmp_path / "doc.md"
    src.write_text(CLEAN_TEXT, encoding="utf-8")
    (report, manifest, examples), out = _build(tmp_path, [src])
    assert out.exists()
    rows = [
        json.loads(line)
        for line in out.read_text(encoding="utf-8").splitlines()
    ]
    assert rows
    for row in rows:
        assert row["data_wall_decision_id"].startswith("decision_")
        assert row["content_hash"]
