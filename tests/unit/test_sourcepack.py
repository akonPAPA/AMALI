"""AMALI-FT-v0 Stage 1: owner sourcepack validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from amali.training.sourcepack import (
    MANIFEST_FILENAME,
    approved_training_files,
    validate_sourcepack,
)

CLEAN_TEXT = (
    "AMALI is an owner-governed local AI infrastructure. The audit ledger "
    "chains every event with SHA-256 so that any mutation of history is "
    "detectable. The Data Wall applies a default-block operation matrix "
    "before data enters any flow."
)

SECRET_TEXT = (
    "Rotation notes: the current gateway key is "
    "sk-abcdefghijklmnopqrstuvwx and must never enter a dataset."
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry(root: Path, rel: str, **overrides) -> dict:
    entry = {
        "path": rel,
        "data_class": "internal",
        "data_sensitivity": "internal",
        "approved_for_training": True,
        "source_type": "owner_written",
        "license_or_origin": "owner original work",
        "notes": "",
        "sha256": _sha256(root / rel),
    }
    entry.update(overrides)
    return entry


def _write_manifest(root: Path, files: list[dict], **overrides) -> None:
    manifest = {
        "schema_version": "1.0.0",
        "sourcepack_id": "test_pack",
        "created_at": "2026-07-09T00:00:00Z",
        "owner": "owner",
        "approved_for_training": True,
        "files": files,
    }
    manifest.update(overrides)
    (root / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _make_pack(root: Path, texts: dict[str, str]) -> None:
    for rel, text in texts.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    _write_manifest(root, [_entry(root, rel) for rel in texts])


def _codes(report) -> set[str]:
    return {i.code for i in report.issues}


def test_valid_sourcepack_passes(tmp_path):
    _make_pack(tmp_path, {"notes/doc.md": CLEAN_TEXT})
    report = validate_sourcepack(tmp_path)
    assert report.status == "VALID"
    assert report.approved_files == 1
    assert not any(i.severity == "error" for i in report.issues)


def test_missing_root_is_owner_data_required(tmp_path):
    report = validate_sourcepack(tmp_path / "does_not_exist")
    assert report.status == "OWNER_DATA_REQUIRED"
    assert "SOURCEPACK_ROOT_MISSING" in _codes(report)


def test_missing_manifest_fails(tmp_path):
    (tmp_path / "doc.md").write_text(CLEAN_TEXT, encoding="utf-8")
    report = validate_sourcepack(tmp_path)
    assert report.status == "MANIFEST_MISSING"


def test_secret_file_fails(tmp_path):
    _make_pack(tmp_path, {"doc.md": SECRET_TEXT})
    report = validate_sourcepack(tmp_path)
    assert report.status == "INVALID"
    assert "SECRET_CONTENT_DETECTED" in _codes(report)


def test_secret_data_class_fails(tmp_path):
    (tmp_path / "doc.md").write_text(CLEAN_TEXT, encoding="utf-8")
    _write_manifest(
        tmp_path, [_entry(tmp_path, "doc.md", data_class="secret")]
    )
    report = validate_sourcepack(tmp_path)
    assert report.status == "INVALID"
    assert "SECRET_CLASS_BLOCKED" in _codes(report)


def test_unlisted_file_fails(tmp_path):
    _make_pack(tmp_path, {"doc.md": CLEAN_TEXT})
    (tmp_path / "sneaky.md").write_text(CLEAN_TEXT + " extra", encoding="utf-8")
    report = validate_sourcepack(tmp_path)
    assert report.status == "INVALID"
    assert "UNLISTED_FILE" in _codes(report)


def test_missing_listed_file_fails(tmp_path):
    (tmp_path / "doc.md").write_text(CLEAN_TEXT, encoding="utf-8")
    entry = _entry(tmp_path, "doc.md")
    entry["path"] = "gone.md"
    _write_manifest(tmp_path, [entry])
    report = validate_sourcepack(tmp_path)
    assert report.status == "INVALID"
    assert "MISSING_FILE" in _codes(report)


def test_hash_mismatch_fails(tmp_path):
    _make_pack(tmp_path, {"doc.md": CLEAN_TEXT})
    (tmp_path / "doc.md").write_text(CLEAN_TEXT + " tampered", encoding="utf-8")
    report = validate_sourcepack(tmp_path)
    assert report.status == "INVALID"
    assert "HASH_MISMATCH" in _codes(report)


def test_duplicate_hash_reported(tmp_path):
    _make_pack(tmp_path, {"a.md": CLEAN_TEXT, "b.md": CLEAN_TEXT})
    report = validate_sourcepack(tmp_path)
    assert report.duplicate_hashes
    assert "DUPLICATE_HASH" in _codes(report)
    # duplicates are a warning, not an error: pack stays valid.
    assert report.status == "VALID"


def test_holdout_path_fails(tmp_path):
    _make_pack(tmp_path, {"holdout/doc.md": CLEAN_TEXT})
    report = validate_sourcepack(tmp_path)
    assert report.status == "INVALID"
    assert "HOLDOUT_PATH_BLOCKED" in _codes(report)


def test_eval_fixture_path_fails(tmp_path):
    _make_pack(tmp_path, {"fixtures/eval/seed.md": CLEAN_TEXT})
    report = validate_sourcepack(tmp_path)
    assert report.status == "INVALID"
    assert "HOLDOUT_PATH_BLOCKED" in _codes(report)


def test_holdout_data_class_fails(tmp_path):
    (tmp_path / "doc.md").write_text(CLEAN_TEXT, encoding="utf-8")
    _write_manifest(
        tmp_path, [_entry(tmp_path, "doc.md", data_class="holdout")]
    )
    report = validate_sourcepack(tmp_path)
    assert report.status == "INVALID"
    assert "HOLDOUT_CLASS_BLOCKED" in _codes(report)


def test_confidential_cannot_be_approved_for_training(tmp_path):
    (tmp_path / "doc.md").write_text(CLEAN_TEXT, encoding="utf-8")
    _write_manifest(
        tmp_path,
        [_entry(tmp_path, "doc.md", data_class="confidential")],
    )
    report = validate_sourcepack(tmp_path)
    assert report.status == "INVALID"
    assert "CONFIDENTIAL_NOT_TRAINABLE" in _codes(report)


def test_model_output_source_type_blocked(tmp_path):
    (tmp_path / "doc.md").write_text(CLEAN_TEXT, encoding="utf-8")
    _write_manifest(
        tmp_path,
        [_entry(tmp_path, "doc.md", source_type="model_output")],
    )
    report = validate_sourcepack(tmp_path)
    assert report.status == "INVALID"
    assert "SYNTHETIC_SOURCE_BLOCKED" in _codes(report)


def test_unsupported_extension_fails(tmp_path):
    (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4 fake")
    _write_manifest(
        tmp_path,
        [
            {
                "path": "doc.pdf",
                "data_class": "internal",
                "data_sensitivity": "internal",
                "approved_for_training": True,
                "source_type": "owner_written",
                "license_or_origin": "owner",
                "notes": "",
                "sha256": hashlib.sha256(b"%PDF-1.4 fake").hexdigest(),
            }
        ],
    )
    report = validate_sourcepack(tmp_path)
    assert report.status == "INVALID"
    assert "UNSUPPORTED_FILE_TYPE" in _codes(report)


def test_jsonl_schema_enforced(tmp_path):
    good = json.dumps(
        {"instruction": "Say what AMALI is.", "expected_response": CLEAN_TEXT}
    )
    bad = json.dumps({"prompt": "wrong keys"})
    (tmp_path / "ex.jsonl").write_text(good + "\n" + bad + "\n", encoding="utf-8")
    _write_manifest(tmp_path, [_entry(tmp_path, "ex.jsonl")])
    report = validate_sourcepack(tmp_path)
    assert report.status == "INVALID"
    assert "JSONL_SCHEMA_INVALID" in _codes(report)


def test_valid_jsonl_passes(tmp_path):
    rows = [
        {"instruction": "Say what AMALI is.", "expected_response": CLEAN_TEXT},
        {
            "instruction": "What score did the unrun benchmark give?",
            "expected_response": "UNKNOWN. No such evaluation has been run.",
            "expected_status": "UNKNOWN",
        },
    ]
    (tmp_path / "ex.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    _write_manifest(tmp_path, [_entry(tmp_path, "ex.jsonl")])
    report = validate_sourcepack(tmp_path)
    assert report.status == "VALID"


def test_pack_not_approved_fails(tmp_path):
    _make_pack(tmp_path, {"doc.md": CLEAN_TEXT})
    _write_manifest(
        tmp_path,
        [_entry(tmp_path, "doc.md")],
        approved_for_training=False,
    )
    report = validate_sourcepack(tmp_path)
    assert report.status == "INVALID"
    assert "SOURCEPACK_NOT_APPROVED" in _codes(report)


def test_invalid_pack_contributes_zero_training_files(tmp_path):
    _make_pack(tmp_path, {"doc.md": CLEAN_TEXT, "bad.md": SECRET_TEXT})
    assert approved_training_files(tmp_path) == []


def test_valid_pack_lists_training_files(tmp_path):
    _make_pack(tmp_path, {"notes/doc.md": CLEAN_TEXT})
    files = approved_training_files(tmp_path)
    assert [f.name for f in files] == ["doc.md"]


def test_path_escape_blocked(tmp_path):
    root = tmp_path / "pack"
    root.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text(CLEAN_TEXT, encoding="utf-8")
    _write_manifest(
        root,
        [
            {
                "path": "../outside.md",
                "data_class": "internal",
                "data_sensitivity": "internal",
                "approved_for_training": True,
                "source_type": "owner_written",
                "license_or_origin": "owner",
                "notes": "",
                "sha256": _sha256(outside),
            }
        ],
    )
    report = validate_sourcepack(root)
    assert report.status == "INVALID"
    assert "PATH_OUTSIDE_SOURCEPACK" in _codes(report)
