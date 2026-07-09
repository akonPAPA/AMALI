"""Owner sourcepack intake and validation (AMALI-FT-v0 Stage 1).

An *owner sourcepack* is the only doorway through which raw owner material
under ``D:\\AMALI\\data\\raw`` may enter the training pipeline. The rules are
deliberately strict and default-deny:

* a ``sourcepack_manifest.json`` must exist and list every file;
* only ``.md``, ``.txt`` and approved-schema ``.jsonl`` files are accepted;
* ``secret``-classified files, secret-shaped content, holdout material and
  eval-fixture paths are always blocked;
* ``confidential`` files may not be approved for training (the Data Wall
  TRAINING flow blocks confidential, so approving one is a manifest error);
* files whose ``source_type`` marks them as model output or synthetic never
  count as owner data;
* every file's SHA-256 must match its manifest entry;
* duplicate content hashes are reported.

Validation never fabricates: a missing root or manifest is reported as
``OWNER_DATA_REQUIRED``/``MANIFEST_MISSING`` — never silently skipped.

No torch. No network. Pure local filesystem checks.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from amali.security.redaction import contains_secret

__all__ = [
    "SOURCEPACK_ROOT",
    "MANIFEST_FILENAME",
    "ALLOWED_EXTENSIONS",
    "SourcepackFileEntry",
    "SourcepackManifest",
    "SourcepackIssue",
    "SourcepackValidationReport",
    "validate_sourcepack",
    "approved_training_files",
]

SCHEMA_VERSION = "1.0.0"
SOURCEPACK_ROOT = Path("D:/AMALI/data/raw")
MANIFEST_FILENAME = "sourcepack_manifest.json"

ALLOWED_EXTENSIONS = (".md", ".txt", ".jsonl")

# data_class buckets an owner may assign; secret and holdout never train.
ALLOWED_DATA_CLASSES = ("public", "internal", "confidential", "secret", "holdout")
TRAINABLE_DATA_CLASSES = ("public", "internal")
ALLOWED_SENSITIVITIES = ("public", "internal", "confidential", "secret", "personal")

# source_type values that mark content as NOT owner data. Model outputs
# pretending to be owner material never count toward the training floor.
FORBIDDEN_SOURCE_TYPES = ("model_output", "llm_output", "synthetic", "generated")

# Path fragments that mark holdout/eval material. Any manifest path or
# on-disk path containing one of these is blocked from training intake.
HOLDOUT_PATH_MARKERS = ("holdout", "fixtures", "eval")

# Approved owner-JSONL example schema: required and permitted keys.
JSONL_REQUIRED_KEYS = frozenset({"instruction", "expected_response"})
JSONL_ALLOWED_KEYS = JSONL_REQUIRED_KEYS | {
    "input_context",
    "expected_status",
    "behavior_tags",
    "required_citations",
    "forbidden_outputs",
    "notes",
}


class SourcepackFileEntry(BaseModel):
    """One manifest row describing one owner file."""

    model_config = {"extra": "forbid"}

    path: str  # relative to the sourcepack root, forward slashes
    data_class: str
    data_sensitivity: str
    approved_for_training: bool
    source_type: str
    license_or_origin: str
    notes: str = ""
    sha256: str


class SourcepackManifest(BaseModel):
    """The owner-authored manifest at the sourcepack root."""

    model_config = {"extra": "forbid"}

    schema_version: str = SCHEMA_VERSION
    sourcepack_id: str
    created_at: str
    owner: str
    approved_for_training: bool
    files: list[SourcepackFileEntry] = Field(default_factory=list)


class SourcepackIssue(BaseModel):
    """One typed validation finding."""

    model_config = {"extra": "forbid"}

    code: str
    severity: str  # "error" | "warning"
    path: str = ""
    detail: str = ""


class SourcepackValidationReport(BaseModel):
    """Typed, honest outcome of one sourcepack validation."""

    model_config = {"extra": "forbid"}

    status: str  # VALID | INVALID | MANIFEST_MISSING | OWNER_DATA_REQUIRED
    sourcepack_root: str
    manifest_path: str = ""
    sourcepack_id: str = ""
    owner: str = ""
    files_listed: int = 0
    files_on_disk: int = 0
    approved_files: int = 0
    issues: list[SourcepackIssue] = Field(default_factory=list)
    duplicate_hashes: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _has_holdout_marker(rel_path: str) -> bool:
    parts = rel_path.lower().replace("\\", "/").split("/")
    return any(
        marker in part for part in parts for marker in HOLDOUT_PATH_MARKERS
    )


def _validate_jsonl_content(text: str, rel_path: str) -> list[SourcepackIssue]:
    issues: list[SourcepackIssue] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(
                SourcepackIssue(
                    code="JSONL_SCHEMA_INVALID",
                    severity="error",
                    path=rel_path,
                    detail=f"line {line_no}: not valid JSON ({exc.msg})",
                )
            )
            continue
        if not isinstance(row, dict):
            issues.append(
                SourcepackIssue(
                    code="JSONL_SCHEMA_INVALID",
                    severity="error",
                    path=rel_path,
                    detail=f"line {line_no}: expected a JSON object",
                )
            )
            continue
        missing = JSONL_REQUIRED_KEYS - row.keys()
        unknown = row.keys() - JSONL_ALLOWED_KEYS
        if missing:
            issues.append(
                SourcepackIssue(
                    code="JSONL_SCHEMA_INVALID",
                    severity="error",
                    path=rel_path,
                    detail=f"line {line_no}: missing keys {sorted(missing)}",
                )
            )
        if unknown:
            issues.append(
                SourcepackIssue(
                    code="JSONL_SCHEMA_INVALID",
                    severity="error",
                    path=rel_path,
                    detail=f"line {line_no}: unknown keys {sorted(unknown)}",
                )
            )
    return issues


def _check_entry(
    entry: SourcepackFileEntry, root: Path
) -> tuple[list[SourcepackIssue], Path | None]:
    """Validate one manifest entry; return (issues, resolved path or None)."""
    issues: list[SourcepackIssue] = []
    rel = entry.path.replace("\\", "/")

    if _has_holdout_marker(rel):
        issues.append(
            SourcepackIssue(
                code="HOLDOUT_PATH_BLOCKED",
                severity="error",
                path=rel,
                detail="path carries a holdout/eval/fixtures marker",
            )
        )

    target = (root / rel).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError:
        issues.append(
            SourcepackIssue(
                code="PATH_OUTSIDE_SOURCEPACK",
                severity="error",
                path=rel,
                detail="manifest path escapes the sourcepack root",
            )
        )
        return issues, None

    if entry.data_class not in ALLOWED_DATA_CLASSES:
        issues.append(
            SourcepackIssue(
                code="UNKNOWN_DATA_CLASS",
                severity="error",
                path=rel,
                detail=f"data_class={entry.data_class!r}",
            )
        )
    if entry.data_sensitivity not in ALLOWED_SENSITIVITIES:
        issues.append(
            SourcepackIssue(
                code="UNKNOWN_DATA_SENSITIVITY",
                severity="error",
                path=rel,
                detail=f"data_sensitivity={entry.data_sensitivity!r}",
            )
        )

    if entry.data_class == "secret" or entry.data_sensitivity == "secret":
        issues.append(
            SourcepackIssue(
                code="SECRET_CLASS_BLOCKED",
                severity="error",
                path=rel,
                detail="secret material is never accepted for training",
            )
        )
    if entry.data_class == "holdout":
        issues.append(
            SourcepackIssue(
                code="HOLDOUT_CLASS_BLOCKED",
                severity="error",
                path=rel,
                detail="holdout material is never accepted for training",
            )
        )
    if (
        entry.data_class == "confidential"
        or entry.data_sensitivity in ("confidential", "personal")
    ) and entry.approved_for_training:
        issues.append(
            SourcepackIssue(
                code="CONFIDENTIAL_NOT_TRAINABLE",
                severity="error",
                path=rel,
                detail=(
                    "the Data Wall TRAINING flow blocks confidential/personal "
                    "data; this entry may not be approved for training"
                ),
            )
        )
    if entry.source_type.lower() in FORBIDDEN_SOURCE_TYPES:
        issues.append(
            SourcepackIssue(
                code="SYNTHETIC_SOURCE_BLOCKED",
                severity="error",
                path=rel,
                detail=(
                    f"source_type={entry.source_type!r}: model/synthetic output "
                    "does not count as owner data"
                ),
            )
        )

    if target.suffix.lower() not in ALLOWED_EXTENSIONS:
        issues.append(
            SourcepackIssue(
                code="UNSUPPORTED_FILE_TYPE",
                severity="error",
                path=rel,
                detail=f"only {ALLOWED_EXTENSIONS} are accepted",
            )
        )

    if not target.is_file():
        issues.append(
            SourcepackIssue(
                code="MISSING_FILE",
                severity="error",
                path=rel,
                detail="manifest entry points to a file that does not exist",
            )
        )
        return issues, None

    actual_hash = _sha256_file(target)
    if actual_hash != entry.sha256:
        issues.append(
            SourcepackIssue(
                code="HASH_MISMATCH",
                severity="error",
                path=rel,
                detail=f"manifest={entry.sha256[:12]}... disk={actual_hash[:12]}...",
            )
        )

    # Content checks only for accepted text types.
    if target.suffix.lower() in ALLOWED_EXTENSIONS:
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            issues.append(
                SourcepackIssue(
                    code="BINARY_CONTENT",
                    severity="error",
                    path=rel,
                    detail="file is not valid UTF-8 text",
                )
            )
            return issues, target
        if contains_secret({"text": text}):
            issues.append(
                SourcepackIssue(
                    code="SECRET_CONTENT_DETECTED",
                    severity="error",
                    path=rel,
                    detail="secret-shaped content found; remove or redact it",
                )
            )
        if target.suffix.lower() == ".jsonl":
            issues.extend(_validate_jsonl_content(text, rel))

    return issues, target


def validate_sourcepack(
    root: str | Path = SOURCEPACK_ROOT,
) -> SourcepackValidationReport:
    """Validate the owner sourcepack under ``root``.

    Status is ``VALID`` only when a manifest exists, every listed file
    checks out, no unlisted files are present, and at least one file is
    approved for training. Anything else is a typed refusal.
    """
    root = Path(root)
    report_base = {"sourcepack_root": str(root)}

    if not root.is_dir():
        return SourcepackValidationReport(
            status="OWNER_DATA_REQUIRED",
            issues=[
                SourcepackIssue(
                    code="SOURCEPACK_ROOT_MISSING",
                    severity="error",
                    path=str(root),
                    detail=(
                        "owner sourcepack root does not exist; create it and "
                        "add approved material plus sourcepack_manifest.json"
                    ),
                )
            ],
            risks=["no owner data available; training must not start"],
            **report_base,
        )

    manifest_path = root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return SourcepackValidationReport(
            status="MANIFEST_MISSING",
            issues=[
                SourcepackIssue(
                    code="MANIFEST_MISSING",
                    severity="error",
                    path=str(manifest_path),
                    detail=(
                        "sourcepack_manifest.json not found; see "
                        "docs/data/SOURCEPACK_TEMPLATE.md"
                    ),
                )
            ],
            risks=["unmanifested raw data is never eligible for training"],
            **report_base,
        )

    try:
        manifest = SourcepackManifest(
            **json.loads(manifest_path.read_text(encoding="utf-8"))
        )
    except (json.JSONDecodeError, ValidationError, UnicodeDecodeError) as exc:
        return SourcepackValidationReport(
            status="INVALID",
            manifest_path=str(manifest_path),
            issues=[
                SourcepackIssue(
                    code="MANIFEST_INVALID",
                    severity="error",
                    path=str(manifest_path),
                    detail=str(exc)[:500],
                )
            ],
            risks=["manifest unreadable; nothing is approved"],
            **report_base,
        )

    issues: list[SourcepackIssue] = []
    if not manifest.approved_for_training:
        issues.append(
            SourcepackIssue(
                code="SOURCEPACK_NOT_APPROVED",
                severity="error",
                path=str(manifest_path),
                detail="manifest approved_for_training is false",
            )
        )

    listed_paths: set[str] = set()
    hash_to_paths: dict[str, list[str]] = {}
    approved: list[Path] = []
    for entry in manifest.files:
        rel = entry.path.replace("\\", "/")
        if rel in listed_paths:
            issues.append(
                SourcepackIssue(
                    code="DUPLICATE_MANIFEST_ENTRY",
                    severity="error",
                    path=rel,
                    detail="path listed more than once in the manifest",
                )
            )
            continue
        listed_paths.add(rel)
        entry_issues, target = _check_entry(entry, root)
        issues.extend(entry_issues)
        hash_to_paths.setdefault(entry.sha256, []).append(rel)
        entry_ok = not any(i.severity == "error" for i in entry_issues)
        if (
            entry_ok
            and target is not None
            and entry.approved_for_training
            and entry.data_class in TRAINABLE_DATA_CLASSES
        ):
            approved.append(target)

    duplicate_hashes = sorted(
        h for h, paths in hash_to_paths.items() if len(paths) > 1
    )
    for dup in duplicate_hashes:
        issues.append(
            SourcepackIssue(
                code="DUPLICATE_HASH",
                severity="warning",
                path=", ".join(hash_to_paths[dup]),
                detail=f"identical content sha256={dup[:12]}...",
            )
        )

    # Files on disk that the manifest does not mention are a hard error:
    # nothing unmanifested may sit inside the intake root.
    on_disk = [
        p for p in sorted(root.rglob("*")) if p.is_file() and p != manifest_path
    ]
    for path in on_disk:
        rel = str(path.relative_to(root)).replace("\\", "/")
        if rel not in listed_paths:
            issues.append(
                SourcepackIssue(
                    code="UNLISTED_FILE",
                    severity="error",
                    path=rel,
                    detail="file exists on disk but is not in the manifest",
                )
            )

    if not manifest.files:
        issues.append(
            SourcepackIssue(
                code="EMPTY_MANIFEST",
                severity="error",
                path=str(manifest_path),
                detail="manifest lists no files",
            )
        )

    has_errors = any(i.severity == "error" for i in issues)
    if not has_errors and not approved:
        issues.append(
            SourcepackIssue(
                code="NO_APPROVED_FILES",
                severity="error",
                path=str(manifest_path),
                detail="no file is approved for training",
            )
        )
        has_errors = True

    risks: list[str] = []
    if has_errors:
        risks.append("sourcepack invalid; training must not use this material")
    if duplicate_hashes:
        risks.append("duplicate content present; dedup will reduce yield")

    return SourcepackValidationReport(
        status="INVALID" if has_errors else "VALID",
        manifest_path=str(manifest_path),
        sourcepack_id=manifest.sourcepack_id,
        owner=manifest.owner,
        files_listed=len(manifest.files),
        files_on_disk=len(on_disk),
        approved_files=len(approved),
        issues=issues,
        duplicate_hashes=duplicate_hashes,
        risks=risks,
        **report_base,
    )


def approved_training_files(
    root: str | Path = SOURCEPACK_ROOT,
) -> list[Path]:
    """Return raw files eligible for training intake.

    Empty unless the whole sourcepack validates: a pack with any error
    contributes nothing, so a single bad file quarantines the pack until
    the owner fixes it.
    """
    root = Path(root)
    report = validate_sourcepack(root)
    if report.status != "VALID":
        return []
    manifest = SourcepackManifest(
        **json.loads((root / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    )
    files: list[Path] = []
    for entry in manifest.files:
        if entry.approved_for_training and entry.data_class in TRAINABLE_DATA_CLASSES:
            files.append(root / entry.path.replace("\\", "/"))
    return sorted(files, key=lambda p: str(p).lower())
