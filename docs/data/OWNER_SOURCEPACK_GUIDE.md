# Owner Sourcepack Guide (AMALI-FT-v0)

An **owner sourcepack** is the only way raw owner material enters the
AMALI training pipeline. Nothing under `D:\AMALI\data\raw\` is used for
training unless it is listed and approved in a `sourcepack_manifest.json`
and the whole pack validates.

## Why this exists

The AMALI-FT-v0 dataset floor is **200 approved non-synthetic examples**.
Synthetic examples never count toward that floor, and the pipeline will
not pad or fabricate. The only honest way to reach the floor is real,
owner-approved local material — this guide describes how to supply it.

## Layout

```
D:\AMALI\data\raw\
├── sourcepack_manifest.json      <- required, see SOURCEPACK_TEMPLATE.md
├── notes\
│   ├── amali_design_notes.md
│   └── operations_runbook.txt
└── examples\
    └── owner_examples.jsonl
```

## Accepted file types

| Type     | Accepted | Notes |
|----------|----------|-------|
| `.md`    | yes      | UTF-8 text |
| `.txt`   | yes      | UTF-8 text |
| `.jsonl` | yes      | approved example schema only (below) |
| PDF, DOCX, images, archives, binaries | **no** | convert to text first |
| scraped web dumps | **no** | |
| logs containing secrets | **no** | |
| eval-suite or holdout files | **never** | contamination |
| model outputs presented as owner data | **never** | `source_type` is checked |

## Classification

Every file must carry a `data_class`:

| data_class     | Trainable? |
|----------------|------------|
| `public`       | yes |
| `internal`     | yes (redaction applies at the Data Wall) |
| `confidential` | **no** — the Data Wall TRAINING flow blocks it; listing one with `approved_for_training: true` is a manifest error |
| `secret`       | **never**; the whole pack is refused |
| `holdout`      | **never**; the whole pack is refused |

`data_sensitivity` uses the AMALI policy values
(`public`, `internal`, `confidential`, `secret`, `personal`).
`personal` and `confidential` sensitivities are not trainable.

## Hard blocks

Validation fails the entire pack (exit nonzero) on any of:

- missing `sourcepack_manifest.json` (`MANIFEST_MISSING`);
- a file on disk that the manifest does not list (`UNLISTED_FILE`);
- a manifest entry whose file is missing (`MISSING_FILE`);
- SHA-256 mismatch between manifest and disk (`HASH_MISMATCH`);
- secret-shaped content — API keys, bearer tokens, private keys,
  credentialed URLs (`SECRET_CONTENT_DETECTED`);
- `secret`/`holdout` classification (`SECRET_CLASS_BLOCKED`,
  `HOLDOUT_CLASS_BLOCKED`);
- a path containing `holdout`, `fixtures`, or `eval`
  (`HOLDOUT_PATH_BLOCKED`);
- non-UTF-8 / binary content (`BINARY_CONTENT`);
- unsupported extension (`UNSUPPORTED_FILE_TYPE`);
- `source_type` of `model_output`, `llm_output`, `synthetic`, or
  `generated` (`SYNTHETIC_SOURCE_BLOCKED`);
- `.jsonl` rows that do not match the approved schema
  (`JSONL_SCHEMA_INVALID`).

Duplicate content (same SHA-256 under two paths) is reported as a
warning (`DUPLICATE_HASH`); dedup happens later in the dataset build.

A pack with **any** error contributes **zero** files to training —
one bad file quarantines the pack until fixed.

## Approved `.jsonl` schema

One JSON object per line. Required keys: `instruction`,
`expected_response`. Optional: `input_context`, `expected_status`,
`behavior_tags`, `required_citations`, `forbidden_outputs`, `notes`.
No other keys.

```json
{"instruction": "Explain what the AMALI audit ledger guarantees.", "expected_response": "Every event is hash-chained with SHA-256 ...", "expected_status": "SUCCESS"}
```

## Workflow

1. Put your files under `D:\AMALI\data\raw\`.
2. Copy `docs/data/SOURCEPACK_TEMPLATE.md`'s manifest into
   `D:\AMALI\data\raw\sourcepack_manifest.json` and fill it in.
   Compute each file's SHA-256, e.g. in PowerShell:
   `Get-FileHash -Algorithm SHA256 .\notes\amali_design_notes.md`.
3. Validate:
   `python scripts/validate_owner_sourcepack.py`
4. On `VALID`, rebuild the dataset:
   `python scripts/build_training_dataset.py`
5. Run the contamination gate:
   `python scripts/check_contamination.py`

Reports land in `artifacts/amali_ft_v0/<timestamp>/`
(`sourcepack_validation_report.json` / `.md`).

## Reaching the floor

The dataset builder chunks documents into 200–2000-character paragraph
chunks; each approved chunk is one non-synthetic example. As a rough
guide, ~1,500–2,000 words of usable prose yields ~5–10 examples, so
plan for on the order of **150–300 pages of owner-approved text** (or a
mix of documents plus owner-written `.jsonl` examples) to clear the
200-example floor. The build report tells you exactly how many more
examples are needed — trust the report, not this estimate.

Owner data categories that fit AMALI's goals (write or curate your own;
never paste model output):

- design notes, architecture decisions, operations runbooks;
- incident write-ups and postmortems (secrets removed);
- project documentation, meeting notes, research summaries you wrote;
- owner-authored Q&A / instruction examples in the approved `.jsonl`
  schema (grounded, citation-anchored, honest-UNKNOWN style).
