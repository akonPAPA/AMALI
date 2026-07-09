# Sourcepack Manifest Template

Save the JSON below as `D:\AMALI\data\raw\sourcepack_manifest.json` and
edit it. Every file under the sourcepack root **must** appear in `files`;
compute real SHA-256 hashes (PowerShell:
`Get-FileHash -Algorithm SHA256 <file>` — lowercase the hex digest).

```json
{
  "schema_version": "1.0.0",
  "sourcepack_id": "owner_pack_2026_07",
  "created_at": "2026-07-09T00:00:00Z",
  "owner": "akonPAPA",
  "approved_for_training": true,
  "files": [
    {
      "path": "notes/amali_design_notes.md",
      "data_class": "internal",
      "data_sensitivity": "internal",
      "approved_for_training": true,
      "source_type": "owner_written",
      "license_or_origin": "owner original work",
      "notes": "design notes written by owner, no third-party content",
      "sha256": "<64-char lowercase hex sha256 of the file>"
    },
    {
      "path": "examples/owner_examples.jsonl",
      "data_class": "internal",
      "data_sensitivity": "internal",
      "approved_for_training": true,
      "source_type": "owner_written",
      "license_or_origin": "owner original work",
      "notes": "owner-authored instruction examples, approved schema",
      "sha256": "<64-char lowercase hex sha256 of the file>"
    }
  ]
}
```

Field reference:

| Field | Values / meaning |
|-------|------------------|
| `sourcepack_id` | your identifier for this pack |
| `created_at` | ISO-8601 UTC timestamp |
| `owner` | who approved this pack |
| `approved_for_training` (top level) | must be `true` for the pack to be usable |
| `path` | forward-slash path relative to the sourcepack root |
| `data_class` | `public` \| `internal` (trainable); `confidential` \| `secret` \| `holdout` (not trainable) |
| `data_sensitivity` | `public` \| `internal` \| `confidential` \| `secret` \| `personal` |
| `approved_for_training` (per file) | `true` only for trainable classes |
| `source_type` | e.g. `owner_written`, `owner_curated`, `licensed`; **never** `model_output` / `synthetic` |
| `license_or_origin` | provenance statement |
| `notes` | free text |
| `sha256` | SHA-256 of the exact file bytes |

Validate with:

```
python scripts/validate_owner_sourcepack.py
```
