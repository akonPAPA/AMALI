# Fix Notebook — tracked P2/P3 issues

Format per entry: severity, file path, root cause, risk, suggested fix,
required proof/tests, owner/target week, status.

---

## P2 — wrapped raw-base eval report filename differs from the readiness spec

- **severity:** P2
- **file path:** `scripts/evaluate_model.py` (REPORT_NAMES), `scripts/compare_models.py`, `scripts/collect_base_model_gate_artifacts.py`
- **root cause:** the wrapped raw-base eval artifact is named
  `eval_report_wrapped_raw_base.json`; the readiness spec names it
  `eval_report_amali_wrapped_raw_base.json`.
- **risk:** none functional (all internal consumers agree on the current
  name); only a naming drift against the written spec.
- **suggested fix:** rename in one commit across evaluate/compare/collect
  and the readiness checker, or keep the current name and amend the spec.
- **required proof/tests:** collector still finds the report; readiness
  checker still reads the wrapped eval status.
- **owner/target week:** unassigned
- **status:** open

## P2 — eval artifacts live under base_model_gate, not base_model_readiness

- **severity:** P2
- **file path:** `scripts/evaluate_model.py`, `scripts/compare_models.py`
- **root cause:** the eval/comparison pipeline predates the
  `artifacts/base_model_readiness/` tree; new pin/smoke/readiness/chat
  artifacts use the new tree, evals stay in `artifacts/base_model_gate/`.
- **risk:** none functional — the readiness checker and collector search
  both trees; purely a layout inconsistency.
- **suggested fix:** migrate eval outputs to the readiness tree once the
  collector's canonical dir is versioned.
- **required proof/tests:** `scripts/check_base_model_readiness.py` and the
  collector resolve the same newest reports after the move.
- **owner/target week:** unassigned
- **status:** open

## P2 — download manifest lands under base_model_gate timestamp dirs

- **severity:** P2
- **file path:** `scripts/download_model.py`
- **root cause:** manifest written to `artifacts/base_model_gate/<ts>/`
  rather than `artifacts/base_model_readiness/<ts>/`.
- **risk:** low; collector finds it either way.
- **suggested fix:** move alongside the other readiness artifacts.
- **required proof/tests:** collector + readiness checker still resolve it.
- **owner/target week:** unassigned
- **status:** open

## P3 — availability "no revision" path returns first classifiable snapshot

- **severity:** P3
- **file path:** `src/amali/model_gateway/availability.py`
  (`check_model_snapshot`, no-revision branch)
- **root cause:** with multiple cached snapshots and no requested revision,
  the newest-by-name AVAILABLE snapshot is reported; "newest" is
  lexicographic over commit SHAs, i.e. arbitrary.
- **risk:** cosmetic only — every promotion-relevant caller passes the
  pinned revision, and floating results are NON_PROMOTABLE by design.
- **suggested fix:** prefer snapshot mtime ordering for the report.
- **required proof/tests:** unit test with two cached snapshots.
- **owner/target week:** unassigned
- **status:** open

## P3 — gitleaks flags deliberate fake test secrets (needs allowlist)

- **severity:** P3
- **file path:** `tests/unit/test_data_wall.py:134`,
  `src/amali/demo/base_ai_gate.py:255`,
  `tests/unit/test_audit_permission_events.py:75`
- **root cause:** these lines intentionally contain fake secret-shaped
  strings (`sk-0123456789abcdef0123`, `ghp_0123456789...`) to prove the
  Data Wall / redaction machinery blocks them; gitleaks flags them as
  generic-api-key across history.
- **risk:** noise only — the values are synthetic sequential fixtures, not
  credentials. Verified 2026-07-09.
- **suggested fix:** add a `.gitleaks.toml` allowlist scoped to those
  exact fixture lines/paths.
- **required proof/tests:** `gitleaks detect` exits 0.
- **owner/target week:** unassigned
- **status:** open

## P3 — semgrep: CI workflow uses mutable action tags

- **severity:** P3 (GitHub CI is disabled for this project)
- **file path:** `.github/workflows/ci.yml:15-16`
- **root cause:** `actions/checkout@v4` / `actions/setup-python@v5` are
  mutable tags.
- **risk:** none while GitHub is disabled; supply-chain hardening if CI
  is ever re-enabled.
- **suggested fix:** pin actions to full commit SHAs (or delete the
  workflow while GitHub is disabled).
- **required proof/tests:** semgrep github-actions ruleset passes.
- **owner/target week:** unassigned
- **status:** open

## P3 — chat smoke AGI-overclaim detector is lexical

- **severity:** P3
- **file path:** `scripts/run_local_amali_chat_smoke.py`
- **root cause:** the overclaim flag matches literal "i am agi" phrasings
  only; paraphrased overclaims are not flagged (they are still visible in
  the recorded transcript).
- **risk:** low — the transcript is recorded verbatim and reviewed by the
  owner; the frozen eval suite scores safety separately.
- **suggested fix:** reuse the eval scorer's forbidden-term machinery.
- **required proof/tests:** unit test with a paraphrased overclaim.
- **owner/target week:** unassigned
- **status:** open
