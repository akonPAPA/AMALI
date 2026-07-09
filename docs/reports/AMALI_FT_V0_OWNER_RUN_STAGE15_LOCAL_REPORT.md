# AMALI-FT-v0 Owner-Run — Stage 15 Report

**Run date:** 2026-07-09 (UTC)
**Overall outcome:** **STOPPED — typed, honest.** Stages 1–14 are
code-complete, tested, and committed. Training was **not** run. The
pipeline stops at `OWNER_DATA_REQUIRED` + `OWNER_MODEL_REVISION_REQUIRED`.
AMALI-FT-v0 does **not** exist yet; nothing was faked.

---

## 1. Branch and commit

- Branch: `iga/amali-ft-v0-owner-run`
- Verified code state: `a8f2f17` ("FT-v0 Stage 14: collector picks up
  FT-v0 artifacts; bnb functional check")
- 7 stage commits on top of `main` @ `e24586b`.
- Remote note: the GitHub `origin` remote was replaced (outside this run)
  by a local bare remote `local -> D:\AMALI\_git-remotes\AMALI-Core.git`;
  the branch is present there at `a8f2f17`. The branch was also pushed to
  GitHub earlier, while `origin` still existed.

## 2. PR #6 merged before branch creation

Yes. `origin/main` was fast-forwarded to `e24586b`
("Iga/base model gate clean (#6)") and the branch was cut fresh from it.

## 3. Base Gate status

PASS throughout:

- full test suite: **388 passed, 0 failed**;
- Gate A demo: all 24 canonical artifacts emitted;
- both preflights: PASS;
- collector: **37/37** canonical artifacts (extended from 31 to include
  the new FT-v0 reports);
- `torch_loaded=False` on `import amali` even with all training deps
  installed.

## 4. Sourcepack status

**`OWNER_DATA_REQUIRED`** — `D:\AMALI\data\raw\` does not exist.

Workflow delivered: `src/amali/training/sourcepack.py`,
`scripts/validate_owner_sourcepack.py`,
`docs/data/OWNER_SOURCEPACK_GUIDE.md`, `docs/data/SOURCEPACK_TEMPLATE.md`,
21 unit tests. Rules enforced: manifest required; SHA-256 per file;
secret / holdout / confidential / synthetic-source / unlisted / missing /
tampered / path-escape all block; one bad file quarantines the whole pack
(zero files contributed). Raw-dir files can enter the dataset **only**
through a VALID sourcepack.

## 5. Dataset status

**`DATASET_NOT_READY`**

| metric | value |
|--------|-------|
| raw sources | 13 |
| candidate examples | 138 |
| approved non-synthetic | **132** |
| synthetic (labeled, never counted) | 6 |
| blocked | 0 |
| redacted | 132 |
| deduped | 0 |
| training floor | 200 |
| **shortfall** | **68 more non-synthetic examples** |

`dataset_build_report.json` now includes `min_required_non_synthetic`,
`sourcepack_validation_status`, `ready_for_training`. Owner `.jsonl`
example files are parsed row-wise as non-synthetic examples (Data-Wall
checked). Owner data categories that fit (write or curate, never model
output): design notes, runbooks, incident write-ups, project docs,
owner-authored Q&A in the approved JSONL schema.

## 6. Contamination

**PASS** — 138 training examples vs 84 frozen eval items, zero exact and
zero near-duplicate leakage. Eval suite frozen and tamper-verified
(`suite_hash bc9f399e9d33…`, `scorer_hash 0ba06aec224c…`).

## 7. Model allowlist / revision

Allowlisted: Qwen/Qwen2.5-1.5B-Instruct (train+eval, default),
Qwen/Qwen2.5-0.5B-Instruct (fallback), Qwen/Qwen2.5-3B-Instruct
(eval-only). `allowed_revision` is still `main` →
**`OWNER_MODEL_REVISION_REQUIRED`**. Promotion refuses floating revisions
(tested). Download manifest contract records files, hashes, license, disk
estimate, `promotion_eligible`, `owner_action_required`; floating-revision
downloads are marked NON_PROMOTABLE.

## 8. Model availability

- deps: **installed** — torch 2.12.1+cu132, transformers, peft, datasets,
  accelerate, huggingface_hub;
- weights: **none cached** (`D:\AI_CACHE\huggingface\hub`) →
  `SKIPPED_LOCAL_MODEL_NOT_AVAILABLE`;
- pinned revision: **none**.

## 9. Training environment

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- CUDA: available; bf16: supported; VRAM: 8.0 GB
- bitsandbytes: **UNUSABLE** — installs and imports on Windows but its
  native CUDA 13.2 binary (`libbitsandbytes_cuda132.dll`) is missing; the
  probe checks functional kernel availability and reports
  **`BITSANDBYTES_UNAVAILABLE`** (no fake QLoRA).
- Fallback decision (recorded in `training_fallback_plan.json`):
  **standard LoRA on the 1.5B**, estimated 7.2 GB peak, *degraded* and
  **requires explicit owner approval** before training.

## 10. Dry-run status

**`DATASET_NOT_READY`** (exit 1). Gates: `eval_suite_frozen` PASS,
`dataset_ready` **FAIL**, `contamination` PASS, `base_model_allowlisted`
PASS, `revision_pinned` **FAIL**, `weights_available_locally` **FAIL**,
`train_deps_installed` PASS, `vram_feasible` PASS, `output_outside_repo`
PASS, `registry_outside_repo` PASS, `sourcepack_valid` NOT_RUN (no
raw-dir examples yet).

## 11. Real training status

**NOT RUN** (stop conditions). No steps, no loss, no wall-clock, no peak
VRAM, no adapter dir, no checkpoint hashes exist — none invented. Typed
`TRAINING_FAILED_OOM` path with a degradation menu
(seq 1024→768→512, rank 16→8, 1.5B→0.5B) is implemented;
`scripts/verify_checkpoint.py` distinguishes CHECKPOINT_INVALID /
CHECKPOINT_TAMPERED / NOT_PROMOTABLE.

## 12. Eval results

All four (`raw_base`, `amali_wrapped_raw_base`, `amali_ft_v0`,
`amali_wrapped_ft_v0`) refuse with typed statuses
(`DEPS_MISSING` at run time / `MODEL_NOT_AVAILABLE`); no invented scores.

## 13. Safety regression

`NOT_RUN` — no evals exist to regress against.

## 14. Promotion decision

**`NOT_READY`** — gates 1–5 and 17–19 PASS; gates 6–16 FAIL because no
training/eval evidence exists. No partial promotion.

## 15. Comparison result

`NOT_RUN` — missing sides listed exactly: `raw_base`,
`amali_wrapped_ft_v0`.

## 16. Demo result

Gate A demo PASS. `--with-local-model --model amali_ft_v0` reports
**`MODEL_NOT_AVAILABLE`** honestly (`gate_b_model_report.json`).

## 17. Model card path

`artifacts/base_model_gate/<timestamp>/model_card.md` — states
DATASET_NOT_READY / promotion NOT_READY, "not trained from scratch", no
performance claims; forbidden-marker check enforced in code.

## 18. Security validation

**PASS** on all four mandatory scans (dataset secrets, artifact secrets,
model-card claims, git tracked paths). Optional tools: gitleaks ran —
6 findings, all deliberate fake tokens in test fixtures; semgrep ran —
CI GitHub Actions use mutable tags (pin to SHAs recommended);
bandit / ruff / mypy honestly `NOT_RUN` (not installed).

## 19. Remaining risks

- bitsandbytes has no CUDA-13.2 Windows binary → QLoRA unavailable on
  this machine; approved path is plain LoRA, est. 7.2 GB peak on an 8 GB
  card (tight; OOM degradation menu ready).
- The torch-free dry-run layer can only check bitsandbytes *presence*;
  the environment probe is the authority on kernel usability.
- Deterministic scorer is lexical (AMALI-system behavior, not general
  ability); no external benchmark run.
- CI workflow actions not yet pinned to commit SHAs.
- GitHub `origin` remote removed locally; sync policy with
  `D:\AMALI\_git-remotes\AMALI-Core.git` is now owner-defined.

## 20. Exact next owner action (in order)

```powershell
# 1. Supply owner data (>= 68 more non-synthetic examples' worth):
#    create D:\AMALI\data\raw\ + sourcepack_manifest.json
#    (see docs/data/OWNER_SOURCEPACK_GUIDE.md), then:
python scripts/validate_owner_sourcepack.py
python scripts/build_training_dataset.py
python scripts/check_contamination.py

# 2. Pin and download the base model (get <commit_sha> from the
#    Hugging Face page of Qwen/Qwen2.5-1.5B-Instruct):
python scripts/download_model.py --model Qwen/Qwen2.5-1.5B-Instruct --revision <commit_sha>
#    then set that sha as allowed_revision in
#    src/amali/model_gateway/base_model_allowlist.py

# 3. Approve the recorded LoRA fallback in training_fallback_plan.json
#    (or install a bitsandbytes build with CUDA 13.2 kernels for QLoRA).

# 4. Gate, then train only on DRY_RUN_PASS:
python scripts/train_amali_adapter.py --dry-run --model Qwen/Qwen2.5-1.5B-Instruct --revision <commit_sha>
python scripts/train_amali_adapter.py --model Qwen/Qwen2.5-1.5B-Instruct --revision <commit_sha>
```
