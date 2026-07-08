# Base AI Gate — Algorithm Proof Documentation (WP12)

Every named algorithm in the AMALI Base AI Gate, mapped to its formula,
invariants, code, tests, artifacts, and claim boundary. All entries are
**CUSTOM_AMALI_COMPOSITION** (deterministic compositions authored for AMALI)
except PIM-Lite, which is **R_AND_D_ONLY** and intentionally unimplemented.

The formulas below are golden-tested: the listed unit tests recompute each
formula by hand and assert the code produces the same number.

---

## RICARDO-HSGR — Reproducible Invariant-Checked Citation-Anchored Retrieval with Deterministic Oversight

- **Purpose:** local retrieval scoring and citation support.
- **Inputs:** query text; corpus of `EvidenceDoc` (content, trust level,
  poison label). **Outputs:** ranked `ScoredDoc` list with per-component
  `ScoreBreakdown` and SHA-256 source hashes.
- **Formula:**
  `overlap = |Q ∩ D| / |Q|`; `exact = 1 if norm(query) ⊆ norm(doc) else 0`;
  `trust ∈ {0.25, 0.5, 0.75, 1.0}`; `poison = 0.5` if injection-suspicious
  or poison label high, else 0;
  `final = clamp(0.55·overlap + 0.15·exact + 0.30·trust − poison, 0, 1)`;
  ties break on ascending `doc_id`.
- **Invariants:** same corpus + query ⇒ same ranking; retrieved text is data,
  never instruction; injection-shaped text is labeled and penalized, never
  executed; every score is re-derivable from the breakdown.
- **Assumptions:** lexical term overlap is a usable relevance proxy at
  Gate A scale. **Failure modes:** paraphrase misses, synonym blindness.
- **Code:** `src/amali/retrieval/hsgr.py`.
  **Tests:** `tests/unit/test_retrieval_hsgr.py` (exact match ranks first,
  poisoning penalty, citation support, formula golden test).
  **Artifact:** `retrieval_report.json` / `.md`. **Metric:**
  `citation_support_score`.
- **Claim boundary:** proves deterministic, injection-aware retrieval — not
  semantic search quality.

## RICARDO-EWA — Reproducible Invariant-Checked Evidence-Weighted Arbitration

- **Purpose:** final response-status arbitration from claim verdicts + risk.
- **Inputs:** `ClaimVerdict` list, secret-exposure flag. **Outputs:**
  `ArbitrationResult` (status, score, gates fired).
- **Formula:** hard gates first, in order —
  HG2 secret exposure ⇒ BLOCKED; HG1 any contradicted ⇒ NEEDS_REVIEW;
  HG3 no claims ⇒ UNKNOWN; HG4 all insufficient ⇒ UNKNOWN;
  HG5 any unsupported ⇒ never SUCCESS (PARTIAL if some supported, else
  UNKNOWN); otherwise SUCCESS. Score (informational, below the gates):
  `ewa_score = mean(support_i)`.
- **Invariants:** no score can override a hard gate; an unsupported factual
  claim can never produce SUCCESS.
- **Code:** `src/amali/arbiter/ewa.py`.
  **Tests:** `tests/unit/test_verifier_and_ewa.py` (hard gates override
  score; unsupported cannot be success). **Artifact:** `ewa_report.json`.
  **Metric:** `ewa_hard_gate_integrity`.
- **Claim boundary:** arbitration correctness over deterministic verdicts;
  verification itself is lexical, not entailment.

## RICARDO-FEC — Reproducible Invariant-Checked Failure-to-Eval Compiler

- **Purpose:** convert runtime failures into redacted, deduplicated eval
  items (permanent regression guards).
- **Formula:** redact first, then dedup key
  `sha256(component \x00 failure_kind \x00 redacted_query)`;
  `eval_id = "eval_" + key[:16]` (stable across runs);
  `conversion_rate = unique / total`.
- **Invariants:** no raw secret survives into an eval item; duplicates merge
  (counted, not stored); ids are stable so suites do not churn.
- **Code:** `src/amali/eval/fec.py`.
  **Tests:** `tests/unit/test_failure_to_eval.py`.
  **Artifact:** `failure_to_eval_report.json`.
  **Metric:** `failure_to_eval_conversion_rate`.

## AMALI-DWAC — Dynamic Weight Activation Controller

- **Purpose:** plan the minimal sufficient pack set under a resource budget.
- **Pseudocode:** sort packs by `(cost_mb, pack_id)`; greedily take a pack
  only if it covers ≥1 uncovered required capability and fits the remaining
  budget; full cover ⇒ feasible plan; else designated fallback if it fits
  (capability gaps recorded); else infeasible.
- **Invariants:** no invalid winner — a returned plan never exceeds budget
  and never names an unknown pack; no useless pack is selected;
  deterministic tie-breaks.
- **Code:** `src/amali/activation/planner.py`.
  **Tests:** `tests/unit/test_activation_planner.py`.
  **Artifact:** `activation_report.json`.
  **Metric:** `activation_plan_feasibility_rate`.
- **Claim boundary:** resource *planning* only; no weights are loaded.

## AMALI-UGE — Uncertainty-Gated Escalation

- **Purpose:** escalate on uncertainty × risk; never de-escalate.
- **Formula:** `u = 1 − ewa_score`; risk ≥ L4 ⇒ NEEDS_REVIEW (unless already
  BLOCKED); risk ≥ L3 and `u > 0.3` ⇒ NEEDS_REVIEW; BLOCKED is terminal.
- **Code:** `src/amali/arbiter/ewa.py::uge_escalate`.
  **Tests:** `tests/unit/test_verifier_and_ewa.py` (L4 always escalates,
  BLOCKED never de-escalates). **Artifact:** `ewa_report.json`.

## AMALI-TRPC — Tool Risk Permission Compiler

- **Purpose:** least-privilege tool permission compilation.
- **Pseudocode:** 14-step default-deny order (canonical docstring in
  `src/amali/policy/trpc.py`): request schema → policy integrity → tool
  resolution → manifest integrity → dangerous patterns → forbidden op →
  op-not-allowed → secret → network → filesystem/traversal → risk L5 →
  TTL → review gate (verified scoped approval only) → least-privilege grant.
- **Invariants:** hard denies precede all permissive branches; forbidden
  overrides allowed; effective risk = max(task, tool) — never lowerable by
  request metadata; every decision emits one audit event; a raw string
  "approved" never verifies.
- **Code:** `src/amali/policy/trpc.py` + `src/amali/tools/`.
  **Tests:** `tests/unit/test_trpc_permission_decisions.py`,
  `tests/unit/test_trpc_negative_security.py`,
  `tests/unit/test_executor_boundary.py`.
  **Artifact:** `runtime_trace.json`. **Metric:** `policy_bypass_rate`.

## HER-MoE System Router (supporting, IGA-03)

- Hard constraints → deterministic scoring → ranked decision → audit; an
  excluded route can never win regardless of score.
- **Code:** `src/amali/router/`. **Tests:**
  `tests/unit/test_her_moe_router.py`. **Artifact:** `runtime_trace.json`.
  **Metric:** `route_zero_violation_rate`.

## AMALI-PIM-Lite — Predictive Integrity Monitor Lite

- **Label: R_AND_D_ONLY. Not implemented in the Base AI Gate.** There is no
  code, no test, and no claim; it appears in `remaining_risks.md` as
  intentionally absent. Implementing it requires an eval foundation first.

---

## Known limitations (all algorithms)

- Retrieval, verification, and answering are lexical; Gate A proves the
  *system flow*, not language ability.
- The deterministic answerer quotes evidence verbatim; Gate B quality is
  unmeasured until owner-approved weights and a frozen suite exist.
- Audit ledger and approvals are in-memory this stage.
