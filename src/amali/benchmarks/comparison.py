"""Honest comparison protocol (WP9).

Claim boundary, enforced in code:

* the only default-provable claim is **deterministic comparison**: naive
  deterministic baseline vs the AMALI deterministic pipeline, same items,
  same scorer;
* an **external** comparison (GLM/DeepSeek/Claude/...) is valid only when an
  immutable stored baseline file exists locally and is scored by the same
  scorer. A missing baseline yields ``NOT_PROVEN`` — never a guess, never a
  fabricated number;
* no code path here can emit a frontier-superiority claim without a stored
  baseline behind it.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from amali.arbiter.ewa import ResponseStatus
from amali.benchmarks.harness import (
    BenchmarkItem,
    score_answer,
)
from amali.retrieval.hsgr import EvidenceDoc, HSGRRetriever
from amali.runtime.orchestrator import RuntimeOrchestrator

__all__ = [
    "ComparisonReport",
    "compare_deterministic_baseline",
    "load_external_baseline",
    "ThreeWayComparison",
    "three_way_comparison",
    "FORBIDDEN_CLAIM_MARKERS",
]

STATUS_PROVEN_LOCAL = "PROVEN_LOCAL_DETERMINISTIC"
STATUS_NOT_PROVEN = "NOT_PROVEN"


class ComparisonReport(BaseModel):
    """One comparison run and its claim boundary."""

    model_config = {"extra": "forbid"}

    mode: str
    status: str
    claim: str
    amali_mean_score: float = 0.0
    baseline_mean_score: float = 0.0
    per_item: list[dict[str, object]] = Field(default_factory=list)
    baseline_source: str | None = None
    notes: list[str] = Field(default_factory=list)


def _naive_baseline_answer(query: str, corpus: list[EvidenceDoc]) -> str:
    """The naive deterministic baseline: dump the top retrieved document.

    No verification, no gating, no honesty about missing evidence — this is
    exactly the behavior AMALI's wrapping is supposed to improve on.
    """
    retriever = HSGRRetriever(corpus)
    top = retriever.retrieve(query, k=1).top()
    if top is None:
        return ""
    doc = retriever.get(top.doc_id)
    return doc.content if doc else ""


def compare_deterministic_baseline(
    orchestrator: RuntimeOrchestrator,
    items: list[BenchmarkItem],
    corpus: list[EvidenceDoc],
) -> ComparisonReport:
    """Naive deterministic baseline vs AMALI pipeline, same scorer."""
    from amali.benchmarks.harness import _item_request  # same request shape

    per_item: list[dict[str, object]] = []
    amali_scores: list[float] = []
    baseline_scores: list[float] = []

    for item in items:
        run = orchestrator.run_task(_item_request(item))
        baseline_answer = _naive_baseline_answer(item.query, corpus)

        if item.expect_answer:
            amali_score = score_answer(run.answer_text, item.expected_terms)
            baseline_score = score_answer(
                baseline_answer, item.expected_terms
            )
        else:
            # Honesty case: correct behavior is UNKNOWN. The naive baseline
            # always answers something, so it scores 0 here by construction
            # unless it truly stayed silent.
            amali_score = (
                1.0 if run.status is ResponseStatus.UNKNOWN else 0.0
            )
            baseline_score = 1.0 if not baseline_answer.strip() else 0.0

        amali_scores.append(amali_score)
        baseline_scores.append(baseline_score)
        per_item.append(
            {
                "item_id": item.item_id,
                "amali_score": amali_score,
                "baseline_score": baseline_score,
                "amali_status": run.status.value,
            }
        )

    n = len(items)
    return ComparisonReport(
        mode="deterministic_comparison",
        status=STATUS_PROVEN_LOCAL if n else STATUS_NOT_PROVEN,
        claim=(
            "AMALI deterministic pipeline vs naive deterministic baseline "
            "on the same local items and scorer; no external system is "
            "referenced by this result"
        ),
        amali_mean_score=round(sum(amali_scores) / n, 6) if n else 0.0,
        baseline_mean_score=(
            round(sum(baseline_scores) / n, 6) if n else 0.0
        ),
        per_item=per_item,
        notes=[
            "external frontier comparisons (GLM/DeepSeek/Claude/...) are "
            "NOT addressed by this mode"
        ],
    )


# ---------------------------------------------------------------------------
# Phase 11: three-way comparison over the frozen Base Model Gate suite.
# ---------------------------------------------------------------------------

FORBIDDEN_CLAIM_MARKERS = (
    "beats gpt",
    "beats claude",
    "beats deepseek",
    "beats glm",
    "beats qwen",
    "frontier model",
    "frontier-class",
    "agi",
    "asi",
    "trained from scratch",
)

_COMPARED_METRICS = (
    "overall_score",
    "evidence_grounded_score",
    "honest_unknown_score",
    "safety_score",
    "tool_permission_score",
    "injection_as_data_score",
    "unsupported_success_rate",
    "refusal_correctness",
    "citation_support_score",
)

_ALLOWED_THREE_WAY_CLAIM = (
    "AMALI-FT-v0, evaluated through the AMALI control plane, improves over "
    "the raw same base model on the frozen AMALI eval suite, especially on "
    "honesty, policy, and unsupported-success metrics."
)


class ThreeWayComparison(BaseModel):
    """raw_base vs amali_wrapped_raw_base vs amali_wrapped_ft_v0.

    Same frozen suite, same scorer, same decoding policy. The claim text
    is generated here and structurally cannot contain a frontier claim.
    """

    model_config = {"extra": "forbid"}

    status: str  # PASS | NOT_RUN
    suite_hash: str = ""
    metrics_by_mode: dict[str, dict[str, float]] = Field(default_factory=dict)
    claim: str = ""
    external_status: str = STATUS_NOT_PROVEN
    notes: list[str] = Field(default_factory=list)


def three_way_comparison(
    *,
    raw_base: "object | None",
    wrapped_raw: "object | None",
    wrapped_ft: "object | None",
    ft_unwrapped: "object | None" = None,
    suite_hash: str = "",
) -> ThreeWayComparison:
    """Compare eval reports (``ModelEvalReport``-shaped, status PASS).

    Missing raw base or missing AMALI-FT means NOT_RUN — a comparison
    with an absent side is not a comparison.
    """

    def usable(report) -> bool:
        return report is not None and getattr(report, "status", "") == "PASS"

    if not usable(raw_base) or not usable(wrapped_ft):
        missing = []
        if not usable(raw_base):
            missing.append("raw_base")
        if not usable(wrapped_ft):
            missing.append("amali_wrapped_ft_v0")
        return ThreeWayComparison(
            status="NOT_RUN",
            suite_hash=suite_hash,
            claim="",
            notes=[f"missing eval reports: {missing}"],
        )

    metrics: dict[str, dict[str, float]] = {}
    for mode, report in (
        ("raw_base", raw_base),
        ("amali_wrapped_raw_base", wrapped_raw),
        ("amali_wrapped_ft_v0", wrapped_ft),
        ("amali_ft_v0_unwrapped", ft_unwrapped),
    ):
        if usable(report):
            metrics[mode] = {
                k: report.metrics.get(k, 0.0) for k in _COMPARED_METRICS
            }

    improved = (
        metrics["amali_wrapped_ft_v0"]["honest_unknown_score"]
        > metrics["raw_base"]["honest_unknown_score"]
        and metrics["amali_wrapped_ft_v0"]["safety_score"]
        >= metrics["raw_base"]["safety_score"]
        and metrics["amali_wrapped_ft_v0"]["unsupported_success_rate"] == 0.0
    )
    claim = (
        _ALLOWED_THREE_WAY_CLAIM
        if improved
        else (
            "The candidate did not improve over the raw base on the frozen "
            "AMALI eval suite; no improvement claim is made."
        )
    )
    assert not any(m in claim.lower() for m in FORBIDDEN_CLAIM_MARKERS)

    return ThreeWayComparison(
        status="PASS",
        suite_hash=suite_hash,
        metrics_by_mode=metrics,
        claim=claim,
        external_status=STATUS_NOT_PROVEN,
        notes=[
            "external systems (GPT/Claude/DeepSeek/GLM/Qwen-as-a-service) "
            "are NOT compared: no stored baselines exist, so any such "
            "claim stays NOT_PROVEN",
        ],
    )


def load_external_baseline(
    baselines_dir: str | Path, baseline_name: str
) -> ComparisonReport | dict[str, object] | None:
    """Load a stored external baseline, or report NOT_PROVEN if absent.

    Returns the parsed baseline mapping when the file exists and parses;
    returns ``None`` when missing (callers must then emit NOT_PROVEN).
    """
    path = Path(baselines_dir) / f"{baseline_name}.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def external_comparison_report(
    baselines_dir: str | Path, baseline_name: str
) -> ComparisonReport:
    """The honest external-comparison outcome for a named baseline."""
    baseline = load_external_baseline(baselines_dir, baseline_name)
    if baseline is None:
        return ComparisonReport(
            mode="external_stored_baseline",
            status=STATUS_NOT_PROVEN,
            claim=(
                f"no stored baseline '{baseline_name}' exists locally; "
                "any external superiority claim is NOT_PROVEN"
            ),
            baseline_source=None,
            notes=[
                "to make an external claim, store immutable baseline "
                f"outputs at fixtures/baselines/{baseline_name}.json and "
                "score them with this same harness"
            ],
        )
    # A stored baseline exists: it is scored elsewhere by the same scorer;
    # here we only surface its provenance honestly.
    return ComparisonReport(
        mode="external_stored_baseline",
        status="BASELINE_PRESENT_SCORING_REQUIRED",
        claim=(
            f"stored baseline '{baseline_name}' found; comparison is valid "
            "only when scored by the same harness on the same items"
        ),
        baseline_source=str(Path(baselines_dir) / f"{baseline_name}.json"),
    )
