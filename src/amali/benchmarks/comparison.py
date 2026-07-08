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
