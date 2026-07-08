"""Claim verification against retrieved evidence (WP3).

Each extracted claim is compared against every evidence document using the
deterministic HSGR citation-support score:

* **SUPPORTED**: some document covers the claim's terms at or above the
  support threshold — the best such document becomes the citation;
* **CONTRADICTED**: a document overlaps the claim strongly *and* carries an
  explicit negation of its terms;
* **UNSUPPORTED**: evidence exists but nothing reaches the threshold;
* **INSUFFICIENT_EVIDENCE**: there was no evidence to check against at all.

The distinction between UNSUPPORTED and INSUFFICIENT_EVIDENCE matters for
arbitration: the first means "the evidence does not back this", the second
means "we could not check".
"""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, Field

from amali.retrieval.hsgr import EvidenceDoc, citation_support_score
from amali.verifier.claims import ExtractedClaim

__all__ = ["ClaimVerdictStatus", "ClaimVerdict", "verify_claims"]

SUPPORT_THRESHOLD = 0.7
_CONTRADICTION_OVERLAP = 0.5

# Explicit negation markers; a contradiction requires one of these *and*
# strong term overlap with the claim.
_NEGATION = re.compile(
    r"(?i)\b(is not|are not|was not|were not|never|no longer|incorrect|false)\b"
)


class ClaimVerdictStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class ClaimVerdict(BaseModel):
    """The audited verdict for one claim."""

    model_config = {"extra": "forbid"}

    claim_index: int
    claim_text: str
    status: ClaimVerdictStatus
    support_score: float = 0.0
    cited_doc_id: str | None = None
    cited_doc_hash: str | None = None


def verify_claims(
    claims: list[ExtractedClaim],
    evidence: list[EvidenceDoc],
) -> list[ClaimVerdict]:
    """Verify every claim against the evidence corpus, deterministically."""
    verdicts: list[ClaimVerdict] = []
    docs = sorted(evidence, key=lambda d: d.doc_id)

    for claim in claims:
        if not docs:
            verdicts.append(
                ClaimVerdict(
                    claim_index=claim.claim_index,
                    claim_text=claim.text,
                    status=ClaimVerdictStatus.INSUFFICIENT_EVIDENCE,
                )
            )
            continue

        best_doc: EvidenceDoc | None = None
        best_support = 0.0
        contradicted_by: EvidenceDoc | None = None

        for doc in docs:
            support = citation_support_score(claim.text, doc)
            if support > best_support:
                best_support = support
                best_doc = doc
            if (
                contradicted_by is None
                and support >= _CONTRADICTION_OVERLAP
                and _NEGATION.search(doc.content)
            ):
                contradicted_by = doc

        # Contradiction wins over support: conflicting evidence must never
        # silently resolve to SUPPORTED (fail-safe ordering).
        if contradicted_by is not None:
            verdicts.append(
                ClaimVerdict(
                    claim_index=claim.claim_index,
                    claim_text=claim.text,
                    status=ClaimVerdictStatus.CONTRADICTED,
                    support_score=best_support,
                    cited_doc_id=contradicted_by.doc_id,
                    cited_doc_hash=contradicted_by.sha256(),
                )
            )
        elif best_support >= SUPPORT_THRESHOLD and best_doc is not None:
            verdicts.append(
                ClaimVerdict(
                    claim_index=claim.claim_index,
                    claim_text=claim.text,
                    status=ClaimVerdictStatus.SUPPORTED,
                    support_score=best_support,
                    cited_doc_id=best_doc.doc_id,
                    cited_doc_hash=best_doc.sha256(),
                )
            )
        else:
            verdicts.append(
                ClaimVerdict(
                    claim_index=claim.claim_index,
                    claim_text=claim.text,
                    status=ClaimVerdictStatus.UNSUPPORTED,
                    support_score=best_support,
                )
            )
    return verdicts
