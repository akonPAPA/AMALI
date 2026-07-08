"""RICARDO-HSGR: Reproducible Invariant-Checked Citation-Anchored Retrieval
with Deterministic Oversight.

Local, deterministic retrieval over a hashed evidence corpus. Retrieved text
is data, never instruction: injection-suspicious documents are labeled and
penalized, and every score is a recorded component breakdown, not an opaque
number.
"""

from amali.retrieval.hsgr import (
    EvidenceDoc,
    HSGRRetriever,
    RetrievalResult,
    ScoredDoc,
    ScoreBreakdown,
    citation_support_score,
)

__all__ = [
    "EvidenceDoc",
    "HSGRRetriever",
    "RetrievalResult",
    "ScoredDoc",
    "ScoreBreakdown",
    "citation_support_score",
]
