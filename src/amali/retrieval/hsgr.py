"""RICARDO-HSGR retrieval core (WP2). Label: CUSTOM_AMALI_COMPOSITION.

Scoring formula (deterministic, all components recorded):

    overlap   = |query_terms ∩ doc_terms| / |query_terms|        in [0, 1]
    exact     = 1.0 if the normalized query is a substring of the
                normalized document else 0.0
    trust     = {LOW: 0.25, MEDIUM: 0.5, HIGH: 0.75, OFFICIAL: 1.0,
                 EXECUTABLE: 1.0}[doc.trust_level]
    poison    = 0.5 if the doc matches an injection/poison pattern
                or carries a high poison label else 0.0

    final = 0.55*overlap + 0.15*exact + 0.30*trust - poison
    (clamped to [0, 1]; ties broken by doc_id ascending)

Invariants:

* the same corpus + query always yields the same ranking (no randomness);
* every document carries a SHA-256 content hash — a changed source is a
  changed citation;
* injection-suspicious text is *labeled and penalized*, never executed or
  reinterpreted as instructions;
* the score breakdown is part of the result, so any ranking can be
  re-derived and audited by hand.
"""

from __future__ import annotations

import hashlib
import re

from pydantic import BaseModel, Field, field_validator

from amali.policy.enums import DataSensitivity
from amali.state.models import TrustLevel

__all__ = [
    "EvidenceDoc",
    "ScoreBreakdown",
    "ScoredDoc",
    "RetrievalResult",
    "HSGRRetriever",
    "citation_support_score",
]

# Instruction-shaped text inside retrieved documents. Presence labels the
# document as injection-suspicious; it never changes system behavior.
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)ignore\s+(all\s+)?(previous|prior|above)\s+instructions"),
    re.compile(r"(?i)disregard\s+(the\s+)?(system|developer)\s+prompt"),
    re.compile(r"(?i)you\s+are\s+now\s+(in\s+)?(developer|admin|root)\s*mode"),
    re.compile(r"(?i)reveal\s+(your\s+)?(system\s+prompt|secrets?|api\s*keys?)"),
    re.compile(r"(?i)grant\s+(me\s+)?(shell|admin|root|network)\s+(access|permission)"),
)

_TRUST_WEIGHT: dict[TrustLevel, float] = {
    TrustLevel.LOW: 0.25,
    TrustLevel.MEDIUM: 0.5,
    TrustLevel.HIGH: 0.75,
    TrustLevel.OFFICIAL: 1.0,
    TrustLevel.EXECUTABLE: 1.0,
}

_POISON_PENALTY = 0.5

_W_OVERLAP = 0.55
_W_EXACT = 0.15
_W_TRUST = 0.30


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def content_hash(text: str) -> str:
    """SHA-256 of the exact document content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EvidenceDoc(BaseModel):
    """One local evidence document with provenance and risk labels."""

    model_config = {"extra": "forbid"}

    doc_id: str
    source_uri: str
    content: str
    trust_level: TrustLevel = TrustLevel.MEDIUM
    sensitivity: DataSensitivity = DataSensitivity.INTERNAL
    poison_label: str = "low"  # low | medium | high (author-asserted)

    @field_validator("doc_id", "source_uri", "content")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("field must not be empty")
        return value

    @field_validator("poison_label")
    @classmethod
    def _poison_in_range(cls, value: str) -> str:
        if value not in {"low", "medium", "high"}:
            raise ValueError("poison_label must be low|medium|high")
        return value

    def sha256(self) -> str:
        return content_hash(self.content)

    def injection_suspicious(self) -> bool:
        """True if the content contains instruction-shaped injection text."""
        return any(p.search(self.content) for p in _INJECTION_PATTERNS)


class ScoreBreakdown(BaseModel):
    """Every component of one document's score, re-derivable by hand."""

    model_config = {"extra": "forbid"}

    overlap: float
    exact: float
    trust: float
    poison_penalty: float
    final: float


class ScoredDoc(BaseModel):
    """One ranked retrieval hit with citation anchors."""

    model_config = {"extra": "forbid"}

    doc_id: str
    source_uri: str
    source_hash: str
    injection_suspicious: bool
    poison_label: str
    breakdown: ScoreBreakdown


class RetrievalResult(BaseModel):
    """The full, auditable result of one retrieval."""

    model_config = {"extra": "forbid"}

    query: str
    ranked: list[ScoredDoc] = Field(default_factory=list)

    def top(self) -> ScoredDoc | None:
        return self.ranked[0] if self.ranked else None


class HSGRRetriever:
    """Deterministic retrieval over an in-memory hashed corpus."""

    def __init__(self, corpus: list[EvidenceDoc]) -> None:
        ids = [d.doc_id for d in corpus]
        if len(set(ids)) != len(ids):
            raise ValueError("doc_ids must be unique")
        # Fixed iteration order regardless of construction order.
        self._corpus = sorted(corpus, key=lambda d: d.doc_id)

    def get(self, doc_id: str) -> EvidenceDoc | None:
        for doc in self._corpus:
            if doc.doc_id == doc_id:
                return doc
        return None

    def score(self, query: str, doc: EvidenceDoc) -> ScoreBreakdown:
        """Compute the recorded component score for one document."""
        q_terms = _terms(query)
        d_terms = _terms(doc.content)
        overlap = (
            len(q_terms & d_terms) / len(q_terms) if q_terms else 0.0
        )
        exact = 1.0 if _normalize(query) in _normalize(doc.content) else 0.0
        trust = _TRUST_WEIGHT[doc.trust_level]
        poison = (
            _POISON_PENALTY
            if (doc.injection_suspicious() or doc.poison_label == "high")
            else 0.0
        )
        raw = _W_OVERLAP * overlap + _W_EXACT * exact + _W_TRUST * trust
        final = max(0.0, min(1.0, raw - poison))
        return ScoreBreakdown(
            overlap=round(overlap, 6),
            exact=exact,
            trust=trust,
            poison_penalty=poison,
            final=round(final, 6),
        )

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        """Rank the corpus for ``query``; deterministic ties on doc_id."""
        scored: list[ScoredDoc] = []
        for doc in self._corpus:
            breakdown = self.score(query, doc)
            scored.append(
                ScoredDoc(
                    doc_id=doc.doc_id,
                    source_uri=doc.source_uri,
                    source_hash=doc.sha256(),
                    injection_suspicious=doc.injection_suspicious(),
                    poison_label=doc.poison_label,
                    breakdown=breakdown,
                )
            )
        scored.sort(key=lambda s: (-s.breakdown.final, s.doc_id))
        return RetrievalResult(query=query, ranked=scored[:k])


def citation_support_score(answer_text: str, doc: EvidenceDoc) -> float:
    """How well a cited document supports an answer sentence, in [0, 1].

    Term coverage of the answer by the document: 1.0 means every content
    term of the answer appears in the document. Deterministic and
    re-derivable; used by the verifier and the retrieval report.
    """
    a_terms = _terms(answer_text)
    if not a_terms:
        return 0.0
    d_terms = _terms(doc.content)
    return round(len(a_terms & d_terms) / len(a_terms), 6)
