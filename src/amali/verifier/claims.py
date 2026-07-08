"""Claim extraction (WP3): factual sentences out of answer text.

Deterministic and conservative: an answer is split into sentences; a
sentence counts as a factual claim only if it has enough content terms to be
checkable. Noise (empty strings, interjections, pure punctuation, questions)
is rejected rather than passed downstream as fake claims.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, field_validator

__all__ = ["ExtractedClaim", "extract_claims"]

_MIN_CONTENT_TERMS = 3

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _content_terms(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text)


class ExtractedClaim(BaseModel):
    """One checkable factual claim extracted from answer text."""

    model_config = {"extra": "forbid"}

    claim_index: int
    text: str

    @field_validator("text")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("claim text must not be empty")
        return value


def extract_claims(answer_text: str) -> list[ExtractedClaim]:
    """Extract checkable factual claims from ``answer_text``.

    Rules:
    * sentences are the claim unit;
    * questions are not claims;
    * a claim needs at least ``_MIN_CONTENT_TERMS`` content terms;
    * order is preserved and indices are stable.
    """
    if not answer_text or not answer_text.strip():
        return []

    claims: list[ExtractedClaim] = []
    index = 0
    for raw in _SENTENCE_SPLIT.split(answer_text.strip()):
        sentence = raw.strip()
        if not sentence:
            continue
        if sentence.endswith("?"):
            continue  # questions assert nothing
        if len(_content_terms(sentence)) < _MIN_CONTENT_TERMS:
            continue  # too short to be checkable; reject noise
        claims.append(ExtractedClaim(claim_index=index, text=sentence))
        index += 1
    return claims
