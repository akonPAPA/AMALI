"""The deterministic, evidence-only answerer (WP8 / Gate A).

This is not a language model and does not pretend to be one. It answers by
*quoting evidence*: the answer text is always a verbatim sentence from a
retrieved document, so every claim it makes is supported by construction —
or it declines and reports UNKNOWN.

Selection rule (deterministic):

* each sentence of each candidate document is scored by query-term
  coverage (``|query_terms ∩ sentence_terms| / |query_terms|``);
* the best sentence wins; ties break on (doc_id, sentence order);
* if the best coverage is below the answer threshold, there is no answer —
  the candidate is marked unanswered and the pipeline reports UNKNOWN.

Injection-suspicious documents are never quoted, even if they score well:
quoting attacker-controlled text into an answer is an injection vector.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from amali.retrieval.hsgr import EvidenceDoc

__all__ = ["AnswerCandidate", "DeterministicAnswerer"]

ANSWER_THRESHOLD = 0.6

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class AnswerCandidate(BaseModel):
    """The answerer's output before verification."""

    model_config = {"extra": "forbid"}

    answered: bool
    text: str = ""
    cited_doc_ids: list[str] = Field(default_factory=list)
    coverage: float = 0.0


class DeterministicAnswerer:
    """Answers strictly by quoting the best-covering evidence sentence."""

    def answer(
        self, query: str, docs: list[EvidenceDoc]
    ) -> AnswerCandidate:
        q_terms = _terms(query)
        if not q_terms:
            return AnswerCandidate(answered=False)

        best: tuple[float, str, str] | None = None  # (coverage, doc_id, text)
        for doc in sorted(docs, key=lambda d: d.doc_id):
            if doc.injection_suspicious():
                continue  # never quote injection-suspicious text
            for sentence in _SENTENCE_SPLIT.split(doc.content.strip()):
                sentence = sentence.strip()
                if not sentence:
                    continue
                coverage = len(q_terms & _terms(sentence)) / len(q_terms)
                key = (coverage, doc.doc_id, sentence)
                if best is None or coverage > best[0]:
                    best = key

        if best is None or best[0] < ANSWER_THRESHOLD:
            return AnswerCandidate(
                answered=False,
                coverage=round(best[0], 6) if best else 0.0,
            )
        coverage, doc_id, sentence = best
        return AnswerCandidate(
            answered=True,
            text=sentence,
            cited_doc_ids=[doc_id],
            coverage=round(coverage, 6),
        )
