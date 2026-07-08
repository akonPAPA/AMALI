"""Contamination gate between training data and the frozen eval suite.

Training on eval content produces fake progress, so this gate runs before
any training and PASSES only with **zero** leakage. Checks:

1. exact content hash overlap (training text == eval text);
2. normalized text hash overlap (case/whitespace/punctuation collapsed);
3. shingled Jaccard near-duplicate similarity >= 0.80;
4. forbidden source-path overlap (training example built from an eval or
   holdout fixture path);
5. eval expected-answer leakage (an eval item's expected terms wholly
   contained in a short training response aimed at the same prompt);
6. rubric leakage (a scoring rubric string appearing in training text).

The gate never silently drops contaminated items to reach PASS — it
reports the exact contaminated ids and fails. Removal is a human decision
that must be visible in the dataset build, not here.

Deterministic; no torch; no network.
"""

from __future__ import annotations

import hashlib
import re

from pydantic import BaseModel, Field

from amali.eval.suite import EvalItem
from amali.training.dataset import TrainingExample

__all__ = [
    "ContaminationFinding",
    "ContaminationReport",
    "check_contamination",
    "normalized_hash",
    "jaccard_shingles",
    "near_duplicate_similarity",
]

NEAR_DUPLICATE_THRESHOLD = 0.80
SHINGLE_SIZE = 3

FORBIDDEN_PATH_MARKERS = ("fixtures/eval", "fixtures\\eval", "holdout")

_WORD = re.compile(r"[a-z0-9]+")


def _normalize(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def normalized_hash(text: str) -> str:
    return hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()


def _shingles(text: str, size: int = SHINGLE_SIZE) -> set[tuple[str, ...]]:
    words = _WORD.findall(text.lower())
    if len(words) < size:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}


def jaccard_shingles(a: str, b: str, size: int = SHINGLE_SIZE) -> float:
    sa, sb = _shingles(a, size), _shingles(b, size)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def near_duplicate_similarity(a: str, b: str, size: int = SHINGLE_SIZE) -> float:
    """max(Jaccard, containment) over word shingles.

    Containment (`|A∩B| / min(|A|,|B|)`) stays high when one text is the
    other plus a few inserted words — exactly the paraphrase shape plain
    Jaccard under-scores.
    """
    sa, sb = _shingles(a, size), _shingles(b, size)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    jaccard = inter / len(sa | sb)
    containment = inter / min(len(sa), len(sb))
    return max(jaccard, containment)


class ContaminationFinding(BaseModel):
    """One detected leak between training and eval material."""

    model_config = {"extra": "forbid"}

    kind: str
    training_example_id: str
    eval_item_id: str = ""
    detail: str = ""
    similarity: float = 0.0


class ContaminationReport(BaseModel):
    """PASS only with zero exact and zero near-duplicate leakage."""

    model_config = {"extra": "forbid"}

    status: str  # PASS | FAIL | NOT_RUN
    training_examples: int
    eval_items: int
    findings: list[ContaminationFinding] = Field(default_factory=list)

    @property
    def contaminated_ids(self) -> list[str]:
        return sorted({f.training_example_id for f in self.findings})


def _training_texts(example: TrainingExample) -> list[str]:
    return [
        t
        for t in (example.instruction, example.input_context, example.expected_response)
        if t
    ]


def _eval_texts(item: EvalItem) -> list[str]:
    return [item.prompt]


def check_contamination(
    training: list[TrainingExample],
    eval_items: list[EvalItem],
    *,
    holdout_texts: list[str] | None = None,
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> ContaminationReport:
    """Compare the training set against the frozen eval suite (and any
    extra holdout texts) and report every leak found."""
    if not training or not eval_items:
        return ContaminationReport(
            status="NOT_RUN",
            training_examples=len(training),
            eval_items=len(eval_items),
        )

    findings: list[ContaminationFinding] = []
    holdout_texts = holdout_texts or []

    eval_exact = {}
    eval_norm = {}
    for item in eval_items:
        for text in _eval_texts(item):
            eval_exact.setdefault(
                hashlib.sha256(text.encode("utf-8")).hexdigest(), item.item_id
            )
            eval_norm.setdefault(normalized_hash(text), item.item_id)
    holdout_norm = {normalized_hash(t): "holdout" for t in holdout_texts}

    for example in training:
        # 4. forbidden source path
        path_l = (example.source_path + " " + example.source_id).lower()
        if any(marker in path_l for marker in FORBIDDEN_PATH_MARKERS):
            findings.append(
                ContaminationFinding(
                    kind="forbidden_source_path",
                    training_example_id=example.example_id,
                    detail=example.source_id,
                )
            )

        for text in _training_texts(example):
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            norm = normalized_hash(text)
            # 1. exact hash overlap
            if digest in eval_exact:
                findings.append(
                    ContaminationFinding(
                        kind="exact_hash_overlap",
                        training_example_id=example.example_id,
                        eval_item_id=eval_exact[digest],
                        similarity=1.0,
                    )
                )
                continue
            # 2. normalized hash overlap
            if norm in eval_norm:
                findings.append(
                    ContaminationFinding(
                        kind="normalized_hash_overlap",
                        training_example_id=example.example_id,
                        eval_item_id=eval_norm[norm],
                        similarity=1.0,
                    )
                )
                continue
            if norm in holdout_norm:
                findings.append(
                    ContaminationFinding(
                        kind="holdout_overlap",
                        training_example_id=example.example_id,
                        detail="normalized match against holdout split",
                        similarity=1.0,
                    )
                )
                continue
            # 3. shingled near-duplicate
            for item in eval_items:
                for eval_text in _eval_texts(item):
                    similarity = near_duplicate_similarity(text, eval_text)
                    if similarity >= threshold:
                        findings.append(
                            ContaminationFinding(
                                kind="near_duplicate",
                                training_example_id=example.example_id,
                                eval_item_id=item.item_id,
                                similarity=round(similarity, 4),
                            )
                        )

        # 5. expected-answer leakage: every expected term of an eval item
        #    present in a focused training response.
        for item in eval_items:
            if not item.expected_terms:
                continue
            response_l = example.expected_response.lower()
            if len(response_l) <= 300 and all(
                term.lower() in response_l for term in item.expected_terms
            ):
                prompt_sim = jaccard_shingles(example.instruction, item.prompt)
                if prompt_sim >= 0.5:
                    findings.append(
                        ContaminationFinding(
                            kind="expected_answer_leakage",
                            training_example_id=example.example_id,
                            eval_item_id=item.item_id,
                            similarity=round(prompt_sim, 4),
                        )
                    )
            # 6. rubric leakage
            if item.scoring_rubric and item.scoring_rubric.lower() in (
                example.instruction + " " + example.input_context + " "
                + example.expected_response
            ).lower():
                findings.append(
                    ContaminationFinding(
                        kind="rubric_leakage",
                        training_example_id=example.example_id,
                        eval_item_id=item.item_id,
                    )
                )

    return ContaminationReport(
        status="PASS" if not findings else "FAIL",
        training_examples=len(training),
        eval_items=len(eval_items),
        findings=findings,
    )
