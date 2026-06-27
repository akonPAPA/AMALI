"""Typed epistemic state models.

Design intent (the architectural claims these models prove):

* Generated text is *not* trusted state. A claim is just text plus an
  explicit epistemic classification; it is never automatically KNOWN.
* Claims are explicitly classified (``EvidenceLevel``).
* Evidence is explicit, carries trust/sensitivity, and records poison and
  injection risk.
* A claim may only be KNOWN when it is backed by at least one piece of
  supporting evidence.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

__all__ = [
    "EvidenceLevel",
    "TaskStatus",
    "Sensitivity",
    "TrustLevel",
    "ClaimRecord",
    "EvidenceRecord",
    "TaskState",
    "AMALIResponse",
]

_RISK_VALUES = frozenset({"low", "medium", "high"})


class EvidenceLevel(str, Enum):
    """Epistemic classification of a claim."""

    KNOWN = "KNOWN"
    INFERRED = "INFERRED"
    UNKNOWN = "UNKNOWN"
    HYPOTHESIS = "HYPOTHESIS"
    RISK = "RISK"


class TaskStatus(str, Enum):
    """Lifecycle status of a task."""

    ACCEPTED = "ACCEPTED"
    PLANNING = "PLANNING"
    ROUTING = "ROUTING"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    BLOCKED = "BLOCKED"
    PARTIAL = "PARTIAL"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class Sensitivity(str, Enum):
    """Data sensitivity classification of evidence."""

    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    CONFIDENTIAL = "CONFIDENTIAL"
    SECRET = "SECRET"
    PERSONAL = "PERSONAL"


class TrustLevel(str, Enum):
    """Trust classification of an evidence source."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    OFFICIAL = "OFFICIAL"
    EXECUTABLE = "EXECUTABLE"


class ClaimRecord(BaseModel):
    """A single classified claim about the world or the task.

    A claim is text plus an explicit ``EvidenceLevel``. It carries no
    authority on its own; it can only become ``KNOWN`` when at least one
    supporting evidence reference is attached.
    """

    claim_id: str
    task_id: str
    text: str
    evidence_level: EvidenceLevel
    source: str
    support_refs: list[str] = Field(default_factory=list)
    contradicted_by: list[str] = Field(default_factory=list)
    status: str = "unsupported"

    @field_validator("text")
    @classmethod
    def _text_not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("claim text must not be empty")
        return value

    @model_validator(mode="after")
    def _known_requires_support(self) -> "ClaimRecord":
        # A KNOWN claim must be backed by evidence. This single rule also
        # guarantees that a generated claim with no support_refs can never
        # be KNOWN, regardless of its declared source.
        if self.evidence_level is EvidenceLevel.KNOWN and not self.support_refs:
            raise ValueError(
                "a KNOWN claim requires at least one support_ref"
            )
        return self


class EvidenceRecord(BaseModel):
    """A reference to a piece of evidence and its trust/risk metadata.

    The kernel never stores raw payloads here; only a URI and/or content
    hash plus explicit trust, sensitivity, and adversarial-risk fields.
    """

    evidence_id: str
    task_id: str
    source_type: str
    trust_level: TrustLevel
    sensitivity: Sensitivity
    source_uri: str | None = None
    source_hash: str | None = None
    supports_claims: list[str] = Field(default_factory=list)
    poison_risk: str = "low"
    injection_risk: str = "low"

    @field_validator("source_type")
    @classmethod
    def _source_type_not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("source_type must not be empty")
        return value

    @field_validator("poison_risk", "injection_risk")
    @classmethod
    def _risk_in_range(cls, value: str) -> str:
        if value not in _RISK_VALUES:
            raise ValueError(
                f"risk must be one of {sorted(_RISK_VALUES)}, got {value!r}"
            )
        return value


class TaskState(BaseModel):
    """Trusted state for a single task.

    Only references (IDs) are held here; the actual claim and evidence
    records live in the store. ``uncertainty_score`` defaults to 1.0
    (maximally uncertain) so that nothing is assumed certain up front.
    """

    task_id: str
    actor_context_ref: str
    project_context_ref: str
    status: TaskStatus = TaskStatus.ACCEPTED
    claim_refs: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    audit_event_refs: list[str] = Field(default_factory=list)
    route_decision_refs: list[str] = Field(default_factory=list)
    uncertainty_score: float = 1.0
    uncertainty_reasons: list[str] = Field(default_factory=list)

    @field_validator("uncertainty_score")
    @classmethod
    def _score_in_unit_interval(cls, value: float) -> float:
        if not (0.0 <= value <= 1.0):
            raise ValueError("uncertainty_score must be between 0.0 and 1.0")
        return value


class AMALIResponse(BaseModel):
    """The public response shape.

    Claims are split by epistemic level and unknowns/risks/missing
    evidence are surfaced explicitly rather than hidden inside prose.
    """

    status: str
    answer: str
    known_claims: list[str] = Field(default_factory=list)
    inferred_claims: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    trace_id: str | None = None
