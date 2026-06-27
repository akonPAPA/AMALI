"""Claim and evidence validation boundaries."""

import pytest
from pydantic import ValidationError

from amali.core.ids import new_claim_id, new_evidence_id, new_task_id
from amali.state.models import (
    ClaimRecord,
    EvidenceLevel,
    EvidenceRecord,
    Sensitivity,
    TrustLevel,
)


def test_known_claim_without_support_refs_fails():
    with pytest.raises(ValidationError):
        ClaimRecord(
            claim_id=new_claim_id(),
            task_id=new_task_id(),
            text="The sky is blue.",
            evidence_level=EvidenceLevel.KNOWN,
            source="model",
        )


def test_known_claim_with_support_ref_allowed():
    claim = ClaimRecord(
        claim_id=new_claim_id(),
        task_id=new_task_id(),
        text="The sky is blue.",
        evidence_level=EvidenceLevel.KNOWN,
        source="sensor",
        support_refs=[new_evidence_id()],
    )
    assert claim.evidence_level is EvidenceLevel.KNOWN


@pytest.mark.parametrize(
    "level",
    [EvidenceLevel.INFERRED, EvidenceLevel.UNKNOWN, EvidenceLevel.HYPOTHESIS],
)
def test_non_known_claim_without_support_refs_allowed(level):
    claim = ClaimRecord(
        claim_id=new_claim_id(),
        task_id=new_task_id(),
        text="Possibly relevant.",
        evidence_level=level,
        source="model",
    )
    assert claim.support_refs == []


def test_empty_claim_text_fails():
    with pytest.raises(ValidationError):
        ClaimRecord(
            claim_id=new_claim_id(),
            task_id=new_task_id(),
            text="   ",
            evidence_level=EvidenceLevel.UNKNOWN,
            source="model",
        )


def test_evidence_requires_source_type():
    with pytest.raises(ValidationError):
        EvidenceRecord(
            evidence_id=new_evidence_id(),
            task_id=new_task_id(),
            source_type="",
            trust_level=TrustLevel.LOW,
            sensitivity=Sensitivity.PUBLIC,
        )


def test_evidence_rejects_unknown_risk_value():
    with pytest.raises(ValidationError):
        EvidenceRecord(
            evidence_id=new_evidence_id(),
            task_id=new_task_id(),
            source_type="file",
            trust_level=TrustLevel.LOW,
            sensitivity=Sensitivity.PUBLIC,
            poison_risk="critical",
        )


def test_evidence_accepts_allowed_risk_values():
    evidence = EvidenceRecord(
        evidence_id=new_evidence_id(),
        task_id=new_task_id(),
        source_type="file",
        trust_level=TrustLevel.OFFICIAL,
        sensitivity=Sensitivity.INTERNAL,
        poison_risk="medium",
        injection_risk="high",
    )
    assert evidence.poison_risk == "medium"
    assert evidence.injection_risk == "high"
