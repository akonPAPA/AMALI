"""Generated text is not trusted state."""

import pytest
from pydantic import ValidationError

from amali.core.ids import new_task_id
from amali.evidence.claims import create_generated_claim
from amali.state.models import ClaimRecord, EvidenceLevel


def test_generated_claim_defaults_to_unknown_unsupported():
    claim = create_generated_claim(task_id=new_task_id(), text="2 + 2 = 5")
    assert claim.evidence_level is EvidenceLevel.UNKNOWN
    assert claim.status == "unsupported"
    assert claim.support_refs == []
    assert claim.source == "generated"


def test_cannot_build_generated_known_claim_without_evidence():
    with pytest.raises(ValidationError):
        ClaimRecord(
            claim_id="claim_x",
            task_id=new_task_id(),
            text="2 + 2 = 4",
            evidence_level=EvidenceLevel.KNOWN,
            source="generated",
        )
