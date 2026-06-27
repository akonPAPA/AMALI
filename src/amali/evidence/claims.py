"""Claim construction helpers.

The key architectural rule lives here: text produced by a model is *not*
trusted state. A generated claim therefore starts as ``UNKNOWN`` and
``unsupported``. It can only be promoted toward ``KNOWN`` after explicit
supporting evidence is attached (which ``ClaimRecord`` validation enforces).
"""

from __future__ import annotations

from amali.core.ids import new_claim_id
from amali.state.models import ClaimRecord, EvidenceLevel

__all__ = ["create_generated_claim"]


def create_generated_claim(task_id: str, text: str) -> ClaimRecord:
    """Build a claim representing model-generated text.

    Such a claim is, by construction, ``UNKNOWN`` and ``unsupported``: no
    generated text is trusted until evidence is attached.
    """
    return ClaimRecord(
        claim_id=new_claim_id(),
        task_id=task_id,
        text=text,
        evidence_level=EvidenceLevel.UNKNOWN,
        source="generated",
        status="unsupported",
    )
