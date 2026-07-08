"""Verifier statuses and EWA hard-gate integrity."""

from amali.arbiter import ResponseStatus, arbitrate, uge_escalate
from amali.policy.enums import RiskLevel
from amali.retrieval import EvidenceDoc
from amali.verifier import (
    ClaimVerdictStatus,
    extract_claims,
    verify_claims,
)


def _doc(doc_id: str, content: str) -> EvidenceDoc:
    return EvidenceDoc(
        doc_id=doc_id, source_uri=f"local://{doc_id}", content=content
    )


EVIDENCE = [
    _doc("genesis", "The audit ledger genesis hash is the string GENESIS."),
    _doc("router", "The router selects routes deterministically by score."),
]


# --- claim extraction -------------------------------------------------------


def test_extracts_factual_sentences():
    claims = extract_claims(
        "The genesis hash is GENESIS. The router is deterministic."
    )
    assert len(claims) == 2
    assert claims[0].claim_index == 0


def test_rejects_empty_and_noise():
    assert extract_claims("") == []
    assert extract_claims("   ") == []
    assert extract_claims("Ok. Yes. Hm!") == []  # too few content terms


def test_questions_are_not_claims():
    assert extract_claims("What is the genesis hash?") == []


# --- verification ----------------------------------------------------------


def test_supported_claim():
    claims = extract_claims("The audit ledger genesis hash is GENESIS.")
    verdicts = verify_claims(claims, EVIDENCE)
    assert verdicts[0].status is ClaimVerdictStatus.SUPPORTED
    assert verdicts[0].cited_doc_id == "genesis"
    assert verdicts[0].cited_doc_hash is not None


def test_unsupported_claim():
    claims = extract_claims("The moon is made of green cheese today.")
    verdicts = verify_claims(claims, EVIDENCE)
    assert verdicts[0].status is ClaimVerdictStatus.UNSUPPORTED


def test_contradicted_claim():
    evidence = EVIDENCE + [
        _doc("negation", "The genesis hash is not GENESIS; that is false.")
    ]
    claims = extract_claims("The genesis hash is GENESIS.")
    verdicts = verify_claims(claims, evidence)
    assert verdicts[0].status is ClaimVerdictStatus.CONTRADICTED
    assert verdicts[0].cited_doc_id == "negation"


def test_insufficient_evidence_when_no_docs():
    claims = extract_claims("The genesis hash is GENESIS.")
    verdicts = verify_claims(claims, [])
    assert verdicts[0].status is ClaimVerdictStatus.INSUFFICIENT_EVIDENCE


# --- EWA hard gates ---------------------------------------------------------


def _verdicts(text: str, evidence=EVIDENCE):
    return verify_claims(extract_claims(text), evidence)


def test_all_supported_is_success():
    result = arbitrate(_verdicts("The audit ledger genesis hash is GENESIS."))
    assert result.status is ResponseStatus.SUCCESS
    assert result.hard_gates_fired == []


def test_unsupported_claim_cannot_be_success():
    result = arbitrate(
        _verdicts(
            "The audit ledger genesis hash is GENESIS. "
            "The moon is made of green cheese today."
        )
    )
    assert result.status is ResponseStatus.PARTIAL
    assert "HG5_UNSUPPORTED_NEVER_SUCCESS" in result.hard_gates_fired
    assert result.status is not ResponseStatus.SUCCESS


def test_contradiction_forces_review():
    evidence = EVIDENCE + [
        _doc("negation", "The genesis hash is not GENESIS; that is false.")
    ]
    result = arbitrate(_verdicts("The genesis hash is GENESIS.", evidence))
    assert result.status is ResponseStatus.NEEDS_REVIEW
    assert "HG1_CONTRADICTED_CLAIM" in result.hard_gates_fired


def test_secret_exposure_blocks_even_with_perfect_score():
    result = arbitrate(
        _verdicts("The audit ledger genesis hash is GENESIS."),
        secret_exposure_flag=True,
    )
    assert result.status is ResponseStatus.BLOCKED
    assert "HG2_SECRET_EXPOSURE" in result.hard_gates_fired


def test_no_claims_is_unknown():
    result = arbitrate([])
    assert result.status is ResponseStatus.UNKNOWN


def test_no_evidence_is_unknown():
    claims = extract_claims("The genesis hash is GENESIS.")
    result = arbitrate(verify_claims(claims, []))
    assert result.status is ResponseStatus.UNKNOWN
    assert "HG4_NO_EVIDENCE" in result.hard_gates_fired


# --- UGE escalation ---------------------------------------------------------


def test_uge_l4_always_escalates_success():
    result = arbitrate(_verdicts("The audit ledger genesis hash is GENESIS."))
    assert result.status is ResponseStatus.SUCCESS
    escalated = uge_escalate(result, task_risk=RiskLevel.L4)
    assert escalated.status is ResponseStatus.NEEDS_REVIEW
    assert "UGE_L4_ALWAYS_REVIEW" in escalated.hard_gates_fired


def test_uge_never_deescalates_blocked():
    result = arbitrate(
        _verdicts("The audit ledger genesis hash is GENESIS."),
        secret_exposure_flag=True,
    )
    escalated = uge_escalate(result, task_risk=RiskLevel.L0)
    assert escalated.status is ResponseStatus.BLOCKED


def test_uge_low_risk_keeps_status():
    result = arbitrate(_verdicts("The audit ledger genesis hash is GENESIS."))
    escalated = uge_escalate(result, task_risk=RiskLevel.L1)
    assert escalated.status is ResponseStatus.SUCCESS
