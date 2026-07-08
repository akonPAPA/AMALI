"""Claim extraction and evidence verification.

An answer is text; its factual claims are extracted, then each claim is
checked against the retrieved evidence corpus. No claim is trusted because
a model (or a deterministic answerer) produced it.
"""

from amali.verifier.claims import ExtractedClaim, extract_claims
from amali.verifier.verify import (
    ClaimVerdict,
    ClaimVerdictStatus,
    verify_claims,
)

__all__ = [
    "ExtractedClaim",
    "extract_claims",
    "ClaimVerdict",
    "ClaimVerdictStatus",
    "verify_claims",
]
