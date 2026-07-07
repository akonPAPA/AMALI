"""Deterministic route scoring (§8.3).

Implements the blueprint's scoring function with non-learned, configurable
weights:

    Score(r | x) =  α·QualityFit + β·EvidenceFit + γ·SafetyFit
                  + η·ValidationStrength + μ·PrivacyFit
                  + ρ·HistoricalSuccess + φ·Availability
                  - λ·Cost - κ·Latency - ω·ResidualRisk

Per §8.3, the initial weights are **config, not truth** — they may be tuned
only after eval data exists (Phase 5+). Every component is deterministic and
confined to [0, 1], so the same inputs always produce the same ranking. No
model is consulted; scoring never sees task text.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel

from amali.model_gateway.enums import ModelProvider
from amali.model_gateway.manifest import ModelManifest
from amali.router.enums import ModelHealth
from amali.router.profile import TaskProfile
from amali.router.route import Route
from amali.policy.enums import RiskLevel, risk_rank

__all__ = ["RouterWeights", "ScoreBreakdown", "score_route"]


class RouterWeights(BaseModel):
    """Non-learned scoring weights (§8.3). Config, not truth."""

    model_config = {"extra": "forbid"}

    alpha_quality: float = 1.0
    beta_evidence: float = 1.0
    gamma_safety: float = 1.0
    eta_validation: float = 1.0
    mu_privacy: float = 1.0
    rho_history: float = 0.5
    phi_availability: float = 1.0
    lambda_cost: float = 0.5
    kappa_latency: float = 0.25
    omega_residual_risk: float = 1.0
    # Normalizer: cost-units value treated as "expensive" (score 1.0).
    cost_ceiling_units: float = 100.0
    version: str = "weights_v0"


class ScoreBreakdown(BaseModel):
    """Every component of one route's score, kept for audit and tuning."""

    model_config = {"extra": "forbid"}

    route_id: str
    quality_fit: float
    evidence_fit: float
    safety_fit: float
    validation_strength: float
    privacy_fit: float
    historical_success: float
    availability: float
    cost: float
    latency: float
    residual_risk: float
    total: float


def _risk_fraction(level: RiskLevel) -> float:
    """Map L0..L5 onto [0, 1]."""
    return risk_rank(level) / risk_rank(RiskLevel.L5)


def score_route(
    route: Route,
    profile: TaskProfile,
    *,
    models: Mapping[str, ModelManifest],
    health: Mapping[str, ModelHealth],
    weights: RouterWeights,
) -> ScoreBreakdown:
    """Score one *eligible* route. Deterministic; no I/O, no randomness."""
    manifests = [models[m] for m in route.models if m in models]

    quality_fit = route.declared_quality

    # EvidenceFit: a retrieval policy matters more when the task needs
    # evidence; a route with no retrieval on an evidence-requiring task is
    # a poor fit but not a violation (the verifier gate still protects).
    has_retrieval = route.retrieval_policy_id is not None
    if profile.evidence_required:
        evidence_fit = 1.0 if has_retrieval else 0.0
    else:
        evidence_fit = 1.0 if has_retrieval else 0.5

    # SafetyFit: inverse of the route's tool-risk ceiling.
    safety_fit = 1.0 - _risk_fraction(route.tool_risk)

    # ValidationStrength: fraction of machine gates the route enforces.
    gates = route.required_gates
    validation_strength = (
        (1.0 if gates.verifier else 0.0)
        + (1.0 if gates.critic else 0.0)
        + (1.0 if gates.security_review else 0.0)
    ) / 3.0

    # PrivacyFit: all-local beats hosted; hosted earns partial credit only
    # for non-sensitive data. (Hard privacy violations were already excluded
    # by the constraints; this only orders the survivors.)
    if manifests and all(
        m.provider is not ModelProvider.HOSTED for m in manifests
    ):
        privacy_fit = 1.0
    else:
        privacy_fit = 0.5

    historical_success = route.historical_success

    # Availability: healthy models are fully available, degraded ones half.
    availability = 1.0
    for model_id in route.models:
        if health.get(model_id, ModelHealth.HEALTHY) is (
            ModelHealth.DEGRADED_ALLOWED
        ):
            availability = min(availability, 0.5)

    cost = min(1.0, route.budget.max_cost_units / weights.cost_ceiling_units)
    latency = 1.0 - route.declared_latency_score
    residual_risk = _risk_fraction(route.tool_risk)

    total = (
        weights.alpha_quality * quality_fit
        + weights.beta_evidence * evidence_fit
        + weights.gamma_safety * safety_fit
        + weights.eta_validation * validation_strength
        + weights.mu_privacy * privacy_fit
        + weights.rho_history * historical_success
        + weights.phi_availability * availability
        - weights.lambda_cost * cost
        - weights.kappa_latency * latency
        - weights.omega_residual_risk * residual_risk
    )

    return ScoreBreakdown(
        route_id=route.route_id,
        quality_fit=quality_fit,
        evidence_fit=evidence_fit,
        safety_fit=safety_fit,
        validation_strength=validation_strength,
        privacy_fit=privacy_fit,
        historical_success=historical_success,
        availability=availability,
        cost=cost,
        latency=latency,
        residual_risk=residual_risk,
        total=total,
    )
