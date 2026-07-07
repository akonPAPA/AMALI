"""Closed vocabularies for the Model Gateway (§6.20 / §4.2).

As in the permission kernel, these are closed enums rather than free
strings: a manifest or request cannot smuggle in an unknown provider,
privacy mode, or model type that the gateway's decision logic has not
accounted for. Values are strings so they stay readable in YAML manifests
and in the audit ledger.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "ModelProvider",
    "PrivacyMode",
    "ModelType",
    "PromotionStage",
    "Capability",
]


class ModelProvider(str, Enum):
    """Where a model physically runs.

    ``local``: runs on the owner's machine, no network.
    ``offline_bundle``: a checksummed, pinned local bundle, no network.
    ``hosted``: reached over the network via a provider API.
    ``experimental``: an R&D backend, never allowed in production routes.
    """

    LOCAL = "local"
    HOSTED = "hosted"
    OFFLINE_BUNDLE = "offline_bundle"
    EXPERIMENTAL = "experimental"


class PrivacyMode(str, Enum):
    """Privacy posture a model manifest declares.

    ``local_only``: context may never leave the machine; hosted providers
    are forbidden for this manifest.
    ``hosted_allowed``: context may be sent to the hosted provider as-is
    (subject to the data-class gate).
    ``redacted_only``: context may be sent to the provider only after
    secret-shaped material is redacted.
    """

    LOCAL_ONLY = "local_only"
    HOSTED_ALLOWED = "hosted_allowed"
    REDACTED_ONLY = "redacted_only"


class ModelType(str, Enum):
    """Role a model plays inside AMALI."""

    GENERATOR = "generator"
    VERIFIER = "verifier"
    CRITIC = "critic"
    ROUTER = "router"
    RISK_CLASSIFIER = "risk_classifier"
    EMBEDDING = "embedding"
    RERANKER = "reranker"
    EVALUATOR = "evaluator"
    DOMAIN_SPECIALIST = "domain_specialist"


class PromotionStage(str, Enum):
    """Owner-governed lifecycle stage of a model artifact (§5.9 / §14)."""

    DEV = "dev"
    EVAL = "eval"
    CANARY = "canary"
    PRODUCTION = "production"
    FROZEN = "frozen"
    RETIRED = "retired"


class Capability(str, Enum):
    """A capability a model advertises."""

    TEXT = "text"
    CODE = "code"
    VISION = "vision"
    EMBEDDINGS = "embeddings"
    RERANK = "rerank"
    TOOL_REASONING = "tool_reasoning"
