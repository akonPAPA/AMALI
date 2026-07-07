"""Model Gateway (§6.20).

Normalizes local/hosted model calls behind one owner-governed boundary,
enforcing versioning, privacy, data-class, budget and provider gates before
any backend is touched, and auditing every outcome. The gateway owns no
truth: it returns an ``unverified`` :class:`ModelOutputRecord` for the
verifier/critic/arbiter layer to act on.
"""

from __future__ import annotations

from amali.model_gateway.enums import (
    Capability,
    ModelProvider,
    ModelType,
    PrivacyMode,
    PromotionStage,
)
from amali.model_gateway.errors import (
    GatewayReason,
    ModelGatewayError,
    ModelGatewayRefused,
)
from amali.model_gateway.gateway import ModelGateway
from amali.model_gateway.manifest import (
    ModelManifest,
    ModelManifestValidationError,
)
from amali.model_gateway.providers import (
    BackendResult,
    LocalDeterministicBackend,
    ModelBackend,
)
from amali.model_gateway.records import ModelOutputRecord, TokenUsage
from amali.model_gateway.requests import Budget, ModelInvocationRequest
from amali.model_gateway.transformers_backend import TransformersBackend

__all__ = [
    "Capability",
    "ModelProvider",
    "ModelType",
    "PrivacyMode",
    "PromotionStage",
    "GatewayReason",
    "ModelGatewayError",
    "ModelGatewayRefused",
    "ModelGateway",
    "ModelManifest",
    "ModelManifestValidationError",
    "BackendResult",
    "LocalDeterministicBackend",
    "ModelBackend",
    "ModelOutputRecord",
    "TokenUsage",
    "Budget",
    "ModelInvocationRequest",
    "TransformersBackend",
]
