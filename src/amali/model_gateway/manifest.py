"""Versioned model manifest (§4.2) and its hard invariants.

A :class:`ModelManifest` declares what a single model *is* and what data it
may touch. It is strictly validated: a manifest that violates a hard
invariant is rejected at construction (pydantic ``ValidationError``) and,
defensively, re-checked by the gateway via :meth:`ModelManifest.assert_valid`
before any invocation.

The invariants encode the security posture of §0–§14:

* ``secret`` and ``personal`` data classes can never be *allowed* — raw
  secret/PII must never flow to a model.
* privacy mode and provider must agree (a ``local_only`` model cannot be
  ``hosted``; a ``hosted`` model cannot claim ``local_only`` privacy).
* an ``offline_bundle`` must be integrity-pinned (checksum + source).
* promotion to ``canary``/``production`` requires an owner approval ref, and
  ``production`` additionally requires eval evidence — no silent promotion.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from amali.model_gateway.enums import (
    Capability,
    ModelProvider,
    ModelType,
    PrivacyMode,
    PromotionStage,
)
from amali.model_gateway.errors import ModelGatewayError
from amali.policy.enums import DataSensitivity

__all__ = ["ModelManifest", "ModelManifestValidationError"]

# Data classes that must never be sent to any model, ever.
_NEVER_ALLOWED_CLASSES = frozenset(
    {DataSensitivity.SECRET, DataSensitivity.PERSONAL}
)


class ModelManifestValidationError(ModelGatewayError):
    """Raised when a model manifest violates a hard invariant."""


class ModelManifest(BaseModel):
    """Versioned declaration of a single model (§4.2)."""

    model_config = {"extra": "forbid"}

    model_id: str
    version: str
    provider: ModelProvider
    model_type: ModelType
    allowed_data_classes: list[DataSensitivity] = Field(default_factory=list)
    forbidden_data_classes: list[DataSensitivity] = Field(
        default_factory=lambda: [DataSensitivity.SECRET, DataSensitivity.PERSONAL]
    )
    context_window: int
    max_tokens: int
    capabilities: list[Capability] = Field(default_factory=list)
    privacy_mode: PrivacyMode
    license: str
    source_uri: str | None = None
    checksum: str | None = None
    promotion_stage: PromotionStage
    eval_report_refs: list[str] = Field(default_factory=list)
    owner_approval_ref: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> "ModelManifest":
        try:
            self.assert_valid()
        except ModelManifestValidationError as exc:
            raise ValueError(str(exc)) from exc
        return self

    def assert_valid(self) -> None:
        """Re-assert every hard manifest invariant.

        Raises :class:`ModelManifestValidationError` on the first violation.
        The gateway calls this defensively so a manifest built via a bypass
        (``model_construct``) or mutated after construction is still caught
        before it can influence an invocation.
        """
        if not self.model_id or not self.model_id.strip():
            raise ModelManifestValidationError("model_id must not be empty")
        if not self.version or not self.version.strip():
            raise ModelManifestValidationError("version must not be empty")
        if not self.license or not self.license.strip():
            raise ModelManifestValidationError("license must not be empty")
        if not self.capabilities:
            raise ModelManifestValidationError(
                "capabilities must be non-empty"
            )

        # Window / token sizing must be sane.
        if self.context_window <= 0:
            raise ModelManifestValidationError("context_window must be > 0")
        if self.max_tokens <= 0:
            raise ModelManifestValidationError("max_tokens must be > 0")
        if self.max_tokens > self.context_window:
            raise ModelManifestValidationError(
                "max_tokens must not exceed context_window"
            )

        # secret/personal can never be allowed, and cannot be both allowed
        # and forbidden.
        allowed = set(self.allowed_data_classes)
        forbidden = set(self.forbidden_data_classes)
        leaked = allowed & _NEVER_ALLOWED_CLASSES
        if leaked:
            raise ModelManifestValidationError(
                "allowed_data_classes must never include "
                f"{sorted(c.value for c in leaked)}"
            )
        overlap = allowed & forbidden
        if overlap:
            raise ModelManifestValidationError(
                "data classes listed as both allowed and forbidden: "
                f"{sorted(c.value for c in overlap)}"
            )
        # secret and personal must be explicitly forbidden.
        missing_forbidden = _NEVER_ALLOWED_CLASSES - forbidden
        if missing_forbidden:
            raise ModelManifestValidationError(
                "forbidden_data_classes must include "
                f"{sorted(c.value for c in _NEVER_ALLOWED_CLASSES)}"
            )

        # Provider and privacy mode must agree.
        if self.privacy_mode is PrivacyMode.LOCAL_ONLY and (
            self.provider is ModelProvider.HOSTED
        ):
            raise ModelManifestValidationError(
                "privacy_mode=local_only forbids provider=hosted"
            )
        if self.provider is ModelProvider.HOSTED and (
            self.privacy_mode is PrivacyMode.LOCAL_ONLY
        ):  # pragma: no cover - symmetric with the check above
            raise ModelManifestValidationError(
                "provider=hosted cannot claim privacy_mode=local_only"
            )

        # An offline bundle must be integrity-pinned.
        if self.provider is ModelProvider.OFFLINE_BUNDLE:
            if not (self.checksum and self.checksum.strip()):
                raise ModelManifestValidationError(
                    "offline_bundle provider requires a checksum"
                )
            if not (self.source_uri and self.source_uri.strip()):
                raise ModelManifestValidationError(
                    "offline_bundle provider requires a source_uri"
                )

        # Owner-governed promotion: no silent path to canary/production.
        if self.promotion_stage in (
            PromotionStage.CANARY,
            PromotionStage.PRODUCTION,
        ):
            if not (self.owner_approval_ref and self.owner_approval_ref.strip()):
                raise ModelManifestValidationError(
                    f"promotion_stage={self.promotion_stage.value} requires "
                    "owner_approval_ref"
                )
        if self.promotion_stage is PromotionStage.PRODUCTION and (
            not self.eval_report_refs
        ):
            raise ModelManifestValidationError(
                "promotion_stage=production requires eval_report_refs"
            )

    def data_class_allowed(self, sensitivity: DataSensitivity) -> bool:
        """Return True iff this model may touch data of this sensitivity."""
        if sensitivity in _NEVER_ALLOWED_CLASSES:
            return False
        if sensitivity in set(self.forbidden_data_classes):
            return False
        return sensitivity in set(self.allowed_data_classes)
