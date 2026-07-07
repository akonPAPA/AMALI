"""Inputs to the Model Gateway: the budget and the invocation request.

A :class:`ModelInvocationRequest` is untrusted at the trust boundary. Two
things are enforced at construction so an invalid or unsafe request can never
reach the gateway's decision logic:

* required identifiers (task, model manifest, prompt manifest + version) must
  be present — a hidden or unversioned prompt is rejected (§6.20 forbids
  "hidden prompt changes").

Note the request separates ``system_prompt`` (trusted framing, never echoed
back in the output record or audit) from ``user_context`` (the untrusted
material whose sensitivity is declared explicitly).

Secret-shaped material is *not* judged here: whether raw secrets are redacted
or blocked depends on the target model's privacy mode, which only the gateway
knows. The gateway therefore owns that decision (§6.20: "sensitive context
redacted or blocked").
"""

from __future__ import annotations

from pydantic import BaseModel, model_validator

from amali.core.errors import ValidationBoundaryError
from amali.policy.enums import DataSensitivity, RequestSource

__all__ = ["Budget", "ModelInvocationRequest"]


class Budget(BaseModel):
    """A token budget the gateway must not exceed.

    The gateway estimates ``prompt_tokens + planned_completion_tokens`` and
    refuses *before* invoking the backend if the estimate exceeds
    ``max_total_tokens``. This is a hard stop, not an after-the-fact metric.
    """

    model_config = {"extra": "forbid"}

    max_total_tokens: int

    @model_validator(mode="after")
    def _validate(self) -> "Budget":
        if self.max_total_tokens <= 0:
            raise ValueError("max_total_tokens must be > 0")
        return self


class ModelInvocationRequest(BaseModel):
    """A validated request to invoke a model through the gateway."""

    model_config = {"extra": "forbid"}

    task_id: str
    actor_ref: str
    model_manifest_id: str
    prompt_manifest_id: str
    prompt_version: str
    system_prompt: str | None = None
    user_context: str
    data_sensitivity: DataSensitivity
    budget: Budget
    request_source: RequestSource = RequestSource.AGENT
    max_output_tokens: int | None = None

    @model_validator(mode="after")
    def _validate(self) -> "ModelInvocationRequest":
        for name in (
            "task_id",
            "actor_ref",
            "model_manifest_id",
            "prompt_manifest_id",
            "prompt_version",
        ):
            value = getattr(self, name)
            if not value or not value.strip():
                raise ValidationBoundaryError(f"{name} must not be empty")

        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise ValidationBoundaryError(
                "max_output_tokens must be > 0 when provided"
            )
        return self
