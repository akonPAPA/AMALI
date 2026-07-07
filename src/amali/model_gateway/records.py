"""Outputs of the Model Gateway: token usage and the output record.

A :class:`ModelOutputRecord` is what the gateway returns. It deliberately
carries only a *hash* of the prompt — never the raw system prompt or user
context — so the record and the audit trail cannot leak prompt material
(§6.20 forbids "raw system prompt leakage").

The record's ``status`` is always ``unverified``: the gateway does not own
final truth. A model's output only becomes usable after the verifier, critic
and Evidence-Weighted Arbiter (§6.25–§6.27) act on it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from amali.model_gateway.enums import ModelProvider, PrivacyMode

__all__ = ["TokenUsage", "ModelOutputRecord"]


class TokenUsage(BaseModel):
    """Token accounting for a single invocation."""

    model_config = {"extra": "forbid"}

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @model_validator(mode="after")
    def _validate(self) -> "TokenUsage":
        if self.prompt_tokens < 0 or self.completion_tokens < 0:
            raise ValueError("token counts must be non-negative")
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError(
                "total_tokens must equal prompt_tokens + completion_tokens"
            )
        return self


class ModelOutputRecord(BaseModel):
    """The result of a gateway invocation. Owns no truth (status=unverified)."""

    model_config = {"extra": "forbid"}

    output_id: str
    trace_id: str
    task_id: str
    model_id: str
    model_version: str
    provider: ModelProvider
    prompt_manifest_id: str
    prompt_version: str
    prompt_sha256: str
    privacy_mode: PrivacyMode
    context_redacted: bool
    usage: TokenUsage
    output_text: str
    structured_output: dict[str, Any] | None = None
    safety_flags: list[str] = Field(default_factory=list)
    status: Literal["unverified"] = "unverified"
    audit_event_id: str | None = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
