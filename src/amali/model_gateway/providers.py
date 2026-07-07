"""Provider abstraction — the seam that avoids provider lock-in (§6.20).

The gateway talks to models only through the :class:`ModelBackend` protocol,
so no provider-specific SDK is baked into the core. Swapping Claude for a
local Llama, or a hosted endpoint for an offline bundle, is a matter of
registering a different backend for a :class:`ModelProvider`.

This local kernel ships one concrete backend: :class:`LocalDeterministicBackend`.
It performs **no network I/O** — like the executor boundary, it is a proven
seam, not a live model. It returns a deterministic completion so the gateway's
gates and records can be exercised end-to-end offline. A real hosted backend
(e.g. an Anthropic Claude adapter) implements the same protocol and is
registered by the owner-governed deployment layer, never hard-wired here.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

__all__ = ["BackendResult", "ModelBackend", "LocalDeterministicBackend"]


class BackendResult(BaseModel):
    """What a backend returns for one completion call."""

    model_config = {"extra": "forbid"}

    output_text: str
    prompt_tokens: int
    completion_tokens: int
    structured_output: dict[str, Any] | None = None
    safety_flags: list[str] = Field(default_factory=list)


@runtime_checkable
class ModelBackend(Protocol):
    """The single method the gateway needs from any provider."""

    def complete(
        self,
        *,
        system_prompt: str | None,
        user_context: str,
        max_output_tokens: int,
    ) -> BackendResult:
        """Produce a completion. Implementations must not mutate inputs."""
        ...


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token). Deterministic, no tokenizer."""
    if not text:
        return 0
    return (len(text) + 3) // 4


class LocalDeterministicBackend:
    """A no-network backend that returns a deterministic completion.

    This is the local-kernel stand-in for a real model. It never reaches the
    network. The completion is a bounded, deterministic transform of the
    input so tests can assert on usage and output without flakiness.
    """

    def complete(
        self,
        *,
        system_prompt: str | None,
        user_context: str,
        max_output_tokens: int,
    ) -> BackendResult:
        prompt_tokens = _estimate_tokens(system_prompt or "") + _estimate_tokens(
            user_context
        )
        # Deterministic "response": a stable acknowledgement plus a bounded
        # echo of the (already policy-cleared) context. No network, no model.
        echo = user_context.strip()
        if len(echo) > 200:
            echo = echo[:200]
        output_text = f"[local-deterministic] {echo}" if echo else "[local-deterministic]"
        completion_tokens = min(
            _estimate_tokens(output_text), max_output_tokens
        )
        return BackendResult(
            output_text=output_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            structured_output=None,
            safety_flags=[],
        )
