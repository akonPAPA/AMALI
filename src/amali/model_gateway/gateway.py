"""The Model Gateway (§6.20).

Normalizes model calls behind one boundary and enforces, in order, the gates
that the blueprint requires *before* any backend is ever touched:

    manifest exists -> manifest valid -> not retired -> prompt versioned
    -> offline blocks hosted -> data class allowed -> no raw secret
    -> privacy-mode redaction -> context fits -> budget not exceeded
    -> backend available -> invoke -> record (unverified) -> audit

Every refusal and every success is written to the hash-chained audit ledger
with redacted, prompt-free metadata. The gateway owns no truth: it returns a
:class:`ModelOutputRecord` with ``status="unverified"``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import datetime, timezone

from amali.audit.ledger import AuditLedger
from amali.core.ids import new_model_call_id, new_trace_id
from amali.model_gateway.audit import emit_model_invocation_audit
from amali.model_gateway.enums import ModelProvider, PrivacyMode, PromotionStage
from amali.model_gateway.errors import GatewayReason, ModelGatewayRefused
from amali.model_gateway.manifest import (
    ModelManifest,
    ModelManifestValidationError,
)
from amali.model_gateway.providers import ModelBackend
from amali.model_gateway.records import ModelOutputRecord, TokenUsage
from amali.model_gateway.requests import ModelInvocationRequest
from amali.security.redaction import contains_secret, redact

__all__ = ["ModelGateway"]


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token). Matches the local backend."""
    if not text:
        return 0
    return (len(text) + 3) // 4


def _prompt_hash(request: ModelInvocationRequest, send_context: str) -> str:
    """Bind the exact prompt the backend receives to a recorded hash.

    Any later "hidden prompt change" would change this hash, so the record
    and audit provide tamper-evidence for the prompt actually used.
    """
    parts = [
        request.prompt_manifest_id,
        request.prompt_version,
        request.system_prompt or "",
        send_context,
    ]
    joined = "\x00".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


class ModelGateway:
    """Owner-governed boundary for invoking models under policy."""

    def __init__(
        self,
        *,
        models: Mapping[str, ModelManifest],
        backends: Mapping[ModelProvider, ModelBackend],
        ledger: AuditLedger,
        offline_mode: bool = False,
    ) -> None:
        self._models = dict(models)
        self._backends = dict(backends)
        self._ledger = ledger
        self._offline_mode = offline_mode

    def invoke(
        self,
        request: ModelInvocationRequest,
        now: datetime | None = None,
    ) -> ModelOutputRecord:
        """Run every gate, then invoke the backend. Refuse (audited) on any
        gate failure. Return an ``unverified`` output record on success."""
        _ = now or datetime.now(timezone.utc)
        trace_id = new_trace_id()

        # 1. Manifest must exist.
        manifest = self._models.get(request.model_manifest_id)
        if manifest is None:
            self._refuse(
                request,
                trace_id,
                GatewayReason.MANIFEST_NOT_FOUND,
                f"no model manifest {request.model_manifest_id!r}",
                manifest=None,
            )

        # 2. Manifest must (still) be valid.
        try:
            manifest.assert_valid()
        except ModelManifestValidationError as exc:
            self._refuse(
                request,
                trace_id,
                GatewayReason.MANIFEST_INVALID,
                str(exc),
                manifest=manifest,
            )

        # 3. A retired model can never be invoked.
        if manifest.promotion_stage is PromotionStage.RETIRED:
            self._refuse(
                request,
                trace_id,
                GatewayReason.MODEL_RETIRED,
                f"model {manifest.model_id!r} is retired",
                manifest=manifest,
            )

        # 4. Prompt must be versioned (also enforced at request construction).
        if not request.prompt_version.strip():  # pragma: no cover - defensive
            self._refuse(
                request,
                trace_id,
                GatewayReason.PROMPT_VERSION_MISSING,
                "prompt_version missing",
                manifest=manifest,
            )

        # 5. Offline mode: hosted providers are blocked before any call.
        if self._offline_mode and manifest.provider is ModelProvider.HOSTED:
            self._refuse(
                request,
                trace_id,
                GatewayReason.OFFLINE_HOSTED_BLOCKED,
                "offline mode forbids hosted provider",
                manifest=manifest,
            )

        # 6. Data-class gate: the model must be permitted this sensitivity.
        if not manifest.data_class_allowed(request.data_sensitivity):
            self._refuse(
                request,
                trace_id,
                GatewayReason.DATA_CLASS_FORBIDDEN,
                f"data class {request.data_sensitivity.value!r} not allowed "
                f"for model {manifest.model_id!r}",
                manifest=manifest,
            )

        # 7. A trusted system prompt must never carry raw secret material —
        #    that would be an authoring bug, and it is never redacted away.
        if contains_secret(request.system_prompt or ""):
            self._refuse(
                request,
                trace_id,
                GatewayReason.RAW_SECRET_BLOCKED,
                "system prompt contains secret-shaped material",
                manifest=manifest,
            )

        # 8. Secret / privacy-mode handling for the user context — the
        #    "redacted or blocked" rule. Only redacted_only models may carry
        #    secret-shaped context, and only after it is redacted; every other
        #    privacy mode blocks it outright.
        context_redacted = False
        send_context = request.user_context
        if manifest.privacy_mode is PrivacyMode.REDACTED_ONLY:
            redacted = redact(request.user_context)
            send_context = (
                redacted if isinstance(redacted, str) else str(redacted)
            )
            context_redacted = send_context != request.user_context
        elif contains_secret(request.user_context):
            self._refuse(
                request,
                trace_id,
                GatewayReason.RAW_SECRET_BLOCKED,
                "user context contains secret-shaped material",
                manifest=manifest,
            )

        # 9. Context must fit the window.
        prompt_tokens = _estimate_tokens(
            request.system_prompt or ""
        ) + _estimate_tokens(send_context)
        if prompt_tokens > manifest.context_window:
            self._refuse(
                request,
                trace_id,
                GatewayReason.CONTEXT_OVERFLOW,
                "estimated prompt tokens exceed context_window",
                manifest=manifest,
            )

        # 10. Budget gate: refuse before the call if the estimate exceeds it.
        planned_completion = manifest.max_tokens
        if request.max_output_tokens is not None:
            planned_completion = min(planned_completion, request.max_output_tokens)
        estimated_total = prompt_tokens + planned_completion
        if estimated_total > request.budget.max_total_tokens:
            self._refuse(
                request,
                trace_id,
                GatewayReason.BUDGET_EXCEEDED,
                f"estimated {estimated_total} tokens exceed budget "
                f"{request.budget.max_total_tokens}",
                manifest=manifest,
                total_tokens=estimated_total,
            )

        # 11. Backend for this provider must be registered.
        backend = self._backends.get(manifest.provider)
        if backend is None:
            self._refuse(
                request,
                trace_id,
                GatewayReason.PROVIDER_UNAVAILABLE,
                f"no backend registered for provider "
                f"{manifest.provider.value!r}",
                manifest=manifest,
            )

        # 12. Bind the exact prompt to a hash, then invoke.
        prompt_sha256 = _prompt_hash(request, send_context)
        result = backend.complete(
            system_prompt=request.system_prompt,
            user_context=send_context,
            max_output_tokens=planned_completion,
        )

        usage = TokenUsage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        )

        record = ModelOutputRecord(
            output_id=new_model_call_id(),
            trace_id=trace_id,
            task_id=request.task_id,
            model_id=manifest.model_id,
            model_version=manifest.version,
            provider=manifest.provider,
            prompt_manifest_id=request.prompt_manifest_id,
            prompt_version=request.prompt_version,
            prompt_sha256=prompt_sha256,
            privacy_mode=manifest.privacy_mode,
            context_redacted=context_redacted,
            usage=usage,
            output_text=result.output_text,
            structured_output=result.structured_output,
            safety_flags=list(result.safety_flags),
        )

        audit_id = emit_model_invocation_audit(
            self._ledger,
            outcome="allow",
            reason=GatewayReason.OK,
            trace_id=trace_id,
            task_id=request.task_id,
            actor_ref=request.actor_ref,
            model_id=manifest.model_id,
            model_version=manifest.version,
            prompt_manifest_id=request.prompt_manifest_id,
            prompt_version=request.prompt_version,
            prompt_sha256=prompt_sha256,
            data_sensitivity=request.data_sensitivity,
            context_redacted=context_redacted,
            total_tokens=usage.total_tokens,
            budget_max_tokens=request.budget.max_total_tokens,
        )
        return record.model_copy(update={"audit_event_id": audit_id})

    def _refuse(
        self,
        request: ModelInvocationRequest,
        trace_id: str,
        reason: GatewayReason,
        message: str,
        *,
        manifest: ModelManifest | None,
        total_tokens: int | None = None,
    ) -> None:
        """Audit a deny event and raise. Never returns."""
        audit_id = emit_model_invocation_audit(
            self._ledger,
            outcome="deny",
            reason=reason,
            trace_id=trace_id,
            task_id=request.task_id,
            actor_ref=request.actor_ref,
            model_id=manifest.model_id if manifest else request.model_manifest_id,
            model_version=manifest.version if manifest else None,
            prompt_manifest_id=request.prompt_manifest_id,
            prompt_version=request.prompt_version,
            prompt_sha256=None,
            data_sensitivity=request.data_sensitivity,
            context_redacted=None,
            total_tokens=total_tokens,
            budget_max_tokens=request.budget.max_total_tokens,
        )
        raise ModelGatewayRefused(reason, message, audit_event_id=audit_id)
