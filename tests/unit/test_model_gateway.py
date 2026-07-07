"""Model Gateway (§6.20) — positive, negative and required-behavior tests.

Covers the four blueprint-required tests explicitly:

* offline-only uses local model only  -> test_offline_mode_blocks_hosted_*
* prompt version recorded              -> test_prompt_version_is_recorded
* budget exceeded stops call           -> test_budget_exceeded_stops_call
* sensitive context redacted or blocked-> test_redacted_only_* / test_*_blocked

Plus manifest invariants, audit-chain integrity on allow and deny, prompt-hash
binding (no hidden prompt change), and the "owns no truth" guarantee.
"""

from __future__ import annotations

import hashlib

import pytest

from amali.audit.ledger import AuditLedger
from amali.model_gateway import (
    Budget,
    Capability,
    LocalDeterministicBackend,
    ModelGateway,
    ModelGatewayRefused,
    ModelInvocationRequest,
    ModelManifest,
    ModelProvider,
    PrivacyMode,
    PromotionStage,
)
from amali.model_gateway.errors import GatewayReason
from amali.model_gateway.manifest import ModelManifestValidationError
from amali.policy.enums import DataSensitivity


# --------------------------------------------------------------------------
# Fixtures / factories
# --------------------------------------------------------------------------


class TripwireBackend:
    """A backend that must never be called; records if it ever is."""

    def __init__(self) -> None:
        self.called = False

    def complete(self, *, system_prompt, user_context, max_output_tokens):
        self.called = True
        raise AssertionError("backend was invoked but must not have been")


class SpyLocalBackend(LocalDeterministicBackend):
    """Local backend that records exactly what it received."""

    def __init__(self) -> None:
        self.last_system_prompt: str | None = None
        self.last_user_context: str | None = None

    def complete(self, *, system_prompt, user_context, max_output_tokens):
        self.last_system_prompt = system_prompt
        self.last_user_context = user_context
        return super().complete(
            system_prompt=system_prompt,
            user_context=user_context,
            max_output_tokens=max_output_tokens,
        )


def local_manifest(**overrides) -> ModelManifest:
    defaults = dict(
        model_id="local_generator_v1",
        version="1.0.0",
        provider=ModelProvider.LOCAL,
        model_type="generator",
        allowed_data_classes=[DataSensitivity.PUBLIC, DataSensitivity.INTERNAL],
        context_window=8192,
        max_tokens=1024,
        capabilities=[Capability.TEXT, Capability.CODE],
        privacy_mode=PrivacyMode.LOCAL_ONLY,
        license="proprietary-local",
        promotion_stage=PromotionStage.EVAL,
    )
    defaults.update(overrides)
    return ModelManifest(**defaults)


def hosted_manifest(**overrides) -> ModelManifest:
    defaults = dict(
        model_id="hosted_opus_v1",
        version="4.8",
        provider=ModelProvider.HOSTED,
        model_type="generator",
        allowed_data_classes=[DataSensitivity.PUBLIC, DataSensitivity.INTERNAL],
        context_window=200000,
        max_tokens=8192,
        capabilities=[Capability.TEXT, Capability.CODE, Capability.TOOL_REASONING],
        privacy_mode=PrivacyMode.HOSTED_ALLOWED,
        license="vendor-hosted",
        promotion_stage=PromotionStage.PRODUCTION,
        owner_approval_ref="approval_owner_1",
        eval_report_refs=["eval_report_1"],
    )
    defaults.update(overrides)
    return ModelManifest(**defaults)


def make_request(**overrides) -> ModelInvocationRequest:
    defaults = dict(
        task_id="task_fixed_1",
        actor_ref="local_dev_agent",
        model_manifest_id="local_generator_v1",
        prompt_manifest_id="prompt_summarize",
        prompt_version="2.1.0",
        system_prompt="You are a careful assistant.",
        user_context="Summarize the AMALI blueprint control planes.",
        data_sensitivity=DataSensitivity.PUBLIC,
        budget=Budget(max_total_tokens=100000),
    )
    defaults.update(overrides)
    return ModelInvocationRequest(**defaults)


def build_gateway(*, models=None, backends=None, offline_mode=False):
    ledger = AuditLedger()
    if models is None:
        models = {m.model_id: m for m in (local_manifest(), hosted_manifest())}
    if backends is None:
        backends = {
            ModelProvider.LOCAL: LocalDeterministicBackend(),
            ModelProvider.HOSTED: LocalDeterministicBackend(),
        }
    gw = ModelGateway(
        models=models,
        backends=backends,
        ledger=ledger,
        offline_mode=offline_mode,
    )
    return gw, ledger


# --------------------------------------------------------------------------
# Positive path
# --------------------------------------------------------------------------


def test_invoke_local_returns_unverified_record_and_audits():
    gw, ledger = build_gateway()
    record = gw.invoke(make_request())

    assert record.status == "unverified"  # gateway owns no truth
    assert record.model_id == "local_generator_v1"
    assert record.model_version == "1.0.0"
    assert record.usage.total_tokens == (
        record.usage.prompt_tokens + record.usage.completion_tokens
    )
    assert record.output_text  # backend produced something
    assert record.audit_event_id is not None
    # Exactly one allow event, chain intact.
    events = ledger.events()
    assert len(events) == 1
    assert events[0].metadata["outcome"] == "allow"
    assert ledger.verify_chain() is True


def test_prompt_version_is_recorded():
    gw, ledger = build_gateway()
    record = gw.invoke(make_request(prompt_version="9.9.9"))
    assert record.prompt_version == "9.9.9"
    assert ledger.events()[0].metadata["prompt_version"] == "9.9.9"


def test_prompt_hash_binds_exact_prompt_no_hidden_change():
    gw, _ = build_gateway()
    req = make_request()
    record = gw.invoke(req)
    expected = hashlib.sha256(
        "\x00".join(
            [
                req.prompt_manifest_id,
                req.prompt_version,
                req.system_prompt or "",
                req.user_context,
            ]
        ).encode("utf-8")
    ).hexdigest()
    assert record.prompt_sha256 == expected


def test_output_record_never_contains_raw_prompt():
    gw, ledger = build_gateway()
    secretish_system = "You are a careful assistant."
    record = gw.invoke(make_request(system_prompt=secretish_system))
    dumped = record.model_dump()
    assert secretish_system not in str(dumped)  # only the hash is kept
    # Audit likewise carries no raw prompt, only the hash.
    assert secretish_system not in str(ledger.events()[0].metadata)


# --------------------------------------------------------------------------
# Required: offline-only uses local model only
# --------------------------------------------------------------------------


def test_offline_mode_blocks_hosted_without_calling_backend():
    tripwire = TripwireBackend()
    gw, ledger = build_gateway(
        backends={
            ModelProvider.LOCAL: LocalDeterministicBackend(),
            ModelProvider.HOSTED: tripwire,
        },
        offline_mode=True,
    )
    with pytest.raises(ModelGatewayRefused) as exc:
        gw.invoke(make_request(model_manifest_id="hosted_opus_v1"))
    assert exc.value.reason is GatewayReason.OFFLINE_HOSTED_BLOCKED
    assert tripwire.called is False  # never reached the backend
    assert ledger.events()[0].metadata["outcome"] == "deny"


def test_offline_mode_still_allows_local():
    gw, _ = build_gateway(offline_mode=True)
    record = gw.invoke(make_request())  # local manifest
    assert record.provider is ModelProvider.LOCAL


# --------------------------------------------------------------------------
# Required: budget exceeded stops call
# --------------------------------------------------------------------------


def test_budget_exceeded_stops_call():
    tripwire = TripwireBackend()
    gw, ledger = build_gateway(
        backends={ModelProvider.LOCAL: tripwire, ModelProvider.HOSTED: tripwire}
    )
    # Budget below even the planned completion (manifest.max_tokens=1024).
    with pytest.raises(ModelGatewayRefused) as exc:
        gw.invoke(make_request(budget=Budget(max_total_tokens=10)))
    assert exc.value.reason is GatewayReason.BUDGET_EXCEEDED
    assert tripwire.called is False
    assert ledger.events()[0].metadata["outcome"] == "deny"


# --------------------------------------------------------------------------
# Required: sensitive context redacted or blocked
# --------------------------------------------------------------------------


def test_redacted_only_model_redacts_secret_context():
    spy = SpyLocalBackend()
    manifest = hosted_manifest(
        model_id="hosted_redacted_v1",
        privacy_mode=PrivacyMode.REDACTED_ONLY,
    )
    gw, _ = build_gateway(
        models={manifest.model_id: manifest},
        backends={ModelProvider.HOSTED: spy},
    )
    record = gw.invoke(
        make_request(
            model_manifest_id="hosted_redacted_v1",
            user_context="here is a token bearer abcdef0123456789ABCDEF please use it",
        )
    )
    assert record.context_redacted is True
    # The backend received a redacted context — the raw token never left.
    assert "abcdef0123456789ABCDEF" not in (spy.last_user_context or "")
    assert "[REDACTED]" in (spy.last_user_context or "")


def test_non_redacting_model_blocks_secret_context():
    tripwire = TripwireBackend()
    # hosted_allowed model (not redacted_only) must block secret-shaped input.
    manifest = hosted_manifest(model_id="hosted_plain_v1")
    gw, _ = build_gateway(
        models={manifest.model_id: manifest},
        backends={ModelProvider.HOSTED: tripwire},
    )
    with pytest.raises(ModelGatewayRefused) as exc:
        gw.invoke(
            make_request(
                model_manifest_id="hosted_plain_v1",
                user_context="bearer abcdef0123456789ABCDEF",
            )
        )
    assert exc.value.reason is GatewayReason.RAW_SECRET_BLOCKED
    assert tripwire.called is False


def test_forbidden_data_class_blocked():
    tripwire = TripwireBackend()
    gw, _ = build_gateway(
        backends={ModelProvider.LOCAL: tripwire, ModelProvider.HOSTED: tripwire}
    )
    with pytest.raises(ModelGatewayRefused) as exc:
        # local manifest allows only public/internal.
        gw.invoke(make_request(data_sensitivity=DataSensitivity.CONFIDENTIAL))
    assert exc.value.reason is GatewayReason.DATA_CLASS_FORBIDDEN
    assert tripwire.called is False


def test_secret_data_class_never_allowed():
    tripwire = TripwireBackend()
    gw, _ = build_gateway(
        backends={ModelProvider.LOCAL: tripwire, ModelProvider.HOSTED: tripwire}
    )
    with pytest.raises(ModelGatewayRefused) as exc:
        gw.invoke(make_request(data_sensitivity=DataSensitivity.SECRET))
    assert exc.value.reason is GatewayReason.DATA_CLASS_FORBIDDEN
    assert tripwire.called is False


# --------------------------------------------------------------------------
# Other gates
# --------------------------------------------------------------------------


def test_unknown_manifest_refused():
    gw, ledger = build_gateway()
    with pytest.raises(ModelGatewayRefused) as exc:
        gw.invoke(make_request(model_manifest_id="does_not_exist"))
    assert exc.value.reason is GatewayReason.MANIFEST_NOT_FOUND
    assert ledger.events()[0].metadata["outcome"] == "deny"


def test_retired_model_refused():
    manifest = local_manifest(
        model_id="retired_v1", promotion_stage=PromotionStage.RETIRED
    )
    gw, _ = build_gateway(models={manifest.model_id: manifest})
    with pytest.raises(ModelGatewayRefused) as exc:
        gw.invoke(make_request(model_manifest_id="retired_v1"))
    assert exc.value.reason is GatewayReason.MODEL_RETIRED


def test_missing_backend_refused():
    manifest = local_manifest()
    gw, _ = build_gateway(models={manifest.model_id: manifest}, backends={})
    with pytest.raises(ModelGatewayRefused) as exc:
        gw.invoke(make_request())
    assert exc.value.reason is GatewayReason.PROVIDER_UNAVAILABLE


def test_deny_events_keep_audit_chain_intact():
    gw, ledger = build_gateway()
    for _ in range(3):
        with pytest.raises(ModelGatewayRefused):
            gw.invoke(make_request(model_manifest_id="missing"))
    assert len(ledger.events()) == 3
    assert ledger.verify_chain() is True


# --------------------------------------------------------------------------
# Manifest invariants
# --------------------------------------------------------------------------


def test_manifest_rejects_secret_in_allowed_classes():
    with pytest.raises(ValueError):
        local_manifest(
            allowed_data_classes=[DataSensitivity.PUBLIC, DataSensitivity.SECRET]
        )


def test_manifest_rejects_local_only_hosted_contradiction():
    with pytest.raises(ValueError):
        hosted_manifest(privacy_mode=PrivacyMode.LOCAL_ONLY)


def test_manifest_requires_owner_approval_for_production():
    with pytest.raises(ValueError):
        hosted_manifest(owner_approval_ref=None)


def test_manifest_requires_eval_for_production():
    with pytest.raises(ValueError):
        hosted_manifest(eval_report_refs=[])


def test_manifest_offline_bundle_requires_integrity_pins():
    with pytest.raises(ValueError):
        local_manifest(
            provider=ModelProvider.OFFLINE_BUNDLE,
            privacy_mode=PrivacyMode.LOCAL_ONLY,
            checksum=None,
        )


def test_manifest_max_tokens_within_window():
    with pytest.raises(ValueError):
        local_manifest(context_window=100, max_tokens=200)


def test_assert_valid_catches_bypass_construction():
    # Build a valid manifest, then mutate it to an invalid state and prove
    # assert_valid re-catches it (the gateway calls this defensively).
    manifest = local_manifest()
    object.__setattr__(manifest, "max_tokens", 999999)
    with pytest.raises(ModelManifestValidationError):
        manifest.assert_valid()
