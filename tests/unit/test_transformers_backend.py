"""TransformersBackend plumbing tests — no model weights, no network.

The backend is dependency-injected, so these tests use a fake tokenizer and
fake model built on real torch tensors. They prove:

* real token accounting (counts come from tensor shapes, not estimates);
* greedy decoding by default (do_sample=False, no temperature leak);
* the gateway's max_output_tokens is passed as max_new_tokens;
* only the *new* tokens are decoded as the completion;
* chat-template use when available, labelled fallback when not;
* end-to-end: the backend satisfies ModelBackend and flows through the
  full ModelGateway gate chain.

Skipped entirely if torch is not installed (optional [local_llm] extra).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from amali.audit.ledger import AuditLedger
from amali.model_gateway import (
    Budget,
    Capability,
    ModelGateway,
    ModelInvocationRequest,
    ModelManifest,
    ModelProvider,
    PrivacyMode,
    PromotionStage,
    TransformersBackend,
)
from amali.policy.enums import DataSensitivity


class FakeTokenizer:
    """Minimal stand-in for a HF tokenizer (deterministic, offline)."""

    eos_token_id = 0
    pad_token_id = None

    def __init__(self, has_chat_template: bool = True) -> None:
        self._has_chat_template = has_chat_template
        self.last_prompt_text: str | None = None
        self.last_decoded_len: int | None = None

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True
    ) -> str:
        if not self._has_chat_template:
            raise ValueError("no chat template defined")
        rendered = "|".join(f"{m['role']}:{m['content']}" for m in messages)
        return rendered + "|assistant:"

    def __call__(self, text: str, return_tensors: str = "pt"):
        self.last_prompt_text = text
        # One token per 10 chars, at least 1 — deterministic.
        n = max(1, len(text) // 10)
        return {"input_ids": torch.arange(n).unsqueeze(0)}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        self.last_decoded_len = int(ids.shape[-1])
        return f"decoded[{self.last_decoded_len}]"


class FakeModel:
    """Appends a fixed number of tokens; records generate() kwargs."""

    def __init__(self, new_tokens: int = 5) -> None:
        self.new_tokens = new_tokens
        self.generate_calls: list[dict] = []

    def generate(self, input_ids, **kwargs):
        self.generate_calls.append(kwargs)
        produced = min(self.new_tokens, kwargs["max_new_tokens"])
        tail = torch.zeros(1, produced, dtype=input_ids.dtype)
        return torch.cat([input_ids, tail], dim=1)


def make_backend(**kwargs):
    tokenizer = kwargs.pop("tokenizer", FakeTokenizer())
    model = kwargs.pop("model", FakeModel())
    backend = TransformersBackend(model=model, tokenizer=tokenizer, **kwargs)
    return backend, model, tokenizer


def test_token_counts_come_from_real_tensor_shapes():
    backend, model, tokenizer = make_backend()
    result = backend.complete(
        system_prompt="be careful",
        user_context="x" * 200,
        max_output_tokens=64,
    )
    expected_prompt = max(1, len(tokenizer.last_prompt_text) // 10)
    assert result.prompt_tokens == expected_prompt
    assert result.completion_tokens == 5  # FakeModel appended 5
    assert result.output_text == "decoded[5]"


def test_only_new_tokens_are_decoded():
    backend, _, tokenizer = make_backend(model=FakeModel(new_tokens=3))
    result = backend.complete(
        system_prompt=None, user_context="hello world", max_output_tokens=64
    )
    assert tokenizer.last_decoded_len == 3
    assert result.completion_tokens == 3


def test_greedy_by_default_no_temperature_leak():
    backend, model, _ = make_backend()
    backend.complete(
        system_prompt=None, user_context="hi", max_output_tokens=8
    )
    kwargs = model.generate_calls[0]
    assert kwargs["do_sample"] is False
    assert "temperature" not in kwargs


def test_max_output_tokens_caps_generation():
    backend, model, _ = make_backend(model=FakeModel(new_tokens=100))
    result = backend.complete(
        system_prompt=None, user_context="hi", max_output_tokens=7
    )
    assert model.generate_calls[0]["max_new_tokens"] == 7
    assert result.completion_tokens == 7


def test_pad_token_falls_back_to_eos():
    backend, model, _ = make_backend()
    backend.complete(
        system_prompt=None, user_context="hi", max_output_tokens=8
    )
    assert model.generate_calls[0]["pad_token_id"] == 0  # eos_token_id


def test_chat_template_used_when_available():
    backend, _, tokenizer = make_backend()
    backend.complete(
        system_prompt="sys rules",
        user_context="user question",
        max_output_tokens=8,
    )
    assert tokenizer.last_prompt_text is not None
    assert "system:sys rules" in tokenizer.last_prompt_text
    assert "user:user question" in tokenizer.last_prompt_text
    assert tokenizer.last_prompt_text.endswith("|assistant:")


def test_fallback_prompt_when_no_chat_template():
    tokenizer = FakeTokenizer(has_chat_template=False)
    backend, _, _ = make_backend(tokenizer=tokenizer)
    backend.complete(
        system_prompt="sys rules",
        user_context="user question",
        max_output_tokens=8,
    )
    text = tokenizer.last_prompt_text
    assert text is not None
    assert "System: sys rules" in text
    assert "User: user question" in text
    assert text.rstrip().endswith("Assistant:")


def test_backend_flows_through_full_gateway_gate_chain():
    manifest = ModelManifest(
        model_id="local_torch_v1",
        version="0.1.0",
        provider=ModelProvider.LOCAL,
        model_type="generator",
        allowed_data_classes=[DataSensitivity.PUBLIC],
        context_window=4096,
        max_tokens=256,
        capabilities=[Capability.TEXT],
        privacy_mode=PrivacyMode.LOCAL_ONLY,
        license="open-weights-local",
        promotion_stage=PromotionStage.DEV,
    )
    backend, _, _ = make_backend()
    ledger = AuditLedger()
    gateway = ModelGateway(
        models={manifest.model_id: manifest},
        backends={ModelProvider.LOCAL: backend},
        ledger=ledger,
        offline_mode=True,  # strictest posture: local-only machine
    )
    record = gateway.invoke(
        ModelInvocationRequest(
            task_id="task_fixed_1",
            actor_ref="local_dev_agent",
            model_manifest_id="local_torch_v1",
            prompt_manifest_id="prompt_test",
            prompt_version="1.0.0",
            system_prompt="be careful",
            user_context="summarize the control planes",
            data_sensitivity=DataSensitivity.PUBLIC,
            budget=Budget(max_total_tokens=4096),
        )
    )
    assert record.status == "unverified"
    assert record.provider is ModelProvider.LOCAL
    assert record.usage.completion_tokens > 0
    assert record.audit_event_id is not None
    assert ledger.verify_chain() is True
