"""Gate B demo smoke must call ModelBackend.complete, not a fictional generate()."""

from amali.demo.base_ai_gate import _gate_b_smoke
from amali.model_gateway.providers import BackendResult


def test_gate_b_smoke_uses_backend_complete(monkeypatch):
    complete_calls: list[dict] = []

    class FakeBackend:
        def complete(
            self,
            *,
            system_prompt,
            user_context,
            max_output_tokens,
        ):
            complete_calls.append(
                {
                    "system_prompt": system_prompt,
                    "user_context": user_context,
                    "max_output_tokens": max_output_tokens,
                }
            )
            return BackendResult(
                output_text="ready",
                prompt_tokens=1,
                completion_tokens=1,
            )

    class FakeTransformersBackend:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            assert model_path == "fake-model"
            return FakeBackend()

    monkeypatch.setattr(
        "amali.model_gateway.transformers_backend.TransformersBackend",
        FakeTransformersBackend,
    )

    result = _gate_b_smoke("fake-model")

    assert result["ran"] is True
    assert result["output_non_empty"] is True
    assert len(complete_calls) == 1
    assert complete_calls[0]["user_context"].startswith("Reply with")
    assert complete_calls[0]["max_output_tokens"] == 8
