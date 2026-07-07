"""Local PyTorch backend for the Model Gateway (HF Transformers).

This is the full-control local runtime: the model is a real ``torch.nn.Module``
loaded from *local* weights, so later phases (LoRA/SFT/DPO experiments, §11;
R&D adapters, §12) can reach the raw model. Nothing here talks to the
network at inference time; by default even weight loading is
``local_files_only=True`` so a missing model fails loudly instead of
silently downloading.

Design rules:

* **Optional dependency.** ``torch``/``transformers`` are imported lazily so
  the AMALI kernel does not require them. Constructing the backend without
  them raises a clear error.
* **Dependency injection.** ``__init__`` takes an already-loaded model and
  tokenizer. Unit tests inject fakes; :meth:`from_pretrained` is the
  convenience loader for real use.
* **Deterministic by default.** Greedy decoding (``do_sample=False``) so the
  same input yields the same output — reproducibility before creativity.
* **Honest accounting.** Token counts come from the tokenizer's real ids,
  not estimates.
"""

from __future__ import annotations

from typing import Any

from amali.model_gateway.providers import BackendResult

__all__ = ["TransformersBackend"]


class TransformersBackend:
    """A :class:`~amali.model_gateway.providers.ModelBackend` running a local
    Hugging Face causal-LM under PyTorch."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        device: str | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._do_sample = do_sample
        self._temperature = temperature

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        local_files_only: bool = True,
        device: str | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> "TransformersBackend":
        """Load a local causal-LM and tokenizer from ``model_path``.

        ``local_files_only=True`` (the default) means this never reaches the
        network: if the weights are not on disk, loading fails loudly. Pass
        ``local_files_only=False`` only for an explicit, owner-approved
        one-time download.
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "TransformersBackend requires the optional [local_llm] "
                "dependencies: pip install amali-core[local_llm]"
            ) from exc

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if resolved_device == "cuda" else torch.float32

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=local_files_only
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        )
        model.to(resolved_device)
        model.eval()
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=resolved_device,
            do_sample=do_sample,
            temperature=temperature,
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self, system_prompt: str | None, user_context: str
    ) -> str:
        """Render system + user into the model's chat template if it has one,
        otherwise fall back to a plain labelled concatenation."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_context})

        apply_template = getattr(self._tokenizer, "apply_chat_template", None)
        if callable(apply_template):
            try:
                return apply_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except (ValueError, TypeError):
                # Tokenizer has no usable chat template; fall through.
                pass
        parts = []
        if system_prompt:
            parts.append(f"System: {system_prompt}")
        parts.append(f"User: {user_context}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # ModelBackend protocol
    # ------------------------------------------------------------------

    def complete(
        self,
        *,
        system_prompt: str | None,
        user_context: str,
        max_output_tokens: int,
    ) -> BackendResult:
        """Tokenize, generate greedily, decode only the new tokens, and
        report real token counts."""
        import torch

        prompt_text = self._build_prompt(system_prompt, user_context)
        encoded = self._tokenizer(prompt_text, return_tensors="pt")
        input_ids = encoded["input_ids"]
        if self._device:
            input_ids = input_ids.to(self._device)
        prompt_tokens = int(input_ids.shape[-1])

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_output_tokens,
            "do_sample": self._do_sample,
        }
        if self._do_sample:
            generate_kwargs["temperature"] = self._temperature
        pad_id = getattr(self._tokenizer, "pad_token_id", None)
        eos_id = getattr(self._tokenizer, "eos_token_id", None)
        if pad_id is None and eos_id is not None:
            generate_kwargs["pad_token_id"] = eos_id

        with torch.no_grad():
            output_ids = self._model.generate(input_ids, **generate_kwargs)

        # Only the tokens *after* the prompt are the completion.
        new_ids = output_ids[0][prompt_tokens:]
        completion_tokens = int(new_ids.shape[-1])
        output_text = self._tokenizer.decode(
            new_ids, skip_special_tokens=True
        ).strip()

        return BackendResult(
            output_text=output_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            structured_output=None,
            safety_flags=[],
        )
