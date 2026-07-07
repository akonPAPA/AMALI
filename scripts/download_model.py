"""One-time, owner-initiated download of local open-weight models.

This script is the *only* sanctioned network moment for model weights.
Everything else in AMALI runs offline: the TransformersBackend loads with
``local_files_only=True`` and will fail loudly rather than silently reach
the network.

Usage (from the repo root, inside the venv):

    python scripts/download_model.py                  # default model
    python scripts/download_model.py Qwen/Qwen2.5-1.5B-Instruct

Sized for the owner's hardware (RTX 4060 Laptop, 8 GB VRAM):

    Qwen/Qwen2.5-3B-Instruct    ~6.2 GB bf16 — default, best quality that fits
    Qwen/Qwen2.5-1.5B-Instruct  ~3.1 GB bf16 — fallback if 3B is tight
    meta-llama/Llama-3.2-3B-Instruct — gated: requires HF account approval

Weights land in the standard Hugging Face cache (~/.cache/huggingface), so
`TransformersBackend.from_pretrained(model_id)` finds them offline afterward.
"""

from __future__ import annotations

import sys

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def main() -> int:
    model_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    print(f"Downloading weights for: {model_id}")
    print("This is a one-time, owner-approved network action.")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "huggingface_hub is missing. Install the local runtime first:\n"
            "    pip install -e .[local_llm]",
            file=sys.stderr,
        )
        return 1

    path = snapshot_download(repo_id=model_id)
    print(f"Done. Cached at: {path}")
    print(
        "From now on, load it fully offline:\n"
        f'    TransformersBackend.from_pretrained("{model_id}")'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
