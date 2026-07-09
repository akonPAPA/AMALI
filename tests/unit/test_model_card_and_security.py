"""Phases 13/14: truthful model card + security/privacy hardening checks."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from amali.audit.ledger import AuditLedger
from amali.training.dataset import build_training_dataset
from amali.training.model_card import ModelCardInputs, generate_model_card

REPO_ROOT = Path(__file__).resolve().parents[2]

SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9]{16,}"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


# --- model card ------------------------------------------------------------------


def test_model_card_contains_base_lineage():
    card = generate_model_card(ModelCardInputs())
    assert "Qwen/Qwen2.5-1.5B-Instruct" in card
    assert "adapter" in card.lower()


def test_model_card_states_not_trained_from_scratch():
    card = generate_model_card(ModelCardInputs())
    assert "not trained from scratch" in card.lower()


def test_model_card_contains_eval_hashes():
    card = generate_model_card(
        ModelCardInputs(
            dataset_manifest_hash="d" * 64, eval_suite_hash="e" * 64
        )
    )
    assert "d" * 64 in card
    assert "e" * 64 in card


def test_model_card_contains_limitations_section():
    card = generate_model_card(ModelCardInputs())
    assert "## Known limitations" in card


def test_model_card_makes_no_external_superiority_claim():
    card = generate_model_card(ModelCardInputs()).lower()
    for forbidden in (
        "beats gpt",
        "beats claude",
        "beats deepseek",
        "frontier model",
        "agi achieved",
    ):
        assert forbidden not in card


def test_model_card_honest_when_nothing_ran():
    card = generate_model_card(ModelCardInputs())
    assert "NOT_RUN" in card
    assert "no training run has happened" in card.lower()
    assert "none are\ninvented" in card.lower() or "none are invented" in (
        card.replace("\n", " ").lower()
    )


def test_model_card_refuses_injected_forbidden_marker():
    with pytest.raises(ValueError):
        generate_model_card(
            ModelCardInputs(
                training_run_facts={"note": "this beats GPT easily"}
            )
        )


# --- security hardening ------------------------------------------------------------


def _no_secrets(text: str, context: str) -> None:
    for pattern in SECRET_PATTERNS:
        assert not pattern.search(text), f"secret-shaped content in {context}"


def test_built_dataset_has_no_secret_shaped_content():
    dataset_path = Path("D:/AMALI/data/processed/amali_bmg_train.jsonl")
    if not dataset_path.exists():
        report, _, _ = build_training_dataset(
            repo_root=REPO_ROOT,
            output_path=dataset_path,
            ledger=AuditLedger(),
        )
    _no_secrets(dataset_path.read_text(encoding="utf-8"), str(dataset_path))


def test_latest_artifacts_have_no_secret_shaped_content():
    base = REPO_ROOT / "artifacts" / "base_model_gate"
    if not base.is_dir():
        return
    directories = sorted(p for p in base.iterdir() if p.is_dir())[-5:]
    for directory in directories:
        for path in directory.glob("*.json"):
            _no_secrets(path.read_text(encoding="utf-8"), str(path))


def test_no_env_or_weights_tracked_by_git():
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=60,
    ).stdout.splitlines()
    bad = [
        p
        for p in tracked
        if p.endswith((".env", ".safetensors", ".bin", ".gguf", ".pt"))
        or p.startswith(("models/", "data/", "artifacts/"))
        or ".egg-info" in p
    ]
    assert bad == []


def test_injected_instruction_in_dataset_is_data_not_instruction():
    dataset_path = Path("D:/AMALI/data/processed/amali_bmg_train.jsonl")
    if not dataset_path.exists():
        build_training_dataset(
            repo_root=REPO_ROOT,
            output_path=dataset_path,
            ledger=AuditLedger(),
        )
    rows = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    injection_rows = [
        r for r in rows if "injection_as_data" in r["behavior_tags"]
    ]
    assert injection_rows, "dataset must carry injection-as-data behavior"
    for row in injection_rows:
        # the expected response treats the payload as data / quotes it,
        # and never complies with its embedded token demand
        for token in row["forbidden_outputs"]:
            assert token not in row["expected_response"]


def test_no_tool_grant_learned_as_behavior():
    dataset_path = Path("D:/AMALI/data/processed/amali_bmg_train.jsonl")
    if not dataset_path.exists():
        build_training_dataset(
            repo_root=REPO_ROOT,
            output_path=dataset_path,
            ledger=AuditLedger(),
        )
    rows = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    refusal_rows = [
        r for r in rows if "refuse_forbidden_tool" in r["behavior_tags"]
    ]
    assert refusal_rows
    for row in refusal_rows:
        assert row["expected_status"] == "REFUSED"
