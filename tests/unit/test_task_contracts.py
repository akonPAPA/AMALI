"""Task contract validation: valid and invalid TaskRequest shapes."""

import pytest
from pydantic import ValidationError

from amali.contracts.task import (
    ActorRef,
    LocalTaskPayload,
    ProjectRef,
    TaskHandle,
    TaskIntent,
    TaskRequest,
    TaskStatus,
)


def _request(**overrides) -> TaskRequest:
    defaults = dict(
        actor=ActorRef(actor_id="local_dev_agent", kind="agent"),
        project=ProjectRef(project_id="amali_core"),
        intent=TaskIntent.ANSWER_QUESTION,
        payload=LocalTaskPayload(query="What is the audit chain genesis hash?"),
        reason="unit test",
    )
    defaults.update(overrides)
    return TaskRequest(**defaults)


def test_valid_request_and_handle():
    request = _request()
    handle = request.mint_handle()
    assert isinstance(handle, TaskHandle)
    assert handle.task_id.startswith("task_")
    assert handle.status is TaskStatus.ACCEPTED
    assert handle.actor_id == "local_dev_agent"
    assert handle.intent is TaskIntent.ANSWER_QUESTION


def test_empty_query_rejected():
    with pytest.raises(ValidationError):
        LocalTaskPayload(query="   ")


def test_empty_actor_rejected():
    with pytest.raises(ValidationError):
        ActorRef(actor_id="", kind="agent")


def test_unknown_actor_kind_rejected():
    with pytest.raises(ValidationError):
        ActorRef(actor_id="a", kind="root")


def test_unknown_intent_rejected():
    with pytest.raises(ValidationError):
        _request(intent="become_admin")


def test_extra_authority_fields_rejected():
    # A request cannot smuggle authority through unknown fields.
    with pytest.raises(ValidationError):
        TaskRequest(
            actor=ActorRef(actor_id="a"),
            project=ProjectRef(project_id="p"),
            intent=TaskIntent.ANSWER_QUESTION,
            payload=LocalTaskPayload(query="q"),
            is_admin=True,
        )


def test_payload_metadata_rejects_secret_material():
    with pytest.raises(ValidationError):
        LocalTaskPayload(
            query="q",
            metadata={"note": "Bearer abcdefgh12345678"},
        )


def test_payload_metadata_rejects_non_primitives():
    with pytest.raises(ValidationError):
        LocalTaskPayload(query="q", metadata={"fn": object()})


def test_no_torch_in_core_import():
    # Core imports must not pull torch (WP0 import boundary). Run in a
    # clean subprocess so other tests that legitimately import torch
    # cannot pollute this check.
    import subprocess
    import sys

    code = (
        "import sys; import amali, amali.contracts.task, "
        "amali.model_gateway, amali.router; "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
