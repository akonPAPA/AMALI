"""Task contracts (WP1): the validated boundary where work enters AMALI.

A :class:`TaskRequest` is untrusted input. Validation here is a security
control, not a convenience:

* the payload is inert data — text and safe primitives only, no authority
  fields, no risk-lowering, no secret material;
* the actor/project are typed references, not free-form trust claims;
* the intent and status vocabularies are closed enums so an unknown value
  cannot be smuggled through;
* a :class:`TaskHandle` (the trusted record) is only ever minted by the
  runtime after validation — callers cannot construct authority.

No torch, no network, no side effects at import time.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator

from amali.core.ids import new_task_id
from amali.policy.enums import DataSensitivity, RiskLevel
from amali.security.redaction import contains_secret

__all__ = [
    "TaskIntent",
    "TaskStatus",
    "ActorRef",
    "ProjectRef",
    "LocalTaskPayload",
    "TaskRequest",
    "TaskHandle",
]

# Primitive types permitted inside payload metadata; anything richer is
# rejected so payloads stay inert, redactable, and auditable.
_SAFE_PRIMITIVES = (str, int, float, bool, type(None))


class TaskIntent(str, Enum):
    """What kind of work the task is asking the system to do."""

    ANSWER_QUESTION = "answer_question"
    RETRIEVE_EVIDENCE = "retrieve_evidence"
    RUN_EVAL = "run_eval"
    RUN_BENCHMARK = "run_benchmark"
    EMIT_ARTIFACT = "emit_artifact"
    REVIEW = "review"


class TaskStatus(str, Enum):
    """Lifecycle status of a task handle (runtime-owned)."""

    ACCEPTED = "ACCEPTED"
    BLOCKED = "BLOCKED"
    RUNNING = "RUNNING"
    PARTIAL = "PARTIAL"
    UNKNOWN = "UNKNOWN"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class ActorRef(BaseModel):
    """A typed reference to the requesting actor. Identity, not authority."""

    model_config = {"extra": "forbid"}

    actor_id: str
    kind: str = "agent"  # user | agent | system | test
    roles: list[str] = Field(default_factory=list)

    @field_validator("actor_id")
    @classmethod
    def _actor_id_not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("actor_id must not be empty")
        return value

    @field_validator("kind")
    @classmethod
    def _kind_known(cls, value: str) -> str:
        if value not in {"user", "agent", "system", "test"}:
            raise ValueError("kind must be one of user|agent|system|test")
        return value


class ProjectRef(BaseModel):
    """A typed reference to the project/workspace context."""

    model_config = {"extra": "forbid"}

    project_id: str
    workspace: str = "local"

    @field_validator("project_id")
    @classmethod
    def _project_id_not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("project_id must not be empty")
        return value


class LocalTaskPayload(BaseModel):
    """The inert data payload of a task.

    ``query`` is the question/instruction text (treated strictly as data by
    downstream stages); ``metadata`` may only hold safe primitives and must
    not carry secret-like material.
    """

    model_config = {"extra": "forbid"}

    query: str
    context_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("query")
    @classmethod
    def _query_not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("query must not be empty")
        return value

    @field_validator("metadata")
    @classmethod
    def _metadata_is_safe(cls, value: dict[str, object]) -> dict[str, object]:
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("metadata keys must be strings")
            if not isinstance(item, _SAFE_PRIMITIVES):
                raise ValueError(
                    "metadata values must be safe primitives "
                    "(str, int, float, bool, None)"
                )
        if contains_secret(value):
            raise ValueError("metadata must not contain secret-like material")
        return value


class TaskRequest(BaseModel):
    """Untrusted request for work, validated at the boundary.

    ``task_risk`` and ``data_sensitivity`` are typed declarations that the
    policy layer treats as *floors*, never ceilings: nothing in the payload
    can lower the effective risk downstream.
    """

    model_config = {"extra": "forbid"}

    actor: ActorRef
    project: ProjectRef
    intent: TaskIntent
    payload: LocalTaskPayload
    task_risk: RiskLevel = RiskLevel.L1
    data_sensitivity: DataSensitivity = DataSensitivity.INTERNAL
    reason: str = ""

    def mint_handle(self) -> "TaskHandle":
        """Create the trusted runtime record for this request.

        Only the runtime should call this, after validation. The handle
        carries a fresh ``task_id`` and starts ``ACCEPTED``.
        """
        return TaskHandle(
            task_id=new_task_id(),
            actor_id=self.actor.actor_id,
            project_id=self.project.project_id,
            intent=self.intent,
            status=TaskStatus.ACCEPTED,
        )


class TaskHandle(BaseModel):
    """The trusted, runtime-owned record of an accepted task."""

    model_config = {"extra": "forbid"}

    task_id: str
    actor_id: str
    project_id: str
    intent: TaskIntent
    status: TaskStatus = TaskStatus.ACCEPTED
    audit_event_refs: list[str] = Field(default_factory=list)
