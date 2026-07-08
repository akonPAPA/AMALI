"""Typed task contracts: the validated entry point into the AMALI runtime.

A task enters the system only as a validated :class:`TaskRequest`. Free text,
actor identity, and payload are data — never instructions and never authority.
"""

from amali.contracts.task import (
    ActorRef,
    LocalTaskPayload,
    ProjectRef,
    TaskHandle,
    TaskIntent,
    TaskRequest,
    TaskStatus,
)

__all__ = [
    "TaskRequest",
    "TaskHandle",
    "TaskIntent",
    "TaskStatus",
    "LocalTaskPayload",
    "ActorRef",
    "ProjectRef",
]
