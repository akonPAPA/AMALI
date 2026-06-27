"""Capability scope models, shared by requests, decisions, and grants.

These live in a low-level module with no upward dependencies so that both
the request side (``amali.tools``) and the decision side (``amali.policy``)
can use them without creating an import cycle.

Both models default every field to the empty / least-capable value: an
omitted scope is the *most* restrictive one, never the most permissive.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["RequestedScope", "GrantedScope"]


class RequestedScope(BaseModel):
    """The capabilities a request is asking for."""

    model_config = {"extra": "forbid"}

    paths: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    env: list[str] = Field(default_factory=list)
    write: bool = False
    network: bool = False
    secret_access: bool = False


class GrantedScope(BaseModel):
    """The capabilities actually granted (least privilege).

    A grant never widens a request; this is the explicit, audited record of
    what the executor boundary is allowed to consume.
    """

    model_config = {"extra": "forbid"}

    paths: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    env: list[str] = Field(default_factory=list)
    write: bool = False
    network: bool = False
    secret_access: bool = False
