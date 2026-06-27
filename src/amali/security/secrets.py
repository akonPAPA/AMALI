"""Opaque secret handles.

AMALI-IGA-02 does **not** retrieve, store, or transmit any real secret
value. It only defines the *handle* abstraction that later stages will use:
an opaque reference that a trusted tool boundary could later resolve, but
which never exposes a raw secret to a model, an agent, a log line, an audit
event, an exception, or a ``repr``.

Because no raw value is ever held, there is nothing to leak. The class is
written defensively anyway so that the invariant remains true even if a
future change is tempted to attach a value.
"""

from __future__ import annotations

from amali.security.redaction import REDACTED

__all__ = ["SecretHandle"]


class SecretHandle:
    """An opaque, non-revealing reference to a secret.

    The handle exposes a stable ``handle_id`` and a logical ``ref`` (e.g.
    ``"env:DATABASE_URL"``) — never a value. ``repr``/``str`` and any
    serialization render a redaction marker.
    """

    __slots__ = ("_handle_id", "_ref")

    def __init__(self, handle_id: str, ref: str) -> None:
        self._handle_id = handle_id
        self._ref = ref

    @property
    def handle_id(self) -> str:
        return self._handle_id

    @property
    def ref(self) -> str:
        """The logical reference (a name/locator), not a secret value."""
        return self._ref

    def reveal(self) -> str:
        """Resolve the raw secret value.

        Intentionally unimplemented in this stage. No secret retrieval
        exists yet, so there is nothing to reveal and nothing to leak.
        """
        raise NotImplementedError(
            "secret resolution is not implemented in AMALI-IGA-02"
        )

    def to_audit(self) -> dict[str, str]:
        """Audit-safe projection: id and ref only, never a value."""
        return {"handle_id": self._handle_id, "ref": self._ref}

    def __repr__(self) -> str:
        # The ref is a name/locator (safe); the value is never shown.
        return (
            f"SecretHandle(handle_id={self._handle_id!r}, "
            f"ref={self._ref!r}, value={REDACTED})"
        )

    __str__ = __repr__
