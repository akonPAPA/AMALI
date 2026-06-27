"""Conservative redaction and non-revealing secret handles."""

from __future__ import annotations

import json

from amali.security.redaction import (
    REDACTED,
    contains_secret,
    looks_like_secret_value,
    redact,
)
from amali.security.secrets import SecretHandle


# --- key-based redaction --------------------------------------------------


def test_redacts_secret_named_keys():
    out = redact(
        {
            "password": "hunter2",
            "api_key": "abc123",
            "authorization": "Bearer xyz",
            "username": "alice",
        }
    )
    assert out["password"] == REDACTED
    assert out["api_key"] == REDACTED
    assert out["authorization"] == REDACTED
    assert out["username"] == "alice"  # non-secret preserved


def test_redaction_handles_nested_dict_and_list():
    data = {
        "outer": {
            "token": "secretval",
            "items": [{"private_key": "k"}, {"ok": "value"}],
        },
        "list": ["plain", {"set-cookie": "sid=1"}],
    }
    out = redact(data)
    assert out["outer"]["token"] == REDACTED
    assert out["outer"]["items"][0]["private_key"] == REDACTED
    assert out["outer"]["items"][1]["ok"] == "value"
    assert out["list"][0] == "plain"
    assert out["list"][1]["set-cookie"] == REDACTED


def test_redaction_does_not_mutate_input():
    data = {"password": "hunter2", "items": [{"token": "t"}]}
    snapshot = json.dumps(data)
    redact(data)
    assert json.dumps(data) == snapshot  # original untouched


# --- value-based redaction ------------------------------------------------


def test_redacts_secret_shaped_values():
    assert looks_like_secret_value("Bearer abcdefgh12345678")
    assert looks_like_secret_value("sk-0123456789abcdef0123")
    assert looks_like_secret_value("ghp_0123456789abcdefghij0123")
    assert looks_like_secret_value(
        "-----BEGIN OPENSSH PRIVATE KEY-----abc"
    )
    assert looks_like_secret_value(
        "postgres://user:p4ssw0rd@db.local:5432/app"
    )
    assert not looks_like_secret_value("just a normal sentence")


def test_redacts_secret_value_under_innocent_key():
    out = redact({"note": "my key is sk-0123456789abcdef0123 ok"})
    assert out["note"] == REDACTED


def test_contains_secret_detects_key_and_value():
    assert contains_secret({"password": "x"})
    assert contains_secret({"note": "Bearer abcdefgh12345678"})
    assert contains_secret(["ok", {"api_key": "z"}])
    assert not contains_secret({"note": "nothing sensitive here"})


# --- secret handle never reveals -----------------------------------------


def test_secret_handle_repr_never_reveals():
    handle = SecretHandle(handle_id="h1", ref="env:DATABASE_URL")
    text = repr(handle)
    assert REDACTED in text
    assert "DATABASE_URL" in text  # the ref/name is fine; the value is not
    assert handle.handle_id == "h1"


def test_secret_handle_audit_projection_is_id_and_ref_only():
    handle = SecretHandle(handle_id="h1", ref="env:API_KEY")
    projection = handle.to_audit()
    assert projection == {"handle_id": "h1", "ref": "env:API_KEY"}


def test_secret_handle_reveal_not_implemented():
    handle = SecretHandle(handle_id="h1", ref="env:API_KEY")
    try:
        handle.reveal()
    except NotImplementedError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("reveal() must not return a value in IGA-02")
