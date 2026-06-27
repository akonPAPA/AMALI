"""Conservative redaction of secret-like material.

This helper is used before anything is written to the audit ledger, an
exception, or a log line. It errs on the side of over-redaction: it is far
better to redact a harmless value than to leak a token.

Two independent strategies are applied:

* **Key-based:** any mapping key whose normalized name looks like a secret
  (``password``, ``api_key``, ``authorization`` ...) has its value replaced
  wholesale, regardless of what the value looks like.
* **Value-based:** any string value that matches a known secret *shape*
  (bearer token, PEM private key, ``sk-``/``ghp_``/``AKIA`` prefixes, a
  connection string with embedded credentials ...) is replaced.

The original object is never mutated; a new structure is returned.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = [
    "REDACTED",
    "redact",
    "looks_like_secret_value",
    "contains_secret",
]

REDACTED = "[REDACTED]"

# Normalized substrings that mark a key as secret-bearing. Matching is done
# against a lowercased, alphanumeric-only form of the key so that
# ``API-Key``, ``api_key`` and ``apiKey`` all collapse to ``apikey``.
_SECRET_KEY_TERMS: tuple[str, ...] = (
    "password",
    "passwd",
    "token",
    "apikey",
    "secret",
    "secrets",
    "privatekey",
    "authorization",
    "cookie",
    "setcookie",
    "accesstoken",
    "refreshtoken",
    "sshkey",
    "bearer",
    "credential",
    "connectionstring",
)

# Value shapes that are almost always secrets.
_SECRET_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9]{16,}"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{8,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\."),  # JWT
    # scheme://user:password@host  -> credentials embedded in a URL/DSN.
    re.compile(r"[a-zA-Z][a-zA-Z0-9+.\-]*://[^/\s:@]+:[^/\s:@]+@"),
)


def _normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def _is_secret_key(key: str) -> bool:
    norm = _normalize_key(key)
    return any(term in norm for term in _SECRET_KEY_TERMS)


def looks_like_secret_value(value: str) -> bool:
    """Return True if a string matches a known secret shape."""
    return any(pattern.search(value) for pattern in _SECRET_VALUE_PATTERNS)


def redact(obj: Any) -> Any:
    """Return a deep copy of ``obj`` with secret-like material removed.

    * ``dict``: secret-named keys have their values fully replaced; other
      values are redacted recursively.
    * ``list`` / ``tuple``: each element is redacted recursively.
    * ``str``: replaced with ``[REDACTED]`` if it matches a secret shape.
    * everything else is returned unchanged.

    The input object is never modified.
    """
    if isinstance(obj, dict):
        result: dict[Any, Any] = {}
        for key, value in obj.items():
            if isinstance(key, str) and _is_secret_key(key):
                result[key] = REDACTED
            else:
                result[key] = redact(value)
        return result
    if isinstance(obj, (list, tuple)):
        return [redact(item) for item in obj]
    if isinstance(obj, str):
        return REDACTED if looks_like_secret_value(obj) else obj
    return obj


def contains_secret(obj: Any) -> bool:
    """Return True if ``obj`` contains any secret-like key or value.

    Used to *reject* untrusted input (e.g. request metadata) that carries
    raw secret material, rather than silently redacting and accepting it.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(key, str) and _is_secret_key(key):
                return True
            if contains_secret(value):
                return True
        return False
    if isinstance(obj, (list, tuple)):
        return any(contains_secret(item) for item in obj)
    if isinstance(obj, str):
        return looks_like_secret_value(obj)
    return False
