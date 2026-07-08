"""RICARDO-FEC (WP7). Label: CUSTOM_AMALI_COMPOSITION.

Compiles runtime failures into redacted, deduplicated eval items so every
observed failure becomes a permanent regression guard.

Algorithm:

1. every incoming :class:`FailureTrace` is redacted first — no raw secret
   or secret-shaped value survives into an eval item;
2. the dedup key is ``sha256(component | failure_kind | redacted_query)``,
   so the same failure observed twice yields one eval item (first trace
   wins; later duplicates are counted, not stored);
3. the resulting :class:`EvalItem` id is derived from the dedup key —
   stable across runs, so eval suites do not churn;
4. ``conversion_rate`` = unique items / total traces (1.0 when every trace
   is unique) is reported for the artifact.

Invariant: an eval item can always be traced back to the failure kind and
component that produced it, but never to raw sensitive content.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field, field_validator

from amali.security.redaction import redact

__all__ = ["FailureTrace", "EvalItem", "FECResult", "compile_failures"]


class FailureTrace(BaseModel):
    """One observed runtime failure, as raised by any component."""

    model_config = {"extra": "forbid"}

    task_id: str
    component: str
    failure_kind: str  # e.g. UNSUPPORTED_CLAIM, WALL_BLOCK, PERMISSION_DENY
    query: str
    observed: str
    expected: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("task_id", "component", "failure_kind", "query", "observed")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("field must not be empty")
        return value


class EvalItem(BaseModel):
    """One redacted, deduplicated regression-guard eval item."""

    model_config = {"extra": "forbid"}

    eval_id: str
    component: str
    failure_kind: str
    prompt: str  # redacted query
    observed: str  # redacted
    expected: str | None = None  # redacted
    duplicate_count: int = 1
    regression_guard: bool = True
    source_task_ids: list[str] = Field(default_factory=list)


class FECResult(BaseModel):
    """The compiler output and its accounting."""

    model_config = {"extra": "forbid"}

    items: list[EvalItem] = Field(default_factory=list)
    total_traces: int = 0
    unique_items: int = 0
    conversion_rate: float = 0.0


def _dedup_key(component: str, failure_kind: str, redacted_query: str) -> str:
    payload = "\x00".join([component, failure_kind, redacted_query])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _redact_text(text: str) -> str:
    # redact() operates on structures; wrap and unwrap a single string.
    return redact({"v": text})["v"]


def compile_failures(traces: list[FailureTrace]) -> FECResult:
    """Compile failures into redacted, deduplicated eval items."""
    by_key: dict[str, EvalItem] = {}
    order: list[str] = []

    for trace in traces:
        red_query = _redact_text(trace.query)
        key = _dedup_key(trace.component, trace.failure_kind, red_query)
        if key in by_key:
            item = by_key[key]
            item.duplicate_count += 1
            if trace.task_id not in item.source_task_ids:
                item.source_task_ids.append(trace.task_id)
            continue
        by_key[key] = EvalItem(
            eval_id=f"eval_{key[:16]}",
            component=trace.component,
            failure_kind=trace.failure_kind,
            prompt=red_query,
            observed=_redact_text(trace.observed),
            expected=(
                _redact_text(trace.expected)
                if trace.expected is not None
                else None
            ),
            source_task_ids=[trace.task_id],
        )
        order.append(key)

    items = [by_key[k] for k in order]
    total = len(traces)
    unique = len(items)
    return FECResult(
        items=items,
        total_traces=total,
        unique_items=unique,
        conversion_rate=round(unique / total, 6) if total else 0.0,
    )
