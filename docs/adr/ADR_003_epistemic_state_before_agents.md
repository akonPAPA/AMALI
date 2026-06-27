# ADR 003 — Epistemic State Before Agents

**Status:** Accepted

## Context

If agent orchestration is built before a rigorous notion of what is known,
inferred, or unknown, the system will confidently act on unverified text.
Multi-agent activity then amplifies hallucination instead of containing it.

## Decision

Build the **epistemic state layer first**: classified claims
(`EvidenceLevel`), explicit evidence with trust/sensitivity/risk, an
uncertainty score on every task, and a rule that a claim can only be
`KNOWN` when backed by supporting evidence. Agent orchestration, routing,
and verification are layered on top of this foundation, not before it.

## Consequences

- The system has a precise, testable representation of certainty before any
  agent acts.
- Generated text is structurally prevented from being treated as fact
  without evidence.
- Agent/routing/verification stages must consume and respect this state,
  which constrains (deliberately) how they are designed later.
