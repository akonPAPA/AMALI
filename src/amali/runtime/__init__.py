"""Gate A runtime: deterministic evidence-only answerer and orchestrator."""

from amali.runtime.answerer import AnswerCandidate, DeterministicAnswerer
from amali.runtime.orchestrator import GateRunResult, RuntimeOrchestrator

__all__ = [
    "DeterministicAnswerer",
    "AnswerCandidate",
    "RuntimeOrchestrator",
    "GateRunResult",
]
