"""Gate A component factory (WP8/WP11).

Builds the standard deterministic Gate A stack from local configuration:
the real manifest registry, a local-only route + model manifest for the
HER-MoE router, a DWAC pack catalogue, and an evidence corpus. No network,
no GPU, no torch — the "model" behind Gate A is the deterministic
evidence-quoting answerer.
"""

from __future__ import annotations

from pathlib import Path

from amali.audit.ledger import AuditLedger
from amali.activation.planner import DWACPlanner, PackSpec
from amali.model_gateway.enums import (
    Capability,
    ModelProvider,
    PrivacyMode,
    PromotionStage,
)
from amali.model_gateway.manifest import ModelManifest
from amali.policy.enums import DataSensitivity, RiskLevel
from amali.policy.registry import ManifestRegistry
from amali.retrieval.hsgr import EvidenceDoc
from amali.router.route import RequiredGates, Route, RouteBudget
from amali.router.router import HERMoERouter
from amali.runtime.orchestrator import RuntimeOrchestrator
from amali.state.models import TrustLevel

__all__ = [
    "build_gate_a_orchestrator",
    "default_corpus",
    "deterministic_model_manifest",
    "deterministic_route",
]

DETERMINISTIC_MODEL_ID = "deterministic_answerer_v0"
DETERMINISTIC_ROUTE_ID = "gate_a_deterministic_answer"


def deterministic_model_manifest() -> ModelManifest:
    """Manifest for the deterministic answerer, declared honestly as local."""
    return ModelManifest(
        model_id=DETERMINISTIC_MODEL_ID,
        version="0.1.0",
        provider=ModelProvider.LOCAL,
        model_type="generator",
        allowed_data_classes=[
            DataSensitivity.PUBLIC,
            DataSensitivity.INTERNAL,
        ],
        context_window=8192,
        max_tokens=1024,
        capabilities=[Capability.TEXT],
        privacy_mode=PrivacyMode.LOCAL_ONLY,
        license="amali-internal-deterministic",
        promotion_stage=PromotionStage.EVAL,
    )


def deterministic_route() -> Route:
    """The single Gate A route: local, bounded, verifier-gated."""
    return Route(
        route_id=DETERMINISTIC_ROUTE_ID,
        agents=["deterministic_answerer"],
        models=[DETERMINISTIC_MODEL_ID],
        tools=["local_readonly_filesystem"],
        budget=RouteBudget(
            max_steps=12,
            max_tool_calls=4,
            max_tokens=4096,
            max_wall_clock_seconds=60,
            max_cost_units=1.0,
        ),
        required_gates=RequiredGates(
            verifier=True,
            critic=False,
            security_review=False,
            human_review=False,
        ),
        max_task_risk=RiskLevel.L2,
        tool_risk=RiskLevel.L1,
    )


def default_packs() -> list[PackSpec]:
    return [
        PackSpec(
            pack_id="deterministic_answerer_pack",
            kind="base_model",
            cost_mb=64,
            capabilities=["answer_from_evidence"],
        ),
        PackSpec(
            pack_id="retrieval_pack",
            kind="expert_pack",
            cost_mb=32,
            capabilities=["retrieve"],
        ),
    ]


def default_corpus() -> list[EvidenceDoc]:
    """The built-in Gate A evidence corpus about AMALI itself."""
    return [
        EvidenceDoc(
            doc_id="audit_genesis",
            source_uri="local://docs/audit_genesis",
            content=(
                "The AMALI audit ledger genesis hash is the string GENESIS. "
                "Every audit event is chained with SHA-256 over the previous "
                "hash and the canonical event body."
            ),
            trust_level=TrustLevel.OFFICIAL,
            sensitivity=DataSensitivity.PUBLIC,
        ),
        EvidenceDoc(
            doc_id="trpc_default_deny",
            source_uri="local://docs/trpc_default_deny",
            content=(
                "The AMALI TRPC permission compiler is default deny. "
                "A tool operation is granted only when every hard check "
                "passes and an audit event is always emitted."
            ),
            trust_level=TrustLevel.OFFICIAL,
            sensitivity=DataSensitivity.PUBLIC,
        ),
        EvidenceDoc(
            doc_id="router_deterministic",
            source_uri="local://docs/router_deterministic",
            content=(
                "The HER-MoE router selects routes deterministically. "
                "Hard constraints exclude routes before scoring and an "
                "excluded route can never win."
            ),
            trust_level=TrustLevel.HIGH,
            sensitivity=DataSensitivity.PUBLIC,
        ),
    ]


def build_gate_a_orchestrator(
    manifests_dir: str | Path,
    *,
    ledger: AuditLedger | None = None,
    corpus: list[EvidenceDoc] | None = None,
) -> RuntimeOrchestrator:
    """Assemble the full deterministic Gate A stack."""
    ledger = ledger or AuditLedger()
    registry = ManifestRegistry.load_from_dir(manifests_dir)
    router = HERMoERouter(
        routes=[deterministic_route()],
        models={DETERMINISTIC_MODEL_ID: deterministic_model_manifest()},
        ledger=ledger,
    )
    planner = DWACPlanner(
        default_packs(), fallback_pack_id="deterministic_answerer_pack"
    )
    return RuntimeOrchestrator(
        ledger=ledger,
        registry=registry,
        router=router,
        planner=planner,
        corpus=corpus if corpus is not None else default_corpus(),
    )
