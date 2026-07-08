"""AMALI-DWAC: Dynamic Weight Activation Controller.

Plans the minimal sufficient set of model/adapter/expert packs for a task
under an explicit resource budget. Deterministic; no learned gating.
"""

from amali.activation.planner import (
    ActivationPlan,
    ActivationRequest,
    DWACPlanner,
    PackSpec,
)

__all__ = ["DWACPlanner", "PackSpec", "ActivationRequest", "ActivationPlan"]
