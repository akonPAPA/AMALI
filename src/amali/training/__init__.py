"""Ready-to-train structures: manifests and contamination checks.

No training happens in this stage. These are the validated structures a
future, owner-approved training phase would consume — and the contamination
check that must pass before any of it is allowed.
"""

from amali.training.readiness import (
    DatasetManifest,
    HoldoutManifest,
    TrainingReadinessReport,
    check_contamination,
    training_readiness,
)

__all__ = [
    "DatasetManifest",
    "HoldoutManifest",
    "TrainingReadinessReport",
    "check_contamination",
    "training_readiness",
]
