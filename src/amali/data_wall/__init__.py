"""Data Wall: flow-aware data admission control.

Every piece of data crossing between AMALI flows (runtime, eval, training,
benchmark, artifact) passes the wall first. The wall decides ALLOW, REDACT,
or BLOCK from a fixed operation matrix — data content never argues its way
through, and holdout data can never reach training.
"""

from amali.data_wall.wall import (
    DataFlow,
    DataWall,
    WallDecision,
    WallOutcome,
)

__all__ = ["DataWall", "DataFlow", "WallDecision", "WallOutcome"]
