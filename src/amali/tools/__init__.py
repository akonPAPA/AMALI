"""Tool request models, grant checks, and the executor boundary.

This package contains no tool execution. It defines the request shape, the
grant-validity check, and the refusal gate that a future sandbox executor
must sit behind.
"""

from amali.tools.executor_boundary import BOUNDARY_RESULT, execute_tool
from amali.tools.grants import GrantCheck, check_grant
from amali.tools.requests import (
    GrantedScope,
    RequestedScope,
    ToolInvocationRequest,
)

__all__ = [
    "ToolInvocationRequest",
    "RequestedScope",
    "GrantedScope",
    "GrantCheck",
    "check_grant",
    "execute_tool",
    "BOUNDARY_RESULT",
]
