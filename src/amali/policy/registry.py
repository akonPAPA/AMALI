"""Local-only manifest registry.

Loads versioned tool manifests and the tool-permission policy from local
YAML files. There is no network, no remote fetch, and no implicit discovery
outside the configured directory.

The registry deliberately separates *parsing* from *validation*:

* ``get_tool`` returns a fully validated :class:`ToolManifest` (pydantic has
  already enforced the hard invariants at construction), or ``None`` if the
  tool id is unknown.
* If a manifest file is present but malformed, loading raises a
  :class:`ManifestValidationError` so the failure is explicit and audited by
  the caller — it never silently degrades to "allow".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from amali.policy.errors import (
    ManifestValidationError,
    PolicyValidationError,
)
from amali.policy.manifests import ToolManifest, ToolPermissionPolicy

__all__ = ["ManifestRegistry"]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ManifestValidationError(
            f"manifest file is not a mapping: {path.name}"
        )
    return data


class ManifestRegistry:
    """In-memory index of tool manifests and the active policy."""

    def __init__(self, tools: dict[str, ToolManifest], policy: ToolPermissionPolicy) -> None:
        self._tools = dict(tools)
        self._policy = policy

    # -- construction ----------------------------------------------------

    @classmethod
    def load_from_dir(cls, base_dir: str | Path) -> "ManifestRegistry":
        """Load all tool manifests and the single policy under ``base_dir``.

        Expected layout::

            base_dir/
              tools/*.yaml
              policies/*.yaml   (exactly one policy)
        """
        base = Path(base_dir)
        tools_dir = base / "tools"
        policies_dir = base / "policies"

        tools: dict[str, ToolManifest] = {}
        for path in sorted(tools_dir.glob("*.yaml")):
            data = _load_yaml(path)
            try:
                manifest = ToolManifest(**data)
            except Exception as exc:  # noqa: BLE001 - re-raise as domain error
                raise ManifestValidationError(
                    f"invalid tool manifest {path.name}: {exc}"
                ) from exc
            if manifest.tool_id in tools:
                raise ManifestValidationError(
                    f"duplicate tool_id across manifests: {manifest.tool_id}"
                )
            tools[manifest.tool_id] = manifest

        policy_files = sorted(policies_dir.glob("*.yaml"))
        if len(policy_files) != 1:
            raise PolicyValidationError(
                f"expected exactly one policy file in {policies_dir}, "
                f"found {len(policy_files)}"
            )
        policy_data = _load_yaml(policy_files[0])
        try:
            policy = ToolPermissionPolicy(**policy_data)
        except Exception as exc:  # noqa: BLE001 - re-raise as domain error
            raise PolicyValidationError(
                f"invalid policy {policy_files[0].name}: {exc}"
            ) from exc

        return cls(tools=tools, policy=policy)

    # -- accessors -------------------------------------------------------

    @property
    def policy(self) -> ToolPermissionPolicy:
        return self._policy

    def get_tool(self, tool_id: str) -> ToolManifest | None:
        """Return the manifest for ``tool_id`` or ``None`` if unknown."""
        return self._tools.get(tool_id)

    def tool_ids(self) -> list[str]:
        return sorted(self._tools)
