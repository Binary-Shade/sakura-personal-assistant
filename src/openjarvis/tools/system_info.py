"""Read-only system awareness tools."""

from __future__ import annotations

import json
from typing import Any

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.system_awareness import list_directory, system_snapshot
from openjarvis.tools._stubs import BaseTool, ToolSpec


@ToolRegistry.register("system_info")
class SystemInfoTool(BaseTool):
    """Return safe local system facts."""

    tool_id = "system_info"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="system_info",
            description=(
                "Return safe read-only facts about the local system: user, "
                "hostname, time, platform, home directory, cwd, and common "
                "directory access."
            ),
            parameters={"type": "object", "properties": {}},
            category="system",
        )

    def execute(self, **params: Any) -> ToolResult:
        data = system_snapshot()
        return ToolResult(
            tool_name="system_info",
            content=json.dumps(data, indent=2),
            success=True,
            metadata=data,
        )


@ToolRegistry.register("list_directory")
class ListDirectoryTool(BaseTool):
    """List directory metadata without reading file contents."""

    tool_id = "list_directory"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="list_directory",
            description=(
                "List bounded read-only metadata for a directory. Returns names, "
                "types, sizes, owners, permissions, and modified times; does not "
                "read file contents and skips hidden/sensitive entries."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to inspect.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum entries to return. Default 50.",
                    },
                },
                "required": ["path"],
            },
            category="filesystem",
        )

    def execute(self, **params: Any) -> ToolResult:
        path = str(params.get("path", "") or "")
        limit = int(params.get("limit", 50) or 50)
        data = list_directory(path, limit=max(1, min(limit, 200)))
        return ToolResult(
            tool_name="list_directory",
            content=json.dumps(data, indent=2),
            success=bool(data.get("ok")),
            metadata=data,
        )


__all__ = ["ListDirectoryTool", "SystemInfoTool"]
