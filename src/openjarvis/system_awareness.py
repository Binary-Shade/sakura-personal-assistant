"""Read-only local system awareness for Sakura."""

from __future__ import annotations

import getpass
import os
import platform
import socket
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


_DEFAULT_COMMON_DIRS = ("Desktop", "Documents", "Downloads", "Pictures", "Music", "Videos")


@dataclass(frozen=True, slots=True)
class DirectoryEntry:
    """Small, non-content metadata for one directory entry."""

    name: str
    kind: str
    size_bytes: int | None
    permissions: str
    owner: str
    modified: str


def _mode_string(path: Path) -> str:
    import stat

    try:
        return stat.filemode(path.stat().st_mode)
    except OSError:
        return "??????????"


def _owner(path: Path) -> str:
    try:
        import pwd

        return pwd.getpwuid(path.stat().st_uid).pw_name
    except Exception:
        return ""


def _is_sensitive_name(path: Path) -> bool:
    lowered = path.name.lower()
    sensitive_parts = (
        ".env",
        "token",
        "secret",
        "credential",
        "key",
        "password",
        "passwd",
        "id_rsa",
        "id_ed25519",
    )
    return any(part in lowered for part in sensitive_parts)


def list_directory(path: str | Path, *, limit: int = 50) -> dict[str, Any]:
    """Return bounded read-only metadata for a directory."""
    target = Path(path).expanduser().resolve()
    if not target.exists():
        return {"ok": False, "error": f"Path does not exist: {target}"}
    if not target.is_dir():
        return {"ok": False, "error": f"Path is not a directory: {target}"}

    entries: list[DirectoryEntry] = []
    hidden = 0
    skipped_sensitive = 0
    try:
        children = sorted(target.iterdir(), key=lambda item: item.name.lower())
    except OSError as exc:
        return {"ok": False, "error": f"Cannot list directory: {exc}"}

    for child in children:
        if child.name.startswith("."):
            hidden += 1
            continue
        if _is_sensitive_name(child):
            skipped_sensitive += 1
            continue
        if len(entries) >= limit:
            break
        try:
            stat_result = child.stat()
        except OSError:
            continue
        entries.append(
            DirectoryEntry(
                name=child.name,
                kind="directory" if child.is_dir() else "file",
                size_bytes=None if child.is_dir() else stat_result.st_size,
                permissions=_mode_string(child),
                owner=_owner(child),
                modified=datetime.fromtimestamp(stat_result.st_mtime).isoformat(
                    timespec="seconds"
                ),
            )
        )

    return {
        "ok": True,
        "path": str(target),
        "entries": [asdict(entry) for entry in entries],
        "shown": len(entries),
        "total_non_hidden_seen": max(0, len(children) - hidden),
        "hidden_skipped": hidden,
        "sensitive_skipped": skipped_sensitive,
        "truncated": len(entries) < max(0, len(children) - hidden - skipped_sensitive),
    }


def system_snapshot(*, cwd: str | Path | None = None) -> dict[str, Any]:
    """Return safe, read-only facts Sakura can use without tool calls."""
    home = Path.home()
    cwd_path = Path(cwd or os.getcwd()).expanduser().resolve()
    common_dirs: list[dict[str, Any]] = []
    for name in _DEFAULT_COMMON_DIRS:
        path = home / name
        common_dirs.append(
            {
                "name": name,
                "path": str(path),
                "exists": path.exists(),
                "readable": os.access(path, os.R_OK),
                "writable": os.access(path, os.W_OK),
            }
        )

    return {
        "user": getpass.getuser(),
        "home": str(home),
        "cwd": str(cwd_path),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "common_dirs": common_dirs,
        "permissions": {
            "read_directory_metadata": True,
            "read_file_contents": "only through explicit file tools",
            "write_files": "requires explicit write tool/action",
            "run_shell_commands": "requires explicit shell tool/action confirmation",
            "sensitive_files": "not included in automatic context",
        },
    }


def format_system_context(snapshot: dict[str, Any] | None = None) -> str:
    """Format a compact prompt section with safe local system facts."""
    data = snapshot or system_snapshot()
    dirs = []
    for item in data["common_dirs"]:
        status = "exists" if item["exists"] else "missing"
        access = []
        if item["readable"]:
            access.append("readable")
        if item["writable"]:
            access.append("writable")
        access_text = ", ".join(access) if access else "no direct access"
        dirs.append(f"- {item['name']}: {item['path']} ({status}; {access_text})")

    return "\n".join(
        [
            "## Local System Context",
            "",
            f"- Current user: {data['user']}",
            f"- Hostname: {data['hostname']}",
            f"- Local time: {data['time']}",
            f"- Platform: {data['platform']}",
            f"- Python: {data['python']}",
            f"- Home directory: {data['home']}",
            f"- Current working directory: {data['cwd']}",
            "- Common directories:",
            *dirs,
            "- Permissions: Sakura may use this context for basic awareness. "
            "For file contents, writes, or shell commands, use explicit tools/actions "
            "and respect configured confirmations and sensitive-file rules.",
        ]
    )


__all__ = ["format_system_context", "list_directory", "system_snapshot"]
