"""Chat startup banner — renders BgStatus as a one-line status string."""

from __future__ import annotations

import os

from openjarvis.cli._bg_state import BgStatus


def render_startup_banner(status: BgStatus) -> str:
    """Return a single-line banner string, or empty if all background work is done."""
    if (
        os.environ.get("SAKURA_HIDE_BACKGROUND_BANNER")
        or os.environ.get("OPENJARVIS_HIDE_BACKGROUND_BANNER")
    ) == "1":
        return ""

    if status.all_ready():
        return ""

    parts: list[str] = []

    if status.rust_extension == "pending":
        parts.append("Rust extension building")
    elif status.rust_extension == "failed":
        parts.append("⚠ Rust extension failed (run `jarvis doctor`)")

    for model_id, state in status.models.items():
        if state == "downloading":
            parts.append(f"{model_id} downloading")
        elif state == "failed":
            parts.append(f"⚠ {model_id} failed (run `jarvis doctor`)")
        # 'ready' models don't show in the banner.

    if not parts:
        return ""
    return "Background: " + " · ".join(parts) + ". Type to start."
