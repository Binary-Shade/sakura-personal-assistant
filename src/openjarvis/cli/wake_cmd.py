"""Wake-word utilities for Sakura voice mode."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import click
from rich.console import Console

from openjarvis.core.config import get_config_dir


@click.group(help="Wake-word tools for Sakura voice mode.")
def wake() -> None:
    """Manage wake-word data and diagnostics."""


@wake.command("record-samples")
@click.option(
    "--kind",
    type=click.Choice(["positive", "negative"], case_sensitive=False),
    default="positive",
    show_default=True,
    help="Record wake phrase samples or non-wake/background samples.",
)
@click.option(
    "--count",
    type=int,
    default=25,
    show_default=True,
    help="Number of clips to record.",
)
@click.option(
    "--phrase",
    default="hey sakura",
    show_default=True,
    help="Wake phrase to show while recording positive samples.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory for samples. Defaults to ~/.openjarvis/wake_samples.",
)
def record_samples(
    kind: str,
    count: int,
    phrase: str,
    output_dir: Path | None,
) -> None:
    """Record WAV clips for custom openWakeWord training."""
    from openjarvis.speech.voice_io import record_until_silence, wav_duration_seconds

    console = Console(stderr=True)
    if count <= 0:
        raise click.BadParameter("count must be greater than zero")

    root = output_dir or (get_config_dir() / "wake_samples")
    target = root / kind.lower()
    target.mkdir(parents=True, exist_ok=True)

    if kind.lower() == "positive":
        console.print(
            f"[cyan]Recording {count} wake samples.[/cyan] Say "
            f"[bold]{phrase!r}[/bold] naturally after each prompt."
        )
    else:
        console.print(
            f"[cyan]Recording {count} negative samples.[/cyan] Use silence, "
            "room noise, normal speech, and similar-but-wrong phrases."
        )

    for index in range(1, count + 1):
        input(f"[{index}/{count}] Press Enter, then speak...")
        audio = record_until_silence(
            silence_seconds=0.7,
            startup_silence_seconds=6.0,
            max_seconds=3.5,
        )
        duration = wav_duration_seconds(audio)
        if duration <= 0:
            console.print("[yellow]Nothing heard; skipping this clip.[/yellow]")
            continue

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = target / f"{kind.lower()}-{stamp}.wav"
        path.write_bytes(audio)
        console.print(f"[green]Saved[/green] {path} ({duration:.2f}s)")

    console.print(f"[cyan]Samples saved under:[/cyan] {root}")


__all__ = ["wake"]
